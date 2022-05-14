#!/usr/bin/env python

import re
from itertools import permutations
import numpy as np
np.set_printoptions(linewidth=400)
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})  # enlarge matplotlib fonts
# Import qubit states Zero (|0>) and One (|1>), and Pauli operators (X, Y, Z)
from qiskit.opflow import Zero, One, I, X, Y, Z, commutator
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
from copy import deepcopy

# Importing standard Qiskit modules
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, IBMQ, execute, transpile, schedule, assemble, Aer
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.monitor import job_monitor
from qiskit.circuit import Parameter, Gate
import qiskit.quantum_info as qi
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter

# Import state tomography modules
from qiskit.ignis.verification.tomography import process_tomography_circuits, state_tomography_circuits, StateTomographyFitter, ProcessTomographyFitter
from qiskit.quantum_info import state_fidelity
from qiskit.ignis.verification.tomography.fitters.lstsq_fit import make_positive_semidefinite
from qiskit.quantum_info.operators.predicates import is_positive_semidefinite_matrix

###### CR gate pulse
from qiskit import pulse, circuit
import qiskit.pulse.library as pulse_lib
from qiskit.pulse import InstructionScheduleMap
from qiskit.pulse.channels import PulseChannel
from scipy.special import erf

# the target problem
ITSS = [Zero^Zero^Zero, Zero^Zero^One, Zero^One^Zero, Zero^One^One, One^Zero^Zero, One^Zero^One, One^One^Zero, One^One^One]
initial_state = One^One^Zero # 6

def get_backend(mode='jakarta', real=False):
    print('getting backend:', mode)
    # load IBMQ Account data
    # IBMQ.save_account(TOKEN)  # replace TOKEN with your API token string (https://quantum-computing.ibm.com/lab/docs/iql/manage/account/ibmq)
    provider = IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-community', group='ibmquantumawards', project='open-science-22')
    jakarta = provider.get_backend('ibmq_jakarta')
    # Simulated backend based on ibmq_jakarta's device noise profile
    if real:
        sim = jakarta
    else:
        sim = QasmSimulator.from_backend(provider.get_backend('ibmq_jakarta'))
    return sim


#######################################################
# MISC utilities
#######################################################
def _get_state_str(s0):
    # simle utility to get the string for a quantum state
    if isinstance(s0, int):
        return str(s0)
    assert len(s0.primitive.keys())==1
    ss = list(s0.primitive.keys())[0]
    return ss
def get_circuit_name(target_time, trotter_steps, order, hs, c2r, dphi, ncx, prefix='fc'):
    if target_time<=1e-10:
        target_time = 0.0
    return "{}_{}_s{}_o{}_n{}_t{:.2f}".format(prefix, hs, trotter_steps, order, ncx, target_time)
def _clean_cdict(cdict, nbit=3):
    if len(cdict)==0:
        return cdict
    k0 = list(cdict.keys())[0]
    if '0x' not in k0:
        return cdict
    d = dict()
    for k,v in cdict.items():
        i = int(k, 16)
        if nbit==3:
            d['{0:03b}'.format(i)] = v
        else:
            d['{0:04b}'.format(i)] = v
    return d
#######################################################
# end MISC utilities
#######################################################

#######################################################
# Utility function to scale the cross-resonance pulses.
# The used equations are the same as in RZXCalibrationBuilder
#######################################################
def _gssq_area_edge(duration, amp, sigma, width):
    amp = np.abs(amp)
    nsigma = (duration-width)/sigma
    return amp*sigma*np.sqrt(2*np.pi)*erf(nsigma)

def _gssq_area(duration, amp, sigma, width):
    amp = np.abs(amp)
    return amp*width + _gssq_area_edge(duration, amp, sigma, width)

def _scale_gssq_wa(theta, duration, amp, sigma, width):
    aref = _gssq_area(duration, amp, sigma, width)
    aedg = _gssq_area_edge(duration, amp, sigma, width)
    atgt = aref * theta/(np.pi/2)
    if atgt >= aedg:
        w = (atgt - aedg) / np.abs(amp)
        d = w//16
        r = w%d
        if r==0:
            return int(w), amp
        elif r>8:
            return int((d+1)*16), amp
        else:
            return int(d*16), amp
    a = atgt/aedg * amp
    return int(0), a
#######################################################
# end cr pulse rescaling utility.
#######################################################


#######################################################
# Utility to general double cross resonance pulse, Rzxz gate
# It implements H = ZXI + IZX, U=exp(-i*theta*H)
# on three qubits.
#######################################################
def _cr_pulse(bkcfg, qtinfo, qcinfo):
    # This function generate a CR pulse with one target and two controls
    def _gssq_pulse(duration, amp, phi, sigma, width, name=None):
        # Build the GaussianSquare pulse
        amp = amp*np.exp(1j*phi)
        return pulse_lib.GaussianSquare(duration, amp, sigma, width=width, name=name, limit_amplitude=False)
    def _cr_control_pulse(u,d,p):
        # Apply CR pulse on control qubit.
        # u: control channel; d: drive channel; p: pulse parameters
        with pulse.build() as schd:
            pulse.play(_gssq_pulse(**p), u)
            pulse.delay(p['duration'], d)
        return schd
    def _cr_cancellation_pulse(d,p):
        # Apply CR pulse on target qubit.
        # d: drive channel; p: pulse parameters
        with pulse.build() as schd:
            if p['amp']==0:
                pulse.delay(p['duration'], d)
            else:
                pulse.play(_gssq_pulse(**p), d)
        return schd
    # Pulse on target, a signle target qubit 
    qtid = qtinfo.get('id')
    qtgssq = qtinfo.get('pulse')
    qtdch = bkcfg.drive(qtid)
    schdt = _cr_cancellation_pulse(qtdch, qtgssq)
    # Pulse on control, two control qubits
    if isinstance(qcinfo, dict):
        qcinfo = [qcinfo]
    with pulse.build() as schdc:
        for f in qcinfo: # iterate over two control qubits
            qcid = f.get('id')
            qcgssq = f.get('pulse')
            qcdch = bkcfg.drive(qcid)
            qcuch = bkcfg.control((qcid, qtid))[0]
            schd = _cr_control_pulse(qcuch, qcdch, qcgssq)
            schdc += schd
    # combine target/control pulses together
    schd = schdt + schdc
    return schd

def _cinv_pulse(bkcfg, istr2schd, qtinfo, qcinfo, invs):
    # Add inverse pulses for echo
    # invs: defines which control qubit get inversed.
    if not np.all(invs):
        return pulse.Schedule()
    assert len(qcinfo) == len(invs)
    duration = 0
    schdc = []
    # Apply the inverse pulse
    for iv,qc in zip(invs, qcinfo):
        with pulse.build() as s:
            if iv:
                pulse.call(istr2schd.get('x', qc['id']))
                duration = max(duration, s.duration)
        schdc.append(s)
    # Add delay if necessarily according to the max delay
    # of the inverse pulse.
    with pulse.build() as schd:
        pulse.delay(duration, bkcfg.drive(qtinfo['id']))
        for (qc,iv,s) in zip(qcinfo,invs,schdc):
            iq = qc['id']
            dd = duration - s.duration
            if dd>0:
                pulse.delay(dd, bkcfg.drive(iq))
            schd += s
    return schd

def _create_cr_schedule(backend, qubits, **params):
    assert len(qubits)>=2 and len(qubits)<=3
    bkcfg = backend.configuration()
    bkdft = backend.defaults()
    istr2schd = bkdft.instruction_schedule_map

    # Build the information dict for target/control qubits
    # according to the given parameters.
    # Use default values if no parameters given
    qtinfo = dict()
    qtinfo['id'] = qubits[0]
    p = dict()
    if len(params)==0 or 'ampt' not in params:
        a = 0.06553534967248906-0.0017696407991610746j
        p['amp'] = np.abs(a)
        p['phi'] = np.angle(a)
        p['sigma'] = 64
        p['duration'] = 704
        p['width'] = 448
    else:
        if 'phit' in params:
            p['amp'] = params['ampt']
            p['phi'] = params['phit']
        else:
            a = params['ampt']
            p['amp'] = np.abs(a)
            p['phi'] = np.angle(a)
        p['sigma'] = params['sigmat']
        p['duration'] = params['durationt']
        p['width'] = params['widtht']
    qtinfo['pulse'] = p

    qcf = dict()
    qcf['id'] = qubits[1]
    p = dict()
    if len(params)==0 or 'ampc0' not in params:
        a = -0.6246826162868288+0.5038482054826277j
        p['amp'] = np.abs(a)
        p['phi'] = np.angle(a)
        p['sigma'] = 64
        p['duration'] = 704
        p['width'] = 448
    else:
        if 'phit' in params:
            p['amp'] = params['ampc0']
            p['phi'] = params['phic0']
        else:
            a = params['ampc0']
            p['amp'] = np.abs(a)
            p['phi'] = np.angle(a)
        p['sigma'] = params['sigmac0']
        p['duration'] = params['durationc0']
        p['width'] = params['widthc0']
    qcf['pulse'] = p
    qcinfo = [qcf]

    if len(qubits)==3:
        qcf = dict()
        qcf['id'] = qubits[2]
        p = dict()
        if len(params)==0 or 'ampc1' not in params:
            a = -0.6246826162868288+0.5038482054826277j
            p['amp'] = np.abs(a)
            p['phi'] = np.angle(a)
            p['sigma'] = 64
            p['duration'] = 704
            p['width'] = 448
        else:
            if 'phit' in params:
                p['amp'] = params['ampc1']
                p['phi'] = params['phic1']
            else:
                a = params['ampc1']
                p['amp'] = np.abs(a)
                p['phi'] = np.angle(a)
            p['sigma'] = params['sigmac1']
            p['duration'] = params['durationc1']
            p['width'] = params['widthc1']
        qcf['pulse'] = p
        qcinfo.append(qcf)

    # First-half cr pulse
    schd1 = _cr_pulse(bkcfg, qtinfo, qcinfo)
    # Inversion  pulse
    schdinv = _cinv_pulse(bkcfg, istr2schd, qtinfo, qcinfo, [1]*(len(qubits)-1))
    qtinfo['pulse']['phi'] += np.pi
    for qcf in qcinfo:
        qcf['pulse']['phi'] += np.pi
    # Second-half cr pulse
    schd2 = _cr_pulse(bkcfg, qtinfo, qcinfo)
    # Restore parameters
    qtinfo['pulse']['phi'] -= np.pi
    for qcf in qcinfo:
        qcf['pulse']['phi'] -= np.pi
    # Assemble final schedule
    with pulse.build() as schd:
        schd += schd1
        schd += schdinv
        schd += schd2
        schd += schdinv
    return schd
#######################################################
# end double cross resonance pulse schedule
#######################################################


#######################################################
# Utility to get CR pulse parameters from backend
# and formulate the parameter sets for the 
# 3-qubit double cross resonance pulse schedules
#######################################################
def _get_cr_params(bk, qubits): # [c,t]
    # Get the tuned CR pulse from backend's cx gate
    cxs = bk.defaults().instruction_schedule_map.get('cx',qubits)
    waveu = None
    waved = None
    for s in cxs.instructions:
        if not hasattr(s[1],'pulse'):
            continue
        wave = s[1].pulse
        if not isinstance(wave, pulse_lib.GaussianSquare):
            continue
        c = s[1].channel
        if isinstance(c, pulse.DriveChannel):
            if c.index!=qubits[1]:
                print("!! CR reverted: dc.index={}, qubits={}".format(c.index, qubits))
            waved = wave
        elif isinstance(c, pulse.ControlChannel):
            waveu = wave
        if waveu is not None and waved is not None:
            break
    assert waveu is not None and waved is not None
    return waveu.parameters, waved.parameters
def _get_pnvals_cr(bk, qubits):
    # Get the backend's tune CR pulse parameters
    # and formulate the dict
    pc, pt = _get_cr_params(bk, qubits)
    pvals = dict()
    pvals['ampt'] = np.abs(pt['amp'])
    pvals['phit'] = np.angle(pt['amp'])
    pvals['sigmat'] = pt['sigma']
    pvals['durationt'] = int(pt['duration'])
    pvals['widtht'] = int(pt['width'])
    pvals['ampc0'] = np.abs(pc['amp'])
    pvals['phic0'] = np.angle(pc['amp'])
    pvals['sigmac0'] = int(pc['sigma'])
    pvals['durationc0'] = int(pc['duration'])
    pvals['widthc0'] = int(pc['width'])
    return pvals
def _add_pnvals_c1(pvals, c2r=1, dphi=0):
    # Add the parameters for the second control qubit
    # Since the backend's cx gates are directional in respect of
    # CR pulse, i.e. there is no CR pulse avaliable for 
    # qubit 5 controlling 3, but only 3 controlling 5.
    # So the second control's pulse parameters are set in 
    # reference to the first one, by scaling the amplitude 
    # possibly a phase shift.
    pvals['ampc1'] = pvals['ampc0'] * c2r
    pvals['phic1'] = pvals['phic0'] + dphi
    pvals['sigmac1'] = pvals['sigmac0']
    pvals['durationc1'] = pvals['durationc0']
    pvals['widthc1'] = pvals['widthc0']
    # target cancellation
    pvals['ampt'] *= (1+c2r)
    return pvals
def get_pnvals(bk, qubits, theta, c2r=1, dphi=0):
    # Build the parameter set for double CR gate
    # based on backend's cx gate
    if not bk.configuration().open_pulse:
        return None
    pnvals = _get_pnvals_cr(bk, [qubits[1], qubits[0]]) # [c,t]
    if len(qubits)>2: # add the parameter for the second control qubit
        pnvals = _add_pnvals_c1(pnvals, c2r, dphi)
    _scale_params(pnvals, theta)
    return pnvals
def _get_max_amp_ratio(bk, qubits):
    # This function helps to make sure the amplitude would not 
    # exceed 1 when scaled for the second qubit
    up, dp = _get_cr_params(bk, qubits)
    ua = up['amp']
    da = dp['amp']
    r = 1/abs(ua)*0.999
    if abs(da)>0 and abs(da)*(1+r)>1:
        r = 1/abs(da)-1
    print('q {}: u amp ({}) {}, d amp ({}) {}, max r={}'.format(qubits, abs(ua), ua, abs(da), da, r))
    return r
def _scale_params(p, theta):
    # Set the pulse parameters to achieve a specific rotation angle theta,
    # based on the parameters extracted from cx gates.
    w, a = _scale_gssq_wa(theta, p['durationt'], p['ampt'], p['sigmat'], p['widtht'])
    p['durationt'] = p['durationt'] - (p['widtht']-w)
    p['widtht'] = w
    p['ampt'] = a
    w, a = _scale_gssq_wa(theta, p['durationc0'], p['ampc0'], p['sigmac0'], p['widthc0'])
    p['durationc0'] = p['durationc0'] - (p['widthc0']-w)
    p['widthc0'] = w
    p['ampc0'] = a
    if p.get('ampc1') is not None:
        w, a = _scale_gssq_wa(theta, p['durationc1'], p['ampc1'], p['sigmac1'], p['widthc1'])
        p['durationc1'] = p['durationc1'] - (p['widthc1']-w)
        p['widthc1'] = w
        p['ampc1'] = a
def create_cr_gate_schd2(backend, qubits, params, name):
    # Build the proper double CR gate ready to be added to a quantumcircuit
    # Return both itself and the cancellation version (*dagger)
    cn = name + ''.join(str(i) for i in qubits)
    cg = dict()
    cg['qubits'] = deepcopy(qubits)
    cg['schedule'] = _create_cr_schedule(backend, qubits, **params)
    cg['gate'] = Gate(cn, len(qubits), [])
    cgd = dict()
    paramsd = deepcopy(params)
    paramsd['phit'] -= np.pi
    paramsd['phic0'] -= np.pi
    paramsd['phic1'] -= np.pi
    cgd['qubits'] = deepcopy(qubits)
    cgd['schedule'] = _create_cr_schedule(backend, qubits, **paramsd)
    cgd['gate'] = Gate(cn+'_dag', len(qubits), [])
    return cg, cgd
#######################################################
# end utilities for double cross resonance gate parameters
#######################################################


#######################################################
# Utilities to build the trotterized circuit
#######################################################
def add_one_trotter_step(bk, qc, qs, cg1, cg1d, cg2, cg2d, tval, hs, order, nr): # t,c,c
    # !!! this is a duplicate for _add_one_trotter_step to be used in jupyternotebook.
    # !!! Only change is to remove the back-to-back rx/ry/rz for illustration.
    # !!! The originl is the same since those would be removed in transpile optimization.
    # Add one trotter step to the give quantum circuit 'qc'
    # qs are the three qubits in order of target, control1, control2
    # cg1/2 are double CR gates with theta=0.5*tval and theta=tval.
    # order=1, 2: is the trotterization order.
    # order=0, -1, -2: are for testing simple double CR gate and circuit based 
    #                  solutions.
    # hs: defines the order of the Hamiltonian, XXI+IXX, YYI+IYY, ZZI+IZZ
    # nr=0,1: is the number of double CR gates for Zero Noise Extrapolation,
    #         for 1, each cg = cg^cgd^cg
    def _add_rzzx(bk, qc, qs, cg, cgd, tval):
        if not bk.configuration().open_pulse:
            qc.rzx(tval*2, qs[1], qs[0])
            qc.rzx(tval*2, qs[2], qs[0])
        else:
            qc.append(cg['gate'], cg['qubits'])
            if cgd is not None:
                qc.append(cgd['gate'], cgd['qubits'])
                qc.append(cg['gate'], cg['qubits'])
    def _addx(bk, qc, qs, cg, cgd, tval):
        qc.ry(-np.pi/2, [qs[1], qs[2]])
        _add_rzzx(bk, qc, qs, cg, cgd, tval)
        qc.ry(np.pi/2, [qs[1], qs[2]])
    def _addy(bk, qc, qs, cg, cgd, tval):
        qc.rx(np.pi/2, [qs[1], qs[2]])
        qc.rz(-np.pi/2, qs[0])
        _add_rzzx(bk, qc, qs, cg, cgd, tval)
        qc.rz(np.pi/2, qs[0])
        qc.rx(-np.pi/2, [qs[1], qs[2]])
    def _addz(bk, qc, qs, cg, cgd, tval):
        qc.ry(np.pi/2, qs[0])
        _add_rzzx(bk, qc, qs, cg, cgd, tval)
        qc.ry(-np.pi/2, qs[0])
    def _addh(bk, qc, qs, cg, cgd, tval, h):
        if h=='x':
            _addx(bk, qc, qs, cg, cgd, tval)
        elif h=='y':
            _addy(bk, qc, qs, cg, cgd, tval)
        elif h=='z':
            _addz(bk, qc, qs, cg, cgd, tval)
    def _append_xyz(qc, XX, YY, ZZ, qr, p):
        for i in p:
            if i==0:
                qc.append(XX, [qr[0], qr[1]])
            elif i==1:
                qc.append(YY, [qr[0], qr[1]])
            elif i==2:
                qc.append(ZZ, [qr[0], qr[1]])
            elif i==3:
                qc.append(XX, [qr[0], qr[2]])
            elif i==4:
                qc.append(YY, [qr[0], qr[2]])
            elif i==5:
                qc.append(ZZ, [qr[0], qr[2]])
            else:
                assert 0
    if order==1: # first order trotterization
        for h in hs:
            _addh(bk, qc, qs, cg2, (cg2d if nr==1 else None), tval, h)
    elif order==2: # second order trotterization
        N = len(hs)
        for i in range(N-1):
            _addh(bk, qc, qs, cg1, (cg1d if nr==1 else None), tval/2, hs[i])
        _addh(bk, qc, qs, cg2, (cg2d if nr==1 else None), tval, hs[N-1])
        for i in range(N-1):
            _addh(bk, qc, qs, cg1, (cg1d if nr==1 else None), tval/2, hs[-2-i])
    elif order==0: # simple double CR for testing
        _add_rzzx(bk, qc, qs, cg2, None, tval)
    elif order==-1 or order==-2: # circuit based, not using pulse
        xx, yy, zz = get_xyz_qc(tval if order==-1 else tval/2)
        p = []
        for h in hs:
            if h=='x':
                p += [0,3]
            elif h=='y':
                p += [1,4]
            elif h=='z':
                p += [2,5]
        _append_xyz(qc, xx, yy, zz, qs, p)
        if order==-2:
            _append_xyz(qc, xx, yy, zz, qs, list(reversed(p)))
    else:
        assert 0
def _add_one_trotter_step(bk, qc, qs, cg1, cg1d, cg2, cg2d, tval, hs, order, nr): # t,c,c
    # Add one trotter step to the give quantum circuit 'qc'
    # qs are the three qubits in order of target, control1, control2
    # cg1/2 are double CR gates with theta=0.5*tval and theta=tval.
    # order=1, 2: is the trotterization order.
    # order=0, -1, -2: are for testing simple double CR gate and circuit based 
    #                  solutions.
    # hs: defines the order of the Hamiltonian, XXI+IXX, YYI+IYY, ZZI+IZZ
    # nr=0,1: is the number of double CR gates for Zero Noise Extrapolation,
    #         for 1, each cg = cg^cgd^cg
    def _add_rzzx(bk, qc, qs, cg, tval):
        if not bk.configuration().open_pulse:
            qc.rzx(tval*2, qs[1], qs[0])
            qc.rzx(tval*2, qs[2], qs[0])
        else:
            qc.append(cg['gate'], cg['qubits'])
    def _addx(bk, qc, qs, cg, tval):
        qc.ry(-np.pi/2, [qs[1], qs[2]])
        _add_rzzx(bk, qc, qs, cg, tval)
        qc.ry(np.pi/2, [qs[1], qs[2]])
    def _addy(bk, qc, qs, cg, tval):
        qc.rx(np.pi/2, [qs[1], qs[2]])
        qc.rz(-np.pi/2, qs[0])
        _add_rzzx(bk, qc, qs, cg, tval)
        qc.rz(np.pi/2, qs[0])
        qc.rx(-np.pi/2, [qs[1], qs[2]])
    def _addz(bk, qc, qs, cg, tval):
        qc.ry(np.pi/2, qs[0])
        _add_rzzx(bk, qc, qs, cg, tval)
        qc.ry(-np.pi/2, qs[0])
    def _addh(bk, qc, qs, cg, tval, h):
        if h=='x':
            _addx(bk, qc, qs, cg, tval)
        elif h=='y':
            _addy(bk, qc, qs, cg, tval)
        elif h=='z':
            _addz(bk, qc, qs, cg, tval)
    def _append_xyz(qc, XX, YY, ZZ, qr, p):
        for i in p:
            if i==0:
                qc.append(XX, [qr[0], qr[1]])
            elif i==1:
                qc.append(YY, [qr[0], qr[1]])
            elif i==2:
                qc.append(ZZ, [qr[0], qr[1]])
            elif i==3:
                qc.append(XX, [qr[0], qr[2]])
            elif i==4:
                qc.append(YY, [qr[0], qr[2]])
            elif i==5:
                qc.append(ZZ, [qr[0], qr[2]])
            else:
                assert 0
    if order==1: # first order trotterization
        for h in hs:
            _addh(bk, qc, qs, cg2, tval, h)
            if nr==1:
                _addh(bk, qc, qs, cg2d, -tval, h)
                _addh(bk, qc, qs, cg2, tval, h)
    elif order==2: # second order trotterization
        N = len(hs)
        for i in range(N-1):
            _addh(bk, qc, qs, cg1, tval/2, hs[i])
            if nr==1:
                _addh(bk, qc, qs, cg1d, -tval/2, hs[i])
                _addh(bk, qc, qs, cg1, tval/2, hs[i])
        _addh(bk, qc, qs, cg2, tval, hs[N-1])
        if nr==1:
            _addh(bk, qc, qs, cg2d, -tval, hs[N-1])
            _addh(bk, qc, qs, cg2, tval, hs[N-1])
        for i in range(N-1):
            _addh(bk, qc, qs, cg1, tval/2, hs[-2-i])
            if nr==1:
                _addh(bk, qc, qs, cg1d, -tval/2, hs[i])
                _addh(bk, qc, qs, cg1, tval/2, hs[i])
    elif order==0: # simple double CR for testing
        _add_rzzx(bk, qc, qs, cg2, tval)
    elif order==-1 or order==-2: # circuit based, not using pulse
        xx, yy, zz = get_xyz_qc(tval if order==-1 else tval/2)
        p = []
        for h in hs:
            if h=='x':
                p += [0,3]
            elif h=='y':
                p += [1,4]
            elif h=='z':
                p += [2,5]
        _append_xyz(qc, xx, yy, zz, qs, p)
        if order==-2:
            _append_xyz(qc, xx, yy, zz, qs, list(reversed(p)))
    else:
        assert 0
def _add_full_trotter(bk, qc, qubits, params1, params2, trotter_steps, tval, hs, order, nr):
    # add full trotterization circuit to quantum circuit qc
    if not bk.configuration().open_pulse:
        cg1 = None
        cg2 = None
        cg1d = None
        cg2d = None
    else:
        cg1, cg1d = create_cr_gate_schd2(bk, qubits, params1, 'cr1')
        cg2, cg2d = create_cr_gate_schd2(bk, qubits, params2, 'cr2')
        qc.add_calibration(cg1['gate'].name, cg1['qubits'], cg1['schedule'])
        qc.add_calibration(cg2['gate'].name, cg2['qubits'], cg2['schedule'])
        qc.add_calibration(cg1d['gate'].name, cg1d['qubits'], cg1d['schedule'])
        qc.add_calibration(cg2d['gate'].name, cg2d['qubits'], cg2d['schedule'])
    for i in range(trotter_steps):
        _add_one_trotter_step(bk, qc, qubits, cg1, cg1d, cg2, cg2d, tval, hs, order, nr)
    return cg1, cg1d, cg2, cg2d
def _add_init_qc(qc, qr, qubits, s0):
    # Add gates to initialize qubits to the required initial state
    ss = _get_state_str(s0)
    nq = len(ss)
    assert nq==len(qubits)
    qs = np.sort(qubits).tolist()
    ss = ss[::-1] # revert for little endian
    for i in range(nq):
        if ss[i]=='1':
            qc.x(qr[qs[i]])
def _add_meas(qc, qubits, meas, cr=None):
    # Add measurements at the end of quantum circuit
    assert len(meas)==len(qubits)
    qs = np.sort(qubits).tolist()
    if cr is None:
        cr = ClassicalRegister(len(qubits))
        qr = qc.qubits
        qc.add_register(cr)
    else:
        qr = qc.qubits
        assert len(cr) == len(qubits)
    for q,m in zip(qs, meas):
        if m=='x' or m=='X':
            qc.h(q)
        elif m=='y' or m=='Y':
            qc.sdg(q)
            qc.h(q)
    qm = [qr[i] for i in qs]
    qc.measure(qm, cr)
def _make_stomo_circuits(bk, qc0, name, do_parity):
    # A workaround for state_tomography_circuits(),
    # to allow for adding a parity bit measurement which
    # does not go into tomography
    smeas = ['X', 'Y', 'Z']
    qms = [1,3,5]
    qp = [6] if do_parity else []
    Nc = len(qms)+len(qp)
    qcs = []
    for m1 in smeas:
        for m2 in smeas:
            for m3 in smeas:
                mstr = "('{}', '{}', '{}')".format(m1, m2, m3)
                qcname = name + '_st' + mstr
                qr = QuantumRegister(bk.configuration().num_qubits)
                cr = ClassicalRegister(Nc)
                qc = QuantumCircuit(qr, cr, name=qcname)
                qc.append(qc0, qr)
                qc = qc.decompose()
                qc.barrier(qr)
                _add_meas(qc, qms, m1+m2+m3, cr[0:3])
                if do_parity:
                    qc.measure(6, cr[3])
                qcs.append(qc)
    return qcs
def _build_full_test_circuit(bk, target_time, trotter_steps, qubits, s0, meas, c2r, dphi, hs='xzy', order=2, nr=0, optlevel=None, stomo=False, name='fc'):
    # This returns the whole circuit to be run on a device,
    # for measurements of final state or state tomography
    # The circuit(s) are transpiled.
    theta = target_time/trotter_steps
    assert isinstance(theta, float)
    assert isinstance(meas, str) or stomo
    #name = _get_circ_name(s0, meas, target_time, c2r, dphi, nr, name)
    name = get_circuit_name(target_time, trotter_steps, order, hs, c2r, dphi, nr, name)
    pvals1 = get_pnvals(bk, qubits, theta, c2r, dphi)
    pvals2 = get_pnvals(bk, qubits, theta*2, c2r, dphi)
    qr = QuantumRegister(bk.configuration().num_qubits)
    qc = QuantumCircuit(qr, name=name)
    _add_init_qc(qc, qr, qubits, s0)
    cg1, cg1d, cg2, cg2d = _add_full_trotter(bk, qc, qubits, pvals1, pvals2, trotter_steps, theta, hs, order, nr)
    if stomo:
        qc = _make_stomo_circuits(bk, qc, name, False)
        if cg1 is not None:
            for q in qc:
                q.add_calibration(cg1['gate'].name, cg1['qubits'], cg1['schedule'])
                q.add_calibration(cg2['gate'].name, cg2['qubits'], cg2['schedule'])
                q.add_calibration(cg1d['gate'].name, cg1d['qubits'], cg1d['schedule'])
                q.add_calibration(cg2d['gate'].name, cg2d['qubits'], cg2d['schedule'])
    else:
        _add_meas(qc, qubits, meas)
    qct = transpile(qc, bk, optimization_level=optlevel)
    return qct
#######################################################
# end Trotterization circuit building utilities
#######################################################

#######################################################
# Utilities to generate experiment circuit sets
#######################################################
def gen_ccr_calibration_circs(bk, target_time, trotter_steps, qubits, initial_state, c2r):
    qcs = []
    tval = target_time/trotter_steps
    for c in c2r:
        qct = _build_full_test_circuit(bk, tval, 1, qubits, initial_state, meas='zzz', c2r=c, dphi=0, order=0, nr=0, optlevel=None, stomo=False, name='ccr')
        qcs.append(qct)
    return qcs
def gen_tomo_experiments(bk, target_time, trotter_steps, qubits, initial_state, order, hs, c2r, dphi, nr):
    # Get all the circuits state for tomography
    # nr: a list for repeatation of double CR gates for ZNE.
    qcs = []
    for c in c2r:
        for n in nr:
            qct = _build_full_test_circuit(bk, target_time, trotter_steps, qubits, initial_state, meas=None, c2r=c, dphi=dphi, hs=hs, order=order, nr=n, optlevel=None, stomo=True)
            qcs += qct
    return qcs
def _gen_mitigation_circs(bk, qubits, trotter_steps, hs, order, nr):
    # Make cricuits for error mitigation.
    # Those circuits aligns with the trotterization setting, but without
    # all coupling pulses between any qubits, i.e. tval = 0.
    # It is to mitigate single qubit noises
    if bk.configuration().open_pulse:
        tval = 0
    else:
        tval = 1e-10
    pvals = get_pnvals(bk, qubits, tval, 1, 0)
    qcs = []
    stls = []
    for s in ITSS:
        assert len(s.primitive.keys())==1
        sl = list(s.primitive.keys())[0]
        stls.append(sl)
        qct = _build_full_test_circuit(bk, tval, trotter_steps, qubits, s, 'zzz', c2r=1, dphi=0, hs=hs, order=order, nr=nr, optlevel=None, stomo=False, name='mit')
        qct.name = qct.name + "cal_" + sl # to match with qiskit internel code
        qcs.append(qct)
    return qcs, stls
def gen_mitigation_circs(bk, qubits, trotter_steps, hs, order, nr):
    qcs = []
    for n in nr:
        qc, stls = _gen_mitigation_circs(bk, qubits, trotter_steps, hs, order, n)
        qcs += qc
    return qcs, stls
#######################################################
# end experiment circuit generation utilities
#######################################################


#######################################################
# Utilities to run jobs
#######################################################
def submit_expr(bk, qc, shots=8192, jname=None, rep=1):
    print("total circs: {}".format(len(qc)))
    assert len(qc) < bk.configuration().max_experiments
    jobs = []
    for i in range(rep):
        qobj = assemble(qc, bk, shots=shots)
        job = bk.run(qobj, job_name=jname)
        #job = execute(qc, bk, shots=shots)
        print('job id: {}'.format(job.job_id()))
        jobs.append(job)
    for job in jobs:
        job_monitor(job)
        try:
            if job.error_message() is not None:
                print(job.error_message())
        except:
            pass
    return jobs
def retrieve_results(jobs):
    results = []
    for job in jobs:
        results.append(job.result())
    return results
#######################################################
# end job utilities
#######################################################


#######################################################
# Utilities to process results
#######################################################
class MYRESULT:
    # This class helps to group results/circuits/err-mit 
    # together for each setting.
    # State tomography is done here for each set of 
    # circuit setting.
    def __init__(self, cname, qcs, result, mfitter):
        self.name = cname
        self.stqcs = qcs
        self._result = result
        self.mfitter = mfitter
        self._process()
        self._cal_density_mat()
    def _process(self):
        if self.mfitter is not None:
            self._result_fitted = self.mfitter.filter.apply(self._result)
        #self._result_filter = filter_result2(self.result, None)
    def _cal_density_mat(self): # standard state tomography
        tomo_fitter = StateTomographyFitter(self.result, self.stqcs)
        self._rho_fit = tomo_fitter.fit(method='lstsq')
        #self._rho_fit = tomo_fitter.fit(method='auto')
    @property
    def result(self):
        if self.mfitter is None:
            return self._result
        else:
            return self._result_fitted
    @property
    def rho(self):
        return self._rho_fit

def get_fitters(result, stls, trotter_steps, hs, order, nr):
    # Make a dict for error mitigation fitters which can be 
    # accessed according to the parameters of the circuit
    fts = dict()
    for n in nr:
        name = get_circuit_name(0.0, trotter_steps, order, hs, 1, 0, n, 'mit')
        ft = CompleteMeasFitter(result, stls, circlabel=name)
        fts[name] = ft
    return fts

def group_results(result, qcs, mfitters):
    # Separate the results according to settings
    # Each setting will result in one MYRESULT instance
    # with density matrix available
    def _get_mfitter(cname, mfitters):
        # Get the correspond error mitigation fitter according to
        # circuit name
        if mfitters is None:
            return None
        fname = re.sub('fc', 'mit', cname)
        fname = re.sub('_t....', '_t0.00', fname)
        fname = re.sub('_r....', '_r1.00', fname)
        fname = re.sub('_d....', '_d0.00', fname)
        if fname in mfitters:
            return mfitters[fname]
        fname = re.sub('_n._', '_n0_', fname)
        if fname in mfitters:
            return mfitters[fname]
        assert 0, 'mfitter for {} not found, by {}'.format(cname, fname)
    def _get_circ(i, cname, qcs):
        # Get the circuit corresponding to this experiment
        if i>=0 and qcs[i].name==cname:
            return qcs[i]
        for q in qcs:
            if q.name==cname:
                return q
        assert 0, "{} not found".format(cname)
    cktres = dict()
    cktqcs = dict()
    for i, res in enumerate(result.results):
        ename = res.header.name
        if 'mit_' in ename:
            continue
        if 'ccr_' in ename:
            continue
        r = deepcopy(res)
        qc = deepcopy(_get_circ(i, ename, qcs))
        ename_new = re.search(r'\(.+\)', ename).group(0)
        cname = re.sub(r'_st\(.+\)', '', ename)
        r.header.name = ename_new
        qc.name = ename_new
        if cname in cktres:
            cktres[cname].append(r)
            cktqcs[cname].append(qc)
        else:
            cktres[cname] = [r]
            cktqcs[cname] = [qc]
    cktres2 = []
    for k, v in cktres.items():
        r = qiskit.result.result.Result(result.backend_name, result.backend_version, result.qobj_id, result.job_id, result.success, v)
        mfitter = _get_mfitter(k, mfitters)
        cktr = MYRESULT(k, cktqcs[k], r, mfitter)
        cktres2.append(cktr)
    return cktres2

def pick_c2r(result, target_time, trotter_steps, st, hs, c2r):
    # Find the amplitude ratio that match best with 
    # ideal double CR gate behavior,
    # i.e. identity wiht 110 initial state
    imax = 0
    vmax = 0
    tval = target_time/trotter_steps
    sstr = _get_state_str(st)
    for i, c in enumerate(c2r):
        cname = get_circuit_name(tval, 1, 0, hs, c, 0, 0, 'ccr')
        cdict = result.get_counts(cname)
        cdict = _clean_cdict(cdict)
        cnt = cdict[sstr]
        if cnt>vmax:
            imax = i
            vmax = cnt
    print('Chosen c2r = {}, {}'.format(c2r[imax], imax))
    return imax
def pick_c2r_result(results, c2r):
    # pick the results according to the c2r value
    sc2r = '_r{:.2f}'.format(c2r)
    res = []
    for r in results:
        if sc2r in r.name:
            res.append(r)
    return res

def filter_rho_423(rho):
    # Remove parity bit from density matrix (the first qubit)
    assert rho.shape==(16,16)
    op = np.zeros((16,16))
    for i in range(8):
        op[i,i] = 1
    r = np.dot(op, np.dot(rho,op))
    t = np.trace(np.dot(op, np.dot(op, rho)))
    r = r/t
    r = r[0:8, 0:8]
    return r
def cal_fidelity_zne(gres, target_state):
    # calculate the fidelity based on density matrix.
    # if there are two entries, apply Zero Noise Extrapolation
    target_state_mat = target_state.to_matrix()
    if len(gres)==1:
        fid = state_fidelity(gres[0].rho, target_state_mat)
        return fid
    # Apply ZNE
    rho0 = gres[0].rho
    rho1 = gres[1].rho
    rho = 1.5*rho0 - 0.5*rho1
    if not is_positive_semidefinite_matrix(rho):
        rho = make_positive_semidefinite(rho)
    if rho.shape==(8,8):
        fid0 = state_fidelity(rho0, target_state_mat)
        fid1 = state_fidelity(rho1, target_state_mat)
        fid = state_fidelity(rho, target_state_mat)
        print("fidelity: {}, {} -> {}".format(fid0, fid1, fid))
    else:
        # remove parity bit
        rho0 = filter_rho_423(rho0)
        rho1 = filter_rho_423(rho1)
        rho = filter_rho_423(rho)
        fid0 = state_fidelity(rho0, target_state_mat)
        fid1 = state_fidelity(rho1, target_state_mat)
        fid = state_fidelity(rho, target_state_mat)
        print("fidelity: {}, {} -> {}".format(fid0, fid1, fid))
    return fid, fid0, fid1
#######################################################
# end results processing utility
#######################################################

if __name__=="__main__":
    target_time = np.pi
    initial_state = One^One^Zero
    bk = get_backend('jakarta',1)
    is_real = not (isinstance(bk, QasmSimulator))
    shots = bk.configuration().max_shots
    rep = 8
    order = 1
    nr = [0,1]
    trotter_steps = 4
    hs = 'xzy'
    qubits = [3, 1, 5] # t, c, c
    #c2r = 0.66
    c2r = np.linspace(0.6, 0.7, 5)
    dphi = 0
    jname = 'plc_{}_o{}_s{}_st{}'.format(hs, order, trotter_steps, shots)
    mfitter = None
    schds = None
    qcs = None

    mqcs, mstls = gen_mitigation_circs(bk, qubits, trotter_steps, hs, order, nr)
    cqcs = gen_ccr_calibration_circs(bk, target_time, trotter_steps, qubits, initial_state, c2r)
    tqcs = gen_tomo_experiments(bk, target_time, trotter_steps, qubits, initial_state, order=order, hs=hs, c2r=c2r, dphi=dphi, nr=nr)
    qcs = mqcs + cqcs + tqcs
    print("Experiment {} has {} circuits: {} mit, {} ccr, {} tomography".format(jname, len(qcs), len(mqcs), len(cqcs), len(tqcs)))

    qsub = qcs
    if bk.configuration().open_pulse:
        schds = schedule(qcs, bk)
        qsub = schds

    if 1:
        jobs = submit_expr(bk, qsub, shots=shots, jname=jname, rep=rep)
        results = retrieve_results(jobs)

        fids = []
        for result in results:
            mfitters = get_fitters(result, mstls, trotter_steps, hs, order, nr)
            gres = group_results(result, qcs, mfitters)
            ic2r = pick_c2r(result, target_time, trotter_steps, initial_state, hs, c2r)
            gres = pick_c2r_result(gres, c2r[ic2r])
            fid, _, __ = cal_fidelity_zne(gres, initial_state)
            fids.append(fid)
        print('state tomography fidelity = {:.4f} \u00B1 {:.4f}'.format(np.mean(fids), np.std(fids)))

        fids = []
        for result in results:
            gres = group_results(result, qcs, None)
            ic2r = pick_c2r(result, target_time, trotter_steps, initial_state, hs, c2r)
            gres = pick_c2r_result(gres, c2r[ic2r])
            fid, _, __ = cal_fidelity_zne(gres, initial_state)
            fids.append(fid)
        print('state tomography fidelity, no mfit = {:.4f} \u00B1 {:.4f}'.format(np.mean(fids), np.std(fids)))
