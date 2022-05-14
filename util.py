#!/usr/bin/env python
# coding: utf-8

# # 1 Introduction
# 
# ### 1-1 Quantum Simulation

from itertools import permutations
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})  # enlarge matplotlib fonts

# Import qubit states Zero (|0>) and One (|1>), and Pauli operators (X, Y, Z)
from qiskit.opflow import Zero, One, I, X, Y, Z, commutator

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

np.set_printoptions(linewidth=400)

# ### 1-2 The $XXX$ Heisenberg Spin Model

HSS = ['IXX', 'IYY', 'IZZ', 'XXI', 'YYI', 'ZZI']
HS = [I^X^X, I^Y^Y, I^Z^Z, X^X^I, Y^Y^I, Z^Z^I]
ITSS = [Zero^Zero^Zero, Zero^Zero^One, Zero^One^Zero, Zero^One^One, One^Zero^Zero, One^Zero^One, One^One^Zero, One^One^One]
initial_state = One^One^Zero # 6

def hcomm(A,B):
    return (A@B) - (B@A)

def int2hs(p):
    ps = list(permutations([0,1,2,3,4,5]))
    hs = [HS[i] for i in ps[p]]
    hss = [HSS[i] for i in ps[p]]
    return ps[p], hss, hs

def get_hs(hs):
    if hs is None:
        hs = HS
    elif isinstance(hs, int):
        hss = list(permutations([0,1,2,3,4,5]))
        hs = [HS[i] for i in hss[hs]]
    elif isinstance(hs[0], int):
        hs = [HS[i] for i in hs]
    return hs

def UT_heis3_unit_1(t, hs=None):
    hs = get_hs(hs)
    h0 = I^I^I
    for h in hs:
        h0 = ((t*h).exp_i()) @ h0
    return h0

def UT_heis3_unit_2(t, hs=None):
    hs = get_hs(hs)
    t /= 2
    h1 = UT_heis3_unit_1(t,hs)
    h2 = UT_heis3_unit_1(t,list(reversed(hs)))
    return h2 @ h1

def UT_heis3(target_time, trotter_steps, hs=None, order=1):
    if order == 1:
        h0 = UT_heis3_unit_1(target_time/trotter_steps,hs)
    elif order == 2:
        h0 = UT_heis3_unit_2(target_time/trotter_steps,hs)
    else:
        assert 0
    h = I^I^I
    for i in range(trotter_steps):
        h = h0 @ h
    return h

def apply_UT(initial_state, target_time, trotter_steps, hs=None, order=1):
    if isinstance(initial_state, int):
        initial_state = ITSS[initial_state]
    t = target_time/trotter_steps;
    h = UT_heis3_unit_1(t,hs) if order==1 else UT_heis3_unit_2(t,hs)
    st = initial_state
    for i in range(trotter_steps):
        st = h @ st
    return st

def apply_UT22(initial_state, target_time, trotter_steps, comm=True):
    A = (X^X^I) + (I^X^X)
    B = (Y^Y^I) + (I^Y^Y)
    C = (Z^Z^I) + (I^Z^Z)
    XZY = (X^Z^Y)
    YZX = (Y^Z^X)
    tval = target_time/trotter_steps
    Ae  = (0.5*tval*A).exp_i()
    Be  = (0.5*tval*B).exp_i()
    Ce  = (tval*C).exp_i()
    XZYe = (-0.5*tval*tval*XZY).exp_i()
    YZXe = (-0.5*tval*tval*YZX).exp_i()
    if comm:
        h1 = YZXe @ XZYe @ Ae @ Be @ Ce @ Be @ Ae
        h2 = Ae @ Be @ Ce @ Be @ Ae @ XZYe @ YZXe
    else:
        h1 = Ae @ Be @ Ce @ Be @ Ae
    st = initial_state
    h = I^I^I
    for i in range(trotter_steps):
        h = h1 @ h
        if st is not None:
            st = h1 @ st
    return st, h

def apply_UTcd(initial_state, target_time, trotter_steps, comm=True):
    C = (X^X^I) + (Y^Y^I) + (Z^Z^I)
    D = (I^X^X) + (I^Y^Y) + (I^Z^Z)
    cm = (X^Z^Y) - (X^Y^Z) - (Y^Z^X) + (Y^X^Z) + (Z^Y^X) - (Z^X^Y)
    #cm = commutator(C,D)/(2*1j)
    tval = target_time / trotter_steps
    Ce = (tval * C).exp_i()
    De = (tval * D).exp_i()
    cme = (-tval*tval*cm).exp_i()
    if comm:
        #h0 = cme @ De @ Ce
        h0 = Ce @ De @ cme
    else:
        h0 = De @ Ce
    h = I^I^I
    st = initial_state
    for i in range(trotter_steps):
        h = h0 @ h
        if st is not None:
            st = h0 @ st
    hg = (target_time * (C+D)).exp_i()
    return st, h, hg
def comp_UTcd_h(target_time, trotter_steps):
    #hg = U_heis3(target_time)
    _, htc, hg = apply_UTcd(None, target_time, trotter_steps, 1)
    _, ht, __ = apply_UTcd(None, target_time, trotter_steps, 0)
    print("H Golden:\n{}\nH CD cm:\n{}\nH CD without [c,d]:\n{}".format(hg.to_matrix(), htc.to_matrix(), ht.to_matrix()))
    print("dmat norm: {}".format(dmat_norm(hg.to_matrix(), htc.to_matrix())))
    print("dmat norm: {}".format(dmat_norm(hg.to_matrix(), ht.to_matrix())))

def comp_rzzx_based(tval):
    Hzx = (Z^X^I) + (I^X^Z)
    Hxx = (X^X^I) + (I^X^X)
    Hyy = (Y^Y^I) + (I^Y^Y)
    Hzz = (Z^Z^I) + (I^Z^Z)
    Hzxe = (tval*Hzx).exp_i()
    Hxxe = (tval*Hxx).exp_i()
    Hyye = (tval*Hyy).exp_i()
    Hzze = (tval*Hzz).exp_i()
    Ry1 = (np.pi/4*(Y^I^I)).exp_i()
    Ry1n = (-np.pi/4*(Y^I^I)).exp_i()
    Ry2 = (np.pi/4*(I^Y^I)).exp_i()
    Ry2n = (-np.pi/4*(I^Y^I)).exp_i()
    Ry3 = (np.pi/4*(I^I^Y)).exp_i()
    Ry3n = (-np.pi/4*(I^I^Y)).exp_i()
    Rx1 = (np.pi/4*(X^I^I)).exp_i()
    Rx1n = (-np.pi/4*(X^I^I)).exp_i()
    Rx3 = (np.pi/4*(I^I^X)).exp_i()
    Rx3n = (-np.pi/4*(I^I^X)).exp_i()
    Rz2 = (np.pi/4*(I^Z^I)).exp_i()
    Rz2n = (-np.pi/4*(I^Z^I)).exp_i()

    Hxxc = Ry1 @ Ry3 @ Hzxe @ Ry3n @ Ry1n
    Hyyc = Rx1n @ Rz2 @ Rx3n @ Hzxe @Rx3 @ Rz2n @ Rx1
    Hzzc = Ry2n @ Hzxe @ Ry2

    Mzxe = Hzxe.eval().to_matrix()
    Mxxe = Hxxe.eval().to_matrix()
    Myye = Hyye.eval().to_matrix()
    Mzze = Hzze.eval().to_matrix()
    Mxxc = Hxxc.to_matrix()
    Myyc = Hyyc.to_matrix()
    Mzzc = Hzzc.to_matrix()

    def _addx(qcx, tval):
        qcx.ry(-np.pi/2, [0,2])
        qcx.rzx(tval*2, 0, 1)
        qcx.rzx(tval*2, 2, 1)
        qcx.ry(np.pi/2, [0,2])
    qcx = QuantumCircuit(3)
    _addx(qcx, tval)

    def _addy(qcy, tval):
        qcy.rx(np.pi/2, [0,2])
        qcy.rz(-np.pi/2, 1)
        qcy.rzx(tval*2, 0, 1)
        qcy.rzx(tval*2, 2, 1)
        qcy.rz(np.pi/2, 1)
        qcy.rx(-np.pi/2, [0,2])
    qcy = QuantumCircuit(3)
    _addy(qcy, tval)

    def _addz(qcz, tval):
        qcz.ry(np.pi/2, 1)
        qcz.rzx(tval*2, 0, 1)
        qcz.rzx(tval*2, 2, 1)
        qcz.ry(-np.pi/2, 1)
    qcz = QuantumCircuit(3)
    _addz(qcz, tval)

    mox = qi.Operator(qcx).data
    moy = qi.Operator(qcy).data
    moz = qi.Operator(qcz).data

    print("XX:\n{}\n{}\n{}\n{}\n{}\n".format(Mxxe, Mxxc, mox, Mxxe-Mxxc, Mxxc-mox))
    print("YY:\n{}\n{}\n{}\n{}\n{}\n".format(Myye, Myyc, moy, Myye-Myyc, Myyc-moy))
    print("ZZ:\n{}\n{}\n{}\n{}\n{}\n".format(Mzze, Mzzc, moz, Mzze-Mzzc, Mzzc-moz))

    qc = QuantumCircuit(3)
    _addx(qc, tval/2)
    _addy(qc, tval/2)
    _addz(qc, tval)
    _addy(qc, tval/2)
    _addx(qc, tval/2)
    mo = qi.Operator(qc).data
    hg = U_heis3(tval)
    mg = hg.to_matrix()
    ht = UT_heis3(tval, 1, [0,3,1,4,2,5], 2)
    mt = ht.to_matrix()
    print("QC:\n{}\n{}\n{}\n{}\n".format(mg, mt, mo, mt-mo))

    return qcx, qcy, qcz, qc

def calc_trotter_error_1(target_time=1, hs=None, nmode=None):
    hs = get_hs(hs)
    m = len(hs)
    res = 0
    for i in range(m-1):
        h1 = hs[i]
        j = i+1
        h2 = hs[j]
        j += 1
        while j<m:
            h2 = h2 + hs[j]
            j += 1
        hc = h2 @ h1 - h1 @ h2
        res += LA.norm(hc.eval().to_matrix(), nmode)
    return res * target_time**2 / 2

def calc_trotter_error_2(target_time=1, hs=None, nmode=None):
    hs = get_hs(hs)
    m = len(hs)
    res = 0
    for i1 in range(m-1):
        h1 = hs[i1]
        i2 = i1+1
        h2 = hs[i2]
        h3 = hs[i2]
        i2 += 1
        while i2 < m:
            h2 += hs[i2]
            h3 += hs[i2]
            i2 += 1
        hc = h2 @ h1 - h1 @ h2
        hc = h3 @ hc - h3 @ hc
        res += LA.norm(hc.eval().to_matrix(), nmode)
        hc = h1 @ h2 - h2 @ h1
        hc = h1 @ hc - hc @ h1
        res += 0.5 * LA.norm(hc.eval().to_matrix(), nmode)
    return res * target_time**3 / 12

def compare_trotter_error_theory(target_time=1, nmode=None):
    ps = list(permutations(list(range(6))))
    res = []
    for p in ps:
        hs = [HS[i] for i in p]
        e1 = calc_trotter_error_1(target_time, hs, nmode)
        e2 = calc_trotter_error_2(target_time, hs, nmode)
        em1 = get_err_mat_norm(target_time, 1, hs, 1, nmode)
        em2 = get_err_mat_norm(target_time, 1, hs, 2, nmode)
        hs.reverse()
        em1r = get_err_mat_norm(target_time, 1, hs, 1, nmode)
        em2r = get_err_mat_norm(target_time, 1, hs, 2, nmode)
        res.append([em1, e1, em2, e2, em1r, em2r])
    return np.asarray(res)

# Returns the matrix representation of the XXX Heisenberg model for 3 spin-1/2 particles in a line
def H_heis3():
    # Interactions (I is the identity matrix; X, Y, and Z are Pauli matricies; ^ is a tensor product)
    XXs = (I^X^X) + (X^X^I)
    YYs = (I^Y^Y) + (Y^Y^I)
    ZZs = (I^Z^Z) + (Z^Z^I)
    
    # Sum interactions
    H = XXs + YYs + ZZs
    
    # Return Hamiltonian
    return H

def dmat_norm(m1, m2, nmode=None):
    return LA.norm(m1-m2, nmode)
# ### 1-3 Time Evolution

# Returns the matrix representation of U_heis3(t) for a given time t assuming an XXX Heisenberg Hamiltonian for 3 spins-1/2 particles in a line
def U_heis3(t):
    # Compute XXX Hamiltonian for 3 spins in a line
    H = H_heis3()
    # Return the exponential of -i multipled by time t multipled by the 3 spin XXX Heisenberg Hamilonian 
    return (t * H).exp_i()

def apply_U(initial_state, target_time):
    if isinstance(initial_state, int):
        initial_state = ITSS[initial_state]
    return U_heis3(float(target_time)) @ initial_state

def get_mat_fid(initial_state, target_time, trotter_steps,hs, order):
    if isinstance(initial_state, int):
        initial_state = ITSS[initial_state]
    s0 = apply_U(initial_state, target_time)
    st = apply_UT(initial_state, target_time, trotter_steps, hs, order)
    return (~s0 @ st).eval()

def get_err_mat_norm(target_time, trotter_steps, hs, order=1, nmode=None):
    U = U_heis3(target_time)
    UT = UT_heis3(target_time, trotter_steps, hs, order)
    return LA.norm(U.to_matrix() - UT.to_matrix(), nmode)

def get_all_err_mat_norm(target_time, trotter_steps, nmode=None):
    hs = list(permutations(HS))
    r = []
    for h in hs:
        n = get_err_mat_norm(target_time, trotter_steps, h)
        nu = get_err_mat_norm(target_time/trotter_steps, 1, h)
        r.append([n,nu])
        print(h, n, nu)
    return r

def plot_mat_fid(xs='time', initial_state=6):
    if isinstance(initial_state, int):
        initial_state = ITSS[initial_state]
    ts = np.linspace(0, np.pi, 10)
    t = np.pi
    ss = np.arange(10) + 1
    s = 4
    hs = list(permutations(HS))
    fs = []
    for h in hs:
        f = []
        if xs == 'time':
            for t in ts:
                f.append(get_mat_fid(initial_state, t, s, h))
        elif xs == 'step':
            for s in ss:
                f.append(get_mat_fid(initial_state, t, s, h))
        fs.append(f)
        print(h, np.mean(np.abs(f)**2))
    fs = np.asarray(fs)
    fr = np.real(fs)
    fi = np.imag(fs)
    ff = np.abs(fs)**2
    fig = plt.figure()
    gs = fig.add_gridspec(3, hspace=0)
    axs = gs.subplots(sharex=True)
    x = ts if xs=='time' else ss
    for f, r, i in zip(ff, fr, fi):
        axs[0].plot(x, f)
        axs[1].plot(x, r)
        axs[2].plot(x, i)
    axs[0].set(ylabel=r'prob')
    axs[0].grid()
    axs[1].set(ylabel=r'real')
    axs[1].grid()
    axs[2].set(xlabel=xs, ylabel=r'imag')
    axs[2].grid()
    fig.suptitle(r'fid of Trotter from $|110\rangle$ under $H_{Heis3}$')
    plt.tight_layout()
    plt.show()
    return ff
#plot_mat_fid()

def plot_trotter_norms():
    ts = np.linspace(0, np.pi, 50)
    h0norm = []
    htnorm = []
    hdnorm = []
    hs = list(permutations(HS))
    for t in ts:
        h0 = U_heis3(float(t)).to_matrix()
        h0norm.append(LA.norm(h0,np.inf))
    for h in hs:
        htn = []
        hdn = []
        for t in ts:
            ht = UT_heis3_unit_1(float(t), h).to_matrix()
            dh = ht - U_heis3(float(t)).to_matrix()
            htn.append(LA.norm(ht,np.inf))
            hdn.append(LA.norm(dh,np.inf))
        htnorm.append(htn)
        hdnorm.append(hdn)
        print(h, np.sum(np.abs(hdn)))
    h0norm = np.asarray(h0norm)
    htnorm = np.asarray(htnorm)
    hdnorm = np.asarray(hdnorm)
    print(h0norm,'\n',htnorm,'\n',hdnorm,htnorm-h0norm)
    fig = plt.figure()
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)
    for htn in htnorm:
        axs[0].plot(ts, htn)
    axs[0].plot(ts, h0norm, 'r-', linewidth=10, alpha=0.2)
    axs[0].set(ylabel=r'norms')
    axs[0].grid()
    for htn,hdn in zip(htnorm,hdnorm):
        axs[1].plot(ts, hdn)
        axs[1].plot(ts, htn-h0norm)
    axs[1].set(ylabel=r'diff')
    axs[1].grid()
    fig.suptitle(r'comparison of Trotter and golden $H_{Heis3}$')
    plt.tight_layout()
    # fig 2
    fig = plt.figure()
    gs = fig.add_gridspec(2)
    axs = gs.subplots()
    hds = np.mean(hdnorm, axis=1)
    axs[0].hist(hds)
    axs[1].plot(hds, 'o')
    #
    plt.show()
    return hds

# ### 1-4 Classical Simulation of $H_{\text{Heis3}}$
# Define initial state |110>

def get_its_amp(t, initial_state=6):
    if isinstance(initial_state, int):
        initial_state = ITSS[initial_state]
    return (~initial_state @ U_heis3(float(t)) @ initial_state).eval()
def get_its_amps(ts, initial_state=6):
    # Compute probability of remaining in |110> state over the array of time points
     # ~initial_state gives the bra of the initial state (<110|)
     # @ is short hand for matrix multiplication
     # U_heis3(t) is the unitary time evolution at time t
     # t needs to be wrapped with float(t) to avoid a bug
     # (...).eval() returns the inner product <110|U_heis3(t)|110>
     #  np.abs(...)**2 is the modulus squared of the innner product which is the expectation value, or probability, of remaining in |110>
    f = [get_its_amp(t) for t in ts]
    fr = np.real(f)
    fi = np.imag(f)
    probs_its = np.abs(f)**2
    return probs_its, fr, fi

def plot_amps(x, r1, r2=None):
    # Plot evolution of |110>
    fig = plt.figure()
    gs = fig.add_gridspec(3, hspace=0)
    axs = gs.subplots(sharex=True)
    axs[0].plot(x, r1[0])
    axs[0].set(ylabel=r'prob')
    axs[0].grid()
    axs[1].plot(x, r1[1])
    axs[1].set(ylabel=r'real')
    axs[1].grid()
    axs[2].plot(x, r1[2])
    axs[2].set(xlabel=r'time',ylabel=r'imag')
    axs[2].grid()
    if r2 is not None:
        axs[0].plot(x, r2[0])
        axs[1].plot(x, r2[1])
        axs[2].plot(x, r2[2])
    fig.suptitle(r'Evolution of state $|110\rangle$ under $H_{Heis3}$')
    plt.tight_layout()
    plt.show()
def plot_theory_amps():
    from qiskit.quantum_info import Statevector
    N = 20
    Trot_steps = 20
    ts = np.linspace(0, np.pi, N)
    ref = get_golden(ts)
    Trot_gate = build_qc0()[0].to_instruction()
    backend = get_backend()
    s0 = Statevector.from_int(0,2**7)
    rq = []
    for i in range(N):
        qr, qc = build_targ_qc(Trot_gate, ts[i], Trot_steps)
        s = s0.evolve(qc)
        s = s[40]
        rq.append(s)
    res = [np.abs(rq)**2, np.real(rq), np.imag(rq)]
    print(ref, '\n', res)
    plot_amps(ts, ref, res)




# ### 1-5 Decomposition of $U_{\text{Heis3}}(t)$ Into Quantum Gates

# # 2 The Open Science Prize
# ### 2-1 Contest Details
# ### 2-2 Import Qiskit
# 
# Feel free to import packages as needed. However, only free packages and those obtained through ```pip install``` or ```conda install``` are allowed.

# Importing standard Qiskit modules
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, IBMQ, execute, transpile, schedule, assemble, Aer
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.monitor import job_monitor
from qiskit.circuit import Parameter, Gate

# Import state tomography modules
from qiskit.ignis.verification.tomography import process_tomography_circuits, state_tomography_circuits, StateTomographyFitter, ProcessTomographyFitter
from qiskit.quantum_info import state_fidelity

# suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Parameterize variable t to be evaluated at t=pi later
t = Parameter('t')

# ### 2-3 Quantum Devices
# Connect to IBM provider and connect to a real or simulated backend. Final submissions must be run on a real backend, but simulated devices are faster for debugging and testing.

# In[6]:



# ### 2-4 Decomposition of $U_{\text{Heis3}}(t)$ into Quantum Gates (Example)
# 
# The following circuit code is written based on the example given in Section 1. This is where you write your solution.

# YOUR TROTTERIZATION GOES HERE -- START (beginning of example)

# In[8]:

def get_xyz_qc(tval=None):
    if tval is None:
        tuse = t
    else:
        tuse = tval
    # Build a subcircuit for XX(t) two-qubit gate
    XX_qr = QuantumRegister(2)
    XX_qc = QuantumCircuit(XX_qr, name='XX')
    XX_qc.ry(np.pi/2,[0,1])
    XX_qc.cnot(0,1)
    XX_qc.rz(2 * tuse, 1)
    XX_qc.cnot(0,1)
    XX_qc.ry(-np.pi/2,[0,1])

    # Build a subcircuit for YY(t) two-qubit gate
    YY_qr = QuantumRegister(2)
    YY_qc = QuantumCircuit(YY_qr, name='YY')
    YY_qc.rx(np.pi/2,[0,1])
    YY_qc.cnot(0,1)
    YY_qc.rz(2 * tuse, 1)
    YY_qc.cnot(0,1)
    YY_qc.rx(-np.pi/2,[0,1])
    
    # Build a subcircuit for ZZ(t) two-qubit gate
    ZZ_qr = QuantumRegister(2)
    ZZ_qc = QuantumCircuit(ZZ_qr, name='ZZ')
    ZZ_qc.cnot(0,1)
    ZZ_qc.rz(2 * tuse, 1)
    ZZ_qc.cnot(0,1)
    return XX_qc, YY_qc, ZZ_qc
 
def build_qcp(num_qubits=3, tval=None, perms=[0,1,2,3,4,5], order=1):
    def _append_xyz(p):
        for i in p:
            if i==0:
                qc.append(XX, [qr[1], qr[0]])
            elif i==1:
                qc.append(YY, [qr[1], qr[0]])
            elif i==2:
                qc.append(ZZ, [qr[1], qr[0]])
            elif i==3:
                qc.append(XX, [qr[1], qr[2]])
            elif i==4:
                qc.append(YY, [qr[1], qr[2]])
            elif i==5:
                qc.append(ZZ, [qr[1], qr[2]])
            else:
                assert 0
    if isinstance(perms, int):
        hs = list(permutations([0,1,2,3,4,5]))
        perms = list(hs[perms])
    if order==2:
        tval = t/2 if tval is None else tval/2
    XX, YY, ZZ = get_xyz_qc(tval)
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr, name='Trot')
    _append_xyz(perms)
    if order==2:
        _append_xyz(list(reversed(perms)))
    return qc.decompose(), XX, YY, ZZ

def build_qc(num_qubits=3, tval=None):
    XX, YY, ZZ = get_xyz_qc(tval)
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr, name='Trot')

    qc.append(ZZ, [qr[1], qr[0]])
    qc.append(ZZ, [qr[1], qr[2]])

    qc.append(YY, [qr[1], qr[0]])
    qc.append(YY, [qr[1], qr[2]])

    qc.append(XX, [qr[1], qr[0]])
    qc.append(XX, [qr[1], qr[2]])

    return qc.decompose(), XX, YY, ZZ

def build_qc1(num_qubits=3, tval=None):
    XX, YY, ZZ = get_xyz_qc(tval)
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr, name='Trot')

    qc.append(ZZ, [qr[0], qr[1]])
    qc.append(ZZ, [qr[1], qr[2]])

    qc.append(YY, [qr[0], qr[1]])
    qc.append(YY, [qr[1], qr[2]])

    qc.append(XX, [qr[0], qr[1]])
    qc.append(XX, [qr[1], qr[2]])

    return qc.decompose(), XX, YY, ZZ

def build_qc2(num_qubits=3, tval=None):
    XX, YY, ZZ = get_xyz_qc(tval)
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr, name='Trot')

    qc.append(ZZ, [qr[0], qr[1]])
    qc.append(YY, [qr[0], qr[1]])
    qc.append(XX, [qr[0], qr[1]])
    qc.append(ZZ, [qr[1], qr[2]])
    qc.append(YY, [qr[1], qr[2]])
    qc.append(XX, [qr[1], qr[2]])

    return qc.decompose(), XX, YY, ZZ

def build_qc3(num_qubits=3, tval=None):
    XX, YY, ZZ = get_xyz_qc(tval)
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr, name='Trot')

    qc.append(ZZ, [qr[0], qr[1]])
    qc.append(YY, [qr[0], qr[1]])
    qc.append(ZZ, [qr[1], qr[2]])
    qc.append(YY, [qr[1], qr[2]])
    qc.append(XX, [qr[1], qr[2]])
    qc.append(XX, [qr[0], qr[1]])

    return qc.decompose(), XX, YY, ZZ

def build_qc4(tval=None):
    qr = QuantumRegister(7)
    qc = QuantumCircuit(qr, name='Trot')
    
    XX, YY, ZZ = get_xyz_qc(tval/2)
    #_, YY, __ = get_xyz_qc(tval)

    #qc.reset([0,4])
    qc.swap(0,1)
    qc.swap(4,5)
    qc.cx(3,1)
    qc.cx(3,5)
    qc.append(ZZ, [0,1])
    qc.append(ZZ, [4,5])
    qc.append(XX, [0,1])
    qc.append(XX, [4,5])
    qc.append(YY, [0,1])
    qc.append(YY, [4,5])
    qc.cx(3,5)
    qc.cx(3,1)
    qc.swap(4,5)
    qc.swap(0,1)
    #qc.reset([0,4])
    return qc.decompose(), XX, YY, ZZ

    #qc.reset([2,6])
    qc.swap(2,1)
    qc.swap(6,5)
    qc.cx(3,1)
    qc.cx(3,5)
    qc.append(YY, [2,1])
    qc.append(YY, [6,5])
    qc.append(XX, [2,1])
    qc.append(XX, [6,5])
    qc.append(ZZ, [2,1])
    qc.append(ZZ, [6,5])
    qc.cx(3,5)
    qc.cx(3,1)
    qc.swap(6,5)
    qc.swap(2,1)
    #qc.reset([2,6])
    return qc.decompose(), XX, YY, ZZ
    
def build_qc0(tval=None):
    # Combine subcircuits into single gate representing one ($n=1$) trotter step.
    # Combine subcircuits into a single multiqubit gate representing a single trotter step
    num_qubits = 3

    XX, YY, ZZ = get_xyz_qc(tval)
    
    Trot_qr = QuantumRegister(num_qubits)
    Trot_qc = QuantumCircuit(Trot_qr, name='Trot')
    
    if 0: # original
        for i in range(0, num_qubits - 1):
            Trot_qc.append(ZZ.to_instruction(), [Trot_qr[i], Trot_qr[i+1]])
            Trot_qc.append(YY.to_instruction(), [Trot_qr[i], Trot_qr[i+1]])
            Trot_qc.append(XX.to_instruction(), [Trot_qr[i], Trot_qr[i+1]])
        return Trot_qc.decompose(), XX, YY, ZZ
    else:
        for i in range(0, num_qubits - 1):
            Trot_qc.append(ZZ, [Trot_qr[i], Trot_qr[i+1]])
            Trot_qc.append(YY, [Trot_qr[i], Trot_qr[i+1]])
            Trot_qc.append(XX, [Trot_qr[i], Trot_qr[i+1]])
        return Trot_qc.decompose(), XX, YY, ZZ


# YOUR TROTTERIZATION GOES HERE -- FINISH (end of example)

# ### 2-5 Trotterized Time Evolution

def add_initial_state_qc(its_id, qc, qr):
    assert qc.num_qubits>=3 and len(qr)==3
    if its_id==0:
        return
    elif its_id==1:
        qc.x(qr[0])
    elif its_id==2:
        qc.x(qr[1])
    elif its_id==3:
        qc.x([qr[0], qr[1]])
    elif its_id==4:
        qc.x(qr[2])
    elif its_id==5:
        qc.x([qr[2], qr[0]])
    elif its_id==6:
        qc.x([qr[2], qr[1]])
    elif its_id==7:
        qc.x(qr)
    else:
        assert 0

def build_targ_qc(Trot_gate, target_time=np.pi, trotter_steps=4, its_id=6, init=True, machine='lima'):
    # The final time of the state evolution
    # target_time = np.pi
    
    # Number of trotter steps
    # trotter_steps = 4  ### CAN BE >= 4
    if machine == 'lima':
        num_qubits = 5
        i1 = 1
        i2 = 3
        i3 = 4
    elif machine == 'jakarta':
        num_qubits = 7
        i1 = 1
        i2 = 3
        i3 = 5
    elif machine == 'ideal':
        num_qubits = 3
        i1 = 0
        i2 = 1
        i3 = 2
    else:
        assert 0
    if num_qubits<Trot_gate.num_qubits:
        num_qubits = Trot_gate.num_qubits
    # Initialize quantum circuit for 3 qubits
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr)
    # Prepare initial state (remember we are only evolving 3 of the 7 qubits on jakarta qubits (q_5, q_3, q_1) corresponding to the state |110>)
    if init:
        add_initial_state_qc(its_id, qc, [qr[i1], qr[i2], qr[i3]])
    # qc.x([i2,i3])  # DO NOT MODIFY (|q_5,q_3,q_1> = |110>)
    # Simulate time evolution under H_heis3 Hamiltonian
    for _ in range(trotter_steps):
        if Trot_gate.num_qubits>3:
            assert num_qubits==7, "{} on {}".format(num_qubits, machine)
            qc.append(Trot_gate, qr)
        else:
            qc.append(Trot_gate, [qr[i1], qr[i2], qr[i3]])
    # Evaluate simulation at target_time (t=pi) meaning each trotter step evolves pi/trotter_steps in time
    qc = qc.bind_parameters({t: target_time/trotter_steps})
    return qr, qc
def build_tomo_qc(qr, qc, machine='lima'):
    if machine == 'lima':
        num_qubits = 5
        i1 = 1
        i2 = 3
        i3 = 4
    elif machine == 'jakarta':
        num_qubits = 7
        i1 = 1
        i2 = 3
        i3 = 5
    elif machine == 'ideal':
        num_qubits = 3
        i1 = 0
        i2 = 1
        i3 = 2
    else:
        assert 0
    # Generate state tomography circuits to evaluate fidelity of simulation
    st_qcs = state_tomography_circuits(qc, [qr[i1], qr[i2], qr[i3]])
    return st_qcs
def build_full_trotter(its_id, target_time, trotter_steps, perms, order, init, machine):
    if order==3:
        qc = build_qc4(t)[0]
    else:
        qc = build_qcp(perms=perms, order=order)[0]
    print(machine)
    qr, qc = build_targ_qc(qc, target_time, trotter_steps, its_id, init, machine)
    return qr, qc

def _add_xx(qc, qs, tval):
    qc.ry(-np.pi/2, [qs[1], qs[2], qs[3]])
    #qc.h([qs[1], qs[2], qs[3]])
    qc.cnot(qs[2], qs[0])
    qc.cnot(qs[1], qs[0])
    qc.rz(2 * tval, qs[0])
    qc.cnot(qs[1], qs[0])
    qc.cnot(qs[3], qs[0])
    qc.rz(2 * tval, qs[0])
    qc.cnot(qs[3], qs[0])
    qc.cnot(qs[2], qs[0])
    qc.ry(np.pi/2, [qs[1], qs[2], qs[3]])
    #qc.h([qs[1], qs[2], qs[3]])
def _add_yy(qc, qs, tval):
    qc.rx(np.pi/2, [qs[1], qs[2], qs[3]])
    qc.cnot(qs[2], qs[0])
    qc.cnot(qs[1], qs[0])
    qc.rz(2 * tval, qs[0])
    qc.cnot(qs[1], qs[0])
    qc.cnot(qs[3], qs[0])
    qc.rz(2 * tval, qs[0])
    qc.cnot(qs[3], qs[0])
    qc.cnot(qs[2], qs[0])
    qc.rx(-np.pi/2, [qs[1], qs[2], qs[3]])
def _add_zz(qc, qs, tval):
    qc.cnot(qs[2], qs[0])
    qc.cnot(qs[1], qs[0])
    qc.rz(2 * tval, qs[0])
    qc.cnot(qs[1], qs[0])
    qc.cnot(qs[3], qs[0])
    qc.rz(2 * tval, qs[0])
    qc.cnot(qs[3], qs[0])
    qc.cnot(qs[2], qs[0])
def _add_xzy(qc, qs, tval):
    qc.ry(-np.pi/2, qs[1])
    qc.rx(np.pi/2, qs[3])
    qc.cnot(qs[1], qs[0])
    qc.cnot(qs[2], qs[0])
    qc.cnot(qs[3], qs[0])
    qc.rz(2 * tval, qs[0])
    qc.cnot(qs[3], qs[0])
    qc.cnot(qs[2], qs[0])
    qc.cnot(qs[1], qs[0])
    qc.rx(-np.pi/2, qs[3])
    qc.ry(np.pi/2, qs[1])
def _add_yzx(qc, qs, tval):
    qc.rx(np.pi/2, qs[1])
    qc.ry(-np.pi/2, qs[3])
    qc.cnot(qs[1], qs[0])
    qc.cnot(qs[2], qs[0])
    qc.cnot(qs[3], qs[0])
    qc.rz(2 * tval, qs[0])
    qc.cnot(qs[3], qs[0])
    qc.cnot(qs[2], qs[0])
    qc.cnot(qs[1], qs[0])
    qc.ry(np.pi/2, qs[3])
    qc.rx(-np.pi/2, qs[1])
def _add_zzz(qc, qs, tval):
    qc.cnot(qs[1], qs[0])
    qc.cnot(qs[2], qs[0])
    qc.cnot(qs[3], qs[0])
    qc.rz(2 * tval, qs[0])
    qc.cnot(qs[3], qs[0])
    qc.cnot(qs[2], qs[0])
    qc.cnot(qs[1], qs[0])
def _add_xxx(qc, qs, tval):
    qc.ry(-np.pi/2, qs[1:])
    qc.cnot(qs[1], qs[0])
    qc.cnot(qs[2], qs[0])
    qc.cnot(qs[3], qs[0])
    qc.rz(2 * tval, qs[0])
    qc.cnot(qs[3], qs[0])
    qc.cnot(qs[2], qs[0])
    qc.cnot(qs[1], qs[0])
    qc.ry(np.pi/2, qs[1:])
def _add_yyy(qc, qs, tval):
    qc.rx(np.pi/2, qs[1:])
    qc.cnot(qs[1], qs[0])
    qc.cnot(qs[2], qs[0])
    qc.cnot(qs[3], qs[0])
    qc.rz(2 * tval, qs[0])
    qc.cnot(qs[3], qs[0])
    qc.cnot(qs[2], qs[0])
    qc.cnot(qs[1], qs[0])
    qc.rx(-np.pi/2, qs[1:])
def _add_one_step_qc(qc, qs, tval, order=2, comm=True):
    if order==1:
        _add_xx(qc,qs,tval)
        _add_yy(qc,qs,tval)
        _add_zz(qc,qs,tval)
    elif order==2:
        _add_xx(qc,qs,tval/2)
        _add_yy(qc,qs,tval/2)
        _add_zz(qc,qs,tval)
        _add_yy(qc,qs,tval/2)
        _add_xx(qc,qs,tval/2)
        if comm:
            _add_xzy(qc,qs,-tval*tval/2)
            _add_yzx(qc,qs,-tval*tval/2)
def _compare_xyz_h(tval):
    dl = [2*i+1 for i in range(8)]
    # x
    qcx = QuantumCircuit(4)
    _add_xx(qcx, [0,1,2,3], tval)
    mx4 = qi.Operator(qcx).data
    hx = (I^X^X^I) + (X^X^I^I)
    hxv = (tval*hx).exp_i()
    print("\nxx:\n", hxv.eval().to_matrix(), "\n", mx4)
    print(np.abs(hxv.eval().to_matrix() - mx4))
    mx3 = np.delete(np.delete(mx4,dl,0), dl, 1)
    hx3 = (tval * ((X^X^I) + (I^X^X))).exp_i()
    print(hx3.eval().to_matrix(), "\n", mx3)
    print(np.abs(hx3.eval().to_matrix() - mx3))
    # y
    qcy = QuantumCircuit(4)
    _add_yy(qcy, [0,1,2,3], tval)
    my4 = qi.Operator(qcy).data
    hy = (I^Y^Y^I) + (Y^Y^I^I)
    hyv = (tval*hy).exp_i()
    print("\nyy:\n", hyv.eval().to_matrix(), "\n", my4)
    print(np.abs(hyv.eval().to_matrix() - my4))
    my3 = np.delete(np.delete(my4,dl,0), dl, 1)
    hy3 = (tval * ((Y^Y^I) + (I^Y^Y))).exp_i()
    print(hy3.eval().to_matrix(), "\n", my3)
    print(np.abs(hy3.eval().to_matrix() - my3))
    # z
    qcz = QuantumCircuit(4)
    _add_zz(qcz, [0,1,2,3], tval)
    mz4 = qi.Operator(qcz).data
    hz = (I^Z^Z^I) + (Z^Z^I^I)
    hzv = (tval*hz).exp_i()
    print("\nzz:\n", hzv.eval().to_matrix(), "\n", mz4)
    print(np.abs(hzv.eval().to_matrix() - mz4))
    mz3 = np.delete(np.delete(mz4,dl,0), dl, 1)
    hz3 = (tval * ((Z^Z^I) + (I^Z^Z))).exp_i()
    print(hz3.eval().to_matrix(), "\n", mz3)
    print(np.abs(hz3.eval().to_matrix() - mz3))
    # xzy
    qcxzy = QuantumCircuit(4)
    _add_xzy(qcxzy, [0,1,2,3], tval)
    mxzy4 = qi.Operator(qcxzy).data
    hxzy = Y^Z^X^I
    hxzyv = (tval*hxzy).exp_i()
    print("\nxzy:\n", hxzyv.eval().to_matrix(), "\n", mxzy4)
    print(np.abs(hxzyv.eval().to_matrix() - mxzy4))
    mxzy3 = np.delete(np.delete(mxzy4,dl,0), dl, 1)
    hxzy3 = (tval * (Y^Z^X)).exp_i()
    print(hxzy3.eval().to_matrix(), "\n", mxzy3)
    print(np.abs(hxzy3.eval().to_matrix() - mxzy3))
    # yzx
    qcyzx = QuantumCircuit(4)
    _add_yzx(qcyzx, [0,1,2,3], tval)
    myzx4 = qi.Operator(qcyzx).data
    hyzx = X^Z^Y^I
    hyzxv = (tval*hyzx).exp_i()
    print("\nyzx:\n", hyzxv.eval().to_matrix(), "\n", myzx4)
    print(np.abs(hyzxv.eval().to_matrix() - myzx4))
    myzx3 = np.delete(np.delete(myzx4,dl,0), dl, 1)
    hyzx3 = (tval * (X^Z^Y)).exp_i()
    print(hyzx3.eval().to_matrix(), "\n", myzx3)
    print(np.abs(hyzx3.eval().to_matrix() - myzx3))
    # xxx
    qcxxx = QuantumCircuit(4)
    _add_xxx(qcxxx, [0,1,2,3], tval)
    mxxx4 = qi.Operator(qcxxx).data
    hxxx = X^X^X^I
    hxxxv = (tval*hxxx).exp_i()
    print("\nxxx:\n", hxxxv.eval().to_matrix(), "\n", mxxx4)
    print(np.abs(hxxxv.eval().to_matrix() - mxxx4))
    mxxx3 = np.delete(np.delete(mxxx4,dl,0), dl, 1)
    hxxx3 = (tval * (X^X^X)).exp_i()
    print(hxxx3.eval().to_matrix(), "\n", mxxx3)
    print(np.abs(hxxx3.eval().to_matrix() - mxxx3))
    # yyy
    qcyyy = QuantumCircuit(4)
    _add_yyy(qcyyy, [0,1,2,3], tval)
    myyy4 = qi.Operator(qcyyy).data
    hyyy = Y^Y^Y^I
    hyyyv = (tval*hyyy).exp_i()
    print("\nyyy:\n", hyyyv.eval().to_matrix(), "\n", myyy4)
    print(np.abs(hyyyv.eval().to_matrix() - myyy4))
    myyy3 = np.delete(np.delete(myyy4,dl,0), dl, 1)
    hyyy3 = (tval * (Y^Y^Y)).exp_i()
    print(hyyy3.eval().to_matrix(), "\n", myyy3)
    print(np.abs(hyyy3.eval().to_matrix() - myyy3))
    # zzz
    qczzz = QuantumCircuit(4)
    _add_zzz(qczzz, [0,1,2,3], tval)
    mzzz4 = qi.Operator(qczzz).data
    hzzz = Z^Z^Z^I
    hzzzv = (tval*hzzz).exp_i()
    print("\nzzz:\n", hzzzv.eval().to_matrix(), "\n", mzzz4)
    print(np.abs(hzzzv.eval().to_matrix() - mzzz4))
    mzzz3 = np.delete(np.delete(mzzz4,dl,0), dl, 1)
    hzzz3 = (tval * (Z^Z^Z)).exp_i()
    print(hzzz3.eval().to_matrix(), "\n", mzzz3)
    print(np.abs(hzzz3.eval().to_matrix() - mzzz3))
    # whole
    qcht2c = QuantumCircuit(4)
    _add_one_step_qc(qcht2c, [0,1,2,3], tval, 2, 1)
    _, ht2c = apply_UT22(ITSS[6], tval, 1)
    mo = qi.Operator(qcht2c).data
    dl = [2*i+1 for i in range(8)]
    mo = np.delete(mo, dl, axis=0)
    mo = np.delete(mo, dl, axis=1)
    print("\nHtrotter (with [A,B]):\n", ht2c.to_matrix(), "\n", mo)
    print(np.abs(ht2c.to_matrix() - mo))
    # while without [A, B]
    qcht2 = QuantumCircuit(4)
    _add_one_step_qc(qcht2, [0,1,2,3], tval, 2, 0)
    _, ht2 = apply_UT22(ITSS[6], tval, 1, 0)
    ht2_2 = UT_heis3(tval, 1, [0,3,1,4,2,5], 2)
    mo = qi.Operator(qcht2).data
    dl = [2*i+1 for i in range(8)]
    mo = np.delete(mo, dl, axis=0)
    mo = np.delete(mo, dl, axis=1)
    print("\nHtrotter (without [A,B]):\n", ht2.to_matrix(), "\n", ht2_2.to_matrix(), "\n", mo)
    print(np.abs(ht2.to_matrix() - mo))
    print(np.abs(ht2_2.to_matrix() - mo))
    # order 1
    qcht1 = QuantumCircuit(4)
    _add_one_step_qc(qcht1, [0,1,2,3], tval, 1, 0)
    ht1 = UT_heis3(tval, 1, [0,3,1,4,2,5])
    mo = qi.Operator(qcht1).data
    dl = [2*i+1 for i in range(8)]
    mo = np.delete(mo, dl, axis=0)
    mo = np.delete(mo, dl, axis=1)
    print("\nHtrotter order 1:\n", ht1.to_matrix(), "\n", mo)
    print(np.abs(ht1.to_matrix() - mo))
    return [[qcx, qcy, qcz, qcxzy, qcyzx, qcxxx, qcyyy, qczzz, qcht2c, qcht2, qcht1], [hxv, hyv, hzv, hxzyv, hyzxv, hxxxv, hyyyv, hzzzv, ht2c, ht2, ht1]]

def build_full_trotter2(its_id, target_time, trotter_steps, init=True, move=True, order=2, comm=True):
    def _pre_qubits(qc):
        qc.swap(5,6)
        qc.swap(3,5)
        qc.swap(4,5)
        qc.swap(1,3)
    def _post_qubits(qc):
        qc.swap(1,3)
        qc.swap(4,5)
        qc.swap(3,5)
        qc.swap(5,6)
    qr = QuantumRegister(7)
    qc = QuantumCircuit(qr)
    qis = [qr[1], qr[3], qr[5]] if move else [qr[3], qr[4], qr[6]]
    if init:
        add_initial_state_qc(its_id, qc, qis)
    tval = target_time/trotter_steps
    if move:
        _pre_qubits(qc)
    for i in range(trotter_steps):
        _add_one_step_qc(qc, [5, 3, 4, 6], tval, order, comm)
    if move:
        _post_qubits(qc)
    return qr, qc
def _get_del_idx(N, qs):
    qsn = list(set(range(N)) - set(qs))
    dl = []
    for i in range(2**N):
        r = 0
        for q in qsn:
            if i & (1<<q):
                r = 1
                break
        if r:
            dl.append(i)
    return dl
def _comp_full_trotter_h(target_time, trotter_steps):
    qs = [1,3,5]
    N = 7
    dl = _get_del_idx(N,qs)
    qr, qc = build_full_trotter2(6, target_time, trotter_steps, 0, 1, 2, 1)
    mo = qi.Operator(qc)
    mo = np.delete(np.delete(mo, dl, axis=0), dl, axis=1)
    assert mo.shape==(2**len(qs), 2**len(qs)), "{}".format(mo.shape)
    hg = U_heis3(target_time)
    _, ht = apply_UT22(None, target_time, trotter_steps)
    print("H Golden:\n{}\nH full trotter:\n{}\nH circuit:\n{}".format(hg.to_matrix(), ht.to_matrix(), mo))
    print(np.abs(ht.to_matrix() - mo))
    print("dmat norm: {}, {}".format(dmat_norm(hg.to_matrix(), ht.to_matrix()), dmat_norm(ht.to_matrix(), mo)))
    del qr
    del qc
    qr, qc = build_full_trotter2(6, target_time, trotter_steps, 0, 1, 2, 0)
    mo = qi.Operator(qc)
    mo = np.delete(np.delete(mo, dl, axis=0), dl, axis=1)
    _, ht = apply_UT22(None, target_time, trotter_steps, 0)
    print("H full trotter without [A,B]:\n{}\nH circuit without [A,B]:\n{}\n".format(ht.to_matrix(), mo))
    ht1 = UT_heis3(target_time, trotter_steps, [0,3,1,4,2,5],1)
    ht2 = UT_heis3(target_time, trotter_steps, [0,3,1,4,2,5],2)
    print("H 1st order trotter:\n{}\nH 2nd order trotter:\n{}\n".format(ht1.to_matrix(), ht2.to_matrix()))
    print("dmat norm: {}, {}".format(dmat_norm(hg.to_matrix(), ht.to_matrix()), dmat_norm(ht.to_matrix(), mo)))
    print("dmat norm: {}, {}".format(dmat_norm(hg.to_matrix(), ht1.to_matrix()), dmat_norm(hg.to_matrix(), ht2.to_matrix())))

# ### 2-6 Execute
# For your final submission, you will need to execute your solution on a real backend with 8 repetitions. For faster debugging, considering using a simulated backend and/or 1 repetition.

def get_backend(mode='ideal', real=False):
    print('getting backend:', mode)
    if 1:
        # load IBMQ Account data
        
        # IBMQ.save_account(TOKEN)  # replace TOKEN with your API token string (https://quantum-computing.ibm.com/lab/docs/iql/manage/account/ibmq)
        provider = IBMQ.load_account()
        # Get backend for experiment
        # provider = IBMQ.get_provider(hub='ibm-q-community', group='ibmquantumawards', project='open-science-22')
        # jakarta = provider.get_backend('ibmq_jakarta')
        # properties = jakarta.properties()
        # Simulated backend based on ibmq_jakarta's device noise profile
        # sim_noisy_jakarta = QasmSimulator.from_backend(provider.get_backend('ibmq_jakarta'))
    
    # Noiseless simulated backend
    if mode=='ideal':
        sim = QasmSimulator()
    elif mode=='lima':
        if real:
            sim = provider.get_backend('ibmq_lima')
        else:
            sim = QasmSimulator.from_backend(provider.get_backend('ibmq_lima'))
    elif mode=='quito':
        if real:
            sim = provider.get_backend('ibmq_quito')
        else:
            sim = QasmSimulator.from_backend(provider.get_backend('ibmq_quito'))
    elif mode=='jakarta':
        provider = IBMQ.get_provider(hub='ibm-q-community', group='ibmquantumawards', project='open-science-22')
        jakarta = provider.get_backend('ibmq_jakarta')
        properties = jakarta.properties()
        # Simulated backend based on ibmq_jakarta's device noise profile
        if real:
            sim = jakarta
        else:
            sim = QasmSimulator.from_backend(provider.get_backend('ibmq_jakarta'))
    else:
        assert 0

    
    # backend = sim_noisy_jakarta
    return sim
    # reps = 8
    # backend = jakarta

# Compute the state tomography based on the st_qcs quantum circuits and the results from those ciricuits
def state_tomo(result, st_qcs):
    # The expected final state; necessary to determine state tomography fidelity
    target_state = (One^One^Zero).to_matrix()  # DO NOT MODIFY (|q_5,q_3,q_1> = |110>)
    # Fit state tomography results
    tomo_fitter = StateTomographyFitter(result, st_qcs)
    rho_fit = tomo_fitter.fit(method='lstsq')
    # Compute fidelity
    fid = state_fidelity(rho_fit, target_state)
    return fid

def run_sim(backend, st_qcs=None):
    #shots = 8192
    shots = 1
    job = execute(st_qcs, backend, shots=shots)
    return job.result()
    
def run_sims(backend, st_qcs, mode, real):
    shots = 8192 if real else 1024
    reps = 1 if mode=='ideal' else 4
    jobs = []
    for _ in range(reps):
        # execute
        job = execute(st_qcs, backend, shots=shots)
        print('Job ID', job.job_id())
        jobs.append(job)
    # We can monitor the status of the jobs using Qiskit's job monitoring tools.
    for job in jobs:
        job_monitor(job)
        try:
            if job.error_message() is not None:
                print(job.error_message())
        except:
            pass
    # ### 2-7 Results Analysis
    # Extract the results for the completed jobs and compute the state tomography fidelity for each repetition. You may choose to include other post-processing analyses here as well.
    # Compute tomography fidelities for each repetition
    fids = []
    for job in jobs:
        fid = state_tomo(job.result(), st_qcs)
        fids.append(fid)
    print('state tomography fidelity = {:.4f} \u00B1 {:.4f}'.format(np.mean(fids), np.std(fids)))
    return np.mean(fids)
    #import qiskit.tools.jupyter
    #get_ipython().run_line_magic('qiskit_version_table', '')

from qiskit.quantum_info import partial_trace
def _sim_qc_stvec(qc, qubits):
    qc = qc.decompose()
    sim = Aer.get_backend('statevector_simulator')
    job = sim.run(qc)
    st = job.result().get_statevector(qc)
    if qubits is not None:
        qs = list(set(range(qc.num_qubits)) - set(qubits))
        st = partial_trace(st, qs).to_statevector()
    return st
def get_fid_sim_stv(its_id, target_time, trotter_steps, perms, order, qubits=None):
    qr, qc = build_full_trotter(its_id, target_time, trotter_steps, perms, order, True, 'ideal')
    st = _sim_qc_stvec(qc, qubits)
    st0 = apply_U(ITSS[its_id], target_time)
    return np.squeeze((~st0).eval().to_matrix().dot(st))[()]

def get_qc_matrix(qc):
    sim = Aer.get_backend('unitary_simulator')
    return sim.run(qc).result().get_unitary(qc)

def get_trotter_matrix(target_time, trotter_steps, perms, order):
    qr, qc = build_full_trotter(0, target_time, trotter_steps, perms, order, False, 'ideal')
    qc = qc.decompose()
    return get_qc_matrix(qc)

def run_one(its_id=6, target_time=np.pi, trotter_steps=4, backend=None, perms=[0,1,2,3,4,5], order=1, machine='ideal', trans=True, real=False, fast=False):
    if backend is None:
        backend = get_backend(machine, real)
    if isinstance(perms, int):
        hs = list(permutations([0,1,2,3,4,5]))
        perms = list(hs[perms])
    print("running init {}, H{}, {}".format(its_id, perms, machine))
    fid_golden = get_mat_fid(its_id, target_time, trotter_steps, list(HS[i] for i in perms), order)
    print("golden initial state residual amplitude: {}".format(fid_golden))
    fid_ideal = get_fid_sim_stv(its_id, target_time, trotter_steps, perms, order)
    print("ideal initial state residual amplitude:  {}".format(fid_ideal))
    tqr, tqc = build_full_trotter(its_id, target_time, trotter_steps, perms, order, True, machine)
    qsize = 0
    qdepth = 0
    qncx = 0
    if trans:
        qct = transpile(tqc, backend)
        qsize = qct.size()
        qdepth = qct.depth()
        qncx = qct.num_nonlocal_gates()
        print("Transpiled Full Trotter citcuit (steps={}): size={} depth={} nonlocal_gates={}".format(trotter_steps, qsize, qdepth, qncx))
    st_qcs = build_tomo_qc(tqr, tqc, machine)
    fid = run_sims(backend, st_qcs, machine,real) if not fast else 0
    print("simu initial state residual amplitude:   {}".format(fid))
    print("init {}, H {}, {}: fid = {:.4g}, {:.4g} (ideal), {:.4g} (golden), size {}, depth {}, ncx {}".format(its_id, perms, machine, fid, fid_ideal, fid_golden, qsize, qdepth, qncx))
    return fid, np.abs(fid_ideal)**2, np.abs(fid_golden)**2, qsize, qdepth, qncx
#run_one(order=3, machine='jakarta')

def run_default(initial_state=6, target_time=np.pi, trotter_steps=4, backend=None, backend_name='ideal', draw=False, orig=False,perms=None,trans=False):
    if isinstance(initial_state, int):
        initial_state = ITSS[initial_state]
    fid_golden = get_mat_fid(initial_state, target_time, trotter_steps, list(HS[i] for i in perms), 1)
    if backend is None:
        backend = get_backend(backend_name)
    # Convert custom quantum circuit into a gate
    if orig:
        Trot_qc = build_qc0()[0]
    elif perms is not None:
        Trot_qc = build_qcp(perms=perms)[0]
    else:
        Trot_qc = build_qc()[0]
    if draw:
        print("Trotter gates:\ntop:")
        print(Trot_qc.draw('text'))
        print("decomposed:")
        print(Trot_qc.decompose().draw('text'))
    machine = backend_name
    qr, qc = build_targ_qc(Trot_qc, target_time, trotter_steps, machine)
    qc = qc.decompose()
    print("Full Trotter citcuit (steps={}): size={} depth={} nonlocal_gates={}".format(trotter_steps, qc.size(), qc.depth(), qc.num_nonlocal_gates()))
    if draw:
        print("Full Trotter gates:\ntop:")
        print(qc.draw('text'))
        print("decomposed:")
        print(qc.decompose().draw('text'))
    qsize = 0
    qdepth = 0
    qncx = 0
    if trans:
        qct = transpile(qc, backend)
        qsize = qct.size()
        qdepth = qct.depth()
        qncx = qct.num_nonlocal_gates()
        if draw:
            print("Transpiled full Trotter circuit:")
            print(qct.draw('text'))
        print("Transpiled Full Trotter citcuit (steps={}): size={} depth={} nonlocal_gates={}".format(trotter_steps, qct.size(), qct.depth(), qct.num_nonlocal_gates()))
    st_qcs = build_tomo_qc(qr,qc, machine)
    get_fid_sim_st(initial_state, st_qcs, target_time)
    if draw:
        # Display circuit for confirmation
        print("Full circuit:\ntop:")
        print(st_qcs[-1].draw())  # only view trotter gates
        print("decomposed:")
        print(st_qcs[-1].decompose().draw())  # view decomposition of trotter gates
    fid = run_sims(backend, st_qcs, backend_name)
    fid_ideal = run_sims(QasmSimulator(), st_qcs, 'ideal')
    return fid, fid_ideal, np.abs(fid_golden)**2, qsize, qdepth, qncx

def run_all(machine='ideal', trotter_steps=4, order=1, fast=False):
    its_id = 6
    t = np.pi
    s = trotter_steps
    hs = list(permutations([0,1,2,3,4,5]))
    res = []
    backend = get_backend(machine)
    for h in hs:
        print('\n', list(HSS[i] for i in h))
        fid, fid_ideal, fid_golden, qsize, qdepth, qncx = run_one(its_id, t, s, backend, h, order, machine, True, False, fast)
        fid_unit = get_mat_fid(its_id, float(np.pi/s), 1, list(HS[i] for i in h), order)
        fid_unit = np.abs(fid_unit)**2
        print(h, fid, fid_ideal, fid_golden, fid_unit, qsize, qdepth, qncx)
        res.append([fid, fid_ideal, fid_golden, fid_unit, qsize, qdepth, qncx])
    np.savetxt('{}_tr{}_i{}.dat'.format(machine, trotter_steps, its_id),res, header='fid fid_ideal fid_golden fid_unit cktsize cktdepth cktncx')
    return res
#r = run_all('jakarta', 5)

def run_tmp():
    t = np.pi/4
    s = 1
    hs = list(permutations([0,1,2,3,4,5]))
    res = []
    for h in hs:
        print('\n', list(HSS[i] for i in h))
        fid = get_mat_fid(initial_state, float(t), s, list(HS[i] for i in h))
        print(h, fid, np.abs(fid)**2)
        res.append([fid])
    return res

def plot_amps(xs='time', backend_name='ideal'):
    initial_state = ITSS[6]
    ts = np.linspace(0, np.pi, 10)
    t = np.pi
    ss = np.arange(10) + 1
    ss = [4]
    s = 4
    hs = list(permutations([0,1,2,3,4,5]))
    fs = []
    backend = get_backend(backend_name)
    for h in hs:
        print('\n', list(HSS[i] for i in h))
        f = []
        if xs == 'time':
            for t in ts:
                f.append(run_default(initial_state, t, s, backend, backend_name, False, False, h, True)[0])
        elif xs == 'step':
            for s in ss:
                f.append(run_default(initial_state, t, s, backend, backend_name, False, False, h, True)[0])
        print(h, np.mean(f))
    fig = plt.figure()
    gs = fig.add_gridspec(1, hspace=0)
    axs = gs.subplots(sharex=True)
    x = ts if xs=='time' else ss
    for f in fs:
        axs[0].plot(x, f)
    axs[0].set(xlabel=xs, ylabel=r'fid')
    plt.show()
    return fs


###### CR gate pulse
from qiskit import pulse, circuit
import qiskit.pulse.library as pulse_lib
from qiskit.pulse import InstructionScheduleMap
from qiskit.pulse.channels import PulseChannel
from scipy.special import erf
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

def _gssq_pulse(duration, amp, phi, sigma, width, name=None):
    """ Wrapper of gaussian square pulse generator.
    ::
        amp = amp * exp(-1j * phi)
    """
    #kwargs['amp'] = kwargs.get('amp', 0) * np.exp(-1j*kwargs.pop('phi', 0))
    amp = amp*np.exp(1j*phi)
    return pulse_lib.GaussianSquare(duration, amp, sigma, width=width, name=name, limit_amplitude=False)

def _cr_control_pulse(u,d,p):
    with pulse.build() as schd:
        pulse.play(_gssq_pulse(**p), u)
        pulse.delay(p['duration'], d)
    return schd

def _cr_cancellation_pulse(d,p):
    with pulse.build() as schd:
        if p['amp']==0:
            pulse.delay(p['duration'], d)
        else:
            pulse.play(_gssq_pulse(**p), d)
    return schd

def _cr_pulse(bkcfg, qtinfo, qcinfo):
    # pulse on target 
    qtid = qtinfo.get('id')
    qtgssq = qtinfo.get('pulse')
    qtdch = bkcfg.drive(qtid)
    schdt = _cr_cancellation_pulse(qtdch, qtgssq)
    # pulse on control
    if isinstance(qcinfo, dict):
        qcinfo = [qcinfo]
    with pulse.build() as schdc:
        for f in qcinfo:
            qcid = f.get('id')
            qcgssq = f.get('pulse')
            qcdch = bkcfg.drive(qcid)
            qcuch = bkcfg.control((qcid, qtid))[0]
            schd = _cr_control_pulse(qcuch, qcdch, qcgssq)
            schdc += schd
    # combine 
    schd = schdt + schdc
    return schd

def _cinv_pulse(bkcfg, istr2schd, qtinfo, qcinfo, invs):
    if not np.all(invs):
        return pulse.Schedule()
    assert len(qcinfo) == len(invs)
    duration = 0
    schdc = []
    for iv,qc in zip(invs, qcinfo):
        with pulse.build() as s:
            if iv:
                pulse.call(istr2schd.get('x', qc['id']))
                duration = max(duration, s.duration)
        schdc.append(s)
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

    # first cr pulse
    schd1 = _cr_pulse(bkcfg, qtinfo, qcinfo)
    # inverse and second 
    schdinv = _cinv_pulse(bkcfg, istr2schd, qtinfo, qcinfo, [1]*(len(qubits)-1))
    qtinfo['pulse']['phi'] += np.pi
    for qcf in qcinfo:
        qcf['pulse']['phi'] += np.pi
    schd2 = _cr_pulse(bkcfg, qtinfo, qcinfo)
    qtinfo['pulse']['phi'] -= np.pi
    for qcf in qcinfo:
        qcf['pulse']['phi'] -= np.pi

    # assemble final schedule
    with pulse.build() as schd:
        #pulse.set_phase(-np.pi/2, bkcfg.drive(1))
        #pulse.shift_phase(-np.pi, bkcfg.drive(3))
        #pulse.shift_phase(-np.pi, bkcfg.control((1,3))[0])
        schd += schd1
        schd += schdinv
        schd += schd2
        schd += schdinv
    return schd

def _init_params(qubits):
    params = dict()
    params['ampt']       = Parameter('ampt')
    params['phit']       = Parameter('phit')
    params['sigmat']     = Parameter('sigmat')
    params['durationt']  = Parameter('durationt')
    params['widtht']     = Parameter('widtht')

    params['ampc0']      = Parameter('ampc0')
    params['phic0']      = Parameter('phic0')
    params['sigmac0']    = Parameter('sigmac0')
    params['durationc0'] = Parameter('durationc0')
    params['widthc0']    = Parameter('widthc0')

    if len(qubits)>2:
        params['ampc1']      = Parameter('ampc1')
        params['phic1']      = Parameter('phic1')
        params['sigmac1']    = Parameter('sigmac1')
        params['durationc1'] = Parameter('durationc1')
        params['widthc1']    = Parameter('widthc1')

    return params

def _get_params(bk, params, qubits, pnvals=None, theta=np.pi/2, c2r=1, dphi=0):
    if pnvals is None:
        pnvals = _get_pnvals(bk, qubits, theta, c2r, dphi)
    else:
        pnvals = deepcopy(pnvals)
        _scale_params(pnvals, theta)
    pvals = dict()
    ns = pnvals.keys()
    for n in ns:
        pvals[params[n]] = pnvals[n]
    return pvals
def _get_pnvals(bk, qubits, theta, c2r=1, dphi=0):
    if not bk.configuration().open_pulse:
        return None
    pnvals = _get_pnvals_cr(bk, [qubits[1], qubits[0]]) # [c,t]
    if len(qubits)>2:
        pnvals = _add_pnvals_c1(pnvals, c2r, dphi)
    _scale_params(pnvals, theta)
    return pnvals
def _get_pnvals_cr(bk, qubits):
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
    pvals['ampc1'] = pvals['ampc0'] * c2r
    pvals['phic1'] = pvals['phic0'] + dphi
    pvals['sigmac1'] = pvals['sigmac0']
    pvals['durationc1'] = pvals['durationc0']
    pvals['widthc1'] = pvals['widthc0']
    # target cancellation
    pvals['ampt'] *= (1+c2r)
    return pvals
def _get_cr_params(bk, qubits): # [c,t]
    #assert qubits[0]<qubits[1]
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
            #assert c.index==qubits[0], "c.index={}, qubits={}".format(c.index, qubits)
            waveu = wave
        if waveu is not None and waved is not None:
            break
    assert waveu is not None and waved is not None
    return waveu.parameters, waved.parameters
def _get_max_amp_ratio(bk, qubits):
    up, dp = _get_cr_params(bk, qubits)
    ua = up['amp']
    da = dp['amp']
    r = 1/abs(ua)*0.999
    if abs(da)>0 and abs(da)*(1+r)>1:
        r = 1/abs(da)-1
    print('q {}: u amp ({}) {}, d amp ({}) {}, max r={}'.format(qubits, abs(ua), ua, abs(da), da, r))
    return r

def _scale_params(p, theta):
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

def _calc_cr_coeff_zx(amp, J, lmbd, ahi, Dw):
    lin = -J*lmbd*amp/Dw*(ahi/(ahi+Dw))
    nonlin = J*(lmbd*amp)**3*(3*ahi**3+11*ahi**2*Dw+15*ahi*Dw**2+9*Dw**3)/(2*Dw**3*(ahi+Dw)**3*(ahi+2*Dw)*(3*ahi+2*Dw))
    return (lin+nonlin)*2
def _calc_cr_coeff_theta(coeff, t, ncr=2):
    return ncr*coeff*t
def _calc_cr_theta(amp, J, lmbd, ahi, Dw, t=1, ncr=2):
    c = _calc_cr_coeff_zx(amp, J, lmbd, ahi, Dw)
    return _calc_cr_coeff_theta(c, t, ncr)
def _get_cr_hwparams(bkcfg, qt, qc):
    hvars = bkcfg.hamiltonian['vars']
    Jstr = 'jq'+str(min(qc,qt))+'q'+str(max(qc,qt))
    J = hvars[Jstr]
    lmbdstr = 'omegad'+str(qc)
    lmbd = hvars[lmbdstr]
    ahistr = 'delta'+str(qc)
    ahi = hvars[ahistr]
    wc = hvars['wq'+str(qc)]
    wt = hvars['wq'+str(qt)]
    Dw = wc - wt
    Dw /= 2 # a qiskit bug in h data?
    #J /= np.pi*2
    #lmbd /= np.pi*2
    #ahi /= np.pi*2
    #Dw /= np.pi*2
    return J, lmbd, ahi, Dw
def _calc_gssq_area_from_cr_theta(theta, J, lmbd, ahi, Dw, ncr=2):
    # only considering the linear part
    c = theta/ncr
    r = -J*lmbd/Dw*(ahi/(ahi+Dw)) * 2
    return c/r

def _create_cr_gate_schd2(backend, qubits, params, name):
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
def _create_cr_gate_schd(backend, qubits, params, name):
    cn = name + ''.join(str(i) for i in qubits)
    cg = dict()
    cg['qubits'] = deepcopy(qubits)
    cg['params'] = deepcopy(params)
    cg['schedule'] = _create_cr_schedule(backend, qubits, **params)
    cg['gate'] = Gate(cn, len(qubits), params.values())
    return cg

def _check_backend_cr_schd(backend, qs): #[c,t]
    cfg = backend.configuration()
    phw = _get_cr_hwparams(cfg, qs[1], qs[0])
    pu, pd = _get_cr_params(bk, qs)
    A = _gssq_area(pu['duration'], pu['amp'], pu['sigma'], pu['width'])
    theta = _calc_cr_theta(A*cfg.dt, *phw)
    print("hw params: {}\nu pulse: {}\nd pulse: {}\narea={}, theta={}\n".format(phw, pu, pd, A, theta))

def _gen_ccr_data(s0=Zero^Zero^Zero):
    def _calc_meas(M, s):
        return ((~s) @ M @ s).eval()
    def _calc_amp(b, s):
        return ((~b) @ s).eval()
    H = ((Z^X^I) + (I^X^Z))
    ts = np.linspace(0, np.pi, 200)
    r = []
    for tt in ts:
        tt = float(tt)
        U = (tt * H).exp_i()
        s = U @ s0
        m1x = _calc_meas(X^I^I, s)
        m1y = _calc_meas(Y^I^I, s)
        m1z = _calc_meas(Z^I^I, s)
        m2x = _calc_meas(I^X^I, s)
        m2y = _calc_meas(I^Y^I, s)
        m2z = _calc_meas(I^Z^I, s)
        m3x = _calc_meas(I^I^X, s)
        m3y = _calc_meas(I^I^Y, s)
        m3z = _calc_meas(I^I^Z, s)
        a0 = _calc_amp(Zero^Zero^Zero, s)
        a1 = _calc_amp(Zero^Zero^One, s)
        a2 = _calc_amp(Zero^One^Zero, s)
        a3 = _calc_amp(Zero^One^One, s)
        a4 = _calc_amp(One^Zero^Zero, s)
        a5 = _calc_amp(One^Zero^One, s)
        a6 = _calc_amp(One^One^Zero, s)
        a7 = _calc_amp(One^One^One, s)
        r.append([tt, m1x, m1y, m1z, m2x, m2y, m2z, m3x, m3y, m3z, a0, a1, a2, a3, a4, a5, a6, a7])
    return np.asarray(r)
def _plot_ccr_meas(r=None, s0=Zero^Zero^Zero):
    if r is None:
        r = _gen_ccr_data(s0)
    ylb = (['m1x', 'm1y', 'm1z', 'm2x', 'm2y', 'm2z', 'm3x', 'm3y', 'm3z'])
    fig = plt.figure()
    gs = fig.add_gridspec(9, 2, hspace=0)
    axs = gs.subplots(sharex=True)
    for i in range(9):
        axs[i, 0].plot(r[:,0], np.real(r[:,i+1]), '.')
        axs[i, 0].plot(r[:,0], np.imag(r[:,i+1]), '.')
        axs[i, 0].set(ylabel=ylb[i])
    axs[8, 0].set(xlabel='time')
    ylb = (['000', '001', '010', '011', '100', '101', '110', '111'])
    for i in range(8):
        axs[i,1].plot(r[:,0], np.real(r[:,i+10]), '.')
        axs[i,1].plot(r[:,0], np.imag(r[:,i+10]), '.')
        axs[i,1].plot(r[:,0], np.abs(r[:,i+10]), '.')
        axs[i,1].set(ylabel=ylb[i])
    axs[8, 1].set(xlabel='time')
    plt.tight_layout()
    plt.show()
    return r

from copy import deepcopy
from qiskit.test.mock import FakeValencia, FakeJakarta
import qiskit.quantum_info as qi

def _add_ccr_circuit2(bk, theta, qubits, params, qc):
    cg = _create_cr_gate_schd2(bk, qubits, params, 'cr')
    qc.append(cg['gate'], cg['qubits'])
    qc.add_calibration(cg['gate'].name, cg['qubits'], cg['schedule'])
def _add_ccr_circuit(bk, theta, qubits, params, pvals, qc=None):
    if qc is None:
        qr = QuantumRegister(7)
        qc = QuantumCircuit(qr, name='ccr')
    else:
        qr = qc.qubits
        assert qc.num_qubits>=len(qubits)
    cg = _create_cr_gate_schd(bk, qubits, params, 'cr')
    qc.append(cg['gate'], cg['qubits'])
    assert cg['gate'].params == list(params.values())
    
    ### Either use add_calibration or both basis_gates and instruction_schedule_map.
    ### Both work. 
    ### bgate+istrmap do not work for three qubits
    ### The instruction_schedule_map cannot be a deepcopy. Possibly a bug in schedule function.
    ### whether assign before transpile or after transpile does not matter. 
    #bgates += [cg['gate'].name]
    #istr2schd.add(cg['gate'].name, cg['qubits'], cg['schedule'], params.keys())
    #
    #### assign before transpile
    #qcv = qc.assign_parameters(pvals)
    #qct = transpile(qcv, bk, bgates)
    #qschd = schedule(qct, bk, istr2schd)
    ### assign after transpile NOT work with real backend
    #qct = transpile(qc, bk, bgates)
    #qcv = qct.assign_parameters(pvals)
    #qschd = schedule(qcv, bk, istr2schd)
    
    ### use add_calibration, instead of bgate/instr map
    qc.add_calibration(cg['gate'].name, cg['qubits'], cg['schedule'], cg['gate'].params)
    #qcv = qc.assign_parameters(pvals)
    #qct = transpile(qcv, bk)
    #qschd = schedule(qct, bk)
    #return qct, qschd
    return qc

def _add_one_trotter_step(bk, qc, qs, cg1, cg1d, cg2, cg2d, tval, hs, order, nr): # t,c,c
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

    if order==1:
        for h in hs:
            _addh(bk, qc, qs, cg2, tval, h)
            if nr==1:
                _addh(bk, qc, qs, cg2d, -tval, h)
                _addh(bk, qc, qs, cg2, tval, h)
    elif order==2:
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
    elif order==0: # for simple ccr testing
        _add_rzzx(bk, qc, qs, cg2, tval)
    elif order==-1 or order==-2: # simple xyz circuit
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
    if not bk.configuration().open_pulse:
        cg1 = None
        cg2 = None
        cg1d = None
        cg2d = None
    else:
        cg1, cg1d = _create_cr_gate_schd2(bk, qubits, params1, 'cr1')
        cg2, cg2d = _create_cr_gate_schd2(bk, qubits, params2, 'cr2')
        qc.add_calibration(cg1['gate'].name, cg1['qubits'], cg1['schedule'])
        qc.add_calibration(cg2['gate'].name, cg2['qubits'], cg2['schedule'])
        qc.add_calibration(cg1d['gate'].name, cg1d['qubits'], cg1d['schedule'])
        qc.add_calibration(cg2d['gate'].name, cg2d['qubits'], cg2d['schedule'])
    for i in range(trotter_steps):
        _add_one_trotter_step(bk, qc, qubits, cg1, cg1d, cg2, cg2d, tval, hs, order, nr)
    return cg1, cg1d, cg2, cg2d

def _get_state_str(s0):
    if isinstance(s0, int):
        return str(s0)
    assert len(s0.primitive.keys())==1
    ss = list(s0.primitive.keys())[0]
    return ss

def _add_init_qc(qc, qr, qubits, s0):
    ss = _get_state_str(s0)
    nq = len(ss)
    assert nq==len(qubits)
    qs = np.sort(qubits).tolist()
    ss = ss[::-1] # revert for little endian
    for i in range(nq):
        if ss[i]=='1':
            qc.x(qr[qs[i]])
def _add_meas(qc, qubits, meas, cr=None):
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

def _get_circ_name(s0, meas, theta, c2r, dphi, prefix='ccr'):
    return "{}_s{}_m{}_t{:.2f}_r{:.2f}_d{:.2f}".format(prefix, _get_state_str(s0), meas, theta, c2r, dphi)

def comp_ccr_trotter(bk, target_time, trotter_steps, hs='xyz', order=2):
    tval = target_time/trotter_steps
    qubits = [3,1,5]
    qr = QuantumRegister(bk.configuration().num_qubits)
    qc = QuantumCircuit(qr)
    _add_full_trotter(bk, qc, qubits, None, None, trotter_steps, tval, hs, order)
    hg = U_heis3(target_time)
    ht = UT_heis3(target_time, trotter_steps, [0,3,1,4,2,5], order)
    mc = qi.Operator(qc).data
    mg = hg.to_matrix()
    mt = ht.to_matrix()
    dl = _get_del_idx(7, [1,3,5])
    mc = np.delete(np.delete(mc, dl, axis=0), dl, axis=1)
    print("H Golden:\n{}\nH trotter:\n{}\nH circuit:\n{}\n{}\n".format(mg,mt,mc,mt-mc))
    print("dmat norm: {}".format(dmat_norm(mg,mt)))
    print("dmat norm: {}".format(dmat_norm(mt,mc)))
    return qc

def _make_stomo_circuits(bk, qc0, name, do_parity):
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
def get_circuit_name(target_time, trotter_steps, order, hs, ncx, prefix='fc'):
    return "{}_{}_s{}_o{}_n{}_t{:.2f}".format(prefix, hs, trotter_steps, order, ncx, target_time)
def _build_full_test_circuit(bk, target_time, trotter_steps, qubits, s0, meas, c2r, dphi, hs='xzy', order=2, nr=0, optlevel=None, stomo=False, name='fc'):
    theta = target_time/trotter_steps
    assert isinstance(theta, float)
    assert isinstance(meas, str) or stomo
    #name = _get_circ_name(s0, meas, target_time, c2r, dphi, nr, name)
    name = get_circuit_name(target_time, trotter_steps, order, hs, nr, name)
    pvals1 = _get_pnvals(bk, qubits, theta, c2r, dphi)
    pvals2 = _get_pnvals(bk, qubits, theta*2, c2r, dphi)
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
    print(name)
    qct = transpile(qc, bk, optimization_level=optlevel)
    return qct
def gen_tomo_experiments(bk, target_time, trotter_steps, qubits, initial_statei, order, hs, c2r, dphi, nr):
    qcs = []
    for n in nr:
        qct = _build_full_test_circuit(bk, target_time, trotter_steps, qubits, initial_state, meas=None, c2r=c2r, dphi=dphi, hs=hs, order=order, nr=n, optlevel=None, stomo=True)
        qcs += qct
    return qcs

def _add_parity(qc):
    qc.swap(0,1)
    qc.cx(0,1)
    qc.swap(1,3)
    qc.cx(1,3)
    qc.swap(3,5)
    qc.cx(3,5)
    qc.swap(5,6)
    qc.swap(3,5)
    qc.swap(1,3)
    qc.swap(0,1)
def _add_one_trotter_step2(qc, tval, hs, order):
    def _append_xyz(qc, XX, YY, ZZ, p):
        for i in p:
            if i==0:
                qc.append(XX, [3,1])
            elif i==1:
                qc.append(YY, [3,1])
            elif i==2:
                qc.append(ZZ, [3,1])
            elif i==3:
                qc.append(XX, [3,5])
            elif i==4:          
                qc.append(YY, [3,5])
            elif i==5:          
                qc.append(ZZ, [3,5])
            else:
                assert 0
    XX, YY, ZZ = get_xyz_qc(tval)
    p = []
    for h in hs:
        if h=='x':
            p += [0,3]
        elif h=='y':
            p += [1,4]
        elif h=='z':
            p += [2,5]
    _append_xyz(qc, XX, YY, ZZ, p)
    if order==2:
        _append_xyz(qc, XX, YY, ZZ, list(reversed(p)))
def _get_circuit_name2(target_time, trotter_steps, order, hs, ncx, prefix='fc'):
    return "{}_{}_s{}_o{}_n{}_t{:.2f}".format(prefix, hs, trotter_steps, order, ncx, target_time)
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import TransformationPass
class CXTranslator(TransformationPass):
    def __init__(self, n):
        super().__init__()
        self.n = n
    def run(self, dag):
        for node in dag.op_nodes():
            if node.op.name=='cx':
                qcnew = QuantumCircuit(2)
                if self.n==-2:
                    dag.remove_op_node(node)
                else:
                    assert self.n > 0
                    for i in range(2*self.n+1):
                        qcnew.cx(0,1)
                    dag.substitute_node_with_dag(node, circuit_to_dag(qcnew))
        return dag
def _build_full_test_circuit2(bk, target_time, trotter_steps, order=2, s0=One^One^Zero, hs='xzy', ncx=0, optlevel=None, do_meas=1, do_init=True, stomo=False, do_parity=True):
    name = _get_circuit_name2(target_time, trotter_steps, order, hs, ncx)
    if order == 1:
        tval = target_time/trotter_steps
    else:
        tval = target_time/trotter_steps/2.0
    qr = QuantumRegister(bk.configuration().num_qubits)
    qc = QuantumCircuit(qr, name=name)
    if do_init:
        if s0.num_qubits==3:
            _add_init_qc(qc, qr, [1,3,5], s0)
        else:
            _add_init_qc(qc, qr, [1,3,5,6], s0)
    for i in range(trotter_steps):
        _add_one_trotter_step2(qc, tval, hs, order)
    if do_parity:
        _add_parity(qc)
    if stomo==2:
        qubits = [1,3,5,6] if do_parity else [1,3,5]
        stqcs = state_tomography_circuits(qc, qubits)
        for stqc in stqcs:
            stqc.name = name + '_st' + stqc.name
        qc = stqcs 
    elif stomo==1:
        qc = _make_stomo_circuits(bk, qc, name, do_parity)
    elif do_meas==1:
        cr = ClassicalRegister(4 if do_parity else 3)
        qc.add_register(cr)
        if do_parity:
            qc.measure([qr[1], qr[3], qr[5], qr[6]], cr)
        else:
            qc.measure([qr[1], qr[3], qr[5]], cr)
    print(name)
    qct = transpile(qc, bk, optimization_level=optlevel)
    if ncx>0 or ncx==-2:
        if isinstance(qct, list):
            for i in range(len(qct)):
                qct[i] = CXTranslator(ncx)(qct[i])
        else:
            qct = CXTranslator(ncx)(qct)
    return qc, qct
def gen_tomo_experiments2(bk, trotter_steps, order, hs, ncx, do_parity):
    qcs = []
    tv = np.pi
    for o in order:
        for ts in trotter_steps:
            for n in ncx:
                _, qct = _build_full_test_circuit2(bk, tv, ts, o, hs=hs, ncx=n, stomo=2, do_parity=do_parity)
                qcs += qct
    return qcs
def gen_experiments2(bk, target_time, trotter_steps, order, hs, ncx):
    qcs = []
    for o in order:
        for ts in trotter_steps:
            for n in ncx:
                for tv in target_time:
                    _, qct = _build_full_test_circuit2(bk, tv, ts, o, hs=hs, ncx=n)
                    qcs.append(qct)
    return qcs
def rearange_result2(result, target_time, trotter_steps, order, hs, ncx):
    nbit = None
    data = []
    for o in order:
        dto = []
        for ts in trotter_steps:
            dtts = []
            for n in ncx:
                dtn = []
                for tv in target_time:
                    cname = _get_circuit_name2(tv, ts, o, hs, n)
                    if nbit is None:
                        nbit = result._get_experiment(cname).header.creg_sizes[0][1]
                    cdict = _get_cdict(result, cname, nbit)
                    mexpc = _get_expectation(cdict, nbit)
                    cnts = _get_count_list(cdict, nbit)
                    dtn.append(mexpc+cnts)
                dtts.append(dtn)
            dto.append(dtts)
        data.append(dto)
    data = np.asarray(data) # order, tsteps, ncx, time, nbit+basis
    return data
import re
def filter_result2(r0, mfts):
    def _value(v0, qs):
        v = 0
        for i,q in enumerate(qs):
            if v0&(1<<q):
                v += (1<<i)
        assert v<2**len(qs)
        return v
    def _get_fitter(nm, mfts):
        if not isinstance(mfts, dict):
            return mfts
        k = re.sub('fc', 'mit', nm)
        k = re.sub('_t.+', '_t0.00', k)
        return mfts[k]
    def _apply_fitter(nm, cdict, mfts):
        mft = _get_fitter(nm, mfts)
        return mft.filter.apply(cdict)
    def _clean(r, qs):
        h = r.header
        cls = []
        for cl in h.clbit_labels:
            if cl[1] in qs:
                cls.append(cl)
        h.clbit_labels = cls
        h.creg_sizes[0][1] = len(qs)
        h.memory_slots = len(qs)
        #h.n_qubits = len(qs)
        #h.qreg_sizes[0][1] = len(qs)
        #qls = []
        #for ql in h.qubit_labels:
        #    if ql[1] in qs:
        #        qls.append(ql)
        #h.qubit_labels = qls
    result = deepcopy(r0)
    qs = [0,1,2]
    qds = [3]
    msk = 0
    for i in qds:
        msk += (1<<i)
    for r in result.results:
        cname = r.header.name
        if 'mit' in cname:
            continue
        cdict0 = _clean_cdict(r.data.counts, len(qs)+len(qds))
        if mfts is not None:
            cdict0 = _apply_fitter(cname, cdict0, mfts)
        if r.metadata['num_clbits']==4:
            cdict1 = {}
            vtot0 = 0
            vtot1 = 0
            for k, v in cdict0.items():
                if '0x' in k:
                    kv = int(k, 16)
                else:
                    kv = int(k, 2)
                vtot0 += v
                if not (kv&msk):
                    k1 = '{0:03b}'.format(_value(kv, qs))
                    cdict1[k1] = v
                    vtot1 += v
            if vtot1 < 0.5*vtot0:
                print('!!! most sample discarded {}: {}/{}\n{}\n{}'.format(cname, vtot1, vtot0, cdict0, cdict1))
            for k in cdict1.keys():
                cdict1[k] *= (vtot0/vtot1)
        else:
            cdict1 = cdict0
        r.data.counts = cdict1
        _clean(r, qs)
    return result

class MYRESULT:
    def __init__(self, cname, qcs, result, mfitter):
        print('Making results for: {}'.format(cname))
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
    def _cal_density_mat(self):
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
    def assign_density_mat(rho):
        self._dm = qi.DensityMatrix(rho)
def _get_mfitter(cname, mfitters):
    if mfitters is None:
        return None
    fname = re.sub('fc', 'mit', cname)
    fname = re.sub('_t.+', '_t0.00', fname)
    if fname in mfitters:
        return mfitters[fname]
    fname = re.sub('_n.+_', '_n-2_', fname)
    if fname in mfitters:
        return mfitters[fname]
    assert 0, 'mfitter for {} not found, by {}'.format(cname, fname)
def group_results(result, qcs, mfitters):
    def _get_circ(i, cname, qcs):
        if i>=0 and qcs[i].name==cname:
            return qcs[i]
        for q in qcs:
            print(cname, q.name)
            if q.name==cname:
                return q
        assert 0, "{} not found".format(cname)
    cktres = dict()
    cktqcs = dict()
    for i, res in enumerate(result.results):
        ename = res.header.name
        if 'mit_' in ename:
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
def plot_expr_result2(r, target_time, trotter_steps, order, hs, ncx, xlabel='time', title=None):
    def _plot_data(ax, d, tvals, tsteps, order, hs, lg=True):
        nts, nncx, ntv = d.shape
        if xlabel=='time':
            for its in range(nts):
                for incx in range(nncx):
                    lb = '{} steps, {}cx'.format(tsteps[its], ncx[incx])
                    ax.plot(tvals, d[its,incx,:], '-o', label=lb)
        elif xlabel=='tsteps':
            for itv in range(ntv):
                for incx in range(nncx):
                    lb = 'time: {}, {}cx'.format(tsteps[its], ncx[incx])
                    ax.plot(tsteps, d[:,incx,itv], '-o', label=lb)
        if lg:
            ax.legend()
    if r.shape[-1]==11:
        nbit = 3
    elif r.shape[-1]==20:
        nbit = 4
    else:
        assert 0, '{}'.format(r.shape)
    N = 2**nbit
    Nr = nbit + N
    Nc = len(order)
    fig = plt.figure()
    gs = fig.add_gridspec(Nr, Nc)
    axs = gs.subplots(sharex=True)
    for ic in range(Nc):
        for ir in range(nbit):
            mstr = 'z'+str(ir)
            ax = _get_ax(ir, ic, Nr, Nc, axs)
            ax.set(ylabel=mstr)
            d = r[ic, :, :, :, ir]
            _plot_data(ax, d, target_time, trotter_steps, order[ic], hs, lg=(not ir))
        for i in range(N):
            ir = i + nbit
            ax = _get_ax(ir, ic, Nr, Nc, axs)
            ylb = '{0:03b}'.format(i)
            ax.set(ylabel=ylb)
            d = r[ic, :, :, :, ir]
            _plot_data(ax, d, target_time, trotter_steps, order[ic], hs, lg=(not ir))
    if title is not None:
        fig.suptitle(title)
    plt.show()

def _build_test_circuit2(bk, theta, qubits, s0, meas, c2r, dphi):
    assert isinstance(theta, float)
    assert isinstance(meas, str)
    name = _get_circ_name(s0, meas, theta, c2r, dphi)
    pnvals = _get_pnvals(bk, qubits, theta, c2r, dphi)
    qr = QuantumRegister(bk.configuration().num_qubits)
    qc = QuantumCircuit(qr, name=name)
    _add_init_qc(qc, qr, qubits, s0)
    _add_ccr_circuit2(bk, theta, qubits, pnvals, qc)
    _add_meas(qc, qubits, meas)
    qct = transpile(qc, bk, output_name=name)
    print(qc.name, qct.name)
    return qct

def _gen_circuits_dphi(bk, theta, qubits, s0, meas, c2r, dphi, tsteps, hs, order):
    circs = []
    for dph in dphi:
        dph = float(dph)
        if tsteps>0:
            c = _build_full_test_circuit(bk, theta, tsteps, qubits, s0, meas, c2r, dph, hs, order)
        else:
            c = _build_test_circuit2(bk, theta, qubits, s0, meas, c2r, dph)
        circs.append(c)
    return circs

def _gen_circuits_c2r(bk, theta, qubits, s0, meas, c2r, dphi, tsteps, hs, order):
    circs = []
    for r in c2r:
        r = float(r)
        c = _gen_circuits_dphi(bk, theta, qubits, s0, meas, r, dphi, tsteps, hs, order)
        circs += c
    return circs

def _gen_circuits_theta(bk, theta, qubits, s0, meas, c2r, dphi, tsteps, hs, order):
    circs = []
    for th in theta:
        th = float(th)
        c = _gen_circuits_c2r(bk, th, qubits, s0, meas, c2r, dphi, tsteps, hs, order)
        circs += c
    return circs

def _gen_circuits_meas(bk, theta, qubits, s0, meas, c2r, dphi, tsteps, hs, order):
    circs = []
    if isinstance(meas, str):
        assert len(meas)==len(qubits)
        meas = [meas]
    for m in meas:
        c = _gen_circuits_theta(bk, theta, qubits, s0, m, c2r, dphi, tsteps, hs, order)
        circs += c
    return circs

def _gen_circuits_state(bk, theta, qubits, ss, meas, c2r, dphi, tsteps, hs, order):
    if not hasattr(theta, '__iter__'):
        theta=[theta]
    circs = []
    for s in ss:
        c = _gen_circuits_meas(bk, theta, qubits, s, meas, c2r, dphi, tsteps, hs, order)
        circs += c
    return circs

def gen_experiments(bk, qubits, theta=None, ss=None, meas=None, c2r=None, dphi=None, tsteps=0, hs='xyz', order=2):
    if theta is None:
        theta = np.linspace(0, np.pi, 25)
        #theta = [np.pi/2]
    if c2r is None:
        try:
            rmax = _get_max_amp_ratio(bk, [min(qubits[1:]), qubits[0]])
        except:
            print('getting rmax for c2r failed.')
            rmax = 1
        print('max c2r: {}'.format(rmax))
        c2r = np.linspace(0.5, min(0.8, rmax), 4)
        #c2r = [0.6]
    if dphi is None:
        #dphi = np.linspace(-np.pi/10, np.pi/10, 7)a
        dphi = [0]
    if ss is None:
        ss = [One^One^Zero]
        #ss = ITSS[0:3]
        #ss = [Zero^Zero^Zero, Zero^Zero^One, One^Zero^Zero]
        #ss = [Zero^One^One, One^One^Zero]a
    if meas is None:
        #meas = ['zxz', 'zyz', 'zzz']
        meas = ['xxx', 'yyy', 'zzz']
    print('ss: {}\nmeas: {}\nc2r: {}\ndphi: {}\ntheta: {}'.format(ss, meas, c2r, dphi, theta))
    circs = _gen_circuits_state(bk, theta, qubits, ss, meas, c2r, dphi, tsteps, hs, order)
    if bk.configuration().open_pulse:
        schds = schedule(circs, bk)
    else:
        schds = None
    return theta, ss, meas, c2r, dphi, circs, schds

def _get_expectation(cdict, nbit):
    expv = [0]*nbit
    if len(cdict)==0:
        return expv
    for k, val in cdict.items():
        k = k[::-1]
        for i in range(nbit):
            if k[i] == '1':
                expv[i] -= val
            else:
                expv[i] += val
    for i in range(nbit):
        expv[i] /= sum(cdict.values())
    return expv

def _get_count_list(cdict, nbit):
    N = 2**nbit
    c = [0]*N
    if len(cdict)==0:
        return c
    S = sum(cdict.values())
    for i in range(N):
        if nbit==3:
            v = cdict.get('{0:03b}'.format(i))
        else:
            v = cdict.get('{0:04b}'.format(i))
        c[i] = 0 if v is None else v/S
    return c

def _get_cdict_from_result_dict(rs, nm):
    for r in rs['results']:
        if r['header']['name']==nm:
            return r['data']['counts']
    assert 0, "{} not found".format(nm)
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
def _get_cdict(r0, cname, nbit=3):
    if isinstance(r0, dict):
        cdict = _get_cdict_from_result_dict(r0, cname)
    else:
        cdict = r0.get_counts(cname)
    cdict = _clean_cdict(cdict, nbit)
    return cdict
def rearange_result(r0, ss, meas, theta, c2r, dphi):
    nbit = ss[0].num_qubits
    rs = []
    for s in ss:
        rm = []
        for m in meas:
            rt = []
            for t in theta:
                rc = []
                for c in c2r:
                    rdp = []
                    for dp in dphi:
                        cname = _get_circ_name(s, m, t, c, dp) + 'T'
                        cdict = _get_cdict(r0, cname)
                        mexpc = _get_expectation(cdict, nbit)
                        cnts = _get_count_list(cdict, nbit)
                        rdp.append(mexpc+cnts)
                    rc.append(rdp)
                rt.append(rc)
            rt = np.asarray(rt)
            rm.append(rt)
        rs.append(rm)
    rs = np.asarray(rs)
    # ss, meas, (nbit+basis), theta, c2r
    rs = np.moveaxis(rs, -1, -4)
    return rs

def _get_ax(ir, ic, Nr, Nc, axs):
    if Nr==1 and Nc==1:
        return axs
    elif Nr==1:
        return axs[ic]
    elif Nc==1:
        return axs[ir]
    else:
        return axs[ir,ic]
def plot_expr_result(s, meas, theta, c2r, dphi, r, xlabel='c2r', title=None):
    def _plot_data(axs, d, theta, c2r, dphi, xlabel):
        Ntheta, Nc2r, Ndphi = d.shape
        if xlabel=='time':
            for cc in range(Nc2r):
                for dd in range(Ndphi):
                    lb = 'c2r={:.2f}, dphi={:.2f}'.format(c2r[cc], dphi[dd])
                    axs.plot(theta, d[:,cc,dd],'-o', label=lb)
        elif xlabel=='c2r':
            for tt in range(Ntheta):
                for dd in range(Ndphi):
                    lb = 'theta={:.2f}, dphi={:.2f}'.format(theta[tt], dphi[dd])
                    axs.plot(c2r, d[tt,:,dd],'-o', label=lb)
        elif xlabel=='dphi':
            for tt in range(Ntheta):
                for cc in range(Nc2r):
                    lb = 'theta={:.2f}, c2r={:.2f}'.format(theta[tt], c2r[cc])
                    axs.plot(dphi, d[tt,cc,:],'-o', label=lb)
        axs.legend()
    nbit = s.num_qubits
    N = 2**nbit
    Nr = nbit + N
    Nc = len(meas)
    fig = plt.figure()
    gs = fig.add_gridspec(Nr, Nc)
    axs = gs.subplots(sharex=True)
    for im in range(Nc):
        for i in range(nbit):
            mstr = meas[im][i] + str(i)
            ax = _get_ax(i, im, Nr, Nc, axs)
            ax.set(ylabel=mstr)
            d = r[im][i]
            _plot_data(ax, d, theta, c2r, dphi, xlabel)
        for i in range(N):
            ii = i+nbit
            ylb = '{0:03b}'.format(i)
            ax = _get_ax(ii, im, Nr, Nc, axs)
            ax.set(ylabel=ylb)
            d = r[im][ii]
            _plot_data(ax, d, theta, c2r, dphi, xlabel)
        ax = _get_ax(Nr-1, im, Nr, Nc, axs)
        ax.set(xlabel=xlabel)
    #plt.tight_layout()
    if title is not None:
        fig.suptitle(title)
    plt.show()

def plot_ccr_trotter_ref(initial_state, tsteps, order=2, shots=8192):
    #tsteps = [4,6]
    #hs = [0,3,1,4,2,5]
    #order = 2
    #shots = 8192
    nbit = 3
    N = 2**nbit
    data = []
    targ_time = np.linspace(0, np.pi, 25)
    hss = [[0,3,1,4,2,5], [0,3,2,5,1,4], [1,4,0,3,2,5], [1,4,2,5,0,3], [2,5,0,3,1,4], [2,5,1,4,0,3]]
    for ts in tsteps:
        rm = []
        for hs in hss:
            print(ts, hs)
            rtv = []
            for tval in targ_time:
                s = apply_UT(initial_state, tval, ts, hs, order)
                cdict = s.eval().primitive.sample_counts(shots)
                mexpc = _get_expectation(cdict, nbit)
                cnts = _get_count_list(cdict, nbit)
                rtv.append(mexpc+cnts)
            rm.append(rtv)
        data.append(rm)
    # tsteps, hss, time, bits
    data = np.asarray(data)

    Nr = nbit + N
    Nc = len(hss)
    fig = plt.figure()
    gs = fig.add_gridspec(Nr, Nc)
    axs = gs.subplots(sharex=True)
    def _plot_data(ax, tsteps, targ_time, d):
        for its, ts in enumerate(tsteps):
            lb = '{} steps'.format(ts)
            ax.plot(targ_time, d[its], '-o', label=lb)
    for i in range(Nc):
        for j in range(nbit):
            mstr = 'z' + str(j)
            ax = _get_ax(j, i, Nr, Nc, axs)
            ax.set(ylabel=mstr)
            d = data[:,i,:,j]
            _plot_data(ax, tsteps, targ_time, d)
        for j in range(N):
            jj = j + nbit
            ylb = '{0:03b}'.format(j)
            ax = _get_ax(jj, i, Nr, Nc, axs)
            ax.set(ylabel=ylb)
            d = data[:,i,:,jj]
            _plot_data(ax, tsteps, targ_time, d)
        ax = _get_ax(Nr-1, i, Nr, Nc, axs)
        ax.set(xlabel='time')
        ax = _get_ax(0, i, Nr, Nc, axs)
        ax.set(title=str(hss[i]))
    plt.show()

from qiskit.providers.ibmq.managed import IBMQJobManager
def run_expr(bk, qc, shots=8192, jname=None):
    print("total circs: {}".format(len(qc)))
    if len(qc) < bk.configuration().max_experiments:
        qobj = assemble(qc, bk, shots=shots)
        job = bk.run(qobj, job_name=jname)
        #job = execute(qc, bk, shots=shots)
        print('job id: {}'.format(job.job_id()))
        r = job.result()
        return job, r
    else:
        jbm = IBMQJobManager()
        job_set = jbm.run(qc, backend=bk, name=jname)
        results = job_set.results()
        r = results.combine_results()
        return job_set, r
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
def _run_qsp_tomo(bk, qc, pvals, theta, bgates):
    qr = qc.qubits
    qrm = [qr[i] for i in np.sort(qubits)]
    if len(qubits)>2:
        H = (Z^X^I) + (I^X^Z)
    else:
        H = Z^X
    ## assign befor tomo
    qcv = qc.assign_parameters(pvals)
    st_qcs = state_tomography_circuits(qcv, qrm)
    pt_qcs = process_tomography_circuits(qcv, qrm)
    pt_qcsv = pt_qcs
    #qrt = qct.qubits
    #pt_qcs = process_tomography_circuits(qct, [qrt[qubits[0]], qrt[qubits[1]]])
    
    ### decided to assign before tomo due to the naming issue.
    ## param assign before transpile
    # pt_qcsv = []
    # for c in pt_qcs:
    #     cv = c.assign_parameters(pvals)
    #     ## tomo circuit got named with the tomo setting, e.g. "(('Zp', 'Zp'), ('X', 'Y'))"
    #     ## the fitter relies on this information.
    #     cv.name = c.name
    #     pt_qcsv.append(cv)
    pt_qcst = transpile(pt_qcsv, bk, bgates)
    pt_qcss = schedule(pt_qcst, bk)
    
    ## param assign after transpile
    ## NOT work with real backend
    #pt_qcst = transpile(pt_qcs, bk, bgates)
    #pt_qcstv = []
    #for c in pt_qcst:
    #    pt_qcstv.append(c.assign_parameters(pvals))
    #pt_qcss = schedule(pt_qcstv, bk)
    
    target_unitary = qi.Operator((0.5*theta*H).exp_i()) # order of qubits 3,2,1, etc
    
    job = execute(pt_qcst, bk, shots=8192)
    qpt_tomo = ProcessTomographyFitter(job.result(), pt_qcst)
    choi_fit_lstsq = qpt_tomo.fit(method='lstsq')
    print('Average gate fidelity: F = {:.5f}'.format(qi.average_gate_fidelity(choi_fit_lstsq, target=target_unitary)))

import pickle
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
def _gen_mitigation_circs2_one(bk, stls, trotter_steps, hs, ncx, order, name, msc, do_parity):
    tval = 1e-9
    qcs = []
    qcts = []
    for sl in stls:
        s0 = One if sl[0]=='1' else Zero
        for c in sl[1:]:
            s0 = s0^(One if c=='1' else Zero)
        qc, qct = _build_full_test_circuit2(bk, tval, trotter_steps, s0=s0, hs=hs, ncx=ncx, order=order, do_meas=msc, do_parity=do_parity)
        qc.name = name+'cal_'+sl
        qct.name = name+'cal_'+sl
        print(qc.name)
        qcs.append(qc)
        qcts.append(qct)
    return qcs, qcts
def _gen_mitigation_circs2(bk, nbit, trotter_steps, hs, order, ncx, msc=1, do_parity=1):
    qcs = []
    qcts = []
    stls = []
    for s in range(2**nbit):
        if nbit == 3:
            sl = '{0:03b}'.format(s)
        else:
            sl = '{0:04b}'.format(s)
        stls.append(sl)
    for o in order:
        for ts in trotter_steps:
            for n in ncx:
                name = _get_circuit_name2(0, ts, o, hs, n, 'mit')
                qc, qct = _gen_mitigation_circs2_one(bk, stls, ts, hs, n, o, name, msc, do_parity)
                qcs += qc
                qcts += qct
    return qcs, qcts, stls
def get_fitters2(result, stls, trotter_steps, hs, order, ncx):
    fts = dict()
    for o in order:
        for ts in trotter_steps:
            for n in ncx:
                name = _get_circuit_name2(0, ts, o, hs, n, 'mit')
                print(name)
                print(stls)
                ft = CompleteMeasFitter(result, stls, circlabel=name)
                fts[name] = ft
    return fts
def _gen_mitigation_circs(bk, qubits, trotter_steps, hs, order, nr, name):
    pvals = _get_pnvals(bk, qubits, 0, 1, 0)
    qcs = []
    stls = []
    for s in ITSS:
        assert len(s.primitive.keys())==1
        sl = list(s.primitive.keys())[0]
        stls.append(sl)
        qct = _build_full_test_circuit(bk, 0, trotter_steps, qubits, s, 'zzz', c2r=1, dphi=0, hs=hs, order=order, nr=nr, optlevel=None, stomo=False, name='mit')
        qct.name = qct.name + "cal_" + sl # to match with qiskit internel code
        qcs.append(qct)
    return qcs, stls
def gen_mitigation_circs(bk, qubits, trotter_steps, hs, order, nr, name):
    qcs = []
    for n in nr:
        qc, stls = _gen_mitigation_circs(bk, qubits, trotter_steps, hs, order, n, name)
        qcs += qc
    return qcs, stls
def get_fitters(result, stls, trotter_steps, hs, order, nr):
    fts = dict()
    for n in nr:
        name = get_circuit_name(0.0, trotter_steps, order, hs, n, 'mit')
        ft = CompleteMeasFitter(result, stls, circlabel=name)
        fts[name] = ft
    return fts
def get_mitigation(bk, tsteps, shots, hs, order, jname):
    clabel = 'mit'
    schds = None
    stls = None
    if 0:
        qubits = [1,3,5]
        qr = QuantumRegister(3)
        mqcs, stls = complete_meas_cal(qr=qr, circlabel=clabel)
        t_mqcs = transpile(mqcs, bk, initial_layout=qubits, optimization_level=0)
        qobj = assemble(t_mqcs, shots=8)
        res = bk.run(qobj, job_name='mfit_{}'.format(jname)).result()
    else:
        mqcs, stls = _gen_mitigation_circs(bk, [3,1,5], tsteps, hs, order, name=clabel)
        if bk.configuration().open_pulse:
            schds = schedule(mqcs, bk)
            job, res = run_expr(bk, schds, shots, 'mfit_{}'.format(jname))
        else:
            job, res = run_expr(bk, mqcs, shots, 'mfit_{}'.format(jname))
    mfitter = CompleteMeasFitter(res, stls, circlabel=clabel)
    with open('mfitter_{}.pk'.format(jname), 'wb') as fp:
        pickle.dump(mfitter, fp)
    return mfitter, mqcs, stls, schds

from qiskit.ignis.verification.tomography.fitters.lstsq_fit import make_positive_semidefinite
from qiskit.quantum_info.operators.predicates import is_positive_semidefinite_matrix
def run_zne(gres):
    rho0 = gres[0].rho
    rho1 = gres[1].rho
    rho = 1.5*rho0 - 0.5*rho1
    if not is_positive_semidefinite_matrix(rho):
        rho = make_positive_semidefinite(rho)
    target_state_mat = (One^One^Zero).to_matrix()
    if rho.shape==(8,8):
        fid0 = state_fidelity(rho0, target_state_mat)
        print(fid0)
        fid1 = state_fidelity(rho1, target_state_mat)
        print(fid1)
        fid = state_fidelity(rho, target_state_mat)
        print("fidelity: {}, {} -> {}".format(fid0, fid1, fid))
    else:
        rho0 = filter_rho_423(rho0)
        rho1 = filter_rho_423(rho1)
        rho = filter_rho_423(rho)
        fid0 = state_fidelity(rho0, target_state_mat)
        print(fid0)
        fid1 = state_fidelity(rho1, target_state_mat)
        print(fid1)
        fid = state_fidelity(rho, target_state_mat)
        print("fidelity: {}, {} -> {}".format(fid0, fid1, fid))
    return fid
def filter_rho_423(rho):
    assert rho.shape==(16,16)
    op = np.zeros((16,16))
    for i in range(8):
        op[i,i] = 1
    r = np.dot(op, np.dot(rho,op))
    t = np.trace(np.dot(op, np.dot(op, rho)))
    r = r/t
    r = r[0:8, 0:8]
    return r

def process_results(job, result, hs, tsteps, order, theta, ss, meas, c2r, dphi, qcs, schds, mfitter, jname):
    d = dict()
    d['job_id'] = job.job_id()
    d['tsteps'] = tsteps
    d['hs'] = hs
    d['order'] = order
    d['theta'] = theta
    d['ss'] = ss
    d['meas'] = meas
    d['c2r'] = c2r
    d['dphi'] = dphi
    d['qc'] = qcs
    d['schd'] = schds
    d['result'] = result
    with open('rundata_{}.pk'.format(jname), 'wb') as fp:
        pickle.dump(d, fp)

    if mfitter is None and bool(mqcs):
        mfitter = CompleteMeasFitter(result, mstls, circlabel='mit')
    
    res = rearange_result(result, ss, meas, theta, c2r, dphi)
    d['resdata'] = res
    if mfitter is not None:
        resmit = mfitter.filter.apply(result)
        resdmit = rearange_result(resmit, ss, meas, theta, c2r, dphi)
        d['resdmit'] = resdmit
        d['mfitter'] = mfitter
    with open('rundata_{}.pk'.format(jname), 'wb') as fp:
        pickle.dump(d, fp)
    
    plt.ion()
    title = '{}, {}, {} steps, {} order'.format(ss[0], hs, tsteps, order)
    plot_expr_result(ss[0], meas, theta, c2r, dphi, res[0], 'time', title)
    if mfitter is not None:
        title = title + ', error mit..ed'
        plot_expr_result(ss[0], meas, theta, c2r, dphi, resdmit[0], 'time', title)

if 0:
    #bk = FakeJakarta()
    bk = get_backend('jakarta',1)
    is_real = not (isinstance(bk, QasmSimulator) or isinstance(bk, FakeJakarta))
    shots = 8192*2 if is_real else 100000
    #shots = 1024
    #bk = get_backend('quito',1)
    cfg = bk.configuration()
    dft = bk.defaults()
    tsteps = 1
    hs = 'xzy'
    order = 0
    ss = None
    theta = np.linspace(0, np.pi*2, 45)
    #theta = None
    c2r = [0.66, 0.77]

#def run_expr_plots(bk, tsteps, shots, hs, order):
    qubits = [3, 1, 5] # [t, c0, c1]
    jname = 'cmo{}_o{}_t{}_s{}'.format(hs, order, tsteps, shots)
    mfitter = None
    schds = None
    qcs = None

    #if 0:
    #    print('running error mit: {}'.format(jname))
    #    mfitter, mqcs, stls, mschds = get_mitigation(bk, tsteps, shots, hs, order, jname)
    #else:
    #    try:
    #        with open('mfitter_{}.pk'.format(jname), 'rb') as fp:
    #            mfitter = pickle.load(fp)
    #    except:
    #        print("loading mfitter.pk failed")
    #        mfitter = None
    #if mfitter is not None:
    #    print("mit matrix:\n{}\n".format(mfitter.cal_matrix))

    #theta, ss, meas, c2r, dphi, qcs, schds = gen_experiments(bk, qubits, theta=np.linspace(0, np.pi, 40), ss=[ITSS[6]], meas=['zzz'], c2r=[0.6], dphi=[0], tsteps=tsteps, hs=hs, order=order)
    if 1:
        mqcs, mstls = _gen_mitigation_circs(bk, qubits, tsteps, hs, order, 'mit')
        if bk.configuration().open_pulse:
            mschds = schedule(mqcs, bk)
        else:
            mschds = []
    else:
        mqcs = []
        mschds = []

    if 1:
        #theta, ss, meas, c2r, dphi, qcs, schds = gen_experiments(bk, qubits, tsteps=tsteps, hs=hs, order=order)
        theta, ss, meas, c2r, dphi, qcs, schds = gen_experiments(bk, qubits, theta=theta, ss=ss, c2r=c2r, tsteps=tsteps, hs=hs, order=order)
        qcs = mqcs + qcs
        if bool(mschds):
            schds = mschds + schds
    if 1:
        print('running circuit: {}'.format(jname))
        if schds:
            print('running schedule')
            job, result = run_expr(bk, schds, shots, jname)
        else:
            print('running circuit')
            job, result = run_expr(bk, qcs, shots, jname)
        process_results(job, result, hs, tsteps, order, theta, ss, meas, c2r, dphi, qcs, schds, mfitter, jname)

if 0: # pulse tomo
    target_time = np.pi
    initial_state = One^One^Zero
    bk = get_backend('jakarta',1)
    is_real = not (isinstance(bk, QasmSimulator) or isinstance(bk, FakeJakarta))
    shots = bk.configuration().max_shots
    rep = 8
    order = 1
    nr = [0,1]
    trotter_steps = 8
    hs = 'xzy'
    qubits = [3, 1, 5] # t, c, c
    c2r = 0.66
    dphi = 0
    jname = 'plc_{}_o{}_s{}_st{}'.format(hs, order, trotter_steps, shots)
    mfitter = None
    schds = None
    qcs = None

    mqcs, mstls = gen_mitigation_circs(bk, qubits, trotter_steps, hs, order, nr, 'mit')
    tqcs = gen_tomo_experiments(bk, target_time, trotter_steps, qubits, initial_state, order=order, hs=hs, c2r=c2r, dphi=dphi, nr=nr)
    qcs = mqcs + tqcs

    qsub = qcs
    if bk.configuration().open_pulse:
        schds = schedule(qcs, bk)
        qsub = schds

    if 1:
        jobs = submit_expr(bk, qsub, shots=shots, jname=jname, rep=rep)
        results = retrieve_results(jobs)

        for result in results:
            mfitters = get_fitters(result, mstls, trotter_steps, hs, order, nr)
            gres = group_results(result, qcs, mfitters)
            fid = run_zne(gres)
            fids.append(fid)
        print('state tomography fidelity = {:.4f} \u00B1 {:.4f}'.format(np.mean(fids), np.std(fids)))

        fids = []
        for result in results:
            gres = group_results(result, qcs, None)
            fid = run_zne(gres)
            fids.append(fid)
        print('state tomography fidelity, no mfit = {:.4f} \u00B1 {:.4f}'.format(np.mean(fids), np.std(fids)))


if 0: # 2
    bk = get_backend('jakarta',0)
    is_real = not (isinstance(bk, QasmSimulator) or isinstance(bk, FakeJakarta))
    shots = 100000
    order = [1,2]
    tsteps = [4, 6, 8, 10]
    ncx = [0]
    time = np.linspace(0, np.pi, 20)
    hs = 'xzy'
    jname = 'xzy_ts4680_o12_mit82'
    mqc, mqct, mstls = _gen_mitigation_circs2(bk, 4, tsteps, hs, order, ncx)
    qct = gen_experiments2(bk, time, tsteps, order, hs, ncx)
    qct = mqct + qct
    job, result = run_expr(bk, qct, shots, jname)
    mfitters = get_fitters2(result, mstls, tsteps, hs, order, ncx)

    res = filter_result2(result, mfitters)
    data = rearange_result2(res, time, tsteps, order, hs, ncx)
    plot_expr_result2(data, time, tsteps, order, hs, ncx, 'time', jname)

    resnf = filter_result2(result, None)
    datanf = rearange_result2(resnf, time, tsteps, order, hs, ncx)
    plot_expr_result2(datanf, time, tsteps, order, hs, ncx, 'time', jname+'_nf')

if 0: # tomo
    #bk = FakeJakarta()
    bk = get_backend('jakarta',0)
    is_real = not (isinstance(bk, QasmSimulator) or isinstance(bk, FakeJakarta))
    shots = bk.configuration().max_shots
    rep = 1
    order = [2]
    tsteps = [4]
    ncx = [0,1]
    ncx_mf = [-2]
    hs = 'xzy'
    do_parity = 1
    jname = 'qc_{}_o{}_t{}_n{}_p{}'.format(hs, order, tsteps, ncx, do_parity)
    nbit = 4 if do_parity else 3
    mqc, mqct, mstls = _gen_mitigation_circs2(bk, nbit, tsteps, hs, order, ncx_mf, 1, do_parity)
    qct = gen_tomo_experiments2(bk, tsteps, order, hs, ncx, do_parity)
    qcs = mqct + qct
    jobs = submit_expr(bk, qcs, shots, jname, rep)
    results = retrieve_results(jobs)
    with open('{}.pk'.format(jname), 'wb') as fp:
        pickle.dump([results, bk, shots, rep, order, tsteps, ncx, ncx_mf, hs, jname, qcs], fp)
    #with open('tmp.pk', 'rb') as fp:
    #    result = pickle.load(fp)
    fids = []
    for result in results:
        mfitters = get_fitters2(result, mstls, tsteps, hs, order, ncx_mf)
        gres = group_results(result, qcs, mfitters)
        fid = run_zne(gres)
        fids.append(fid)
    print('state tomography fidelity = {:.4f} \u00B1 {:.4f}'.format(np.mean(fids), np.std(fids)))
    
    fids = []
    for result in results:
        gres = group_results(result, qcs, None)
        fid = run_zne(gres)
        fids.append(fid)
    print('state tomography fidelity, no mfit = {:.4f} \u00B1 {:.4f}'.format(np.mean(fids), np.std(fids)))
