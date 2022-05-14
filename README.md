# double_cross_resonance_gate
## Three qubit cross resonance gate with two controls and one target

The 'cross resonance' operation has been the fundamental mechanism to realize the CNOT gate, which is the basic gate to entagle twoo qubits. It is described with a ZX Hamiltonian, applying Z operator ot the control qubit and X on the target qubit. The 'double cross resonance' gate extends this operation to three qubits, following a Hamiltonian of ZXI+IXZ. This can be further extended to 'tripple/quadruple cross resonance' gates and so on. The intention of devicing those multiple qubit gates is to reduce the quantum circuit depth for certain calculations/simulations.

This shared package is a work for 'IBM open-science-prize-2021'. It implements a prototype of the double cross resonance gate and uses it to simulate the Heisenberg model of three spins. The purpose of this post is to share the idea of this multi-qubit gate. It is preliminarily tuned and please bare with the uncleaned original code. 

The details are presented in 'main.ipynb'. All other are supporting code or saved data.
