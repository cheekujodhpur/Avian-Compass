import theano
import qutip as qt
import numpy as np
from theano import sparse

Sx = sparse.csr_matrix(name='sx', dtype='complex128');
Sy = sparse.csr_matrix(name='sy', dtype='complex128');
Sz = sparse.csr_matrix(name='sz', dtype='complex128');

G = sparse.csr_matrix(name='g', dtype='complex128');
G = Sx*Sx.T + Sx*Sy.T + Sx*Sz.T +\
    Sy*Sx.T + Sy*Sy.T + Sy*Sz.T +\
    Sz*Sx.T + Sz*Sy.T + Sz*Sz.T

f = theano.function([Sx, Sy, Sz], G)

sx_a 	 = qt.tensor(qt.sigmax(), qt.identity(288)).data
sy_a 	 = qt.tensor(qt.sigmay(), qt.identity(288)).data
sz_a     = qt.tensor(qt.sigmaz(), qt.identity(288)).data

sx_b 	 = qt.tensor(qt.sigmax(), qt.identity(192)).data
sy_b 	 = qt.tensor(qt.sigmay(), qt.identity(192)).data
sz_b     = qt.tensor(qt.sigmaz(), qt.identity(192)).data

g_a = f(sx_a, sy_a, sz_a)
g_b = f(sx_b, sy_b, sz_b)
