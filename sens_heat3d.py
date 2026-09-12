import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
import pyamg
class SensHeat3D:
    def __init__(self):
        self.n1, self.n2, self.n3 =  75, 75, 150
        self.L1, self.L2, self.L3 = 0.5, 0.5, 1.0
        self.h = self.L1 / self.n1 
        self.k, self.k0, self.Q, self.p = 1.0, 0.001, 2.0, 3.0
        self.Ke = (self.h/36)*np.array([
            [ 12,   0,   0,  -3,   0,  -3,  -3,  -3],
            [  0,  12,  -3,   0,  -3,   0,  -3,  -3],
            [  0,  -3,  12,   0,  -3,  -3,   0,  -3],
            [ -3,   0,   0,  12,  -3,  -3,  -3,   0],
            [  0,  -3,  -3,  -3,  12,   0,   0,  -3],
            [ -3,   0,  -3,  -3,   0,  12,  -3,   0],
            [ -3,  -3,   0,  -3,   0,  -3,  12,   0],
            [ -3,  -3,  -3,   0,  -3,   0,   0,  12]])
        self._setup_fem()
    def _setup_fem(self):
        N = (self.n1+1)*(self.n2+1)*(self.n3+1)
        nodes = np.arange(N).reshape(self.n3+1, self.n2+1, self.n1+1)
        n0 = nodes[:-1, :-1, :-1].flatten() 
        sx, sy, sz = 1, self.n1+1, (self.n1+1)*(self.n2+1)        
        offsets = np.array([0, sx, sy, sx+sy, sz, sz+sx, sz+sy, sz+sx+sy])
        self.ind_edge = n0[:, None] + offsets
        self.iK = np.repeat(self.ind_edge, 8, axis=0).flatten()
        self.jK = np.repeat(self.ind_edge, 8, axis=1).flatten()
        ind_fixed = np.array([0])
        self.free = np.setdiff1d(np.arange(N), ind_fixed)
        wx = np.full(self.n1+1, 1.0); wx[[0, -1]] = 0.5
        wy = np.full(self.n2+1, 1.0); wy[[0, -1]] = 0.5
        wz = np.full(self.n3+1, 1.0); wz[[0, -1]] = 0.5
        W = np.einsum('k,j,i->kji', wz, wy, wx).flatten()
        self.f = W * (self.h**3) * self.Q
    def _solve_forward(self, x):
        sK = (self.Ke.flatten()[:, None]*(self.k0+(self.k-self.k0)*x**self.p)).flatten('F')
        self.K = sparse.coo_matrix((sK, (self.iK, self.jK))).tocsr()
        K_free = self.K[self.free, :][:, self.free]
        f_free = self.f[self.free]
        ml = pyamg.smoothed_aggregation_solver(K_free)
        M = ml.aspreconditioner(cycle='V')
        u_free, _ = spla.cg(K_free, f_free, M=M, rtol=1e-5, atol=1e-5)        
        self.d = np.zeros(len(self.f))
        self.d[self.free] = u_free
    def compute_obj_sens(self, x):
        self._solve_forward(x)
        f_TC = np.dot(self.f, self.d)
        gradf_TC = -self.p*(self.k-self.k0)*x**(self.p-1)*np.sum((self.d[self.ind_edge]@self.Ke)*self.d[self.ind_edge], axis=1)
        return f_TC, gradf_TC