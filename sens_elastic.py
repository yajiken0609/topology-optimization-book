import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
class SensElastic:
    def __init__(self):
        self.n1, self.n2 = 100, 50
        self.L1, self.L2 = 2.0, 1.0
        self.h = self.L1/self.n1
        self.nnx, self.nny = self.n1+1, self.n2+1
        self.ndof = 2*self.nnx*self.nny
        self.E, self.E0, self.nu, self.ft, self.p = 1.0, 1e-9, 0.3, -1.0, 3.0
        self._setup_Ke()
        self._setup_indices()
    def _setup_Ke(self):
        k1 = np.array([
            [ 12,  3, -6, -3, -6, -3,  0,  3],
            [  3, 12,  3,  0, -3, -6, -3, -6],
            [ -6,  3, 12, -3,  0, -3, -6,  3],
            [ -3,  0, -3, 12,  3, -6,  3, -6],
            [ -6, -3,  0,  3, 12,  3, -6, -3],
            [ -3, -6, -3, -6,  3, 12,  3,  0],
            [  0, -3, -6,  3, -6,  3, 12, -3],
            [  3, -6,  3, -6, -3,  0, -3, 12]])
        k2 = np.array([
            [-4,  3, -2,  9,  2, -3,  4, -9],
            [ 3, -4, -9,  4, -3,  2,  9, -2],
            [-2, -9, -4, -3,  4,  9,  2,  3],
            [ 9,  4, -3, -4, -9, -2,  3,  2],
            [ 2, -3,  4, -9, -4,  3, -2,  9],
            [-3,  2,  9, -2,  3, -4, -9,  4],
            [ 4,  9,  2,  3, -2, -9, -4, -3],
            [-9, -2,  3,  2,  9,  4, -3, -4]])        
        self.Ke = (k1+self.nu*k2)*(1/(1-self.nu**2)/24)
    def _setup_indices(self):
        ind_node = np.arange((self.nny)*(self.nnx)).reshape((self.nny), (self.nnx))
        ind_elem = (2*(ind_node[:-1, :-1])).ravel()
        self.ind_edge = ind_elem[:, None]+[0, 1, 2, 3, 2*(self.nnx)+2, 2*(self.nnx)+3, 
                                           2*(self.nnx), 2*(self.nnx)+1]    
        self.iK = np.repeat(self.ind_edge, 8, axis=0).flatten()
        self.jK = np.repeat(self.ind_edge, 8, axis=1).flatten()
    def _solve_forward(self, x):
        sK = (self.Ke.flatten()[:, None]*(self.E0+(self.E-self.E0)*x**self.p)).flatten('F')
        self.K = sparse.coo_matrix((sK, (self.iK, self.jK))).tocsc()
        self.d = np.zeros(self.ndof)
        self.d[self.free] = spla.spsolve(self.K[self.free][:, self.free], self.f[self.free])
    def compute_obj_sens(self, x):
        self._solve_forward(x)
        f_MC = np.dot(self.f, self.d)
        gradf_MC = -self.p*(self.E-self.E0)*x**(self.p-1)*np.sum((self.d[self.ind_edge]@self.Ke)*self.d[self.ind_edge], axis=1)
        return f_MC, gradf_MC
# 親クラスを継承して片持ちはりのクラスを作成
class Cantilever(SensElastic):
    def __init__(self):
        super().__init__()
        self._setup_boundary_conditions()
    def _setup_boundary_conditions(self):
        self.fixed = np.union1d(np.arange(0, 2*self.nnx*(self.nny-1)+1, 2*self.nnx), 
                                np.arange(1, 2*self.nnx*(self.nny-1)+2, 2*self.nnx)) 
        self.free = np.setdiff1d(np.arange(self.ndof), self.fixed)
        self.ind_t = 2*self.nnx*(self.nny//2+1)-1
        self.f = np.zeros(self.ndof)
        self.f[self.ind_t] = self.ft
# 親クラスを継承しMBBはりの子クラスを定義
class MBB(SensElastic):
    def __init__(self):
        super().__init__()
        self._setup_boundary_conditions()
    def _setup_boundary_conditions(self):
        ind_left_x = np.arange(0, 2*self.nnx*self.nny, 2*self.nnx)
        ind_right_bottom_y = 2*self.n1+1
        self.fixed = np.union1d(ind_left_x, ind_right_bottom_y)
        self.free = np.setdiff1d(np.arange(self.ndof), self.fixed)
        self.ind_t = 2*((self.nny-1)*self.nnx)+1
        self.f = np.zeros(self.ndof)
        self.f[self.ind_t] = self.ft