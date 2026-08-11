import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
class SensHeat:
    def __init__(self):
        self.n1, self.n2 = 100, 100
        self.L1, self.L2 = 1.0, 1.0
        self.h = self.L1/self.n1
        self.k, self.k0, self.Q, self.p = 1.0, 0.001, 2.0, 3.0
        self.Ke = (1/6) * np.array([
            [4, -1, -2, -1],
            [-1, 4, -1, -2],
            [-2, -1, 4, -1],
            [-1, -2, -1, 4]])
        self._setup_fem()
    def _setup_fem(self):
        ind_node = np.arange((self.n1+1)*(self.n2+1)).reshape(self.n2+1, self.n1+1)
        ind_elem = ind_node[:-1, :-1].ravel()
        self.ind_edge = ind_elem[:, None] + [0, 1, self.n1+2, self.n1+1]
        self.iK = np.repeat(self.ind_edge, 4, axis=0).flatten()
        self.jK = np.repeat(self.ind_edge, 4, axis=1).flatten()
        ind_fixed = self.n1//2
        self.free = np.setdiff1d(np.arange((self.n1+1)*(self.n2+1)), ind_fixed)
        ind_bound = np.concatenate([
            np.arange(1, self.n1+1), 
            np.arange(2*(self.n1+1)-1, (self.n1+1)*(self.n2+1), self.n1+1),
            np.arange((self.n1+1)*self.n2+1, (self.n1+1)*(self.n2+1)-1), 
            np.arange(0, (self.n1+1)*(self.n2+1), self.n1+1)])
        ind_inside = np.setdiff1d(np.arange((self.n1+1)*(self.n2+1)), ind_bound)
        ind_bound_adiabatic = np.setdiff1d(ind_bound, ind_fixed)
        ind_bound_except_corners = np.setdiff1d(ind_bound, [0, self.n1, self.n2*(self.n1+1), (self.n1+1)*(self.n2+1)-1])
        self.f = np.zeros((self.n1+1)*(self.n2+1))
        self.f[ind_inside] = self.h**2*self.Q
        self.f[ind_bound_adiabatic] += self.h**2*self.Q/4
        self.f[ind_bound_except_corners] += self.h**2*self.Q/4
    def _solve_forward(self, x):
        sK = (self.Ke.flatten()[:, None]*(self.k0+(self.k-self.k0)*x**self.p)).flatten('F')
        self.K = sparse.coo_matrix((sK, (self.iK, self.jK))).tocsc()
        self.d = np.zeros((self.n1+1)*(self.n2+1))
        self.d[self.free] = spla.spsolve(self.K[self.free][:, self.free], self.f[self.free])
    def compute_obj_sens(self, x):
        self._solve_forward(x)
        f_TC = np.dot(self.f, self.d)
        gradf_TC = -self.p*(self.k-self.k0)*x**(self.p-1)*np.sum((self.d[self.ind_edge]@self.Ke)*self.d[self.ind_edge], axis=1)
        return f_TC, gradf_TC
# 親クラスを継承して部分領域の温度と感度を取得する子クラスを定義
class SubsetTemp(SensHeat):
    def __init__(self):
        super().__init__()
        self._setup_fem()
    def _setup_fem(self):
        super()._setup_fem()
        x_p, y_p = np.meshgrid(np.linspace(0, self.L1, self.n1+1), np.linspace(0, self.L2, self.n2+1))        
        c1x, c1y, c2x, c2y = 0.3, 0.3, 0.7, 0.7
        rc = 0.3
        ind_obj_d1 = (np.sqrt((x_p-c1x)**2+(y_p-c1y)**2)<rc).flatten()
        ind_obj_d2 = (np.sqrt((x_p-c2x)**2+(y_p-c2y)**2)<rc).flatten()
        self.ind_obj_domain = ind_obj_d1 | ind_obj_d2 
        self.f_obj = np.zeros((self.n1+1)*(self.n2+1))
        self.f_obj[self.ind_obj_domain] = self.h**2
    def _solve_adjoint(self):
        self.adj_d = np.zeros((self.n1+1)*(self.n2+1))
        self.adj_d[self.free] = spla.spsolve(self.K[self.free][:, self.free], self.f_obj[self.free])
    def compute_obj_sens(self, x):
        self._solve_forward(x)
        f_T = np.dot(self.f_obj, self.d)
        self._solve_adjoint()
        gradf_T = -self.p*(self.k-self.k0)*x**(self.p-1)*np.sum((self.adj_d[self.ind_edge]@self.Ke)*self.d[self.ind_edge], axis=1)
        return f_T, gradf_T
# 親クラスを継承して対称境界条件のクラスを作成
class SensHeatSym(SensHeat):
    def __init__(self):
        super().__init__()
        self.n1 = self.n1//2
        self.L1 = self.L1/2
        self.h = self.L1/self.n1
        self._setup_fem()
    def _setup_fem(self):
        super()._setup_fem()
        ind_fixed = np.array([0])
        self.free = np.setdiff1d(np.arange((self.n1+1)*(self.n2+1)), ind_fixed)
        ind_bound = np.concatenate([
            np.arange(1, self.n1+1), 
            np.arange(2*(self.n1+1)-1, (self.n1+1)*(self.n2+1), self.n1+1),
            np.arange((self.n1+1)*self.n2+1, (self.n1+1)*(self.n2+1)-1), 
            np.arange(0, (self.n1+1)*(self.n2+1), self.n1+1)])
        ind_inside = np.setdiff1d(np.arange((self.n1+1)*(self.n2+1)), ind_bound)
        ind_bound_adiabatic = np.setdiff1d(ind_bound, ind_fixed)
        ind_bound_except_corners = np.setdiff1d(ind_bound, [0, self.n1, self.n2*(self.n1+1), (self.n1+1)*(self.n2+1)-1])
        self.f = np.zeros((self.n1+1)*(self.n2+1))
        self.f[ind_inside] = self.h**2*self.Q
        self.f[ind_bound_adiabatic] += self.h**2*self.Q/4
        self.f[ind_bound_except_corners] += self.h**2*self.Q/4