import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
class SensElasticCompmech:
    def __init__(self):
        self.n1, self.n2 = 100, 50
        self.L1, self.L2 = 1.0, 0.5
        self.h = self.L1/self.n1
        self.nnx, self.nny = self.n1+1, self.n2+1
        self.ndof = 2*self.nnx*self.nny
        self.E, self.E0, self.nu, self.p = 1.0, 1e-9, 0.3, 3.0
        self.f_in, self.f_out, self.k_in, self.k_out = 1.0, -1.0, 1.0, 1e-2
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
        K_b = sparse.coo_matrix((sK, (self.iK, self.jK))).tocsc()
        K_in = sparse.coo_matrix((np.array([self.k_in]), (np.array([self.ind_in]), np.array([self.ind_in]))), shape=(self.ndof, self.ndof))
        K_out = sparse.coo_matrix((np.array([self.k_out]), (np.array([self.ind_out]), np.array([self.ind_out]))), shape=(self.ndof, self.ndof))
        self.K = K_b + K_in + K_out
        self.d = np.zeros(self.ndof)
        self.d[self.free] = spla.spsolve(self.K[self.free][:, self.free], self.f[self.free])
    def _solve_adjoint(self):
        self.adj_d = np.zeros(self.ndof)
        self.adj_d[self.free] = spla.spsolve(self.K[self.free][:, self.free], self.adj_f[self.free])
    def compute_obj_sens(self, x):
        self._solve_forward(x)
        f_CM = -self.f_out*self.d[self.ind_out]
        self._solve_adjoint()
        gradf_CM = -self.p*(self.E-self.E0)*x**(self.p-1)*np.sum((self.adj_d[self.ind_edge]@self.Ke)*self.d[self.ind_edge], axis=1)
        return f_CM, gradf_CM
# 親クラスを継承して変位インバータの子クラスを定義
class Invertor(SensElasticCompmech):
    def __init__(self):
        super().__init__()
        self._setup_boundary_conditions()
    def _setup_boundary_conditions(self):
        self.ind_in = 2*(self.n2*self.nnx) 
        self.ind_out = 2*(self.n2*self.nnx+self.n1)
        fix_bl = [0, 1]
        fix_top = 2*np.arange(self.n2*self.nnx, self.ndof//2)+1
        self.fixed = np.union1d(fix_bl, fix_top)
        self.free = np.setdiff1d(np.arange(self.ndof), self.fixed)
        self.f = np.zeros(self.ndof)
        self.f[self.ind_in] = self.f_in
        self.adj_f = np.zeros(self.ndof)
        self.adj_f[self.ind_out] = -self.f_out
# 親クラスを継承しグリッパの子クラスを定義
class Gripper(SensElasticCompmech):
    def __init__(self):
        super().__init__()
        self.f_out = 1.0
        self._setup_boundary_conditions()
        self._setup_passive_regions()
    def _setup_boundary_conditions(self):
        self.ind_in = 2*(self.n2*self.nnx) 
        self.ind_out = 2*(int(self.n2*0.6)*self.nnx+self.n1)+1
        fix_bl = [0, 1]
        fix_top = 2*np.arange(self.n2*self.nnx, self.ndof//2)+1
        self.fixed = np.union1d(fix_bl, fix_top)
        self.free = np.setdiff1d(np.arange(self.ndof), self.fixed)
        self.f = np.zeros(self.ndof)
        self.f[self.ind_in] = self.f_in
        self.adj_f = np.zeros(self.ndof)
        self.adj_f[self.ind_out] = -self.f_out
    def _setup_passive_regions(self):
        el_x, el_y = np.meshgrid(np.linspace(self.h/2, self.L1-self.h/2, self.n1),
                                 np.linspace(self.h/2, self.L2-self.h/2, self.n2))
        is_right = el_x > 0.75*self.L1
        self.pas_solid = (is_right&(el_y>=0.5*self.L2)&(el_y<=0.6*self.L2)).flatten()
        self.pas_void  = (is_right&(el_y>0.6*self.L2)).flatten()
    def compute_obj_sens(self, x):
        x[self.pas_solid] = 1.0
        x[self.pas_void] = 0.0
        f_CM, gradf_CM = super().compute_obj_sens(x)        
        gradf_CM[self.pas_solid|self.pas_void] = 0.0
        return f_CM, gradf_CM