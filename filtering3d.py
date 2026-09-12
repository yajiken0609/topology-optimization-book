import numpy as np
from scipy import ndimage
class Filtering:
    def __init__(self, n1, n2, n3, sigma):
        self.shape = (n1, n2, n3)
        self.sigma = sigma
    def apply(self, x):
        x_3d = x.reshape(self.shape, order='F')
        x_tilde_3d = ndimage.gaussian_filter(x_3d, sigma=self.sigma, mode='reflect')
        return x_tilde_3d.flatten(order='F')
    def grayscale_indicator(self, x):
        return np.sum(4*x*(1-x))/len(x)*100
# 親クラスを継承してヘビサイドプロジェクションのクラスを作成
class Projection(Filtering):
    def __init__(self, n1, n2, n3, sigma, eta, beta, beta_max=16, coeff=2, interval=25):
        super().__init__(n1, n2, n3, sigma)
        self.eta, self.beta, self.beta_max = eta, beta, beta_max
        self.coeff, self.interval, self.x_filt = coeff, interval, None
    def _proj(self, x, d=False):
        num = np.tanh(self.beta*self.eta)+np.tanh(self.beta*(x-self.eta))
        den = np.tanh(self.beta*self.eta)+np.tanh(self.beta*(1-self.eta))
        return self.beta*(1-np.tanh(self.beta*(x-self.eta))**2)/den if d else num/den
    def apply(self, itr, x):
        self.x_filt = super().apply(x)
        if itr > 0 and itr % self.interval == 0: 
            self.beta = min(self.beta*self.coeff, self.beta_max)
        return self._proj(self.x_filt)
    def modify_sensitivity(self, grad_proj, g=False):
        grad = super().apply(grad_proj*self._proj(self.x_filt, d=True))
        return grad[None, :] if g else grad
# Projectionクラスを継承してAMFilterクラスを作成
class AMFilter(Projection):
    def __init__(self, n1, n2, n3, r, eta, beta, P=40, eta_s=0.5, eps=1e-4):
        super().__init__(n1, n2, n3, r, eta, beta)
        self.P, self.eps, self.Q = P, eps, P + np.log(5)/np.log(eta_s)
        self.shifts = [(0,0), (1,0), (-1,0), (0,1), (0,-1)]
    def _shift(self, a, sx, sy):
        r = np.roll(a, (sx, sy), axis=(0, 1))
        if sx: r[0 if sx==1 else -1, :] = 0
        if sy: r[:, 0 if sy==1 else -1] = 0
        return r
    def _xi(self, x, Xi, d=0):
        sq = np.sqrt((x-Xi)**2+self.eps)
        if d == 0: return 0.5*(x+Xi-sq+np.sqrt(self.eps))
        return 0.5*(1+(-1 if d==1 else 1)*(x-Xi)/sq)
    def apply(self, itr, x):
        xp = super().apply(itr, x).reshape(self.shape, order='F')
        z, Xi = np.zeros_like(xp), np.ones_like(xp)
        for k in range(self.shape[2]):
            if k > 0:
                s = sum(self._shift(z[:,:,k-1]**self.P, sx, sy) for sx, sy in self.shifts)
                Xi[:,:,k] = s**(1.0/self.Q)
            z[:,:,k] = self._xi(xp[:,:,k], Xi[:,:,k])
        self.store = {'xp': xp, 'z': z, 'Xi': Xi}
        return z.flatten(order='F')
    def modify_sensitivity(self, grad_z, g=False):
        gz, lam = grad_z.reshape(self.shape, order='F'), np.zeros(self.shape[:2])
        g_xp = np.zeros_like(gz)
        xp, z, Xi = self.store['xp'], self.store['z'], self.store['Xi']
        EPS_P = 1e-6
        for k in reversed(range(self.shape[2])):
            lam += gz[:,:,k]
            g_xp[:,:,k] = lam*self._xi(xp[:,:,k], Xi[:,:,k], d=1)
            if k > 0:
                c = lam*self._xi(xp[:,:,k], Xi[:,:,k], d=2)*(self.P/self.Q)*(Xi[:,:,k]+EPS_P)**(1-self.Q)
                lam = sum(self._shift(c, -sx, -sy) for sx, sy in self.shifts)*z[:,:,k-1]**(self.P-1)
        return super().modify_sensitivity(g_xp.flatten(order='F'), g=g)