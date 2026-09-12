import numpy as np
from scipy import sparse
class Filtering:
    def __init__(self, n1, n2, r):
        self.n1, self.n2, self.r = n1, n2, r
        self._initialize_filter()
    def _get_weight(self, dist):
        return self.r - dist
    def _initialize_filter(self):
        n1, n2, r = self.n1, self.n2, self.r
        dy, dx = np.mgrid[-int(r):int(r)+1, -int(r):int(r)+1]
        dist = np.sqrt(dx**2 + dy**2)
        mask = dist <= r
        ox, oy, w = dx[mask], dy[mask], self._get_weight(dist[mask])
        idx = np.arange(n1*n2)
        jx = np.repeat(idx%n1, len(w))+np.tile(ox, len(idx))
        jy = np.repeat(idx//n1, len(w))+np.tile(oy, len(idx))
        valid = (jx>=0) & (jx<n1) & (jy>=0) & (jy<n2)
        self.H = sparse.coo_matrix((np.tile(w, len(idx))[valid], (np.repeat(idx, len(w))[valid], jy[valid]*n1 + jx[valid])), shape=(n1*n2, n1*n2))
        self.Hs = np.array(self.H.sum(1)).flatten()
    def apply(self, x):
        return self.H.dot(x)/self.Hs
    def grayscale_indicator(self, x):
        return np.sum(4*x*(1-x))/(self.n1*self.n2)*100
# 親クラスを継承してヘビサイドプロジェクションのクラスを作成
class Projection(Filtering):
    def __init__(self, n1, n2, r, eta, beta, beta_max=16, coeff=2, interval=25):
        super().__init__(n1, n2, r)
        self.eta, self.beta, self.beta_max = eta, beta, beta_max
        self.coeff, self.interval, self.x_filt = coeff, interval, None
    def _proj(self, x, d=False):
        num = np.tanh(self.beta*self.eta)+np.tanh(self.beta*(x-self.eta))
        den = np.tanh(self.beta*self.eta)+np.tanh(self.beta*(1-self.eta))
        return self.beta*(1-np.tanh(self.beta*(x-self.eta))**2)/den if d else num/den
    def apply(self, itr, x):
        self.x_filt = super().apply(x)
        if itr > 0 and itr % self.interval == 0: self.beta = min(self.beta*self.coeff, self.beta_max)
        return self._proj(self.x_filt)
    def modify_sensitivity_gradf(self, gradf):
        return self.H@(gradf*self._proj(self.x_filt, d=True)/self.Hs)
    def modify_sensitivity_gradgi(self, gradgi):
        return (self.H@(gradgi*self._proj(self.x_filt, d=True)/self.Hs).T).T
# 親クラスを継承して最大寸法制約のクラスを作成
class MaxLength(Filtering):
    def __init__(self, n1, n2, r, b):
        super().__init__(n1, n2, r)
        self.b = b
    def get_alpha(self):
        return 1-2/np.pi*(np.arccos(self.b/self.r)-self.b/self.r*np.sqrt(1-(self.b/self.r)**2))
    def _get_weight(self, dist):
        return np.ones_like(dist)
    def modify_sensitivity_gradgi(self, gradgi):
        return (self.H@(gradgi/self.Hs).T).T