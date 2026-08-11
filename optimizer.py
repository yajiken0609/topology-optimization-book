import numpy as np
class SLP:
    def __init__(self, x, m, move_lim=0.05, x_min=0.0, x_max=1.0):
        self.x_old1 = x.copy()
        self.x_old2 = x.copy()
        self.m = m
        self.move_lim = move_lim
        self.x_min = x_min
        self.x_max = x_max
    def _sub_problem(self, itr, x, gi, gradf, gradgi):
        MOVE_INIT, MOVE_FAST, MOVE_SLOW = 0.5, 1.2, 0.7
        if itr < 2:
            self.move = np.full(len(x), MOVE_INIT*self.move_lim)
        else:
            xx = (x-self.x_old1)*(self.x_old1-self.x_old2)
            move_coef = np.where(xx > 0, MOVE_FAST, np.where(xx < 0, MOVE_SLOW, 1.0))
            self.move = np.clip(move_coef*self.move, None, self.move_lim)
        a = np.clip(x-self.move*(self.x_max-self.x_min), self.x_min, None)
        b = np.clip(x+self.move*(self.x_max-self.x_min), None, self.x_max)
        self.x, self.gi, self.gradf, self.gradgi = x, gi, gradf, gradgi
        return a, b
    def _apx_term(self, x):
        df_apx, dgi_apx = self.gradf, self.gradgi
        gi_apx = self.gi+np.dot(self.gradgi, (x-self.x))
        diag_Hf_apx, diag_Hgi_apx = np.zeros_like(df_apx), np.zeros_like(dgi_apx)
        return df_apx, gi_apx, dgi_apx, diag_Hf_apx, diag_Hgi_apx
    @staticmethod        
    def _kkt(x, gi, gradf, gradgi, lam_s, s, y, lam_y, lam_a, lam_b, a, b, rho, PENAL_Y):
        kkt = np.concatenate([
        gradf+np.sum(lam_s[:, None]*gradgi, axis=0)-lam_a+lam_b,
        PENAL_Y-lam_s-lam_y, gi+s-y,
        lam_a*(x-a)-rho, lam_b*(b-x)-rho, lam_s*s-rho, lam_y*y-rho])
        return kkt
    def _solve_sub_problem(self, a, b):
        RHO_TOL, TAU_TOL, KKT_COEF = 1e-7, 0.99, 0.9
        RHO_DECAY, STEP_DECAY = 0.1, 0.5
        INITR_MAX = [200, 200, 200]
        PENAL_Y = 1e4*np.ones(self.m)
        rho = 1.0
        xj = (a+b)/2
        lam_s = s = y = np.ones(self.m)
        lam_y = 0.5*PENAL_Y
        lam_a = np.maximum(np.ones_like(xj), 1/(xj-a))
        lam_b = np.maximum(np.ones_like(xj), 1/(b-xj))
        ind_w = np.cumsum([len(lam_s), len(s), len(y), len(lam_y), 
                           len(lam_a), len(lam_b)])[:-1]
        initr_0 = 0
        while rho > RHO_TOL:
            if initr_0 > INITR_MAX[0]: print(f"initr_0 > {INITR_MAX[0]}"); break
            df_apx, gi_apx, dgi_apx, _, _ = self._apx_term(xj)
            tmp_kkt = self._kkt(xj, gi_apx, df_apx, dgi_apx, lam_s, s, y, lam_y, 
                                lam_a, lam_b, a, b, rho, PENAL_Y)
            norm_l2_kkt = np.linalg.norm(tmp_kkt, 2)
            norm_inf_kkt = np.linalg.norm(tmp_kkt, np.inf)
            initr_1 = 0
            while norm_inf_kkt > KKT_COEF*rho:
                if initr_1 > INITR_MAX[1]: print(f"initr_1 > {INITR_MAX[1]}"); break
                df_apx, gi_apx, dgi_apx, diag_Hf_apx, diag_Hgi_apx = self._apx_term(xj)               
                delta_x = df_apx+np.sum(lam_s[:, None]*dgi_apx, 
                                        axis=0)-rho/(xj-a)+rho/(b-xj)
                delta_ly = gi_apx-y+rho/lam_s+y*(PENAL_Y-lam_s-rho/y)/lam_y
                diag_Psi = diag_Hf_apx+np.sum(lam_s[:, None]*diag_Hgi_apx, 
                                              axis=0)+lam_a/(xj-a)+lam_b/(b-xj)
                diag_D_ly = s/lam_s+y/lam_y
                dlam_s = np.linalg.solve(np.diag(diag_D_ly)
                                         +dgi_apx/diag_Psi[None, :]@dgi_apx.T, 
                                         delta_ly-np.dot(dgi_apx/diag_Psi[None, :],delta_x))
                dx = -delta_x/diag_Psi-np.sum(dgi_apx/diag_Psi[None, :]
                                              *dlam_s[:, None], axis=0)
                ds = -s/lam_s*dlam_s-s+rho/lam_s
                dy = y*dlam_s/lam_y-y*(PENAL_Y-lam_s-rho/y)/lam_y
                dlam_y = -lam_y*dy/y-lam_y+rho/y
                dlam_a = -lam_a/(xj-a)*dx-lam_a+rho/(xj-a)
                dlam_b = lam_b/(b-xj)*dx-lam_b+rho/(b-xj)
                w = np.concatenate([lam_s, s, y, lam_y, lam_a, lam_b])
                dw = np.concatenate([dlam_s, ds, dy, dlam_y, dlam_a, dlam_b])
                inv_t = np.max(-dw/(TAU_TOL*w))
                inv_t = np.maximum(inv_t, np.max(-dx/(TAU_TOL*(xj-a))))
                inv_t = np.maximum(inv_t, np.max(dx/(TAU_TOL*(b-xj))))
                t = 1/np.maximum(inv_t, 1)
                x_old, w_old = xj.copy(), w.copy()
                tmp_norm_l2_kkt = 2*norm_l2_kkt
                initr_2 = 0
                while tmp_norm_l2_kkt > norm_l2_kkt:
                    if initr_2 > INITR_MAX[2]: print(f"initr_2 > {INITR_MAX[2]}"); break
                    xj, w = x_old+t*dx, w_old+t*dw
                    df_apx, gi_apx, dgi_apx, _, _ = self._apx_term(xj)
                    lam_s, s, y, lam_y, lam_a, lam_b = np.split(w, ind_w)
                    tmp_kkt = self._kkt(xj, gi_apx, df_apx, dgi_apx, lam_s, s, y, lam_y, 
                                        lam_a, lam_b, a, b, rho, PENAL_Y)
                    tmp_norm_l2_kkt = np.linalg.norm(tmp_kkt, 2)
                    t *= STEP_DECAY
                    initr_2 += 1
                norm_l2_kkt = tmp_norm_l2_kkt
                norm_inf_kkt = np.linalg.norm(tmp_kkt, np.inf)
                initr_1 += 1
            rho = RHO_DECAY*rho
            initr_0 += 1
        return xj     
    def run(self, itr, x, gi, gradf, gradgi):
        a, b = self._sub_problem(itr, x, gi, gradf, gradgi)
        x_new = self._solve_sub_problem(a, b)
        self.x_old2, self.x_old1 = self.x_old1, x.copy()
        return x_new
# SLPのクラスを継承してMMAのクラスを作成
class MMA(SLP):
    def __init__(self, x, m, move_lim=0.05, x_min=0.0, x_max=1.0):
        super().__init__(x, m, move_lim, x_min, x_max)
        self.low = np.ones_like(x)
        self.upp = np.ones_like(x)    
    @staticmethod     
    def _P_Q(low, upp, x, grad, is_f=True):
        PQ_COEF, PQ_X = 1e-3, 1e-5
        c = (1+PQ_COEF, PQ_COEF, 1) if is_f else (1, 0, 0)
        P = (upp-x)**2*(c[0]*grad*(grad>0)-c[1]*grad*(grad<0)+c[2]*PQ_X)
        Q = (x-low)**2*(c[1]*grad*(grad>0)-c[0]*grad*(grad<0)+c[2]*PQ_X)
        return P, Q
    def _relax_term(self, x, is_f=True):
        P, Q = (self.P0, self.Q0) if is_f else (self.Pi, self.Qi)
        res = np.sum(P/(self.upp-x)+Q/(x-self.low), axis=-1)
        grad = P/(self.upp-x)**2-Q/(x-self.low)**2
        diag_H = 2*P/(self.upp-x)**3+2*Q/(x-self.low)**3
        return res, grad, diag_H     
    def _sub_problem(self, itr, x, gi, gradf, gradgi):        
        ASY_INIT, ASY_FAST, ASY_SLOW = 0.5, 1.2, 0.7
        LU_COEF_MIN, LU_COEF_MAX, AB_COEF = 0.1, 10, 0.1
        if itr < 2:
            asy_coeff = np.full(len(x), ASY_INIT)
        else:
            xx = (x-self.x_old1)*(self.x_old1-self.x_old2)
            asy_coeff = np.where(xx > 0, ASY_FAST, np.where(xx < 0, ASY_SLOW, 1.0))
        self.low = x-asy_coeff*(self.x_old1-self.low if itr > 1 else self.x_max-self.x_min)
        self.upp = x+asy_coeff*(self.upp-self.x_old1 if itr > 1 else self.x_max-self.x_min)
        self.low = np.clip(self.low, x-LU_COEF_MAX*(self.x_max-self.x_min), 
                           x-LU_COEF_MIN*(self.x_max-self.x_min))
        self.upp = np.clip(self.upp, x+LU_COEF_MIN*(self.x_max-self.x_min), 
                           x+LU_COEF_MAX*(self.x_max-self.x_min))
        a = np.clip(np.maximum(self.low+AB_COEF*(x-self.low), 
                               x-self.move_lim*(self.x_max-self.x_min)), self.x_min, None)
        b = np.clip(np.minimum(self.upp-AB_COEF*(self.upp-x), 
                               x+self.move_lim*(self.x_max-self.x_min)), None, self.x_max)
        self.P0, self.Q0 = self._P_Q(self.low, self.upp, x, gradf)
        self.Pi, self.Qi = self._P_Q(self.low, self.upp, x, gradgi, False)
        self.ri = gi-self._relax_term(x, False)[0]
        return a, b
    def _apx_term(self, x):
        _, df_apx, diag_Hf_apx = self._relax_term(x)
        resi, dgi_apx, diag_Hgi_apx = self._relax_term(x, False)
        gi_apx = self.ri+resi
        return df_apx, gi_apx, dgi_apx, diag_Hf_apx, diag_Hgi_apx
# MMAのクラスを継承してDualMMAのクラスを作成
class DualMMA(MMA):
    def __init__(self, x, m, move_lim=0.05, x_min=0.0, x_max=1.0):
        super().__init__(x, m, move_lim, x_min, x_max)
    def _solve_sub_problem(self, a, b):
        DUAL_TOL, H_REG_COEF, STEP_DECAY = 1e-6, 1e-3, 0.5
        PENAL_Y = 1e3
        INITR_MAX = [20, 20]
        def eval_state(l, a, b): 
            P, Q = self.P0+l@self.Pi, self.Q0+l@self.Qi
            sP, sQ = np.sqrt(P), np.sqrt(Q)
            xj = np.clip((sP*self.low+sQ*self.upp)/(sP+sQ), a, b)
            gi_apx = self.ri+np.sum(self.Pi/(self.upp-xj)+self.Qi/(xj-self.low), axis=1)
            yi = np.maximum(0, l-PENAL_Y)
            dual_grad = gi_apx-yi    
            return xj, dual_grad, sP, sQ, yi
        lam = np.full(self.m, 1.0)
        for _ in range(INITR_MAX[0]):
            xj, dual_grad, sP, sQ, yi = eval_state(lam, a, b)
            if np.max(np.abs(dual_grad)) < DUAL_TOL: break
            act = (xj > a) & (xj < b)
            inv_Ux = 1/(self.upp[act]-xj[act])
            inv_xL = 1/(xj[act]-self.low[act])
            dg = self.Pi[:, act]*inv_Ux**2-self.Qi[:, act]*inv_xL**2
            d2L = 2*(sP[act]**2*inv_Ux**3+sQ[act]**2*inv_xL**3)
            H = -dg@(dg.T/d2L[:, None])
            H -= np.diag((yi > 0).astype(float))
            H -= np.eye(self.m)*H_REG_COEF
            mask = (dual_grad > -DUAL_TOL) | (lam > DUAL_TOL)            
            if not mask.any(): break            
            dlam = np.zeros(self.m)
            dlam[mask] = np.linalg.solve(H[np.ix_(mask, mask)], -dual_grad[mask])            
            t, err = 1.0, (dual_grad[mask]**2).sum()
            for _ in range(INITR_MAX[1]):
                lam_try = np.maximum(0, lam+t*dlam)
                _, grad_try, _, _, _ = eval_state(lam_try, a, b)
                if (grad_try[mask]**2).sum() < err:
                    lam = lam_try; break
                t *= STEP_DECAY            
        return xj