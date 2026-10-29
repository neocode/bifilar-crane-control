"""Maximal output-admissible (positively invariant) set of the terminal LQR law.

Gilbert & Tan (1991): O_inf = {xi : Hc A_cl^t xi <= hc, t = 0, 1, ...}; the set is finitely
determined, i.e. the constraints for t > t* are redundant. Redundancy is checked with LPs.
"""
import numpy as np
from scipy.optimize import linprog


def max_admissible_set(Acl, Hc, hc, t_max=5000, tol=1e-9):
    n = Acl.shape[0]
    Hs, hs = [Hc.copy()], [hc.copy()]
    Ap = np.eye(n)
    for t in range(1, t_max + 1):
        Ap = Ap @ Acl
        Hn = Hc @ Ap
        Hall, hall = np.vstack(Hs), np.concatenate(hs)
        redundant = True
        for i in range(Hn.shape[0]):
            res = linprog(-Hn[i], A_ub=Hall, b_ub=hall, bounds=[(None, None)] * n, method='highs')
            if res.status != 0 or -res.fun > hc[i] + tol:
                redundant = False
                break
        if redundant:
            return np.vstack(Hs), np.concatenate(hs), t - 1
        Hs.append(Hn)
        hs.append(hc.copy())
    raise RuntimeError('O_inf not finitely determined within t_max')


class TerminalSet:
    def __init__(self, Acl, Hc, hc):
        self.H, self.h, self.tstar = max_admissible_set(Acl, Hc, hc)

    def contains(self, xi, tol=1e-9):
        return bool(np.all(self.H @ xi <= self.h + tol))
