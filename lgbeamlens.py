# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 15:56:12 2020

@author: luizv
"""

import numpy as np
import mpmath
from field import SphericalElectricField
import warnings

# Memoization decorator
def memoized(func):
    CACHE = {}
    def memoized_func(*args, **kwargs):
        kwvals = [val for _, val in kwargs.items()]
        if (*args, *kwvals) not in CACHE:
            value = func(*args, **kwargs)
            CACHE[(*args, *kwvals)] = value
            return value
        return CACHE[(*args, *kwvals)]
    return memoized_func


class LGBeamLens(SphericalElectricField):
    def __init__(self, wave_number, bscs={'TM': {}, 'TE': {}},
                 p=0, l=2, w0=3.2 * 632.8E-9, na=.95, 
                 r_focus=1E-3, alpha=1, beta=0):
        super().__init__(wave_number, bscs=bscs)
        self.p, self.l = p, l
        if self.l == 0:
            raise NotImplementedError("l = 0 not ready yet.")
        self.w0 = w0
        self.r_focus, self.na = r_focus, na
        self.alpha, self.beta = alpha, beta
        self.wavelength = 2 * np.pi / self.wave_number
        self.s = 1 / self.wave_number / self.w0
        self.degrees.add(l - 1)
        self.degrees.add(l + 1)
        
    @property
    def k(self):
        return self.wave_number

    def a(self, eta):
        return eta * self.r_focus * self.s * np.sqrt(2)
    
    def F(self, n, u):
        return ( (-1) ** ((n - u) / 2) \
                * mpmath.gammaprod([], [(n - u) / 2 + 1, (n + u) / 2 + 1]))
    
    def premul(self, n, m, mode="TM"):
        if mode != "TM":
            raise NotImplementedError("Only TM mode available yet.")
            
        if not (n - m) % 2:
            if m == self.l + 1:
                pow2 = self.l + 2
                pown1 = (n + self.l + 1) / 2
                gammatop = [(n - self.l + 1) / 2]
                gammabtm = [(n + self.l) / 2 + 1]
            elif m == self.l - 1:
                pow2 = self.l
                pown1 = (n + self.l - 1) / 2
                gammatop = [(n - self.l + 3) / 2]
                gammabtm = [(n + self.l) / 2]
            else:
                return 0
            
        else:
            if m == self.l + 1:
                pow2 = self.l + 3
                pown1 = (n + self.l) / 2
                gammatop = [(n - self.l) / 2]
                gammabtm = [(n + self.l + 3) / 2]
            elif m == self.l - 1:
                pow2 = self.l + 1
                pown1 = (n + self.l) / 2 - 1
                gammatop = [(n - self.l) / 2 + 1]
                gammabtm = [(n + self.l + 1) / 2]
            else:
                return 0
            
        return ( 1j ** n * mpmath.sqrt(mpmath.pi) / mpmath.power(2, pow2) \
                   * (-1) ** pown1 * mpmath.gammaprod(gammatop, gammabtm))
        
    def integrand(self, eta, l, p, n, kind="A"):
        if kind not in ("A", "B", "C", "D", "E"):
            raise NotImplementedError("Integrand types supported \
                                      go only from A to E")
        if kind == "C":
            return (mpmath.sqrt(self.k ** 2 - eta ** 2) / eta \
                    * self.integrand(eta, l, p, n, kind="A"))
        if kind == "D":
            return (mpmath.sqrt(self.k ** 2 - eta ** 2) / eta \
                    * self.integrand(eta, l, p, n, kind="B"))
        
        pm = 1
        exponent = n - 1
        if kind == "B":
            pm = -1
        elif kind == "E":
            pm = 0
            exponent = n
            
        if p == 0:
            lgr_factor = 1
        else:
            lgr_factor = (self.a(eta) ** 2 \
                       * mpmath.laguerre(p - 1, l, self.a(eta) ** 2))
        
        
        kz = mpmath.sqrt(self.k ** 2 - eta ** 2)
        return ( mpmath.power(eta, np.abs(l) + 1) / mpmath.sqrt(kz) \
               * mpmath.exp(-self.a(eta) ** 2 / 2) * (1 + pm * kz / self.k) \
               * mpmath.power(eta / self.k, exponent) * lgr_factor)
    
    def integral(self, l, p, n, kind="A"):
        integrand = lambda eta: self.integrand(eta, l, p, n, kind=kind)
        interval = [0, self.k * self.na]
        return mpmath.quad(integrand, interval)
    
    def new_term(self, l, p, n, premul, params):
        """
        i.e.: params = {"A": (l, n - 1, l, 1), "B": (l - 2, n - 1, l - 2, 1)} 
        [n-m even, m=l-1]
        """
        term = 0
        for kind, vals in params.items():
            if n > vals[0]:
                nf = vals[1]
                lf = vals[2]
                sgn = vals[3]
                term += sgn * self.F(nf, lf) \
                      * self.integral(l, p, n, kind=kind)
        return term * premul
    
    @memoized
    def maclaurin(self, n, p, m):
        if p < 0:
            return 0
        
        premul = 1j * mpmath.sqrt(2 / mpmath.pi)
        if not (n - m) % 2:    # even
            if m == self.l + 1:
                premul *= (self.alpha - 1j * self.beta) / mpmath.power(2, n)
                params = {"A":  (self.l, n - 1, self.l, 1),
                          "B":  (self.l + 2, n - 1, self.l + 2, 1)}
            elif m == self.l - 1:
                premul *= (self.alpha + 1j * self.beta) / mpmath.power(2, n)
                params = {"A":  (self.l, n - 1, self.l, 1),
                          "B":  (self.l - 2, n - 1, self.l - 2, 1)}
            else:
                premul, params = 0, {}
        else:
            if m == self.l + 1:
                premul *= (1j * self.alpha + self.beta) \
                        / mpmath.power(2, n - 1)
                params = {"C":  (self.l + 1, n - 2, self.l, 1),
                          "D":  (self.l + 3, n - 2, self.l + 2, 1),
                          "E":  (self.l + 1, n - 1, self.l + 1, -1)}
            elif m == self.l - 1:
                premul *= (1j * self.alpha - self.beta) \
                        / mpmath.power(2, n - 1)
                params = {"C":  (self.l + 1, n - 2, self.l, 1),
                          "D":  (self.l - 1, n - 2, self.l - 2, 1),
                          "E":  (self.l - 1, n - 1, self.l - 1, 1)}
            else:
                premul, params = 0, {}
        
        if p == 0:
            return self.new_term(self.l, 0, n, premul, params)
        
        return (  (2 * p - 1 + self.l) / p * self.maclaurin(n, p - 1, m) \
                - (p - 1 + self.l) / p * self.maclaurin(n, p - 2, m) \
                - self.new_term(self.l, p, n, premul, params) / p )
        
    
    def bsc(self, n, m, mode="TM"):
        g = 0
        q = 0
        if n < m:
            return 0
        if m not in [self.l + 1, self.l - 1]:
            return 0
        
        while q <= n / 2:
            inc = mpmath.power(2, (.5 + n - 2 * q)) \
                * mpmath.gammaprod([.5 + n - q], [q + 1]) \
                * self.maclaurin(n - 2 * q, self.p, m)
            g += inc
            q += 1
        return self.premul(n, m, mode=mode) * g
    
    def radial_integrand_lm1(self, eta, r, theta, phi):
        k = self.k
        kz = mpmath.sqrt(k ** 2 - eta ** 2)
        l, p = self.l, self.p
        a, b = self.alpha, self.beta
        result = (mpmath.power(eta, np.abs(l) + 1) / mpmath.sqrt(kz) \
               * mpmath.laguerre(p, l, self.a(eta) ** 2) \
               * mpmath.exp(-self.a(eta) ** 2 / 2 + 1j * (l - 1) * phi \
                            + 1j * kz * r * np.cos(theta)) \
               * ((a + 1j * b) / 2 * (1 - kz / k) \
               * mpmath.besselj(l - 2, eta * r * np.sin(theta)) \
               * np.sin(theta) \
               + eta / k * (1j * a - b) \
               * mpmath.besselj(l - 1, eta * r * np.sin(theta)) \
               * np.cos(theta) \
               + (a + 1j * b) / 2 * (1 + kz / k) \
               * mpmath.besselj(l, eta * r * np.sin(theta)) * np.sin(theta)))
        return result
        
    def radial_integrand_lp1(self, eta, r, theta, phi):
        k = self.k
        kz = mpmath.sqrt(k ** 2 - eta ** 2)
        l, p = self.l, self.p
        a, b = self.alpha, self.beta
        result = (mpmath.power(eta, np.abs(l) + 1) / mpmath.sqrt(kz) \
               * mpmath.laguerre(p, l, self.a(eta) ** 2) \
               * mpmath.exp(-self.a(eta) ** 2 / 2 + 1j * (l + 1) * phi \
                            + 1j * kz * r * np.cos(theta)) \
               * ((a - 1j * b) / 2 * (1 + kz / k) \
               * mpmath.besselj(l, eta * r * np.sin(theta)) \
               * np.sin(theta) \
               - eta / k * (1j * a + b) \
               * mpmath.besselj(l + 1, eta * r * np.sin(theta)) \
               * np.cos(theta) \
               + (a - 1j * b) / 2 * (1 - kz / k) \
               * mpmath.besselj(l + 2, eta * r * np.sin(theta)) \
               * np.sin(theta)))
        return result
    
    def exact_radial_field(self, r, theta, phi):
        intrv = [0, self.k * self.na]
        lm1 = lambda eta: self.radial_integrand_lm1(eta, r, theta, phi)
        lp1 = lambda eta: self.radial_integrand_lp1(eta, r, theta, phi)
        return mpmath.quad(lm1, intrv) + mpmath.quad(lp1, intrv)
    
    def r_max(self, N):
        c = 2 - N
        a = 4.05
        y = (mpmath.power(2, 1 / 3) \
          * mpmath.power(mpmath.sqrt(12 * a ** 3 + 81 * c ** 2) - 9 * c, 
                         2 / 3) \
          - 2 * mpmath.power(3, 1 / 3) * a) \
          / (mpmath.power(6, 2 / 3) \
          * mpmath.power(mpmath.sqrt(12 * a ** 3 + 81 * c ** 2) - 9 * c,
                         1 / 3))
        return y ** 3 / self.k
    
    def compute_bscs(self, nmax, mode="TM"):
        bscdict = self.tm_bscs if mode.upper() == "TM" else self.te_bscs
        for m in (self.l - 1, self.l + 1):
            for n in range(1, nmax + 1):
                bscdict[(n, m)] = self.bsc(n, m, mode=mode)
    
    def enumerate_bscs(self, m, mode="TM"):
        bscdict = self.tm_bscs if mode.upper() == "TM" else self.te_bscs
        pairs = []
        for (nn, mm), val in bscdict.items():
            if mm == m: pairs.append(((nn, mm), val))
        return sorted(pairs, key=lambda x: x[0][0])

    def ila_integrand_lm1_a(self, eta, rloc):
        k, p, l = self.k, self.p, self.l
        kz = mpmath.sqrt(k ** 2 - eta ** 2)
        res = mpmath.power(eta, np.abs(l) + 1) / mpmath.sqrt(kz) \
            * mpmath.laguerre(p, l, self.a(eta) ** 2) * mpmath.exp(-self.a(eta) ** 2 / 2) \
            * (1 - kz / k) * mpmath.besselj(l - 2, eta * rloc / k)
        return res

    def ila_integrand_lm1_b(self, eta, rloc):
        k, p, l = self.k, self.p, self.l
        kz = mpmath.sqrt(k ** 2 - eta ** 2)
        res = mpmath.power(eta, np.abs(l) + 1) / mpmath.sqrt(kz) \
            * mpmath.laguerre(p, l, self.a(eta) ** 2) * mpmath.exp(-self.a(eta) ** 2 / 2) \
            * (1 + kz / k) * mpmath.besselj(l, eta * rloc / k)
        return res

    # def ila_integrand_lm1(self, eta, rloc):
    #     k, p, l = self.k, self.p, self.l
    #     kz = mpmath.sqrt(k ** 2 - eta ** 2)
    #     res = mpmath.power(eta, np.abs(l) + 1) / mpmath.sqrt(kz) \
    #         * mpmath.laguerre(p, l, self.a(eta) ** 2) * mpmath.exp(-self.a(eta) ** 2 / 2) \
    #         * ((1 - kz / k) * mpmath.besselj(l - 2, eta * rloc / k) + (1 + kz / k) * mpmath.besselj(l, eta * rloc / k))
    #     return res

    def ila_integrand_lp1_a(self, eta, rloc):
        k, p, l = self.k, self.p, self.l
        kz = mpmath.sqrt(k ** 2 - eta ** 2)
        res = mpmath.power(eta, np.abs(l) + 1) / mpmath.sqrt(kz) \
            * mpmath.laguerre(p, l, self.a(eta) ** 2) * mpmath.exp(-self.a(eta) ** 2 / 2) \
            * (1 - kz / k) * mpmath.besselj(l + 2, eta * rloc / k)
        return res

    def ila_integrand_lp1_b(self, eta, rloc):
        k, p, l = self.k, self.p, self.l
        kz = mpmath.sqrt(k ** 2 - eta ** 2)
        res = mpmath.power(eta, np.abs(l) + 1) / mpmath.sqrt(kz) \
            * mpmath.laguerre(p, l, self.a(eta) ** 2) * mpmath.exp(-self.a(eta) ** 2 / 2) \
            * (1 + kz / k) * mpmath.besselj(l, eta * rloc / k)
        return res

    def ila_integrand_lp1(self, eta, rloc):
        k, p, l = self.k, self.p, self.l
        kz = mpmath.sqrt(k ** 2 - eta ** 2)
        res = mpmath.power(eta, np.abs(l) + 1) / mpmath.sqrt(kz) \
            * mpmath.laguerre(p, l, self.a(eta) ** 2) * mpmath.exp(-self.a(eta) ** 2 / 2) \
            * ((1 - kz / k) * mpmath.besselj(l + 2, eta * rloc / k) + (1 + kz / k) * mpmath.besselj(l, eta * rloc / k))
        return res

    def ila_er_lm1(self, rloc):
        alpha, beta = self.alpha, self.beta
        interval = [0, self.k * self.na]
        integrand = lambda eta: self.ila_integrand_lm1_a(eta, rloc)
        res = (alpha - 1j * beta) / 2 * mpmath.quad(integrand, interval)
        integrand = lambda eta: self.ila_integrand_lm1_b(eta, rloc)
        res += (alpha - 1j * beta) / 2 * mpmath.quad(integrand, interval)
        return res

    def ila_er_lp1(self, rloc):
        alpha, beta = self.alpha, self.beta
        interval = [0, self.k * self.na]
        integrand = lambda eta: self.ila_integrand_lp1_a(eta, rloc)
        res = (alpha + 1j * beta) / 2 * mpmath.quad(integrand, interval)
        integrand = lambda eta: self.ila_integrand_lp1_b(eta, rloc)
        res += (alpha + 1j * beta) / 2 * mpmath.quad(integrand, interval)
        return res
    
    def ila_bsc_lm1(self, n, rloc):
        res = mpmath.power(-1j / rloc, np.abs(-self.l + 1) - 1) \
            * self.ila_er_lm1(rloc)
        return res

    def ila_bsc_lp1(self, n, rloc):
        res = mpmath.power(-1j / rloc, np.abs(-self.l - 1) - 1) \
            * self.ila_er_lp1(rloc)
        return res

    def ila_bsc(self, n, m, rloc_type=1):
        # rloc1 = (n + 1 / 2), rloc2 = \sqrt((n - 1)(n + 2))
        if rloc_type == 1:
            rloc = n + .5
        elif rloc_type == 2:
            rloc = np.sqrt((n - 1) * (n + 2))
        else:
            warnings.warn("Rloc type {} isn't known. Using type 1.".format(rloc_type))
            rloc = n + .5

        if m == self.l - 1:
            return self.ila_bsc_lm1(n, rloc)
        elif m == self.l + 1:
            return self.ila_bsc_lp1(n, rloc)
        else:
            return 0