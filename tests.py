import sys
sys.path.insert(0, r"D:\USP\TCC\glmtscatt")
sys.path.insert(0, r"..\python\plotting")

import mpmath
from lgbeamlens import LGBeamLens
import matplotlib.pyplot as plt
import numpy as np
from custom_plot import SavedSinglePlot
import pickle

wavelength = 632.8E-9
k = 2 * mpmath.pi / wavelength

def test_single_tm_bsc(n, m):
    beam = LGBeamLens(k)
    bsc = beam.bsc(n, m, mode='TM')
    print("g_", n,"^", m, "=", np.abs(bsc))
    return bsc

def test_ssp():
    xs = np.linspace(-np.pi, np.pi, 100)
    yss = [np.sin(xs), np.cos(xs)]
    ssp = SavedSinglePlot(
        xs * 1E6, yss, xlabel=r"$x$ [\si{\pico\meter}]",
        style=["science", "ieee"], legend=False
    )
    ssp.show()
    return ssp

def test_pickled_dpss_beams_bscs():
    with open("bscs_15_50_100_beams_lm1s_lp1s.pickle", "rb") as f:
        foo = pickle.load(f)
    beams = foo[0]
    nmax = 2000
    yss = []
    for beam in beams:
        yss.append(np.abs([beam.tm_bscs[(n, beam.l - 1)] for n in range(1, nmax)]))
    xs = range(1, nmax)
    ssp = SavedSinglePlot(
        xs, yss, xlabel=r'$n$', ylabel=r'$g_{n}^{l - 1}$',
        labels=["dps = {}".format(d) for d in (15, 50, 100)],
        style=["science", "ieee"]
    )
    ssp.show()
    return ssp

def test_pickled_dpss():
    with open("bscs_15_50_100_beams_lm1s_lp1s.pickle", "rb") as f:
        foo = pickle.load(f)
    lm1s = foo[1][0]
    yss = []
    nmax = 4001
    dpss = (100, 50, 15)
    for dps in dpss:
        yss.append(np.abs(lm1s[dps][:nmax - 1]))
    top = 1.5 * float(max(yss[0][:nmax // 2]))
    xs = range(1, nmax)
    ssp = SavedSinglePlot(
        xs, yss, xlabel=r'$n$', ylabel=r'$g_{n, \text{TM}}^{l - 1}$',
        labels=["{} dps".format(d) for d in dpss],
        style=["science", "ieee"], ylim_top=top, ylim_bottom=-1
    )
    ssp.savefig("bscs_lm1s_15-50-100.png")
    return ssp

if __name__ == "__main__":
    ssp = test_pickled_dpss()