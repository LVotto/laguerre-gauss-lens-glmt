import sys                                  # My work. My _precious_ work.
sys.path.insert(0, "D:\\USP\\TCC\\glmtscatt")
sys.path.insert(0, r"..\python\plotting")

import mpmath                               # Important for precision control
from lgbeamlens import LGBeamLens           # Beam class
import matplotlib.pyplot as plt             # Das Plotting
import matplotlib as mpl                    # Die Plotting
import numpy as np                          # My best-friend
import time                                 # Time is money
import pickle                               # Saving data that takes too long to compute
from custom_plot import SavedSinglePlot     # Makes plotting easier

DEFAULT_DPS = 15
mpmath.mp.dps = DEFAULT_DPS         # precision in decimal places (dps)
STYLE = ["default"]                 # faster and does not use tex necessarily
# STYLE = ["science", "ieee"]       # for publishable figs (recommended for less than 5 lines)
# STYLE = ["science", "notebook"]   # publishable (recommended for more than 5 lines in the same plot)
mpl.style.use(STYLE)

# The default beam:
wavelength = 632.8E-9
k = 2 * mpmath.pi / wavelength
DEFAULT_PARAMETERS = {
    "bscs": {'TM': {}, 'TE': {}},
    "p": 0, "l": 2, "w0": 3.2 * wavelength,
    "na": .95, "r_focus": 1e-3,
    "alpha": 1, "beta": 0
}
default_beam = LGBeamLens(k, **DEFAULT_PARAMETERS)

def lpm1_string(m, l, slm1="lm1", slp1="lp1", sels=None):
    if m == l - 1:
        return slm1
    
    if m == l + 1:
        return slp1
    
    return "" if sels is None else str(m)
    

def timed_plot_lglens(m=1, nmax=4000, dps=15, params=DEFAULT_PARAMETERS,
                     save=True, filename=None, filename_tail=None, load=False,
                     show=False, method="FS"):
    mpmath.mp.dps = dps
    beam_dps = LGBeamLens(k, **params)
    if filename is None:
        method_str = method.lower()
        filename_tail = "_" + filename_tail if filename_tail is not None else ""
        filename = "./pickles/timed{}_m{}_l{}_p{}_dps{}_nmax{}".format(method_str, m,
                    beam_dps.l, beam_dps.p, dps, nmax) + filename_tail
    
    if load:
        if not filename.endswith(".pickle"):
            filename = filename + ".pickle"
        with open(filename, "rb") as f:
            result = pickle.load(f)
        return result
    
    times = []
    start = time.time()
    bsclist = []
    ns = [n for n in range(1, nmax + 1)]
    if method.upper() == "FS":
        bsc_func = beam_dps.bsc
    elif method.upper() == "ILA":
        bsc_func = beam_dps.ila_bsc
    else:
        warning.warn("Method {} not known. Using FS method.".format(method.upper()))
        bsc_func = beam_dps.bsc

    for n in ns:
        bsclist.append(bsc_func(n, m))
        times.append(time.time() - start)
    if show:    
        plt.plot(ns, times)
        plt.xlabel("n")
        plt.ylabel("Elapsed Time [s]")
        plt.grid()
        plt.show()

        superscript = lpm1_string(m, beam_dps.l, 
                                slm1="l - 1", slp1="l + 1")

        plt.title(r"$p = {},\ l = {}$".format(beam_dps.p, beam_dps.l))
        plt.plot(ns, np.abs(bsclist))
        plt.xlabel("n")
        plt.ylabel("$|g_{n, TM}^{" + "{}".format(superscript) + "}|$")
        top = 1.5 * float(max(np.abs(bsclist[:1000])))
        plt.ylim(top=top, bottom=-1)
        plt.show()


    tabled_ns = np.array([10, 100, 500, 1000, 2000, 3000, 4000])
    tabled_ns = list(tabled_ns[tabled_ns <= nmax])
    tabled_ts = [times[n - 1] for n in tabled_ns]
    header = """ Number of BSCs | Time """
    header = header + "\n" + ":---:|:---:"
    for i, t in enumerate(tabled_ts):
        tt = time.gmtime(t)
        tstring = time.strftime("%Hh %Mmin %Ss", tt)
        header = header + "\n" + str(tabled_ns[i]) + "|" + tstring
    
    result = times, bsclist, header
    if save:
        if not filename.endswith(".pickle"):
            filename = filename + ".pickle"
        with open(filename, "wb") as f:
            pickle.dump(result, f)
    
    return result

l = DEFAULT_PARAMETERS["l"]

# dpss = [15, 50, 100]
ps = [0, 1, 2]
nmax = 4000
ms = [l - 1, l + 1]

dpss = [12]
# ps = [0]
# nmax = 10

# method = "FS"
method = "ILA"

for dps in dpss:
    mpmath.mp.dps = dps
    for p in ps:
        for m in ms:
            print("{} dps | p = {} | m = {}".format(dps, p, m))
            params = dict(DEFAULT_PARAMETERS)
            params.update({"p": p})
            res = timed_plot_lglens(m=m, params=params, dps=dps, show=False, nmax=nmax, method=method)
            print("Took {} seconds".format(res[0][-1]))
plt.plot([1, 1], [2, 1])
plt.show()