from simulate import FeFeSimulator
import numpy as np
import matplotlib.pyplot as plt
efile="raw_data.csv"
ifile="/home/henryll/Documents/Experimental_data/Lucy/WT data/All_WT_1.csv"
bfile="/home/henryll/Documents/Experimental_data/Lucy/WT data/All_Blank.csv"
with open(ifile, "r") as f:
    data = np.loadtxt(f, skiprows=1, delimiter=",")

time=data[:,0]
current=data[:,1]
params=["init_1",  "k_inact", "k_react", "k_react_exp", "k_deg","current_conversion", "cap_scaling"]+["k_AB", "k_BA"]
sclass=FeFeSimulator(efile, bfile)
potential=sclass.potential
sclass.set_param_names(params)
boundaries=dict(k_deg     = [0, 5e-4],
                init_1      = [0,1],
                init_2      = [0,1],
                k_inact = [1e-3, 5],
                k_react = [1e-3, 5],
                k_react_exp  = [0,25],
                k_AB=[1e-3, 5],
                k_BA=[1e-3, 5],
                current_conversion=[0, 5000],
                cap_scaling=[0, 20]
                )
sclass.set_boundaries(boundaries)
cvals=[2000, 3000, 4000, 5000]
fig,ax=plt.subplots(1,3)
ax[0].plot(time, current, label="Current")
ax[0].plot(time, sclass.cap_current, color="darkslategrey", label="Blank current")
ax[0].legend()
ax[0].twinx().plot(time, potential, color="black", linestyle="--")


values=[2.15238940e-11, 8.78613051e-02, 1.12859893e-02, 5.96802490e-12,
 2.01187488e-04, 1.00000000e+03, 2.41178966e+00, 5.70865829e-11,
 1.87009721e-01, 8.09773547e-02]
values=[9.31716244e-11, 4.99999929e+00, 4.99999995e+00, 4.01081356e+00, 2.01548680e-04, 1.01857597e+02, 2.41219941e+00, 1.00000002e-03, 1.60576973e+00]
vals=dict(zip(sclass.param_names, values
))

scurrent=sclass.dimensional_simulate(vals, time)
ax[1].plot(time, current, label="Current")
ax[1].plot(time, scurrent, label="Simulation")
ax[1].legend()
ax[1].twinx().plot(time, potential, color="black", linestyle="--")


#plt.plot(scurrent+sclass.cap_current*vals["cap_scaling"], color="black", linestyle="--")


ax[2].plot(time, sclass.quantities["A"], label="active")
ax[2].plot(time, sclass.quantities["B"], label="active2")
ax[2].plot(time, sclass.quantities["C"], label="inact")
ax[2].twinx().plot(time, potential, color="black", linestyle="--")
ax[2].legend()
plt.show()
