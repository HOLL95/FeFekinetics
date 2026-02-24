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
params=["init_1",  "k_inact", "k_react", "k_react_exp", "k_deg","current_conversion", "cap_scaling"]+[ "k_AB", "k_BA"]
sclass=FeFeSimulator(efile, bfile)
potential=sclass.potential
split_points=np.where(np.abs(np.diff(potential))>0.2)[0]
sclass.set_param_names(params)
boundaries=dict(k_deg     = [0, 5e-4],
                init_1      = [0,1],
                init_2      = [0,1],
                k_inact = [1e-3, 5],
                k_react = [1e-3, 5],
                k_react_exp  = [-25,25],
                k_AB=[1e-3, 5],
                k_BA=[1e-3, 5],
                current_conversion=[0, 5000],
                cap_scaling=[0, 20]
                )
sclass.set_boundaries(boundaries)
values=[0.10411234786702483, 0.9182703490458808, 0.0010000000124533851, -10.250282567458138, 0.0004999997407632446, 1958.1222971838154, 1.919758321750141, 0.0480141951174752, 0.0010000000305276065]



vals=dict(zip(sclass.param_names, values
))
for i in range(1, len(split_points), 2):
    fig,ax=plt.subplots(1,2)
    chunk=[split_points[i-1], split_points[i]]
    chunk_times=time[chunk[0]+1:chunk[1]]
    chunk_current=current[chunk[0]+1:chunk[1]]
    ax[0].plot(chunk_times, chunk_current, label="Data")
    scurrent=sclass.dimensional_simulate(vals,chunk_times)

    ax[0].plot(chunk_times, scurrent,label="Simulation")
    ax[0].plot(chunk_times, sclass.cap_current[chunk[0]+1:chunk[1]], color="black", linestyle="--", label="Blank")
    ax[0].legend()
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Current")
    ax[1].plot(chunk_times, sclass.quantities["A"], label="active")
    ax[1].plot(chunk_times, sclass.quantities["B"], label="half active")
    ax[1].plot(chunk_times, sclass.quantities["C"], label="inactive")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Species proportion")
    ax[1].legend()
    print(f"Optimising chunk {i}, {"="*30}")
    plt.show()
    sclass.optimise(chunk_times, chunk_current, repeats=5, parallel=True)

    
