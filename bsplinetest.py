import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import make_smoothing_spline


strain = np.array ([
    0,0.089912,0.114247,0.128433,
    0.14018,
    0.156295,
    0.173353,
    0.189883,
    0.208322,
    0.228676,
    0.244638,
    0.260589,
    0.276511,
    0.295806,
    0.314587,
    0.341543,
    0.371842,
    0.39922,
    0.449642,
    0.508663,
    0.593114,
    0.593114,
    0.643514,
    0.712156,
    0.821144,
    0.918628,
])

stress = np.array ([0,
5.922207,
7.379479,
8.03854,
8.603932,
9.278435,
9.903342,
10.3993,
10.85558,
11.29202,
11.54992,
11.77806,
11.92685,
12.09548,
12.17483,
12.25418,
12.27402,
12.19467,
12.01612,
11.68879,
11.21267,
11.21267,
10.97461,
10.6572,
10.22075,
9.883503,
   ])

firstpoint = False

sorted_indices = np.argsort(strain)
strain_sorted = strain[sorted_indices]
stress_sorted = stress[sorted_indices]
strain_sorted=np.unique(strain_sorted)
stress_sorted=np.unique(stress_sorted)

print(strain_sorted)
print(stress_sorted)

smoothing_factor = 0.0000000000000000005
spl = make_smoothing_spline(strain_sorted, stress_sorted, lam=smoothing_factor)
smoothed_stress_spline = spl(strain_sorted)


newstress = np.array(smoothed_stress_spline)

slope=np.diff(newstress)/np.diff(strain_sorted)


for i in range(len(slope)-1):
    if slope[i]>0 and slope[i+1]<0:
        ax=float(strain_sorted[i+1])
        ay=float(newstress[i+1])
        print("point a is", "(", ax, ",", ay, ")")
        firstpoint = True
        break

if firstpoint:
    for i in range(len(slope)-1):
        if slope[i] < 0 and slope[i + 1] > 0:
           bx=float(strain_sorted[i+1])
           by=float(newstress[i+1])
           print("point b is", "(", bx, ",", by, ")")
           break

plt.figure(figsize=(10,6))
plt.plot(strain_sorted,smoothed_stress_spline, label="smoothed curve", linewidth=4 )
plt.xlabel('Strain')
plt.ylabel('Stress')
plt.title('Stress-Strain Curves')
plt.grid(True)
plt.plot(strain_sorted,stress_sorted, label="raw curve", linewidth=0.5)
plt.scatter(ax,ay,color='red',s=200)
plt.scatter(bx,by,color='red',s=200)
plt.plot([strain_sorted[0],ax],[newstress[0],ay],color='green', label="line predictions",linewidth=4 )
plt.plot([ax,bx],[ay,by],color='green',linewidth=4 )
plt.hlines(y=by,xmin=bx,xmax=strain_sorted[-1], color='green',linewidth=4)

plt.legend()
plt.show()

slopeoflinea=(ay-newstress[0])/(ax-strain_sorted[0])
print("slope of the first line is", slopeoflinea)

slopeoflineb=(by-ay)/(bx-ax)
print("slope of the second line is", slopeoflineb)

print("the residual is", by)




