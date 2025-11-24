import numpy as np
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(current_dir,"..","src","lab02"))
sys.path.append(src_path)
from linalg import spline_function


def main():
    with open("water_density_vs_temp_usgs.txt","r") as file:
        water = file.read()
        water = water.split()
        water = [float(i) for i in water]
    #print(water)
    xd_w = water[::2]
    yd_w = water[1::2]
    #print (x_dw,y_dw)

    with open("air_density_vs_temp_eng_toolbox.txt","r") as file:
        air = file.read()
        air = air.split()
        air = [float(i) for i in air]
    xd_a = air [::2]
    yd_a = air[1::2]
    #print (x_da,y_da)

    w_temp = np.linspace(xd_w[0], xd_w[-1],100)
    a_temp = np.linspace(xd_a[0], xd_a[-1],100)

    fig, axes = plt.subplots(3,2,figsize=(12,12))


    for col, (xd,yd, temp, title) in enumerate ([
        (xd_w,yd_w,w_temp, "water density"),
        (xd_a,yd_a,a_temp, "air density")]):
        
        for order in(1,2,3):
            f = spline_function(xd,yd,order=order)
            y = f(temp)
            
            
            

            ax = axes[order-1,col]
            ax.plot(xd,yd,"o",label="measured values")
            ax.plot (temp,y,label=f"order {order}  spline")
            ax.set_xlabel("Temperature(C)")
            ax.set_ylabel("Density (Kg/m^3)")
            ax.set_title(f"order {order} spline of {title}")
            ax.legend()

            
    plt.tight_layout()

    plt.show()
                        






main()
