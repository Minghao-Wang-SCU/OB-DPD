#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 18:13:01 2018

@author: liuyang
"""

import MDAnalysis
import numpy as np
import matplotlib.pyplot as plt
#import subprocess
#splt.style.use('seaborn-talk')
plt.rcParams['font.family'] = 'sans'
plt.rcParams['font.size'] = 28
plt.rcParams['axes.labelsize'] = 25
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 19
plt.rcParams['ytick.labelsize'] = 19
plt.rcParams['legend.fontsize'] = 19
plt.rcParams['figure.titlesize'] = 28
plt.rcParams["errorbar.capsize"]=8
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
#import os
#output_name='CG1_CG2_distance_as_function_of_time.xvg'
start_frame=1
end_frame=100

size_length=20

bin_size=int(size_length/100)
bin_size=0.1
def compute_partial_density(add, tpr, xtc, label_=None,linestyle="-", color=None, atom_name="1"):

    
    u = MDAnalysis.Universe(add + tpr, add + xtc)

    membrane=u.select_atoms('name ' + atom_name)


    radius=0
    number_atoms_box_whole_trj=[]
    for frame in u.trajectory[start_frame:end_frame]:
        membrane_center=membrane.atoms.center_of_geometry()
        reference_vector=membrane.positions-membrane_center
        distance_box=[]
        for line in reference_vector:
            distance=np.linalg.norm(line)
            distance_box.append(distance)

        for line in reference_vector:
            distance=np.linalg.norm(line)
            distance_box.append(distance)
        number_atoms_box=[]
        for radius in np.arange(0,size_length,bin_size):
            number_atoms=0
            for line in distance_box:
                if line > radius and line <= radius+bin_size:
                    number_atoms+=1
            number_atoms_box.append(number_atoms/(4/3*np.pi*((radius+bin_size)**3-radius**3)))
        number_atoms_box_whole_trj.append(number_atoms_box)
    number_atoms_box_whole_trj=np.array(number_atoms_box_whole_trj).T
    partial_density=[]
    for line in number_atoms_box_whole_trj:
        partial_density.append(np.mean(line))

#    partial_density=[a for a in partial_density if a !=0 ]

    xdata=np.linspace(0,size_length,len(partial_density)) 
    plt.plot(xdata, partial_density, label= label_, linewidth=6,linestyle=linestyle)
    tpr=tpr.split('.')[0]
    tpr=tpr.split('/')[1]
    with open (directory+"/partial_density_"+atom_name+tpr+".xvg",'w') as myfile:
        for x, y in zip(xdata,partial_density):
            line=str(x)+' '+str(y)+'\n'
            myfile.writelines(line)



    
add1="/usr/local/yang/lizifeng_dpd/vesicle_less2"
add2="/usr/local/yang/lizifeng_dpd/vesicle_less_oxidize2"


pdb = '/home/ls/wmh/DPD/one-button-dpd/main/P2/3 polymer_number 70/all.pdb'
gro = '/home/ls/wmh/DPD/one-button-dpd/main/P2/3 polymer_number 70/all.gro'
index = 1
compute_partial_density(pdb, gro," P2 70","-", plt.cm.cool_r(index/2), "1")



# for directory, label in zip([ add1],['PCP']):
    
#     compute_partial_density(directory,"/start.pdb", "/run.trr",label+" PCL","-", plt.cm.cool_r(index/2), "1")
#     compute_partial_density(directory, "/start.pdb", "/run.trr",label+" CDI","-", plt.cm.cool_r(index/2), "2")
#     compute_partial_density(directory,"/start.pdb", "/run.trr",label+" PEG","-", plt.cm.cool_r(index/2), "3")

#     index +=1
#     ##################################################
# index=1
# for directory, label in zip([ add2],['PCP-O']):

#     compute_partial_density(directory,"/start.pdb", "/run.trr",label+" PCL",":", plt.cm.cool_r(index/2), "1")
#     compute_partial_density(directory, "/start.pdb", "/run.trr",label+" CDI-O",":", plt.cm.cool_r(index/2), "2")
#     compute_partial_density(directory,"/start.pdb", "/run.trr",label+" PEG",":", plt.cm.cool_r(index/2), "3")

#     index +=1


# plt.xlabel("Radial distance (nm)")
# plt.tight_layout()
# #plt.xlim(-2,24)
# plt.ylabel("Partial density (nm^(-3))")
# #plt.legend(loc = "left right")
# plt.legend()

# plt.show()
