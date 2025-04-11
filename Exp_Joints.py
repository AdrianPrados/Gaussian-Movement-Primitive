from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pyLasaDataset as lasa
from ProGP import ProGpMp
import numpy as np
import time
from ObstacleAvoidance3D import *

import os
from scipy.io import loadmat, savemat
import pandas as pd


#! ---- Code for real data from robots----
dt = 0.01
gap = 4
demostraciones = 3
data_dict = {}
values = []
lengths = []

for i in range(1, demostraciones +1):
    file_name = f"/home/nox/Escritorio/motion_primitives/GaussianMotion/ExpJointAdam/Water/AguaLeft_{i}.csv"
    data = pd.read_csv(file_name)
    lengths.append(len(data['joint_0'][::gap]))

min_length = min(lengths)

for i in range(1, demostraciones +1):
    file_name = f"/home/nox/Escritorio/motion_primitives/GaussianMotion/ExpJointAdam/Water/AguaLeft_{i}.csv"
    data = pd.read_csv(file_name)
    
    j0_data = np.array(data['joint_0'][::gap])[:min_length]
    j1_data = np.array(data['joint_1'][::gap])[:min_length]
    j2_data = np.array(data['joint_2'][::gap])[:min_length]
    j3_data = np.array(data['joint_3'][::gap])[:min_length]
    j4_data = np.array(data['joint_4'][::gap])[:min_length]
    j5_data = np.array(data['joint_5'][::gap])[:min_length]
    #j6_data = np.array(data['joint_6'][::gap])[:min_length] #j6 for IIWA
    
    data_dict[f"Grasping_{i}"] = np.array([j0_data, j1_data, j2_data, j3_data, j4_data, j5_data])
    values.append(np.array([j0_data, j1_data, j2_data, j3_data, j4_data, j5_data]))
    
for key, value in data_dict.items():
    pos = value
    t = np.linspace(0, 6, min_length).reshape(1, min_length)
    X_ = t.T
    Y_ = pos.T
    print(Y_.shape)
    if key == "Grasping_1":
        size = Y_.shape[0]
        X = X_
        Y = Y_
    else:
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))
        
np.random.seed(30)
font_size = 18
print(type(pos))
#* Selecting the target position and time and via points
#! Points for Real data (constructed by hand, you can use what ever you want)

target_t = t[0][-1]
target_position=np.array([values[1][0][-1]+0.76, values[1][1][-1]+0.84, values[1][2][-1]+0.64,values[1][3][-1]-0.8, values[1][4][-1]-0.75, values[1][5][-1]-1.2])

via_point0_t = t[0][0]
via_point0_position= np.array([j0_data[0]+0.84, j1_data[0]-0.68, j2_data[0]+0.79, j3_data[0]+0.87, j4_data[0]-0.77, j5_data[0]-0.63])

via_point1_t = t[0][100]
via_point1_position = np.array([values[1][0][50]+0.27, values[1][1][50], values[1][2][50]+0.45,values[1][3][50]-0.5, values[1][4][50]+0.46, values[1][5][50]+0.15])

via_point2_t = t[0][50]
via_point2_position = np.array([values[1][0][20]+0.27, values[1][1][20], values[1][2][20]+0.45,values[1][3][20]-0.5, values[1][4][20]+0.46, values[1][5][20]+0.15])
#via_point1_position = np.array([1.28, -2.89, -1.23,-1.82, -0.81, 0.32])

#* Vias points
X_ = np.array([via_point0_t,via_point2_t, target_t]).reshape(-1, 1)
Y_ = np.array([via_point0_position,via_point2_position, target_position])

#time.sleep(1000)
# predicting for dim0   --> size=demos[0]['pos'][0].T[:, 0::gap].T.shape[0]
observation_noise = 1.0
#gp_mp= ProGpMp(X, Y, X_, Y_,dim=3, demos=demostraciones,size = demos[0].pos[:, 0::gap].T.shape[0] , observation_noise=observation_noise) # For RAIL data use: demos[0].pos[:, 0::gap].T.shape[0]
gp_mp= ProGpMp(X, Y, X_, Y_,dim=6, demos=demostraciones,size = size, observation_noise=observation_noise) # For real data

gp_mp.BlendedGpMp(gp_mp.ProGP) #? If you use more than one GpMp is mandatory to use BlendedGpMp, input: list[]
test_x = np.arange(0.0, target_t, dt)
#test_x = np.arange(Y_[0,0],Y_[2,0],(1/pos0.size))
print(test_x)
print(len(test_x))

#alpha_list = (np.tanh((test_x - 0.5) * 5) + 1.0) / 2
#print(type(alpha_list))
""" alpha_list=np.ones(len(test_x))
alpha_list = np.vstack((alpha_list, alpha_list))
alpha_list = np.vstack((np.ones(np.shape(test_x)[0]), np.ones(np.shape(test_x)[0]))) """

# alpha_list = np.vstack((np.ones(np.shape(test_x)[0]), np.ones(np.shape(test_x)[0])))
mean_blended, var_blended = gp_mp.predict_BlendedPos(test_x.reshape(-1, 1))
print("Valores del path final")
print(var_blended[0].reshape(-1, 1))
print(var_blended[1].reshape(-1, 1))
print(var_blended[2].reshape(-1, 1))
print(var_blended[3].reshape(-1, 1))
print(var_blended[4].reshape(-1, 1))
print(var_blended[5].reshape(-1, 1))
#print(var_blended[6].reshape(-1, 1))
#time.sleep(1000)
var_blended[0]=np.where(var_blended[0]<0,0,var_blended[0])
var_blended[1]=np.where(var_blended[1]<0,0,var_blended[1])
var_blended[2]=np.where(var_blended[2]<0,0,var_blended[2])
var_blended[3]=np.where(var_blended[3]<0,0,var_blended[3])
var_blended[4]=np.where(var_blended[4]<0,0,var_blended[4])
var_blended[5]=np.where(var_blended[5]<0,0,var_blended[5])
#var_blended[6]=np.where(var_blended[6]<0,0,var_blended[6])


font_size = 25
fig = plt.figure(figsize=(16, 8), dpi=100)
plt.subplots_adjust(left=0.1, right=0.9, wspace=0.5, hspace=0.5, bottom=0.15, top=0.99)
#* Visualization in 3D

#ax1 = fig.add_subplot(121, projection='3d')
""" ax1.scatter(Y_[:, 0], Y_[:, 1], Y_[:, 2], s=600, c='blue', marker='x')
ax1.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=20, c='blue', marker='o', alpha=0.3)
ax1.plot(mean_blended[0], mean_blended[1], mean_blended[2], c='black', linewidth=5, label='$ProGpMp$')

ax1.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
ax1.set_xlabel('$x$/mm', fontsize=font_size)
ax1.set_ylabel('$y$/mm', fontsize=font_size)
ax1.set_zlabel('$z$/mm', fontsize=font_size) """

#* 2D visualization
ax2 = fig.add_subplot(231)
ax2.plot(test_x, mean_blended[0], c='red', linewidth=3, label='$q0_{ProGP}$')
ax2.fill_between(test_x, mean_blended[0] - 5 * np.sqrt(var_blended[0]), mean_blended[0] + 5 * np.sqrt(var_blended[0]), color='red', alpha=0.3)
ax2.scatter(X_[:, 0], Y_[:, 0], s=200, c='red', marker='x')
ax2.scatter(X[:, 0], Y[:, 0], s=10, c='red', marker='o', alpha=0.3)
ax2.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
#ax2.set_xlabel('', fontsize=font_size)
ax2.set_ylabel('q0', fontsize=font_size)

ax3 = fig.add_subplot(232)
ax3.plot(test_x, mean_blended[1], c='blue', linewidth=3, label='$q1_{ProGP}$')
ax3.fill_between(test_x, mean_blended[1] - 5 * np.sqrt(var_blended[1]), mean_blended[1] + 5 * np.sqrt(var_blended[1]), color='blue', alpha=0.3)
ax3.scatter(X_[:, 0], Y_[:, 1], s=200, c='blue', marker='x')
ax3.scatter(X[:, 0], Y[:, 1], s=10, c='blue', marker='o', alpha=0.3)
ax3.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
#ax3.set_xlabel('(c)', fontsize=font_size)
ax3.set_ylabel('q1', fontsize=font_size)

ax4 = fig.add_subplot(233)
ax4.plot(test_x, mean_blended[2], c='green', linewidth=3, label='$q2_{ProGP}$')
ax4.fill_between(test_x, mean_blended[2] - 5 * np.sqrt(var_blended[2]), mean_blended[2] + 5 * np.sqrt(var_blended[2]), color='green', alpha=0.3)
ax4.scatter(X_[:, 0], Y_[:, 2], s=200, c='green', marker='x')
ax4.scatter(X[:, 0], Y[:, 2], s=10, c='green', marker='o', alpha=0.3)
ax4.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
#ax4.set_xlabel('(d)', fontsize=font_size)
ax4.set_ylabel('q2', fontsize=font_size)

ax4 = fig.add_subplot(234)
ax4.plot(test_x, mean_blended[3], c='purple', linewidth=3, label='$q3_{ProGP}$')
ax4.fill_between(test_x, mean_blended[3] - 5 * np.sqrt(var_blended[3]), mean_blended[3] + 5 * np.sqrt(var_blended[3]), color='purple', alpha=0.3)
ax4.scatter(X_[:, 0], Y_[:, 3], s=200, c='purple', marker='x')
ax4.scatter(X[:, 0], Y[:, 3], s=10, c='purple', marker='o', alpha=0.3)
ax4.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
#ax4.set_xlabel('(d)', fontsize=font_size)
ax4.set_ylabel('q3', fontsize=font_size)

ax4 = fig.add_subplot(235)
ax4.plot(test_x, mean_blended[4], c='orange', linewidth=3, label='$q4_{ProGP}$')
ax4.fill_between(test_x, mean_blended[4] - 5 * np.sqrt(var_blended[4]), mean_blended[4] + 5 * np.sqrt(var_blended[4]), color='orange', alpha=0.3)
ax4.scatter(X_[:, 0], Y_[:, 4], s=200, c='orange', marker='x')
ax4.scatter(X[:, 0], Y[:, 4], s=10, c='orange', marker='o', alpha=0.3)
ax4.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
#ax4.set_xlabel('(d)', fontsize=font_size)
ax4.set_ylabel('q4', fontsize=font_size)

ax4 = fig.add_subplot(236)
ax4.plot(test_x, mean_blended[5], c='hotpink', linewidth=3, label='$q5_{ProGP}$')
ax4.fill_between(test_x, mean_blended[5] - 5 * np.sqrt(var_blended[5]), mean_blended[5] + 5 * np.sqrt(var_blended[5]), color='hotpink', alpha=0.3)
ax4.scatter(X_[:, 0], Y_[:, 5], s=200, c='hotpink', marker='x')
ax4.scatter(X[:, 0], Y[:, 5], s=10, c='hotpink', marker='o', alpha=0.3)
ax4.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
#ax4.set_xlabel('(d)', fontsize=font_size)
ax4.set_ylabel('q5', fontsize=font_size)

""" ax4 = fig.add_subplot(337)
ax4.plot(test_x, mean_blended[6], c='hotpink', linewidth=3, label='$q6_{ProGP}$')
ax4.fill_between(test_x, mean_blended[6] - 5 * np.sqrt(var_blended[6]), mean_blended[6] + 5 * np.sqrt(var_blended[6]), color='hotpink', alpha=0.3)
ax4.scatter(X_[:, 0], Y_[:, 6], s=200, c='hotpink', marker='x')
ax4.scatter(X[:, 0], Y[:, 6], s=10, c='hotpink', marker='o', alpha=0.3)
ax4.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
#ax4.set_xlabel('(d)', fontsize=font_size)
ax4.set_ylabel('q6', fontsize=font_size) """

plt.show()
old_mean = []
old_mean = mean_blended.copy()

# Matrix Nx6
data_matrix = np.column_stack((mean_blended[0], mean_blended[1], mean_blended[2], mean_blended[3], mean_blended[4], mean_blended[5]))

# Save files as .mat
savemat("mean_blended_data.mat", {"mean_blended": data_matrix})

print("Archivo guardado exitosamente como 'mean_blended_data.mat'")