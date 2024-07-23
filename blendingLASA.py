import matplotlib.pyplot as plt
import pyLasaDataset as lasa
from ProGP import ProGpMp
from ProGP import BlendDifferentGaussians
import numpy as np
import time
from ObstacleAvoidance import *

np.random.seed(30)
font_size = 18
data = lasa.DataSet.Multi_Models_3 #? Here put the dataset that you want to use
dt = data.dt
demos = data.demos
gap = 30
#lasa.utilities.plot_model(lasa.DataSet.BendedLine) #? If you want to plot the data uncomment this line 
lasa.utilities.plot_model(lasa.DataSet.Multi_Models_3) #? If you want to plot the data uncomment this line 
#! --------------------------------------------- loading and training the model data------------------------------------
#*Loading all the LASA dataset that you want
demostraciones = 3

for i in range(demostraciones):
    demo = demos[i]
    pos = demo.pos[:, 0::gap]  # np.ndarray, shape: (2,1000/gap)
    vel = demo.vel[:, 0::gap]  # np.ndarray, shape: (2,1000/gap)
    acc = demo.acc[:, 0::gap]  # np.ndarray, shape: (2,1000/gap)
    t = demo.t[:, 0::gap]  # np.ndarray, shape: (1,1000/gap)
    X_ = t.T
    Y_ = pos.T
    if i == 0:
        X = X_
        Y = Y_
    else:
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))
print("Prove of the data:",demos[0].pos[:, 0::gap].T.shape[0])

#* Selecting the target position and time and via points
""" target_t = 4.22313 #! Time in X axis
target_position = np.array([0.0, 0.0]) #! Target position in Y axis
via_point0_t = 0.0
via_point0_position = np.array([28.364, 0.1552])
via_point1_t = 2.1559
via_point1_position = np.array([12.9869, -3.32277]) """

#* Constructing the training set with the target and via points
target_t = sum(demos[i].t[:, 0::gap][0, -1] for i in range(demostraciones)) / demostraciones
target_position = sum(demos[i].pos[:, 0::gap][:, -1] for i in range(demostraciones)) / demostraciones
via_point0_t = sum(demos[i].t[:, 0::gap][0, 0] for i in range(demostraciones)) / demostraciones
via_point0_position = sum(demos[i].pos[:, 0::gap][:, 0] for i in range(demostraciones)) / demostraciones
via_point1_t = sum(demos[i].t[:, 0::gap][0, demos[i].pos[:, 0::gap].shape[1] * 2 // 4] for i in range(demostraciones)) / demostraciones
via_point1_position = sum(demos[i].pos[:, 0::gap][:, demos[i].pos[:, 0::gap].shape[1] * 2 // 4] for i in range(demostraciones)) / demostraciones
#* Vias points
X_ = np.array([via_point0_t, target_t]).reshape(-1, 1)
Y_ = np.array([via_point0_position, target_position])


#* Predicting for dim0
observation_noise = 1.0
gp_mp= ProGpMp(X, Y, X_, Y_,dim=2, demos=demostraciones, size=demos[0].pos[:, 0::gap].T.shape[0], observation_noise=observation_noise)

gp_mp.BlendedGpMp(gp_mp.ProGP) #? If you use more than one GpMp is mandatory to use BlendedGpMp, input: list[]
test_x = np.arange(0.0, target_t, dt)
#test_x = np.arange(Y_[0,0],Y_[2,0],(1/pos0.size))
print(test_x)
print(len(test_x))

mean_blended, var_blended = gp_mp.predict_BlendedPos(test_x.reshape(-1, 1))
print("Valores del path final")
print(var_blended[0].reshape(-1, 1))
print(var_blended[1].reshape(-1, 1))
var_blended[0]=np.where(var_blended[0]<0,0,var_blended[0])
var_blended[1]=np.where(var_blended[1]<0,0,var_blended[1])


#! --------------------------------------------- loading and training the model data 2------------------------------------
demostrac_fin = demostraciones + 4
for i in range(demostraciones+1,demostrac_fin):
    print("Demostracion",i)
    demo2 = demos[i]
    pos2 = demo2.pos[:, 0::gap]  # np.ndarray, shape: (2,1000/gap)
    vel2 = demo2.vel[:, 0::gap]  # np.ndarray, shape: (2,1000/gap)
    acc2 = demo2.acc[:, 0::gap]  # np.ndarray, shape: (2,1000/gap)
    t2 = demo2.t[:, 0::gap]  # np.ndarray, shape: (1,1000/gap)
    X2_ = t2.T
    Y2_ = pos2.T
    if i == demostraciones+1:
        X2 = X2_
        Y2 = Y2_
    else:
        X2 = np.vstack((X2, X2_))
        Y2 = np.vstack((Y2, Y2_))
#* Selecting the target position and time and via points
""" target_t = 4.22313 #! Time in X axis
target_position = np.array([0.0, 0.0]) #! Target position in Y axis
via_point0_t = 0.0
via_point0_position = np.array([28.364, 0.1552])
via_point1_t = 2.1559
via_point1_position = np.array([12.9869, -3.32277]) """

#* Constructing the training set with the target and via points
target_t2 = sum(demos[i].t[:, 0::gap][0, -1] for i in range(demostraciones+1,demostrac_fin)) / (demostrac_fin - (demostraciones+1))
target_position2 = sum(demos[i].pos[:, 0::gap][:, -1] for i in range(demostraciones+1,demostrac_fin)) / (demostrac_fin - (demostraciones+1))
via_point0_t2 = sum(demos[i].t[:, 0::gap][0, 0] for i in range(demostraciones+1,demostrac_fin)) / (demostrac_fin - (demostraciones+1))
via_point0_position2 = sum(demos[i].pos[:, 0::gap][:, 0] for i in range(demostraciones+1,demostrac_fin)) / (demostrac_fin - (demostraciones+1))
via_point1_t2 = sum(demos[i].t[:, 0::gap][0, demos[i].pos[:, 0::gap].shape[1] * 2 // 4] for i in range(demostraciones+1,demostrac_fin)) / (demostrac_fin - (demostraciones+1))
via_point1_position2 = sum(demos[i].pos[:, 0::gap][:, demos[i].pos[:, 0::gap].shape[1] * 2 // 4] for i in range(demostraciones+1,demostrac_fin)) / (demostrac_fin - (demostraciones+1))
#* Vias points
X2_ = np.array([via_point0_t2, via_point1_t2,target_t2]).reshape(-1, 1)
Y2_ = np.array([via_point0_position2,via_point1_position2, target_position2])


#* Predicting for dim0
observation_noise = 1.0
gp_mp2= ProGpMp(X2, Y2, X2_, Y2_,dim=2, demos=demostrac_fin - (demostraciones+1), size=demos[4].pos[:, 0::gap].T.shape[0], observation_noise=observation_noise)

gp_mp2.BlendedGpMp(gp_mp2.ProGP) #? If you use more than one GpMp is mandatory to use BlendedGpMp, input: list[]
test_2x = np.arange(0.0, target_t2, dt)
#test_x = np.arange(Y_[0,0],Y_[2,0],(1/pos0.size))


mean_blended2, var_blended2 = gp_mp2.predict_BlendedPos(test_2x.reshape(-1, 1))
print("Valores del path final")
print(var_blended2[0].reshape(-1, 1))
print(var_blended2[1].reshape(-1, 1))
var_blended2[0]=np.where(var_blended2[0]<0,0,var_blended2[0])
var_blended2[1]=np.where(var_blended2[1]<0,0,var_blended2[1])


print("HASTA AQUI")
print(gp_mp.ProGP)
print("*************")
print(gp_mp2.ProGP)


#!  -------------------------------------------------------blending between the diferent data-----------------------------------------------------
blended_dim0 = BlendDifferentGaussians([gp_mp.ProGP[0], gp_mp2.ProGP[0]])
blended_dim1 = BlendDifferentGaussians([gp_mp.ProGP[1], gp_mp2.ProGP[1]])
test_x_model2_length = np.shape(test_2x)[0]
via_point1_t_model1 = via_point1_t
index = 0
for i in range(test_x_model2_length):
    if test_2x[i] > via_point1_t_model1:
        index = i
        break
alpha_model1 = np.ones(test_x_model2_length)
model12model2_length = test_x_model2_length - 1 - index
mu_dim0_list = np.empty(test_x_model2_length)
var_dim0_list = np.empty(test_x_model2_length)
mu_dim1_list = np.empty(test_x_model2_length)
var_dim1_list = np.empty(test_x_model2_length)
alpha_list = np.empty(test_x_model2_length)
for i in range(test_x_model2_length):
    if i <= index:
        alpha = 1
    else:
        alpha = 1 - np.tanh((i - index - 1) / model12model2_length * 5)
    alpha_list[i] = alpha
    mu_dim0, var_dim0 = blended_dim0.predict_single_blended_determined_input(test_2x[i], np.array([alpha, 1 - alpha]))
    mu_dim1, var_dim1 = blended_dim1.predict_single_blended_determined_input(test_2x[i], np.array([alpha, 1 - alpha]))
    mu_dim0_list[i] = mu_dim0
    var_dim0_list[i] = var_dim0
    mu_dim1_list[i] = mu_dim1
    var_dim1_list[i] = var_dim1
#!  -------------------------------------------------------plotting-----------------------------------------------------

plt.figure(figsize=(16, 8), dpi=100)
plt.subplots_adjust(left=0.05, right=0.99, wspace=0.8, hspace=0.8, bottom=0.1, top=0.99)
plt1 = plt.subplot2grid((8, 16), (0, 0), rowspan=8, colspan=8)
size = 30
plt1.scatter(Y_[:, 0], Y_[:, 1], s=400, c='blue', marker='x')
plt1.scatter(Y[:, 0], Y[:, 1], s=15, c='blue', marker='o', alpha=0.3)
plt1.scatter(Y2_[:, 0], Y2_[:, 1], s=400, c='red', marker='x')
plt1.scatter(Y2[:, 0], Y2[:, 1], s=15, c='red', marker='o', alpha=0.3)

plt1.plot(mean_blended[0], mean_blended[1], c='blue', linewidth=4, label='$GMP1$')
plt1.plot(mean_blended2[0], mean_blended2[1], c='red', linewidth=4, label='$GMP2$')
plt1.plot(mu_dim0_list, mu_dim1_list, ls='-', c='grey', linewidth=4, label='$GMP_{merged}$')

plt1.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
plt1.tick_params(labelsize=font_size)
plt1.set_xlabel('$x$/mm\n(a)', fontsize=font_size)
plt1.set_ylabel('$y$/mm', fontsize=font_size)

plt2 = plt.subplot2grid((8, 16), (0, 8), rowspan=3, colspan=8)
plt2.plot(test_x, mean_blended[0], c='blue', linewidth=4, label='$x_{GMP1}$')
plt2.fill_between(test_x, mean_blended[0] - 5 * np.sqrt(var_blended[0]), mean_blended[0] + 5 * np.sqrt(var_blended[0]), color='blue', alpha=0.3)
plt2.scatter(X_[:, 0], Y_[:, 0], s=400, c='blue', marker='x')
plt2.scatter(X[:, 0], Y[:, 0], s=15, c='blue', marker='o', alpha=0.3)
plt2.plot(test_2x, mean_blended2[0], c='red', linewidth=4, label='$x_{GMP2}$')
plt2.fill_between(test_2x, mean_blended2[0] - 5 * np.sqrt(var_blended2[0]), mean_blended2[0] + 5 * np.sqrt(var_blended2[0]), color='red', alpha=0.3)
plt2.scatter(X2_[:, 0], Y2_[:, 0], s=400, c='red', marker='x')
plt2.scatter(X2[:, 0], Y2[:, 0], s=15, c='red', marker='o', alpha=0.3)
plt2.plot(test_2x, mu_dim0_list, c='grey', linewidth=3, label='$x_{merged}$')
plt2.fill_between(test_2x, mu_dim0_list - 5 * np.sqrt(var_dim0_list), mu_dim0_list + 5 * np.sqrt(var_dim0_list), color='grey', alpha=alpha)
plt2.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
plt2.tick_params(labelsize=font_size)
plt2.set_xlabel('(b)', fontsize=font_size)
plt2.set_ylabel('$x$/mm', fontsize=font_size)

plt3 = plt.subplot2grid((8, 16), (3, 8), rowspan=3, colspan=8)
plt3.plot(test_x, mean_blended[1], c='blue', linewidth=4, label='$y_{GMP1}$')
plt3.fill_between(test_x, mean_blended[1] - 5 * np.sqrt(var_blended[1]), mean_blended[1] + 5 * np.sqrt(var_blended[1]), color='blue', alpha=0.3)
plt3.scatter(X_[:, 0], Y_[:, 1], s=400, c='blue', marker='x')
plt3.scatter(X[:, 0], Y[:, 1], s=15, c='blue', marker='o', alpha=0.3)
plt3.plot(test_2x, mean_blended2[1], c='red', linewidth=4, label='$y_{GMP2}$')
plt3.fill_between(test_2x, mean_blended2[1] - 5 * np.sqrt(var_blended2[1]), mean_blended2[1] + 5 * np.sqrt(var_blended2[1]), color='red', alpha=0.3)
plt3.scatter(X2_[:, 0], Y2_[:, 1], s=400, c='red', marker='x')
plt3.scatter(X2[:, 0], Y2[:, 1], s=15, c='red', marker='o', alpha=0.3)
plt3.plot(test_2x, mu_dim1_list, c='grey', linewidth=3, label='$y_{merged}$')
plt3.fill_between(test_2x, mu_dim1_list - 15* np.sqrt(var_dim1_list), mu_dim1_list + 15* np.sqrt(var_dim1_list), color='grey', alpha=alpha)
plt3.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
plt3.tick_params(labelsize=font_size)
plt3.set_xlabel('(c)', fontsize=font_size)
plt3.set_ylabel('$y$/mm', fontsize=font_size)

plt4 = plt.subplot2grid((8, 16), (6, 8), rowspan=2, colspan=8)
plt4.plot(test_2x, alpha_list, linewidth=3, label='$\\alpha_1$', c='blue')
plt4.plot(test_2x, 1 - alpha_list, linewidth=3, label='$\\alpha_2$', c='red')
plt4.legend(loc='upper left', frameon=False, handlelength=1, ncol=2, columnspacing=1)
plt4.tick_params(labelsize=font_size)
plt4.set_ylabel('$\\alpha$', fontsize=font_size)
plt4.set_xlabel('time/s\n(d)', fontsize=font_size)

plt.show()
