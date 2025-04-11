# Exampe of use of the method for 1D blending of two different Gaussian processes
#* The blending is done by using the method BlendDifferentGaussians
#*That can be used for the blending of N Gaussina in joint space (each joint use one different gaussians)
import numpy as np
from ProGP import ProGpMp
from ProGP import BlendDifferentGaussians
import matplotlib.pyplot as plt
import time

np.random.seed(3)
font_size = 30

#*------ Example of merging two Gaussians data

size_set = 10
size_via_points = 2


x1 = np.random.uniform(0, 1, size_set)
y1 = 0.8 * np.sin(2 * np.pi * x1) + 0.5 * np.sin(np.pi * x1) + np.random.normal(0, 0.2, size_set)

x1_ = np.random.uniform(0.5, 1.0, size_via_points)
y1_ = 0.8 * np.sin(2 * np.pi * x1_) + 0.5 * np.sin(np.pi * x1_)

progp_mp1 = ProGpMp(x1.reshape(-1, 1), y1, x1_.reshape(-1, 1), y1_, dim=1, observation_noise=0.4, demos=1, size=size_set)
print('progp_mp1 training...')
progp_mp1.train()


x2 = np.random.uniform(0, 1, size_set)
y2 = 1.2 * np.cos(2 * np.pi * x2) + 0.3 * np.cos(np.pi * x2) + np.random.normal(0, 0.2, size_set)

x2_ = np.random.uniform(0, 0.5, size_via_points)
y2_ = 1.2 * np.cos(2 * np.pi * x2_) + 0.3 * np.cos(np.pi * x2_)
progp_mp2 = ProGpMp(x2.reshape(-1, 1), y2, x2_.reshape(-1, 1), y2_,dim=1, observation_noise=0.4,demos=1,size= size_set)
print('progp_mp2 training...')
progp_mp2.train()

#! Blending case 

blended_progpmp = BlendDifferentGaussians([progp_mp1, progp_mp2])

test_x = np.arange(0, 1, 0.01)
mean1, var1 = progp_mp1.predict_determined_input_1D(test_x.reshape(-1, 1))
mean1 = mean1.reshape(-1)
var1 = var1.reshape(-1)

mean2, var2 = progp_mp2.predict_determined_input_1D(test_x.reshape(-1, 1))
mean2 = mean2.reshape(-1)
var2 = var2.reshape(-1)

""" alpha_list = (np.tanh((test_x - 0.5) * 5) + 1.0) / 2
alpha_list = np.vstack((alpha_list, 1 - alpha_list)) """
alpha_list = 0.3687 * (1 - np.cos(np.pi * test_x))

alpha_list = np.vstack((alpha_list, 1 - alpha_list))
# alpha_list = np.vstack((np.ones(np.shape(test_x)[0]), np.ones(np.shape(test_x)[0])))
mean_blended, var_blended = blended_progpmp.predict_blended_determined_input(test_x.reshape(-1, 1), alpha_list)
mean_blended.reshape(-1)
var_blended.reshape(-1)

#! Plotting
linewidth = 3
alpha = 0.3

plt.figure(figsize=(16, 8), dpi=100)
plt.subplots_adjust(left=0.05, right=0.99, wspace=0.8, hspace=0.8, bottom=0.1, top=0.99)
plt1 = plt.subplot2grid((8, 8), (0, 0), rowspan=4, colspan=4)
size = 30
plt1.scatter(x1, y1, c='red', marker='o',s=np.ones(size_set) * size, alpha=alpha)
plt1.scatter(x1_, y1_, c='red', marker='x', s=np.ones(size_via_points) * size*3)
plt1.scatter(x2, y2, c='blue', marker='o',s=np.ones(size_set) * size, alpha=alpha)
plt1.scatter(x2_, y2_, c='blue', marker='x', s=np.ones(size_via_points) * size*3)


plt1.plot(test_x, mean1, c='red', label='GMP1', linewidth=linewidth)
plt1.fill_between(test_x, mean1 - 2 * var1, mean1 + 2 * var1, color='red', alpha=alpha)

plt1.plot(test_x, mean2, c='blue', label='GMP2', linewidth=linewidth)
plt1.fill_between(test_x, mean2 - 2 * var2, mean2 + 2 * var2, color='blue', alpha=alpha)

plt1.plot(test_x, mean_blended, c='grey', label='GMP Merged')
plt1.fill_between(test_x, mean_blended - 3 * var_blended, mean_blended + 3 * var_blended, color='grey', alpha=0.5)

plt1.legend(loc='upper right', frameon=False, handlelength=1, ncol=2, columnspacing=1)
plt1.tick_params(labelsize=font_size)


plt2 = plt.subplot2grid((8, 8), (4, 0), rowspan=4, colspan=4)
length = np.shape(test_x)[0]
plt2.plot(test_x, alpha_list[0, :], c='red', label='$\\beta_1$', linewidth=linewidth)
plt2.plot(test_x, alpha_list[1, :], c='blue', label='$\\beta_2$', linewidth=linewidth)
plt2.legend(loc='center right', frameon=False, handlelength=1, ncol=2, columnspacing=1)
plt2.tick_params(labelsize=font_size)
plt2.set_xlabel('(a): Merging case 1', fontsize=font_size)


alpha_list = np.vstack((np.ones(np.shape(test_x)[0]), np.ones(np.shape(test_x)[0])))
mean_blended, var_blended = blended_progpmp.predict_blended_determined_input(test_x.reshape(-1, 1), alpha_list)
plt3 = plt.subplot2grid((8, 8), (0, 4), rowspan=4, colspan=4)
size = 30
plt3.scatter(x1, y1, c='red', marker='o', s=np.ones(size_set) * size,alpha=alpha)
plt3.scatter(x1_, y1_, c='red', marker='x', s=np.ones(size_via_points) * size*3)
plt3.scatter(x2, y2, c='blue', marker='o', s=np.ones(size_set) * size,alpha=alpha)
plt3.scatter(x2_, y2_, c='blue', marker='x', s=np.ones(size_via_points) * size*3)
plt3.plot(test_x, mean1, c='red', label='GMP1', linewidth=linewidth)
plt3.fill_between(test_x, mean1 - 2 * var1, mean1 + 2 * var1, color='red', alpha=alpha)
plt3.plot(test_x, mean2, c='blue', label='GMP2', linewidth=linewidth)
plt3.fill_between(test_x, mean2 - 2 * var2, mean2 + 2 * var2, color='blue', alpha=alpha)
plt3.plot(test_x, mean_blended, c='grey', label='$GMP_{merged}$')
plt3.fill_between(test_x, mean_blended - 3 * var_blended, mean_blended + 3 * var_blended, color='grey', alpha=0.5)
plt3.legend(loc='upper right', frameon=False, handlelength=1, ncol=2, columnspacing=1)
plt3.tick_params(labelsize=font_size)

plt4 = plt.subplot2grid((8, 8), (4, 4), rowspan=4, colspan=4)
length = np.shape(test_x)[0]
plt4.plot(test_x, alpha_list[0, :], c='red', label='$\\beta_1$', linewidth=linewidth)
plt4.plot(test_x, alpha_list[1, :], c='blue', label='$\\beta_2$', linewidth=linewidth)
plt4.legend(loc='upper right', frameon=False, handlelength=1, ncol=2, columnspacing=1)
plt4.set_xlabel('(b): Merging case 2', fontsize=font_size)
plt4.tick_params(labelsize=font_size)

plt.show()

#*------ Example of merging three Gaussians data


size_set = 10
# GP 1
x1 = np.random.uniform(0, 1, size_set)
y1 = np.sin(2 * np.pi * x1) + np.cos(2 * np.pi * x1) + np.random.normal(0, 0.8, size_set)
x1_ = np.random.uniform(0.5, 1.0, 2)
y1_ = np.sin(2 * np.pi * x1_) + np.cos(2 * np.pi * x1_)
progp_mp1 = ProGpMp(x1.reshape(-1, 1), y1, x1_.reshape(-1, 1), y1_, dim=1, observation_noise=0.4, demos=1, size=size_set)
print('progp_mp1 training...')
progp_mp1.train()

# GP 2
x2 = np.random.uniform(0, 1, size_set)
y2 = 2 * np.sin(2 * np.pi * x2) + 3 * np.cos(2 * np.pi * x2) + np.random.normal(0, 0.8, size_set)
x2_ = np.random.uniform(0, 0.5, 2)
y2_ = 2 * np.sin(0.5 * np.pi * x2_) + 3 * np.cos(2 * np.pi * x2_)
progp_mp2 = ProGpMp(x2.reshape(-1, 1), y2, x2_.reshape(-1, 1), y2_, dim=1, observation_noise=0.4, demos=1, size=size_set)
print('progp_mp2 training...')
progp_mp2.train()

# GP 3
x3 = np.random.uniform(0, 1, size_set)
y3 = -np.sin(4 * np.pi * x3) + 2 * np.cos(np.pi * x3) + np.random.normal(0, 0.8, size_set)
x3_ = np.random.uniform(0.25, 0.75, 2)
y3_ = -np.sin(4 * np.pi * x3_) + 2 * np.cos(np.pi * x3_)
progp_mp3 = ProGpMp(x3.reshape(-1, 1), y3, x3_.reshape(-1, 1), y3_, dim=1, observation_noise=0.4, demos=1, size=size_set)
print('progp_mp3 training...')
progp_mp3.train()

#! Blending case 
blended_progpmp = BlendDifferentGaussians([progp_mp1, progp_mp2, progp_mp3])

test_x = np.arange(0, 1, 0.01)

mean1, var1 = progp_mp1.predict_determined_input_1D(test_x.reshape(-1, 1))
mean2, var2 = progp_mp2.predict_determined_input_1D(test_x.reshape(-1, 1))
mean3, var3 = progp_mp3.predict_determined_input_1D(test_x.reshape(-1, 1))

# Define alpha weights
alpha1 = np.tanh(10 * (test_x - 0.3)) + 1     # range [0, 2]
alpha2 = np.cos(2 * np.pi * test_x) + 1       # range [0, 2]
alpha3 = np.sin(2 * np.pi * test_x) + 1
alpha_stack = np.vstack((alpha1, alpha2, alpha3))
alpha_list = alpha_stack / np.sum(alpha_stack, axis=0)

mean_blended, var_blended = blended_progpmp.predict_blended_determined_input(test_x.reshape(-1, 1), alpha_list)

#! Plotting
linewidth = 3
alpha = 0.3
size = 30

plt.figure(figsize=(18, 10), dpi=100)
plt1 = plt.subplot2grid((8, 8), (0, 0), rowspan=4, colspan=8)
plt1.scatter(x1, y1, c='red', marker='o', s=size, alpha=alpha)
plt1.scatter(x1_, y1_, c='red', marker='x', s=size*3)
plt1.scatter(x2, y2, c='blue', marker='o', s=size, alpha=alpha)
plt1.scatter(x2_, y2_, c='blue', marker='x', s=size*3)
plt1.scatter(x3, y3, c='green', marker='o', s=size, alpha=alpha)
plt1.scatter(x3_, y3_, c='green', marker='x', s=size*3)

plt1.plot(test_x, mean1.reshape(-1), c='red', label='GMP1', linewidth=linewidth)
plt1.fill_between(test_x, mean1.reshape(-1) - 2 * var1.reshape(-1), mean1.reshape(-1) + 2 * var1.reshape(-1), color='red', alpha=alpha)

plt1.plot(test_x, mean2.reshape(-1), c='blue', label='GMP2', linewidth=linewidth)
plt1.fill_between(test_x, mean2.reshape(-1) - 2 * var2.reshape(-1), mean2.reshape(-1) + 2 * var2.reshape(-1), color='blue', alpha=alpha)

plt1.plot(test_x, mean3.reshape(-1), c='green', label='GMP3', linewidth=linewidth)
plt1.fill_between(test_x, mean3.reshape(-1) - 2 * var3.reshape(-1), mean3.reshape(-1) + 2 * var3.reshape(-1), color='green', alpha=alpha)

plt1.plot(test_x, mean_blended, c='black', label='$GMP_{merged}$')
plt1.fill_between(test_x, mean_blended - 3 * var_blended, mean_blended + 3 * var_blended, color='black', alpha=0.5)

plt1.legend(loc='upper right', frameon=False, handlelength=1, ncol=3, columnspacing=1)
plt1.tick_params(labelsize=font_size)

plt2 = plt.subplot2grid((8, 8), (4, 0), rowspan=4, colspan=8)
plt2.plot(test_x, alpha_list[0, :], c='red', label='$\\beta_1$', linewidth=linewidth)
plt2.plot(test_x, alpha_list[1, :], c='blue', label='$\\beta_2$', linewidth=linewidth)
plt2.plot(test_x, alpha_list[2, :], c='green', label='$\\beta_3$', linewidth=linewidth)
plt2.legend(loc='upper right', frameon=False, handlelength=1, ncol=3, columnspacing=1)
plt2.set_xlabel('Merging Case: 3 Gaussians', fontsize=font_size)
plt2.tick_params(labelsize=font_size)

plt.show()



