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


size_set = 10
x1 = np.random.uniform(0, 1, size_set)
y1 = np.sin(2 * np.pi * x1) + np.cos(2 * np.pi * x1) + np.random.normal(0, 0.8, size_set)
print(x1.reshape(-1, 1).shape)
print("****")
print(y1.shape)


size_via_points = 2
x1_ = np.random.uniform(0.5, 1.0, size_via_points)
y1_ = np.sin(2 * np.pi * x1_) + np.cos(2 * np.pi * x1_)

print(x1_.shape)
print(y1_.shape)
time.sleep(1000)
progp_mp1 = ProGpMp(x1.reshape(-1, 1), y1, x1_.reshape(-1, 1), y1_,dim=1, observation_noise=0.4,demos=1,size= size_set)
print('progp_mp1 training...')
progp_mp1.train()

x2 = np.random.uniform(0, 1, size_set)
y2 = 2 * np.sin(2 * np.pi * x2) + 3 * np.cos(2 * np.pi * x2) + np.random.normal(0, 0.8, size_set)
size_via_points = 2
x2_ = np.random.uniform(0, 0.5, size_via_points)
y2_ = 2 * np.sin(0.5 * np.pi * x2_) + 3 * np.cos(2 * np.pi * x2_)
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

alpha_list = (np.tanh((test_x - 0.5) * 5) + 1.0) / 2
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

plt1.plot(test_x, mean_blended, c='grey', label='Blended GMP')
plt1.fill_between(test_x, mean_blended - 3 * var_blended, mean_blended + 3 * var_blended, color='grey', alpha=0.5)

plt1.legend(loc='upper right', frameon=False, handlelength=1, ncol=2, columnspacing=1)
plt1.tick_params(labelsize=font_size)


plt2 = plt.subplot2grid((8, 8), (4, 0), rowspan=4, colspan=4)
length = np.shape(test_x)[0]
plt2.plot(test_x, alpha_list[0, :], c='red', label='$\\alpha_1$', linewidth=linewidth)
plt2.plot(test_x, alpha_list[1, :], c='blue', label='$\\alpha_2$', linewidth=linewidth)
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
plt3.plot(test_x, mean_blended, c='grey', label='Blended GMP')
plt3.fill_between(test_x, mean_blended - 3 * var_blended, mean_blended + 3 * var_blended, color='grey', alpha=0.5)
plt3.legend(loc='upper right', frameon=False, handlelength=1, ncol=2, columnspacing=1)
plt3.tick_params(labelsize=font_size)

plt4 = plt.subplot2grid((8, 8), (4, 4), rowspan=4, colspan=4)
length = np.shape(test_x)[0]
plt4.plot(test_x, alpha_list[0, :], c='red', label='$\\alpha_1$', linewidth=linewidth)
plt4.plot(test_x, alpha_list[1, :], c='blue', label='$\\alpha_2$', linewidth=linewidth)
plt4.legend(loc='upper right', frameon=False, handlelength=1, ncol=2, columnspacing=1)
plt4.set_xlabel('(b): Merging case 2', fontsize=font_size)
plt4.tick_params(labelsize=font_size)

plt.show()