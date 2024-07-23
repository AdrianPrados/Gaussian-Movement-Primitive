from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pyLasaDataset as lasa
from ProGP import ProGpMp
import numpy as np
import time
from ObstacleAvoidance3D import *

import os
from scipy.io import loadmat
import pandas as pd


#! ---- Code for real dat laod with Panda ----
""" dt = 0.01
demostraciones = 5
data_dict = {}
values = []
lengths = []

for i in range(1, 6):
    file_name = f"/home/nox/Escritorio/motion_primitives/GaussianMotion/ExpCartesianIiwa/obstacle_{i}.csv"
    data = pd.read_csv(file_name)
    lengths.append(len(data['x'][::30]))

min_length = min(lengths)

for i in range(1, 6):
    file_name = f"/home/nox/Escritorio/motion_primitives/GaussianMotion/ExpCartesianIiwa/obstacle_{i}.csv"
    data = pd.read_csv(file_name)
    
    x_data = np.array(data['x'][::30])[:min_length]*10
    y_data = np.array(data['y'][::30])[:min_length]*10
    z_data = np.array(data['z'][::30])[:min_length]*10
    
    data_dict[f"obstacle_{i}"] = np.array([x_data, y_data, z_data])
    values.append(np.array([x_data, y_data, z_data]))


for key, value in data_dict.items():
    pos = value
    t = np.linspace(0, 6, min_length).reshape(1, min_length)
    X_ = t.T
    Y_ = pos.T
    print(Y_.shape)
    if key == "obstacle_1":
        X = X_
        Y = Y_
    else:
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_)) """

#!---- Code for RAIL datatset ----
def leer_archivos_mat(ruta_carpeta,number):
    data = {}
    for archivo in os.listdir(ruta_carpeta):
        if archivo.endswith(str(number)+'.mat'):
            # Extract name
            nombre_archivo, extension = os.path.splitext(archivo)
            try:
                clave = int(nombre_archivo)
            except ValueError:
                continue  # Ignore archives withou numeric number
            
            # Complete path
            ruta_archivo = os.path.join(ruta_carpeta, archivo)
            
            # Load data from .mat
            datos = loadmat(ruta_archivo)
            
            # Save dta in dict
            data[clave] = datos

    return data



# Ptah to folder with .mat
ruta_carpeta = '/home/nox/Escritorio/motion_primitives/GaussianMotion/RAIL/PUSHING'
number = 2
data = leer_archivos_mat(ruta_carpeta,number)

# Example of use: print the data from 1.mat
#print(data[1]['dataset']['pos'][0][0].T)  #data[1]['dataset']['pos'][numero de demostraciones][0].T

dt = 0.01
gap = 30

#*Loading all the data
demostraciones = 5
demos = data[number]['dataset']
for i in range(demostraciones):
    demo = demos[i]
    pos = demo['pos'][0].T[:, 0::gap]*10  # np.ndarray, shape: (3,1000/gap)
    vel = demo['vel'][0].T[:, 0::gap]  # np.ndarray, shape: (3,1000/gap)
    acc = demo['acc'][0].T[:, 0::gap]  # np.ndarray, shape: (3,1000/gap)
    t = demo['time'][0].T[:, 0::gap]  # np.ndarray, shape: (1,1000/gap)
    print ("Tiempo: ",type(t))
    X_ = t.T
    Y_ = pos.T
    if i == 0:
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
""" target_t = t[0][-1]
target_position=np.array([values[2][0][-1], values[2][1][-1], values[2][2][-1]])

via_point0_t = t[0][0]
via_point0_position= np.array([x_data[0], y_data[0], z_data[0]])

via_point1_t = t[0][20]
via_point1_position = np.array([values[3][0][20],values[3][1][20], values[3][2][20]]) """

#* Constructing the training set with the target and via points
#! Points for RAIL data
target_t = sum(demos[i]['time'][0].T[:, 0::gap][0, -1] for i in range(demostraciones)) / demostraciones
target_position = sum(demos[i]['pos'][0].T[:, 0::gap][:, -1]*10 for i in range(demostraciones)) / demostraciones
via_point0_t = sum(demos[i]['time'][0].T[:, 0::gap][0, 0] for i in range(demostraciones)) / demostraciones
via_point0_position = sum(demos[i]['pos'][0].T[:, 0::gap][:, 0]*10 for i in range(demostraciones)) / demostraciones
via_point1_t = sum(demos[i]['time'][0].T[:, 0::gap][0, demos[i]['pos'][0].T[:, 0::gap].shape[1] * 2 // 4] for i in range(demostraciones)) / demostraciones
via_point1_position = sum(demos[i]['pos'][0].T[:, 0::gap][:, demos[i]['pos'][0].T[:, 0::gap].shape[1] * 2 // 4]*10 for i in range(demostraciones)) / demostraciones
#Vias points
X_ = np.array([via_point0_t,via_point1_t, target_t]).reshape(-1, 1)
Y_ = np.array([via_point0_position,via_point1_position, target_position])


#time.sleep(1000)
# predicting for dim0   --> size=demos[0]['pos'][0].T[:, 0::gap].T.shape[0]
observation_noise = 1.0
gp_mp= ProGpMp(X, Y, X_, Y_,dim=3, demos=demostraciones,size = demos[0].pos[:, 0::gap].T.shape[0] , observation_noise=observation_noise)

gp_mp.BlendedGpMp(gp_mp.ProGP) #? If you use more than one GpMp is mandatory to use BlendedGpMp, input: list[]
test_x = np.arange(0.0, target_t, dt)
#test_x = np.arange(Y_[0,0],Y_[2,0],(1/pos0.size))
print(test_x)
print(len(test_x))

#alpha_list = (np.tanh((test_x - 0.5) * 5) + 1.0) / 2
#print(type(alpha_list))
alpha_list=np.ones(len(test_x))
alpha_list = np.vstack((alpha_list, alpha_list))
alpha_list = np.vstack((np.ones(np.shape(test_x)[0]), np.ones(np.shape(test_x)[0])))

# alpha_list = np.vstack((np.ones(np.shape(test_x)[0]), np.ones(np.shape(test_x)[0])))
mean_blended, var_blended = gp_mp.predict_BlendedPos(test_x.reshape(-1, 1))
print("Valores del path final")
print(var_blended[0].reshape(-1, 1))
print(var_blended[1].reshape(-1, 1))
print(var_blended[2].reshape(-1, 1))
#time.sleep(1000)
var_blended[0]=np.where(var_blended[0]<0,0,var_blended[0])
var_blended[1]=np.where(var_blended[1]<0,0,var_blended[1])
var_blended[2]=np.where(var_blended[2]<0,0,var_blended[2])


font_size = 25
fig = plt.figure(figsize=(16, 8), dpi=100)
plt.subplots_adjust(left=0.1, right=0.9, wspace=0.5, hspace=0.5, bottom=0.15, top=0.99)
#* Visualization in 3D

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(Y_[:, 0], Y_[:, 1], Y_[:, 2], s=600, c='blue', marker='x')
ax1.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=20, c='blue', marker='o', alpha=0.3)
ax1.plot(mean_blended[0], mean_blended[1], mean_blended[2], c='blue', linewidth=5, label='$ProGpMp$')

ax1.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
ax1.set_xlabel('$x$/mm', fontsize=font_size)
ax1.set_ylabel('$y$/mm', fontsize=font_size)
ax1.set_zlabel('$z$/mm', fontsize=font_size)

#* 2D visualization
ax2 = fig.add_subplot(322)
ax2.plot(test_x, mean_blended[0], c='red', linewidth=3, label='$x_{ProGP}$')
ax2.fill_between(test_x, mean_blended[0] - 5 * np.sqrt(var_blended[0]), mean_blended[0] + 5 * np.sqrt(var_blended[0]), color='red', alpha=0.3)
ax2.scatter(X_[:, 0], Y_[:, 0], s=200, c='red', marker='x')
ax2.scatter(X[:, 0], Y[:, 0], s=10, c='red', marker='o', alpha=0.3)
ax2.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
ax2.set_xlabel('(b)', fontsize=font_size)
ax2.set_ylabel('$x$/mm', fontsize=font_size)

ax3 = fig.add_subplot(324)
ax3.plot(test_x, mean_blended[1], c='blue', linewidth=3, label='$y_{ProGP}$')
ax3.fill_between(test_x, mean_blended[1] - 5 * np.sqrt(var_blended[1]), mean_blended[1] + 5 * np.sqrt(var_blended[1]), color='blue', alpha=0.3)
ax3.scatter(X_[:, 0], Y_[:, 1], s=200, c='blue', marker='x')
ax3.scatter(X[:, 0], Y[:, 1], s=10, c='blue', marker='o', alpha=0.3)
ax3.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
ax3.set_xlabel('(c)', fontsize=font_size)
ax3.set_ylabel('$y$/mm', fontsize=font_size)

ax4 = fig.add_subplot(326)
ax4.plot(test_x, mean_blended[2], c='green', linewidth=3, label='$z_{ProGP}$')
ax4.fill_between(test_x, mean_blended[2] - 5 * np.sqrt(var_blended[2]), mean_blended[2] + 5 * np.sqrt(var_blended[2]), color='green', alpha=0.3)
ax4.scatter(X_[:, 0], Y_[:, 2], s=200, c='green', marker='x')
ax4.scatter(X[:, 0], Y[:, 2], s=10, c='green', marker='o', alpha=0.3)
ax4.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
ax4.set_xlabel('(d)', fontsize=font_size)
ax4.set_ylabel('$z$/mm', fontsize=font_size)

plt.show()
old_mean = []
old_mean = mean_blended.copy()


#*Osbatcle avoidance
obstacles=[]
collision_points = []
save_points =[]
positions = []
collided = []
#! Example 1
obstacle=Obstacle(center=[60,5,5],radius=15,force=[0,0,0],threshold=5) #(110,-20,0)
obstacles.append(obstacle)
obstacle2=Obstacle(center=[60,-10,22],radius=5,force=[0,0,0],threshold=5) #(110,-20,0)
obstacles.append(obstacle2)
#! Example 2
""" obstacle=Obstacle(center=[61,51,5],radius=10,force=[0,0,0],threshold=1) #(110,-20,0)
obstacles.append(obstacle) """
""" obstacle2=Obstacle(center=[50,5,18],radius=6,force=[0,0,0],threshold=5) #(110,-20,0)
obstacles.append(obstacle2) """
""" obstacle2 = Obstacle(center=[150,20,0],radius=20,force=[0,0,0],threshold=5)
obstacles.append(obstacle2) """
""" obstacle3 = Obstacle(center=[160,-20,0],radius=30,force=[0,0,0],threshold=5)
obstacles.append(obstacle3) """
""" obstacle2 = Obstacle(center=[270,175,0],radius=40,force=[0,0,0],threshold=20)
obstacles.append(obstacle2)
obstacle3 = Obstacle(center=[240,50,0],radius=40,force=[0,0,0],threshold=20) """
#obstacles.append(obstacle3)
#! Real data
""" obstacle=Obstacle(center=[85.5,0,30],radius=10,force=[0,0,0],threshold=1) #(110,-20,0)
obstacles.append(obstacle) """


for i in range(len(mean_blended[0])):
    positions.append([mean_blended[0][i]*10,mean_blended[1][i]*10,mean_blended[2][i]*10])
print(positions)
for obs in obstacles:
    for point in positions:
        within_radius = obs.is_within_radius(point)
        if within_radius:
            save_points.append(point)
            inside= obs.is_inside_obstacle(point)
            if inside:
                collision_points.append(point)

#* Find where is located the first collision point and the last one and take all the points between them
ind = np.where(np.all(np.isin(positions,collision_points),axis=1))
ind_borders = np.where(np.all(np.isin(positions,save_points),axis=1))
print("Indice de colision:",ind)
if len(ind[0]) != 0:
    for i in range(ind[0][0],ind[0][-1],5):
        collided.append(positions[i])
    collided.insert(0,positions[ind_borders[0][0]-2])
    collided.append(positions[ind_borders[0][-1]+5])
    print("Cuantos tengo: ",collided)
    head = Node(np.array([positions[ind_borders[0][0] - 4][0], positions[ind_borders[0][0] - 4][1], 0]), None, None, True)
    #print("Collision points:",collision_points)
    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
    ax.set_xlabel('$x$', fontsize=font_size)
    ax.set_ylabel('$y$', fontsize=font_size)
    ax.set_zlabel('$z$', fontsize=font_size)
    for obs in obstacles:
        #obs.draw(color='purple')
        #obs.draw_thresh(color='pink',linestyle='--')
        obs.drawSphere(color='purple',alpha=0.5,ax=ax)
    for point in positions:
        ax.plot(point[0], point[1],point[2], 'bo',c='blue')
        #plt.plot(point[0], point[1], 'bo')
    for point in collision_points:
        ax.plot(point[0], point[1],point[2], 'ro',c='red')
        #plt.plot(point[0], point[1], 'ro')
    plt.title('Points inside obstacles')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()
    #head = Node(np.array(collided[0]), None, None, True)
    local_path = execute_ObsAv(collided,head,obstacles)
    mean_blended[0]=np.delete(mean_blended[0],np.arange(ind_borders[0][0]-50,ind_borders[0][-1]+50))
    mean_blended[1]=np.delete(mean_blended[1],np.arange(ind_borders[0][0]-50,ind_borders[0][-1]+50))
    mean_blended[2]=np.delete(mean_blended[2],np.arange(ind_borders[0][0]-50,ind_borders[0][-1]+50))

    x_val = [local_path[i][0] for i in range(len(local_path))]
    y_val = [local_path[i][1] for i in range(len(local_path))]
    z_val = [local_path[i][2] for i in range(len(local_path))]


    mean_blended[0]=np.insert(mean_blended[0],ind_borders[0][0]-50,x_val)
    mean_blended[1]=np.insert(mean_blended[1],ind_borders[0][0]-50,y_val)
    mean_blended[2]=np.insert(mean_blended[2],ind_borders[0][0]-50,z_val)

    #* Plotting the local path
    obstacles=[]
    #! Example1
    obstacle=Obstacle(center=[6,0.5,0.5],radius=1.5,force=[0,0,0],threshold=0.5) #(110,-20,0)
    obstacles.append(obstacle)
    obstacle2=Obstacle(center=[6,-1,2.2],radius=1.0,force=[0,0,0],threshold=0.5) #(110,-20,0)
    obstacles.append(obstacle2)
    #! Example2
    """ obstacle=Obstacle(center=[6,1.5,1.0],radius=0.8,force=[0,0,0],threshold=0.5) #(110,-20,0)
    obstacles.append(obstacle)
    obstacle2=Obstacle(center=[5,0.5,1.8],radius=0.8,force=[0,0,0],threshold=0.5) #(110,-20,0)
    obstacles.append(obstacle2) """
    #! Test real values
    """ obstacle=Obstacle(center=[8.55,0,3],radius=1,force=[0,0,0],threshold=0.1) #(110,-20,0)
    obstacles.append(obstacle) """


    fig3 = plt.figure(dpi=100)
    size = 30
    axf = fig3.add_subplot(projection='3d')
    axf.scatter(Y_[:, 0], Y_[:, 1], Y_[:, 2], s=200, c='blue', marker='x')
    axf.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=10, c='blue', marker='o', alpha=0.3)

    
    plt.plot(mean_blended[0], mean_blended[1],mean_blended[2], c='blue', linewidth=6, label='$Avoided GMP$')
    plt.plot(old_mean[0], old_mean[1],old_mean[2], c='red', linewidth=6, label='$Not avoided GMP$')
    for obs in obstacles:
        obs.drawSphere(color='purple',alpha=0.5,ax=axf)
    plt.show()
    
    #! Save data in .csv (if nedeed)
    timeData = np.linspace(0, 6, 405)
    print(timeData.shape)
    print(t.T.reshape(-1).shape)
    print(mean_blended[0].shape)
    data_out = {
    'x': mean_blended[0]/10,
    'y': mean_blended[1]/10,
    'z': mean_blended[2]/10,
    'time': timeData
    }
    df = pd.DataFrame(data_out)

    # Save dataFrame
    output_file = '/home/nox/Escritorio/motion_primitives/GaussianMotion/ExpCartesianIiwa/Collisionoutput.csv'
    df.to_csv(output_file, index=False)
    
    
else:
    print("Without collision :)") 
