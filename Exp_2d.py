import matplotlib.pyplot as plt
import pyLasaDataset as lasa
from ProGP import ProGpMp
import numpy as np
import time
from ObstacleAvoidance import *

np.random.seed(30)
font_size = 18
# using multi_models 3, or you will change the loading-data code
#data = lasa.DataSet.Multi_Models_1
data = lasa.DataSet.DoubleBendedLine
dt = data.dt
print(dt)
demos = data.demos
gap = 30
#lasa.utilities.plot_model(lasa.DataSet.BendedLine)
#! --------------------------------------------- Loading and training the model 1 data------------------------------------
#*Loading all the data
demostraciones = 7
#print(demos[0].pos)

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
#print("Prueba:",demos[0].pos[:, 0::gap].T.shape[0])

#* Selecting the target position and time and via points
""" target_t = 4.22313 #! Time in X axis
target_position = np.array([0.0, 0.0]) #! Target position in Y axis
via_point0_t = 0.0
via_point0_position = np.array([28.364, 0.1552])
via_point1_t = 2.1559
via_point1_position = np.array([12.9869, -3.32277]) """

via1t = 1.5
via2t=3.0
via3t=4.3

via1p = np.array([-30.0, 14.0])
via2p = np.array([-24.0, -3.0])
via3p = np.array([-6, 14.4])

#* Constructing the training set with the target and via points
target_t = sum(demos[i].t[:, 0::gap][0, -1] for i in range(demostraciones)) / demostraciones
target_position = sum(demos[i].pos[:, 0::gap][:, -1] for i in range(demostraciones)) / demostraciones
via_point0_t = sum(demos[i].t[:, 0::gap][0, 0] for i in range(demostraciones)) / demostraciones
via_point0_position = sum(demos[i].pos[:, 0::gap][:, 0] for i in range(demostraciones)) / demostraciones
via_point1_t = sum(demos[i].t[:, 0::gap][0, demos[i].pos[:, 0::gap].shape[1] * 2 // 4] for i in range(demostraciones)) / demostraciones
via_point1_position = sum(demos[i].pos[:, 0::gap][:, demos[i].pos[:, 0::gap].shape[1] * 2 // 4] for i in range(demostraciones)) / demostraciones
via_point2_t = sum(demos[i].t[:, 0::gap][0, demos[i].pos[:, 0::gap].shape[1] * 2 // 5] for i in range(demostraciones)) / demostraciones
via_point2_position = sum(demos[i].pos[:, 0::gap][:, demos[i].pos[:, 0::gap].shape[1] * 2 // 5] for i in range(demostraciones)) / demostraciones
#Vias points
X_ = np.array([via_point0_t, target_t]).reshape(-1, 1)
Y_ = np.array([via_point0_position, target_position])

print(X_.shape)
print(Y_.shape)
print(demos[0].pos[:, 0::gap].T.shape[0])
time.sleep(100)

#* Predicting for dim0
observation_noise = 1.0
gp_mp= ProGpMp(X, Y, X_, Y_,dim=2, demos=demostraciones, size=demos[0].pos[:, 0::gap].T.shape[0], observation_noise=observation_noise)

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
var_blended[0]=np.where(var_blended[0]<0,0,var_blended[0])
var_blended[1]=np.where(var_blended[1]<0,0,var_blended[1])
#time.sleep(1000)
#print("VAriables blended:",var_blended)


plt.figure(figsize=(16, 8), dpi=100)
plt.subplots_adjust(left=0.05, right=0.99, wspace=0.8, hspace=0.8, bottom=0.1, top=0.99)
plt1 = plt.subplot2grid((8, 16), (0, 0), rowspan=8, colspan=8)
size = 80
plt1.scatter(Y_[:, 0], Y_[:, 1], s=400, c='green', marker='x')
plt1.scatter(Y[:, 0], Y[:, 1], s=30, c='green', marker='o', alpha=0.3)
#plt1.plot(gp_mp1_predict_y_dim0, gp_mp1_predict_y_dim1, ls='-', c='blue', linewidth=2, label='$p_{gpmp1}$')

plt1.plot(mean_blended[0], mean_blended[1], c='black', linewidth=4, label='$GMP$')
#plt1.fill_between(mean_blended[0], mean_blended[1] - 5 * np.sqrt(abs(var_blended[0])), mean_blended[1] + 5 * np.sqrt(abs(var_blended[1])), color='blue', alpha=0.5)
#plt1.fill_between(X[:,0], Y[:,0] - 5 * np.sqrt(abs(X[:,0])), Y[:,0] + 5 * np.sqrt(abs(X[:,0])), color='red', alpha=0.5)

plt1.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
plt1.tick_params(labelsize=font_size)
plt1.set_xlabel('$x$/mm\n(a)', fontsize=font_size)
plt1.set_ylabel('$y$/mm', fontsize=font_size)

print("Var Blendes:",var_blended[0])
print("Var Blendes2:",var_blended[1])
plt2 = plt.subplot2grid((8, 16), (0, 9), rowspan=3, colspan=8)
plt2.plot(test_x, mean_blended[0], c='red', linewidth=3, label='$x_{GMP}$')
plt2.fill_between(test_x, mean_blended[0] - 5 * np.sqrt(var_blended[0]), mean_blended[0] + 5 * np.sqrt(var_blended[0]), color='red', alpha=0.3)
plt2.scatter(X_[:, 0], Y_[:, 0], s=400, c='red', marker='x')
plt2.scatter(X[:, 0], Y[:, 0], s=15, c='red', marker='o', alpha=0.3)
plt2.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
plt2.tick_params(labelsize=font_size)
plt2.set_xlabel('$time$(s)\n(b)', fontsize=font_size)
plt2.set_ylabel('$x$/mm', fontsize=font_size)

plt3 = plt.subplot2grid((8, 16), (4, 9), rowspan=3, colspan=8)
plt3.plot(test_x, mean_blended[1], c='blue', linewidth=3, label='$y_{GMP}$')
plt3.fill_between(test_x, mean_blended[1] - 5 * np.sqrt(var_blended[1]), mean_blended[1] + 5 * np.sqrt(var_blended[1]), color='blue', alpha=0.3)
plt3.scatter(X_[:, 0], Y_[:, 1], s=400, c='blue', marker='x')
plt3.scatter(X[:, 0], Y[:, 1], s=15, c='blue', marker='o', alpha=0.3)
plt3.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
plt3.tick_params(labelsize=font_size)
plt3.set_xlabel('$time$(s)\n(c)', fontsize=font_size)
plt3.set_ylabel('$y$/mm', fontsize=font_size)
plt.show()
old_mean = []
old_mean = mean_blended.copy()


#* Prueba de osbatcle avoidance
obstacles=[]
collision_points = []
save_points =[]
positions = []
collided = []
#! Example obstacle 1
""" obstacle=Obstacle(center=[130,-40,0],radius=25,force=[0,0,0],threshold=5) #(110,-20,0)
obstacles.append(obstacle)
obstacle2 = Obstacle(center=[150,20,0],radius=20,force=[0,0,0],threshold=5)
obstacles.append(obstacle2) """
#! Example obstacle 2
""" obstacle=Obstacle(center=[130,-40,0],radius=25,force=[0,0,0],threshold=5) #(110,-20,0)
obstacles.append(obstacle)
obstacle2 = Obstacle(center=[170,20,0],radius=20,force=[0,0,0],threshold=5)
obstacles.append(obstacle2) """
""" obstacle3 = Obstacle(center=[160,-20,0],radius=30,force=[0,0,0],threshold=5)
obstacles.append(obstacle3) """
""" obstacle2 = Obstacle(center=[270,175,0],radius=40,force=[0,0,0],threshold=20)
obstacles.append(obstacle2)
obstacle3 = Obstacle(center=[240,50,0],radius=40,force=[0,0,0],threshold=20) """
#obstacles.append(obstacle3)
#! Example for DoubleBendenLine
obstacle=Obstacle(center=[-190,-130,0],radius=25,force=[0,0,0],threshold=5) #(110,-20,0)
obstacles.append(obstacle)
obstacle2 = Obstacle(center=[-140,-160,0],radius=20,force=[0,0,0],threshold=5)
obstacles.append(obstacle2)
obstacle3 = Obstacle(center=[-47,-146,0],radius=25,force=[0,0,0],threshold=5)
obstacles.append(obstacle3)


for i in range(len(mean_blended[0])):
    positions.append([mean_blended[0][i]*10,mean_blended[1][i]*10,0])
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
    for i in range(ind[0][0],ind[0][-1],2):
        collided.append(positions[i])
    collided.insert(0,positions[ind_borders[0][0]-2])
    collided.append(positions[ind_borders[0][-1]+5])
    #print(ind_borders[0][-1]+2)
    print("Cuantos tengo: ",collided)
    head = Node(np.array([positions[ind_borders[0][0] - 4][0], positions[ind_borders[0][0] - 4][1], 0]), None, None, True)
    #print("Collision points:",collision_points)
    plt.xlabel('X')
    plt.ylabel('Y')
    for obs in obstacles:
        obs.draw(color='purple')
        obs.draw_thresh(color='pink',linestyle='--')
    for point in positions:
        plt.plot(point[0], point[1], 'bo')
    for point in collision_points:
        plt.plot(point[0], point[1], 'ro')
    plt.title('Points inside obstacles')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()
    #head = Node(np.array(collided[0]), None, None, True)
    local_path = execute_ObsAv(collided,head,obstacles)
    mean_blended[0]=np.delete(mean_blended[0],np.arange(ind_borders[0][0]-36,ind_borders[0][-1]+38)) #-6 +8
    mean_blended[1]=np.delete(mean_blended[1],np.arange(ind_borders[0][0]-36,ind_borders[0][-1]+38))

    x_val = [local_path[i][0] for i in range(len(local_path))]
    y_val = [local_path[i][1] for i in range(len(local_path))]
    z_val = [local_path[i][2] for i in range(len(local_path))]


    mean_blended[0]=np.insert(mean_blended[0],ind_borders[0][0]-36,x_val)
    mean_blended[1]=np.insert(mean_blended[1],ind_borders[0][0]-36,y_val)

    #* Plotting the local path
    obstacles=[]
    #! Case 1
    """ obstacle=Obstacle(center=[13,-4,0],radius=2.5,force=[0,0,0],threshold=0.5)
    obstacles.append(obstacle)
    obstacle2 = Obstacle(center=[15,2,0],radius=2,force=[0,0,0],threshold=0.5)
    obstacles.append(obstacle2) """

    #! Case 2
    """ obstacle2 = Obstacle(center=[17,2,0],radius=2,force=[0,0,0],threshold=0.5)
    obstacles.append(obstacle2) """
    #! Case 3
    obstacle=Obstacle(center=[-19,-13,0],radius=2.5,force=[0,0,0],threshold=0.5) #(110,-20,0)
    obstacles.append(obstacle)
    obstacle2 = Obstacle(center=[-14,-16,0],radius=2.0,force=[0,0,0],threshold=0.5)
    obstacles.append(obstacle2)
    obstacle3 = Obstacle(center=[-4.7,-14.6,0],radius=2.5,force=[0,0,0],threshold=0.5)
    obstacles.append(obstacle3)

    plt.figure(dpi=100)
    #plt.subplots_adjust(left=0.05, right=0.99, wspace=0.8, hspace=0.8, bottom=0.1, top=0.99)
    #plt3 = plt.subplot2grid((8, 16), (0, 0), rowspan=8, colspan=8)
    size = 30
    plt.scatter(Y_[:, 0], Y_[:, 1], s=400, c='blue', marker='x')
    plt.scatter(Y[:, 0], Y[:, 1], s=30, c='blue', marker='o', alpha=0.3)
    #plt1.plot(gp_mp1_predict_y_dim0, gp_mp1_predict_y_dim1, ls='-', c='blue', linewidth=2, label='$p_{gpmp1}$')
    plt.plot(mean_blended[0], mean_blended[1], c='blue', linewidth=2, label='$Avoided GMP$')
    plt.plot(old_mean[0], old_mean[1], c='red', linewidth=2, label='$Not avoided GMP$')
    for obs in obstacles:
        obs.draw(color='purple')
        obs.draw_thresh(color='pink',linestyle='--')
    #plt.fill_between(old_mean[0], old_mean[1] - 5 * np.sqrt(abs(var_blended[0])), old_mean[1] + 5 * np.sqrt(abs(var_blended[1])), color='blue', alpha=0.5)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()
else:
    print("Without collision :)")

