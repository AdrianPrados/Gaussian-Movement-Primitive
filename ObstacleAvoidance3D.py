import numpy as np
import pdb
import time
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, BSpline
import bezier


def normalize(vector):
    return vector / np.linalg.norm(vector)


def execute_ObsAv(positions,head,obstacles):
    node = head
    flag=[False]*len(positions)
    path = [(0,0,0)]*len(positions)
    exit_node=False

    for pos in positions:
        node.nxt = Node(np.array(pos), node, None)
        node = node.nxt

    node.locked = True #!Block last node

    node = head
    fig=plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(projection='3d')
    
    node = node.nxt
    init= time.time()
    while exit_node==False:
        #plt.gca().set_aspect('equal', adjustable='box')
        #plt.gca().add_patch(plt.Circle((150, 130), 40, color='purple', fill=False))
        """ for obs in obstacles:
            obs.draw(color='purple')
            obs.draw_thresh(color='pink',linestyle='--') """
        i=0
        valor_final = 0
        node = head
        while (node):
            """ if node.prv:
                plt.plot([node.prv.pos[0], node.pos[0]], [node.prv.pos[1], node.pos[1]], color='black') """
            if node.nxt:
                ax.plot([node.pos[0], node.nxt.pos[0]], [node.pos[1], node.nxt.pos[1]],[node.pos[2], node.nxt.pos[2]], color='red')

            forceVector = node.pos + node.calcForces(obstacles) * 100
            ax.plot([node.pos[0], forceVector[0]], [node.pos[1], forceVector[1]], [node.pos[2], forceVector[2]],color='green', linewidth=1)
            ax.plot(node.pos[0], node.pos[1],node.pos[2], marker='o', markersize=3, color='blue')
            node.applyForces(obstacles)
            add_points = node.smoothCurve(1000,node) #* Smaller values increase de smoothnes 5-> para 2D
            #add_points = False
            if add_points[0]!=False:
                
                flag_added= node.is_outside_obstacle(add_points[0],obstacles)
                path.insert(i,add_points[1])
                flag.insert(i,flag_added)
                i = i+1
                valor_final = i
                print("Valor final de datos: ",valor_final)
            else:
                flag_added = True

            flag_node= node.is_outside_obstacle(node,obstacles)
            if flag_node == True and flag_added == True:
                flag[i]=True
            else:
                flag[i]=False
            path[i]=node.pos
            node = node.nxt
            if i < len(path)-1:
                i=i+1
            else:
                i=0

        if all(flag)==False:
            plt.show(block=False)
            plt.pause(0.1)
            plt.clf()
        if all(flag)==True:
            print("C'est fini")
            print("Datos finales:",valor_final)
            end = time.time()
            print("Final time: ",end - init)
            exit_node=True
            """ print("Length Path puntos: ",len(path))
            print("Path puntos: ",path) """
            for i in range(1): #* Reordering n initial elements that have been blocked (for force 30, n=1)
                aux = path[0]
                path.remove(path[0])
                path.append(aux)
            #print("Posiciones: ",positions)
            for h, point in enumerate(path):
                ax.text(point[0], point[1],point[1], str(h), fontsize=8, color='black')
            ax.plot([point[0] for point in path], [point[1] for point in path],[point[2] for point in path], 'go')
            plt.show()
            path = [tuple(map(lambda x: x/10, point)) for point in path]
            path.remove(path[-1])
            break
    return path
class Obstacle:
    def __init__(self, center, radius,force,threshold):
        self.center_draw=center
        self.center = np.array(center)
        self.radius = radius
        self.ObsForce=np.array(force)
        self.threshold=threshold

    def drawSphere(self,color,alpha,ax):
        # Crear malla de puntos para la esfera
        phi, theta = np.mgrid[0.0:2.0 * np.pi:20j, 0.0:np.pi:10j]
        x = self.center_draw[0] + self.radius * np.sin(theta) * np.cos(phi)
        y = self.center_draw[1] + self.radius * np.sin(theta) * np.sin(phi)
        z = self.center_draw[2] + self.radius * np.cos(theta)

        # Dibujar la esfera
        ax.plot_surface(x, y, z, color=color, alpha=alpha)

    def draw(self, color,linestyle='-'):
        plt.gca().add_patch(plt.Circle((self.center_draw), self.radius, color=color, fill=False,linestyle=linestyle))

    def draw_thresh(self, color,linestyle='-'):
        plt.gca().add_patch(plt.Circle((self.center_draw), self.radius + self.threshold, color=color, fill=False,linestyle=linestyle))

    def calculate_distance(self, point):
        # Calcular la distancia euclidiana entre el punto y el centro del obstáculo
        distance = np.linalg.norm(np.array(point) - np.array(self.center))
        return distance

    def is_within_radius(self, point):
        # Verificar si el punto está dentro del radio del obstáculo considerando un umbral
        distance = self.calculate_distance(point)
        return distance <= (self.radius + self.threshold)
    def is_inside_obstacle(self, point):
        # Verificar si el punto está dentro del radio del obstáculo
        distance = self.calculate_distance(point)
        return distance <= self.radius

class Node:
    def __init__(self, pos = None, prv = None, nxt = None,
                locked = False, mass = 1.0, jointRigidity = 1.0):
        self.pos = pos          # X, Y, V numpy array
        self.prv = prv          # previous node
        self.nxt = nxt          # next node
        self.locked = locked    # if true, forces have no effect on this node
        self.mass = mass        # 1 / how much a force effects that node
        self.k = jointRigidity  # coefficient for the tension force
        self.dist = 0           # way to first node
        self.minDist = 1       # minimal distance between nodes
        if self.prv:
            self.dist = self.getDistance()

    def insertNodeProg(self, pos, locked = False, mass = 1.0, jointRigidity = 1.0):
        # insert new node after
        self.nxt = Node(pos, self, self.nxt, locked, self.mass, jointRigidity)
        self.nxt.nxt.prv = self.nxt
        return self.nxt,self.nxt.pos

    def insertNodeRetro(self, pos, locked = False, mass = 1.0, jointRigidity = 1.0):
        # insert new node after
        self.prv = Node(pos, self.prv, self, locked, self.mass, jointRigidity)
        self.prv.prv.nxt = self.prv
        return self.prv,self.prv.pos


    def calcForces(self,obstacles):
        totForce = np.array([0, 0, 0])
        if not self.locked:
            totForce = totForce + self.getTensionForce()
            totForce = totForce + self.getObstacleForce(obstacles=obstacles)
            #totForce = totForce + self.getDVForce(5)
            totForce = totForce + self.getVForce()
        return totForce


    def applyForces(self,obstacles):
        self.pos = self.pos + (1 / self.mass) * self.calcForces(obstacles)
        print("Posiciones: ",self.pos)
        
    def getTensionForce(self):
        pos = self.pos[0:3]
        tensionForce = np.array([0, 0, 0])

        if self.prv:
            prv = self.prv.pos[0:3]
            kPrv = 1 / (1 / self.k + 1 / self.prv.k)
            tensionForce = tensionForce + kPrv * (prv - pos) / \
                        np.linalg.norm((prv - pos))

        if self.nxt:
            nxt = self.nxt.pos[0:3]
            kNxt = 1 / (1 / self.k + 1 / self.nxt.k)
            tensionForce = tensionForce + kNxt * (nxt- pos) / \
                        np.linalg.norm((nxt- pos))

        return tensionForce

    def getObstacleForce(self,obstacles):
        total_force = obstacles[0].ObsForce * 0
        beta=0.0 #*for 3D it is better to use a value of 0.8 to 1.0
        for obstacle in obstacles:
            pos = self.pos[0:3]
            obstacleForce = obstacle.ObsForce
            obsPos = obstacle.center
            dist = np.linalg.norm(obsPos - pos)
            if dist < (obstacle.radius + obstacle.threshold):
                obstacleForce = (pos - obsPos) / (dist*dist)
            else:
                obstacleForce = (pos - obsPos) / (dist*dist) * beta
                #obstacleForce = obstacles[0].ObsForce * 0
                #obstacleForce = (pos - obsPos)*self.mass / (dist*dist)
            total_force = total_force + obstacleForce
        return np.append(total_force * 20, 0)[0:3] #* 15 is the force multiplier, that very sensitive to the force applied

    def getDVForce(self, aMax):
        force = np.array([0, 0, 0])
        accRetro = self.getAccelerationRetro()
        accProg = self.getAccelerationProg()
#       force[2] = force[2] + (accRetro + accProg)

        if accRetro > aMax:
            force[2] = force[2] + accRetro - aMax
        elif accRetro < -aMax:
            force[2] = force[2] + accRetro + aMax
        
        if accProg > aMax:
            force[2] = force[2] + accProg - aMax
        elif accProg < -aMax:
            force[2] = force[2] + accProg + aMax
        return force
        

#       force = np.array([0, 0, 0])
#       dVR = self.getDVDXYRetro()
#       dVP = self.getDVDXY()
#       if dVR > aMax:
#           force[2] = -dVR
#       elif dVP < -aMax:
#           force[2] = dVP
#       return force

    def smoothCurve(self, fPedMax, node):
        print("Curvature: ",self.getCurvature())
        if self.getCurvature() >= fPedMax:
            if np.linalg.norm(self.pos - self.nxt.pos) > self.minDist:
                nxt,nxt_pos = self.insertNodeProg(self.pos + (self.nxt.pos - self.pos) / 3)
                #print("Valores de retorno: ",nxt,nxt_pos)
                return [nxt,nxt_pos,1]
            elif np.linalg.norm(self.pos - self.prv.pos) > self.minDist:
                prv,prv_pos = self.insertNodeRetro(self.pos + (self.prv.pos - self.pos) / 3)
                return [prv,prv_pos,-1]
            else:
                return [node,self.pos,0]
            """ elif np.linalg.norm(self.pos - self.prv.pos) > self.minDist:
                prv,prv_pos = self.insertNodeRetro(self.pos + (self.prv.pos - self.pos) / 3)
                return [prv,prv_pos,-1] """
        else:
            return [False]


    def getXY(self):
        return (int(self.pos[0]), int(self.pos[1]))

    def getV(self):
        return int(self.pos[2])

    def getDVDXY(self):
        derivative = 0
        diff = np.array([0, 0, 0])

        if self.nxt:
            diff = self.nxt.pos - self.pos
            derivative = diff[2]
            diff[2] = 0
            derivative = derivative / np.linalg.norm(diff)
        return derivative

    def getDVDXYRetro(self):
        derivative = 0
        diff = np.array([0, 0, 0])

        if self.prv:
            diff = self.pos - self.prv.pos
            derivative = diff[2]
            diff[2] = 0
            derivative = derivative / np.linalg.norm(diff)
        return derivative

    def getAccelerationProg(self):
        acc = 0

        if self.nxt:
            diff = self.nxt.pos - self.pos
            diff[2] = 0
            acc = (self.nxt.pos[2]**2 - self.pos[2]**2) / np.linalg.norm(diff) / 2
        return acc

    def getAccelerationRetro(self):
        acc = 0

        if self.prv:
            diff = self.prv.pos - self.pos
            diff[2] = 0
            acc = (self.prv.pos[2]**2 - self.pos[2]**2) / np.linalg.norm(diff) / 2
        return acc


    def getCurvature(self):
        curvature = 0
        if self.nxt and self.prv:
            pos = self.pos[0:3]
            nxt = self.nxt.pos[0:3]
            prv = self.prv.pos[0:3]

            curvature = normalize(nxt - pos) - normalize(pos - prv) #* Optimización de curvatura
            #curvature = (nxt - pos) - (pos - prv)
            print("Curvature dentro de funcion: ",curvature)
            curvature = self.pos[2] * np.linalg.norm(curvature)
        return curvature

    def getDistance(self):
        pos = self.pos[0:3]
        prv = self.prv.pos[0:3]
        return np.linalg.norm(pos - prv) + self.prv.dist

    def getVForce(self):
        return np.array([0, 0, 0.1])
    def is_outside_obstacle(self,node,obstacles):
        i=0
        not_collision=[False]*len(obstacles)
        for obstacle in obstacles:
            obstacle_center = obstacle.center
            obstacle_radius = obstacle.radius + obstacle.threshold
            if np.linalg.norm(node.pos[0:3] - obstacle_center) <= obstacle_radius:
                not_collision[i]=False
            else:
                #node.locked=True
                not_collision[i]=True
            i=i+1
        if all(not_collision)==True:
            return True
        else:
            return False



if __name__ == "__main__":
    obstacles=[]
    collision_points = []
    head = Node(np.array([90, 90, 0]), None, None, True)
    obstacle=Obstacle(center=[150,130,0],radius=40,force=[0,0,0],threshold=20)
    obstacles.append(obstacle)
    obstacle2 = Obstacle(center=[270,175,0],radius=40,force=[0,0,0],threshold=20)
    obstacles.append(obstacle2)
    obstacle3 = Obstacle(center=[240,50,0],radius=40,force=[0,0,0],threshold=20)
    obstacles.append(obstacle3)

    positions = [[131, 100,   0],
                [150,  123,   0],
                [200, 150,   0],
                [210, 160,   0],
                [220, 170,   0],
                [250, 190,   0],
                [240, 230,   0]] #Initial path positions
    for obs in obstacles:
        for point in positions:
            within_radius = obs.is_within_radius(point)
            if within_radius:
                collision_points.append(point)

    print("Collision points:",collision_points)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.gca().add_patch(plt.Circle((150, 130), 40, color='purple', fill=False))
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
    execute_ObsAv(positions,head,obstacles)