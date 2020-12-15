import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import csv
from scipy.spatial import cKDTree
from matplotlib.animation import FuncAnimation

# parameter
N_SAMPLE = 1000  # number of sample_points
N_KNN = 10  # number of edge from one sampled point
MAX_EDGE_LEN = 30.0  # [m] Maximum edge length

show_animation = True

class PRM:

    class Node:
       
        def __init__(self, x, y, cost, parent_index):
            self.x = x
            self.y = y
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return (
                str(self.x)
                + ","
                + str(self.y)
                + ","
                + str(self.cost)
                + ","
                + str(self.parent_index)
            )

    def __init__(self,
                 start,
                 goal,
                 x_obstacle,
                 y_obstacle,
                 robot_radius,
                 grid_size
                ):
        self.robot_radius = robot_radius
        self.grid_size=grid_size
        self.create_obstacle_map(x_obstacle, y_obstacle)
        # self.start = self.Node(self.calc_xy(start[0], self.x_min), self.calc_xy(start[1], self.y_min),0.0,-1)
        # self.end = self.Node(self.calc_xy(goal[0], self.x_min), self.calc_xy(goal[1], self.y_min),0.0,-1)
        self.start = self.Node(start[0], start[1],0.0,-1)
        self.end = self.Node(goal[0], goal[1],0.0,-1)
        self.x_obstacle = x_obstacle
        self.y_obstacle = y_obstacle

    @staticmethod
    def calc_distance_and_angle(sx, sy, gx, gy):
        dx = gx - sx
        dy = gy - sy
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    def calc_angle(self,x_out,y_out):
        angle=[]
        for i in range(len(x_out)-1):
            if x_out[i+1]==x_out[i] and y_out[i+1] < y_out[i]:
                angle.append(0)  # top
            if x_out[i+1]==x_out[i] and y_out[i+1] > y_out[i]:
                angle.append(np.pi) # down

            if x_out[i+1] < x_out[i] and y_out[i+1] == y_out[i]:
                angle.append(-np.pi/2) #right
            if x_out[i+1] > x_out[i] and y_out[i+1] == y_out[i]:
                angle.append(np.pi/2) #left

            if x_out[i+1]>x_out[i] and y_out[i+1]<y_out[i]:
                angle.append(np.pi/4) #leftdown
            if x_out[i+1]<x_out[i] and y_out[i+1]<y_out[i]:
                angle.append(-np.pi/4) #rightdown
            if x_out[i+1]>x_out[i] and y_out[i+1]>y_out[i]:
                angle.append(np.pi*2/3) #leftup
            if x_out[i+1]<x_out[i] and y_out[i+1]>y_out[i]:
                angle.append(-np.pi*2/3) #rightdown
        return angle

    def calc_xy(self, position, min_position):
        return round((position-min_position) / self.grid_size)

    def calc_grid_idx(self, node):
        return (node.y-self.y_min) * self.x_width + (node.x-self.x_min)

    def calc_grid_position(self, idx, p):
        return idx * self.grid_size + p

    def create_obstacle_map(self, x_obstacle, y_obstacle):
        self.x_min = round(min(x_obstacle))
        self.y_min = round(min(y_obstacle))
        self.x_max = round(max(x_obstacle))
        self.y_max = round(max(y_obstacle))
        self.x_width = round((self.x_max - self.x_min) / self.grid_size)
        self.y_width = round((self.y_max - self.y_min) / self.grid_size)
        self.obstacle_pos = [[False for i in range(self.y_width)] for i in range(self.x_width)]
        for idx_x in range(self.x_width):
            x = self.calc_grid_position(idx_x, self.x_min)
            for idx_y in range(self.y_width):
                y = self.calc_grid_position(idx_y, self.y_min)
                for idx_x_obstacle, idx_y_obstacle in zip(x_obstacle, y_obstacle):
                    d = math.sqrt((idx_x_obstacle - x)**2 + (idx_y_obstacle - y)**2)
                    if d <= self.robot_radius:
                        self.obstacle_pos[idx_x][idx_y] = True
                        break

    def prm_planning(self):

        obstacle_kd_tree = cKDTree(np.vstack((self.x_obstacle, self.y_obstacle)).T)

        sample_x, sample_y = self.sample_points(obstacle_kd_tree)
        if show_animation:
            plt.plot(sample_x, sample_y, ".b")

        road_map = self.generate_road_map(sample_x, sample_y, obstacle_kd_tree)

        rx, ry, visited_point_x, visited_point_y = self.dijkstra_planning(road_map, sample_x, sample_y)
        angle =self.calc_angle(rx, ry)
        # return (rx, ry, visited_point_x, visited_point_y,angle,sample_x,sample_y)
        return (rx, ry, visited_point_x, visited_point_y,angle)


    def is_collision(self, sx, sy, gx, gy, obstacle_kd_tree):
        x = sx
        y = sy
        d, yaw = self.calc_distance_and_angle(sx, sy, gx, gy)

        if d >= MAX_EDGE_LEN:
            return True

        D = self.robot_radius
        n_step = round(d / D)

        for _ in range(n_step):
            dist, _ = obstacle_kd_tree.query([x, y])
            if dist <= self.robot_radius:
                return True  # collision
            x += D * math.cos(yaw)
            y += D * math.sin(yaw)

        # goal point check
        dist, _ = obstacle_kd_tree.query([gx, gy])
        if dist <= self.robot_radius:
            return True  # collision

        return False  # OK


    def generate_road_map(self, sample_x, sample_y, obstacle_kd_tree):

        road_map = []
        n_sample = len(sample_x)
        sample_kd_tree = cKDTree(np.vstack((sample_x, sample_y)).T)

        for (_, ix, iy) in zip(range(n_sample), sample_x, sample_y):

            _, indexes = sample_kd_tree.query([ix, iy], k=n_sample)
            edge_id = []

            for i in range(1, len(indexes)):
                nx = sample_x[indexes[i]]
                ny = sample_y[indexes[i]]

                if not self.is_collision(ix, iy, nx, ny, obstacle_kd_tree):
                    edge_id.append(indexes[i])

                if len(edge_id) >= N_KNN:
                    break

            road_map.append(edge_id)

        # plot_road_map(road_map, sample_x, sample_y)

        return road_map


    def dijkstra_planning(self, road_map, sample_x, sample_y):

        start_node = self.start
        goal_node = self.end

        open_set, closed_set = dict(), dict()
        open_set[len(road_map) - 2] = start_node

        path_found = True

        visited_point_x = []
        visited_point_y = []

        while True:
            if not open_set:
                print("Cannot find path")
                path_found = False
                break

            c_id = min(open_set, key=lambda o: open_set[o].cost)
            current = open_set[c_id]

            visited_point_x.append(current.x)
            visited_point_y.append(current.y)

            if c_id == (len(road_map) - 1):
                print("goal is found!")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]
            # Add it to the closed set
            closed_set[c_id] = current

            # expand search grid based on motion model
            for i in range(len(road_map[c_id])):
                n_id = road_map[c_id][i]
                dx = sample_x[n_id] - current.x
                dy = sample_y[n_id] - current.y
                d = math.hypot(dx, dy)
                node = self.Node(sample_x[n_id], sample_y[n_id], current.cost + d, c_id)

                if n_id in closed_set:
                    continue
                # Otherwise if it is already in the open set
                if n_id in open_set:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id].cost = node.cost
                        open_set[n_id].parent_index = c_id
                else:
                    open_set[n_id] = node

        if path_found is False:
            return [], [], [], []

        # generate final course
        rx, ry = [goal_node.x], [goal_node.y]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(n.x)
            ry.append(n.y)
            parent_index = n.parent_index

        return (rx, ry, visited_point_x, visited_point_y)


    def sample_points(self, obstacle_kd_tree):
        sample_x, sample_y = [], []

        while len(sample_x) <= N_SAMPLE:
            tx = (random.random() * (self.x_max - self.x_min)) + self.x_min
            ty = (random.random() * (self.y_max - self.y_min)) + self.y_min

            dist, _ = obstacle_kd_tree.query([tx, ty])

            if dist >= self.robot_radius:
                sample_x.append(tx)
                sample_y.append(ty)

        sample_x.append(self.start.x)
        sample_y.append(self.start.y)
        sample_x.append(self.end.x)
        sample_y.append(self.end.y)

        return sample_x, sample_y


def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = 20#25.0  # [m]
    sy = 175#174.0  # [m]
    gx = 166#180.0  # [m]
    gy = 18#34.0  # [m]
    robot_size = 3.0  # [m]

    grid_size = 2.0
    robot_radius = 3.0

    image = cv2.imread("maze2.jpg")
    image = cv2.resize(image, (200, 200))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    (width, length) = gray.shape
    x_obstacle, y_obstacle = [], []
    for i in range(width):
        for j in range(length):
            if gray[i][j] <= 150:
                y_obstacle.append(i)
                x_obstacle.append(j)

    fig = plt.figure()
    plt.plot(x_obstacle, y_obstacle, ".k")
    plt.plot(sx, sy, "^r")
    plt.plot(gx, gy, "^c")
    plt.grid(True)
    plt.axis("equal")

    prm = PRM([sx,sy],[gx,gy],x_obstacle,y_obstacle,robot_radius,grid_size)
    # rx, ry = prm_planning(sx, sy, gx, gy, x_obstacle, y_obstacle, robot_size)
    rx, ry, visited_point_x, visited_point_y,angle = prm.prm_planning()

    path=list(zip(rx,ry))
    # with open('prm_maze2.csv','w+') as csvfile:
    #     filewriter= csv.writer(csvfile)
    #     filewriter.writerows(path)
    patch = patches.Rectangle((sx, sy), robot_radius, 7, 0, fc='m')

    rx.reverse()
    ry.reverse()
    angle.reverse()

    def init():
        # graph, = plt.plot([], [], 'og')

        plt.gca().add_patch(patch)
        return patch

    def cal_center_robot(x_out, y_out, angle):
        if angle == -np.pi/2 or angle == np.pi or angle == -np.pi*2/3 or angle == -np.pi/4:
            return x_out + robot_radius / 2, y_out + robot_radius /2
        if angle == np.pi/2 or angle == 0 or angle == np.pi*2/3 or angle == np.pi/4:
            return x_out - robot_radius / 2, y_out - robot_radius /2

    def animate(i):
        # graph.set_data(x_out_path[i], y_out_path[i])
        #patch.set_xy([x_out_path[i]+2, y_out_path[i]+2])
        x,y =cal_center_robot(rx[i],ry[i],angle[i])
        patch.set_xy([x,y])
        patch.angle = np.rad2deg(angle[i])
        # plt.plot(x_out_path[i], y_out_path[i], "*", color='yellow')
        plt.plot(rx[i], ry[i], "o", color='red')
        return patch

    plt.plot(rx, ry, "-", color='red')
    # ani = FuncAnimation(fig, animate,  init_func=init, frames=250, interval=40)
    plt.plot(visited_point_x, visited_point_y, ".y")
    plt.show()


if __name__ == "__main__":
    main()
