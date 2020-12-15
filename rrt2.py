import cv2
import math
import random
import pdb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.animation import FuncAnimation
show_animation = True


class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None
            # self.path = path

    def __init__(self,
                 start,
                 goal,
                 x_obstacle,
                 y_obstacle,
                 expand_dis=4,
                 path_resolution=2,
                 goal_sample_rate=5,
                 max_iter=500):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        """
        self.robot_radius = expand_dis
        self.grid_size = path_resolution
        self.create_obstacle_map(x_obstacle, y_obstacle)
        self.start = self.Node(self.calc_xy(start[0], self.x_min), self.calc_xy(start[1], self.y_min))
        self.end = self.Node(self.calc_xy(goal[0], self.x_min), self.calc_xy(goal[1], self.y_min))
        self.min_rand_x = self.x_min
        self.max_rand_x = self.x_max
        self.min_rand_y = self.y_min
        self.max_rand_y = self.y_max
        self.expand_dis = expand_dis
        self.close = 1
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        # self.obstacle_list = obstacle_list
        self.node_list = []


    def planning(self, animation=True):
        """
        rrt path planning
        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        self.list_of_angle = [0]
        self.visited_nodes = []
        # for _ in range(self.max_iter):
        while True:
            rnd_node = self.get_random_node()
            self.visited_nodes.append([rnd_node.x,rnd_node.y])
            # pdb.set_trace()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node, angle = self.steer(nearest_node, rnd_node, self.expand_dis)

            # if self.check_collision(new_node, self.obstacle_list):
            #     self.node_list.append(new_node)
            if self.check_validity(new_node):
                self.node_list.append(new_node)
                self.list_of_angle.append(angle)

            # print("node_list",self.node_list[1].x,self.node_list[1].y)
            print("len list",len(self.node_list))

            # tmp = self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y)
            # print("tmp",tmp)
            # print("node z calc", self.node_list[-1].x, self.node_list[-1].y)
            if self.calc_dist_to_goal(self.node_list[-1].x,
                                      self.node_list[-1].y) <= self.close:
                final_node, _ = self.steer(self.node_list[-1], self.end,
                                        self.expand_dis)
                # print("final node", final_node.x, final_node.y)
                if self.check_validity(final_node):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None  # cannot find path

    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)
        # print(n_expand)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, angle = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node
        # print("new node",new_node.x,new_node.y)
        new_node.x,new_node.y = round(new_node.x),round(new_node.y)
        # print("new node round",new_node.x,new_node.y)
        return new_node, angle

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        path_angles = [[0]]
        node = self.node_list[goal_ind]
        # print("node_list",node)
        # print("len list generate_final_course",len(self.node_list))
        while node.parent is not None:
            path.append([self.calc_grid_position(node.x,self.x_min), self.calc_grid_position(node.y,self.y_min)])
            # path.append([node.x, node.y])
            path_angles.append(self.list_of_angle[self.node_list.index(node)])
            node = node.parent
        # path.append([node.x, node.y])
        path.append([self.calc_grid_position(node.x,self.x_min), self.calc_grid_position(node.y,self.y_min)])
        path_angles.append(self.list_of_angle[self.node_list.index(node)])

        # print("generate path",path)
        return path, path_angles, self.visited_nodes

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                round(random.uniform(self.min_rand_x, self.max_rand_x)),
                round(random.uniform(self.min_rand_y, self.max_rand_y)))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd


    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind


    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta


    def calc_grid_position(self, idx, p):
        return idx * self.grid_size + p

    def calc_xy(self, position, min_position):
        return round((position-min_position) / self.grid_size)


    def calc_grid_idx(self, node):
        return (node.y-self.y_min) * self.x_width + (node.x-self.x_min)


    def check_validity(self, node):
        x_position = self.calc_grid_position(node.x, self.x_min)
        y_position = self.calc_grid_position(node.y, self.y_min)
        # print("nody", node.x, node.y)
        # print("position", x_position,y_position)
        if x_position < self.x_min:
            return False
        elif y_position < self.y_min:
            return False
        elif x_position >= self.x_max:
            return False
        elif y_position >= self.y_max:
            return False
        if self.obstacle_pos[node.x][node.y]:
            return False
        return True

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


def main():
    print("start " + __file__)

    image = cv2.imread('maze3.jpg')
    image = cv2.resize(image, (200,200))
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    (width, length) = gray.shape
    x_obstacle, y_obstacle = [], []

    for i in range(width):
        for j in range(length):
            if gray[i][j] <= 150:
              y_obstacle.append(i)
              x_obstacle.append(j)

    # ====Search Path with RRT====
    # obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
    #                 (9, 5, 2), (8, 10, 1)]  # [x, y, radius]
    # Set Initial parameters
    # start=[20, 175]
    # end = [166, 18]
    start=[20, 180]
    end = [172, 18]
    robot_radius=3
    rrt = RRT(
        # start=[0, 0],
        start=start,
        # goal=[gx, gy],
        goal=end,
        x_obstacle=x_obstacle,
        y_obstacle=y_obstacle)
    # pdb.set_trace()
    path, angles, visited_nodes = rrt.planning(animation=show_animation)
    # print(len(path),len(angles))
    path_x = [x for (x, y) in path]
    path_y = [y for (x, y) in path]
    visited_x = [x for (x, y) in visited_nodes]
    visited_y = [y for (x, y) in visited_nodes]
    print("len", len(visited_x))
    import csv

    
    path = path[1:]
    path=list(zip(path_x,path_y))
    visited_nodes=list(zip(visited_x,visited_y))
    with open('rrt3path.csv','w+') as csvfile:
        filewriter= csv.writer(csvfile)
        filewriter.writerows(path)

    with open('rrt3node.csv','w+') as csvfile:
        filewriter= csv.writer(csvfile)
        filewriter.writerows(visited_nodes)

    fig = plt.figure(figsize=(12,12))
    # sample, = plt.plot(samplex, sampley, ".b")
    visited, = plt.plot(visited_x, visited_y, ".y")
    obstacle, = plt.plot(x_obstacle, y_obstacle, ".k")
    start, = plt.plot(start[0], start[1], "og")
    end, =plt.plot(end[0], end[1], "xb")
    path, = plt.plot(path_x,path_y,"-r")
    plt.grid(True)
    plt.axis("equal")
    plt.title("Ścieżka wygenerowana dla algorytmu RRT.")
    plt.legend([start,end,path,visited],['pozycja startowa','pozycja docelowa','wygenerowana ścieżka','przeszukane punkty'],loc='upper right')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('RRT_maze3.jpg')
    plt.show()
    # angle = rrt.calc_angle(path_x,path_y)
    # fig=plt.figure()
    # plt.plot(visited_x, visited_y, '.g')
    # plt.plot(x_obstacle, y_obstacle, ".k")
    # plt.plot(start[0], start[1], "og")
    # plt.plot(end[0], end[1], "xb")
    # plt.grid(True)
    # plt.axis("equal")


    # # print("path", path)
    # path = path[1:]
    # # plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
    # # plt.show()

    # patch = patches.Rectangle((start[0], start[1]), robot_radius, 7, 0, fc='m')
    # path_x.reverse()
    # path_y.reverse()
    # # angle.reverse()
    # angles.reverse()
    # def init():
    #     # graph, = plt.plot([], [], 'og')

    #     plt.gca().add_patch(patch)
    #     return patch

    # def cal_center_robot(x_out, y_out, angle):
    #     if angle == -np.pi/2 or angle == np.pi or angle == -np.pi*2/3 or angle == -np.pi/4:
    #         return x_out + robot_radius / 2, y_out + robot_radius /2
    #     if angle == np.pi/2 or angle == 0 or angle == np.pi*2/3 or angle == np.pi/4:
    #         return x_out - robot_radius / 2, y_out - robot_radius /2

    # def animate(i):
    #     # graph.set_data(x_out_path[i], y_out_path[i])
    #     #patch.set_xy([x_out_path[i]+2, y_out_path[i]+2])
    #     x,y =cal_center_robot(path_x[i],path_y[i],angles[i])
    #     patch.set_xy([x,y])
    #     # patch.angle = np.rad2deg(angle[i])
    #     patch.angle = angles[i]
    #     # plt.plot(x_out_path[i], y_out_path[i], "*", color='yellow')
    #     plt.plot(path_x[i], path_y[i], "o", color='red')
    #     return patch

    # # ani = FuncAnimation(fig, animate,  init_func=init, frames=250, interval=40)
    # # plt.plot(visited_point_x, visited_point_y, ".y")
    # # plt.show()

    # # if path is None:
    # #     print("Cannot find path")
    # # else:
    # #     print("found path!!")
    # #     plt.plot(x_obstacle, y_obstacle, ".k")
    # #     plt.show()

    #     # Draw final path
    # if show_animation:
    # #     rrt.draw_graph()
    #     plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
        

    #     plt.grid(True)
    #     plt.pause(0.01)  # Need for Mac
    #     plt.show()


if __name__ == '__main__':
    main()
