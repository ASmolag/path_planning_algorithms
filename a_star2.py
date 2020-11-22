import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
show_animation = False
import pdb
class AStarPath:
    def __init__(self, robot_radius, grid_size, x_obstacle, y_obstacle):
        self.grid_size = grid_size
        self.robot_radius = robot_radius
        self.create_obstacle_map(x_obstacle, y_obstacle)
        self.path = self.get_path()

    class Node:
        def __init__(self, x, y, cost, path):
            self.x = x
            self.y = y
            self.cost = cost
            self.path = path
        
        def __str__(self):
            return str(self.x)+'+'+str(self.y)+','+str(self.cost)+','+str(self.path)

        # def __repr__(self):
        #     return str(self.x)+'+'+str(self.y)+','+str(self.cost)+','+str(self.path)

    
    @staticmethod
    def calc_heuristic(num1, num2):
        weight = 1.0
        return weight * math.sqrt((num1.x-num2.x)**2 + (num1.y-num2.y)**2)

    def calc_grid_position(self, idx, p):
        return idx * self.grid_size + p

    def calc_xy(self, position, min_position):
        return round((position-min_position) / self.grid_size)
    
    def calc_grid_idx(self, node):
        return (node.y-self.y_min) * self.x_width + (node.x-self.x_min)
    
    def check_validity(self, node):
        x_position = self.calc_grid_position(node.x, self.x_min)
        y_position = self.calc_grid_position(node.y, self.y_min)

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
    
    @staticmethod
    def get_path():
        path = [[1, 0, 1],
                [0, 1, 1],
                [-1, 0, 1],
                [0, -1, 1],
                [-1, -1, math.sqrt(2)],
                [-1, 1, math.sqrt(2)],
                [1, -1, math.sqrt(2)],
                [1, 1, math.sqrt(2)]]

        return path

    def calc_final_path(self, end_node, record_closed):
        x_out_path, y_out_path = [self.calc_grid_position(end_node.x, self.x_min)], [self.calc_grid_position(end_node.y, self.y_min)]
        path = end_node.path
        while path != -1:
            n = record_closed[path]
            x_out_path.append(self.calc_grid_position(n.x, self.x_min))
            y_out_path.append(self.calc_grid_position(n.y, self.y_min))
            path = n.path

        return x_out_path, y_out_path
    
    def cal_angle(self, x_out, y_out):
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

    def a_star_search(self, start_x, start_y, end_x, end_y):
        # pdb.set_trace()
        start_node = self.Node(self.calc_xy(start_x, self.x_min), self.calc_xy(start_y, self.y_min), 0.0, -1)
        end_node = self.Node(self.calc_xy(end_x, self.x_min), self.calc_xy(end_y, self.y_min), 0.0, -1)

        record_open, record_closed = dict(), dict()
        record_open[self.calc_grid_idx(start_node)] = start_node

        while True:
            if len(record_open) == 0:
                print('Check Record Validity')
                break

            total_cost = min(record_open, key=lambda x: record_open[x].cost + self.calc_heuristic(end_node, record_open[x]))
            cost_collection = record_open[total_cost]
        
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(cost_collection.x, self.x_min),  self.calc_grid_position(cost_collection.y, self.y_min), "xy")
                if len(record_closed.keys())%10 == 0:
                    plt.pause(0.001)
            
            if cost_collection.x == end_node.x and cost_collection.y == end_node.y:
                print("Finished!")
                end_node.path = cost_collection.path
                end_node.cost = cost_collection.cost
                break

            del record_open[total_cost]
            record_closed[total_cost] = cost_collection

            for i, _ in enumerate(self.path):
                node = self.Node(cost_collection.x + self.path[i][0], cost_collection.y + self.path[i][1], cost_collection.cost + self.path[i][2], total_cost)
                idx_node = self.calc_grid_idx(node)

                if not self.check_validity(node):
                    continue

                if idx_node in record_closed:
                    continue

                if idx_node not in record_open:
                    record_open[idx_node] = node
                else:
                    if record_open[idx_node].cost > node.cost:
                        record_open[idx_node] = node
        #pdb.set_trace()
        x_out_path, y_out_path = self.calc_final_path(end_node, record_closed)
        visited_points_x = [self.calc_grid_position(v.x,self.x_min) for v in record_closed.values()]
        visited_points_y = [self.calc_grid_position(v.y, self.y_min) for v in record_closed.values()]

        angle=self.cal_angle(x_out_path,y_out_path)
        return x_out_path, y_out_path, visited_points_x, visited_points_y, angle


def main():
    from matplotlib.animation import FuncAnimation
    start_x = 20#15
    start_y = 180#10
    end_x = 170 #55
    end_y = 17#175
    grid_size = 2.0
    robot_radius = 3.0
   # pdb.set_trace()
    image = cv2.imread('maze3.jpg')
    image = cv2.resize(image, (200,200))
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    (width, length) = gray.shape
    x_obstacle, y_obstacle = [], []
    for x in range(50):
        x_obstacle.append(x)
        y_obstacle.append(190)
    for i in range(width):
        for j in range(length):
            if gray[i][j] <= 150:
              y_obstacle.append(i)
              x_obstacle.append(j)
    # pdb.set_trace()
    # if show_animation:
    fig = plt.figure()

    plt.plot(x_obstacle, y_obstacle, ".k")
    plt.plot(start_x, start_y, "og")
    plt.plot(end_x, end_y, "xb")
    plt.grid(True)
    plt.axis("equal")
    # pdb.set_trace()
    a_star = AStarPath(robot_radius, grid_size, x_obstacle, y_obstacle)
    x_out_path, y_out_path, visited_points_x, visited_points_y, angle = a_star.a_star_search(start_x, start_y, end_x, end_y)
    # print(record_closed)
    #pdb.set_trace()
    # visited_points_x = [v.x for v in record_closed.values()]
    # visited_points_y = [v.y for v in record_closed.values()]

    # if show_animation:
    x_out_path.reverse()
    y_out_path.reverse()
    angle.reverse()
    
    # plt.plot(x_out_path, y_out_path, "r")

    # graph = plt.plot([], [], 'og')
    # ax = fig.add_subplot(111)
    patch = patches.Rectangle((start_x, start_y), robot_radius, 7, 0, fc='m')

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
        x,y =cal_center_robot(x_out_path[i],y_out_path[i],angle[i])
        patch.set_xy([x,y])
        patch.angle = np.rad2deg(angle[i])
        # plt.plot(x_out_path[i], y_out_path[i], "*", color='yellow')
        plt.plot(x_out_path[i], y_out_path[i], "o", color='red')
        return patch

    ani = FuncAnimation(fig, animate,  init_func=init, frames=250, interval=50)
    plt.plot(visited_points_x, visited_points_y, ".y")
    plt.show()


if __name__ == '__main__':
    main()