import cv2
import time
import functools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import math
from bezier import *

from a_star2 import AStarPath

# import rrt
# import prm


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start = time.perf_counter()
        value = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value

    return wrapper_timer


@timer
def read_obstacles_from_map(filename):
    x_obstacle, y_obstacle = [], []
    image = cv2.imread(filename)
    image = cv2.resize(image, (200, 200))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    (width, length) = gray.shape
    for x in range(50):
        x_obstacle.append(x)
        y_obstacle.append(190)
    for i in range(width):
        for j in range(length):
            if gray[i][j] <= 150:
                y_obstacle.append(i)
                x_obstacle.append(j)

    return x_obstacle, y_obstacle

def plot_function(nazwa,x_obstacle,y_obstacle,start_x,start_y,end_x,end_y,x_path,y_path, visited_x, visited_y):
    fig = plt.figure(figsize=(12,12))
    visited, = plt.plot(visited_x, visited_y, ".y")
    obstacle, = plt.plot(x_obstacle, y_obstacle, ".k")
    start, = plt.plot(start_x, start_y, "og")
    end, =plt.plot(end_x, end_y, "xb")
    path, = plt.plot(x_path,y_path,"-r")
    plt.grid(True)
    plt.axis("equal")
    plt.title(f"Ścieżka wygenerowana dla algorytmu {nazwa}.")
    plt.legend([start,end,path,visited],['pozycja startowa','pozycja docelowa','wygenerowana ścieżka','przeszukane punkty'],loc='upper right')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'{nazwa}_maze3.jpg')
    plt.show()
    
def plot_function_smooth(nazwa,nazwa2,x_obstacle,y_obstacle,path_smooth,points_smooth, original_path):
    x1, y1 = points_smooth[:,0], points_smooth[:,1]
    px, py = path_smooth[:,0], path_smooth[:,1]
    x_p = [x for x,y in original_path]
    y_p = [y for x,y in original_path]
    fig = plt.figure(figsize=(12,12))
    obstacle, = plt.plot(x_obstacle, y_obstacle, ".k")
    orginal_path, = plt.plot(x_p, y_p, "y-")
    bezier_path, =plt.plot(px, py, 'b-')
    bezier_points, = plt.plot(x1, y1, 'ro')
    orginal_points, = plt.plot(x_p,y_p, 'cx')
    plt.grid(True)
    plt.axis("equal")
    plt.title(f"Wygładzona ścieżka dla {nazwa} - {nazwa2}.")
    plt.legend([orginal_path,bezier_path,bezier_points,orginal_points],['ścieżka początkowa','ścieżka wygładzona','pozostałe punkty','początkowe punkty'],loc='upper right')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'{nazwa}_{nazwa2}_maze3.jpg')
    plt.show()

def animate_path(nazwa,start_x,start_y,end_x,end_y,x_obstacle,y_obstacle, x_out_path, y_out_path, angles, robot_radius):
    fig = plt.figure()
    plt.plot(x_obstacle, y_obstacle, ".k")
    start, = plt.plot(start_x, start_y, "og")
    end, =plt.plot(end_x, end_y, "xb")
    plt.grid(True)
    plt.axis("equal")
    plt.title(f"Animacja ścieżki dla algorytmu {nazwa}.")
    plt.legend([start,end],['pozycja startowa','pozycja docelowa'],loc='upper right')
    plt.xlabel('x')
    plt.ylabel('y')
    patch = patches.Rectangle((start_x, start_y), 7, robot_radius, 0, fc="m")
            
    def init():
        plt.gca().add_patch(patch)
        return patch

    def cal_center_robot(x_out, y_out, angle):
        if angle > 0:
            return x_out + robot_radius / 2, y_out + robot_radius /2
        if angle <= 0 :
            return x_out - robot_radius / 2, y_out - robot_radius /2

    def animate(i):
        x, y = cal_center_robot(x_out_path[i], y_out_path[i], angles[i])
        patch.set_xy([x, y])
        patch.angle = np.rad2deg(angles[i])
        plt.plot(x_out_path[i], y_out_path[i], "o", color='red')
        return patch

    ani = FuncAnimation(fig, animate, init_func=init, frames=len(x_out_path), interval=50)
    plt.show()

def prepare_for_bezier(x,y,x_obstacle,y_obstacle,epsilon,distance):
    path = np.array([x,y])
    path = np.transpose(path)
    obstacles = list(zip(x_obstacle,y_obstacle))
    new_path = remove_points(path, obstacles,epsilon,distance)
    new_path = np.array(new_path)
    indexes = np.unique(new_path, axis=0, return_index=True)[1]
    new_path = np.array([new_path[index] for index in sorted(indexes)])
    # new_path = np.flip(new_path, axis=0)
    return new_path, path

def main():
    maze_filename = "maze3.jpg"
    show_animation = True
    timing_num = 1
    elapsed_times = []
    elapsed_times_rrt = []
    elapsed_times_prm = []

    start_x = 20  # 15
    start_y = 180  # 10
    end_x = 170  # 55
    end_y = 17  # 175
    grid_size = 2.0
    robot_radius = 4.0

    x_obstacle, y_obstacle = read_obstacles_from_map(maze_filename)

    for i in range(timing_num):
        start = time.perf_counter()
        a_star = AStarPath(robot_radius, grid_size, x_obstacle, y_obstacle)
        (
            x_out_path,
            y_out_path,
            visited_points_x,
            visited_points_y,
            angle,
        ) = a_star.a_star_search(start_x, start_y, end_x, end_y)
        elapsed_time = time.perf_counter() - start
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        elapsed_times.append(elapsed_time)

    average_time = sum(elapsed_times) / timing_num  # len(elapsed_times)
    print("average_time", average_time)

    x_out_path.reverse()
    y_out_path.reverse()
    angle.reverse()
    angles=calc_list_of_angles(list(zip(x_out_path,y_out_path)))
    plot_function('A*',x_obstacle,y_obstacle,start_x,start_y,end_x,end_y,x_out_path,y_out_path, visited_points_x, visited_points_y)
    animate_path('A*',start_x,start_y,end_x,end_y,x_obstacle,y_obstacle, x_out_path, y_out_path, angles, robot_radius)

    #bezier
    new_path, path = prepare_for_bezier(x_out_path,y_out_path,x_obstacle,y_obstacle,10,6)
    bezier_path=evaluate_bezier(new_path,20)
    bezier_angles = calc_list_of_angles(bezier_path)
    plot_function_smooth('A*','bezier',x_obstacle,y_obstacle,bezier_path,new_path,path)
    animate_path('A*',start_x,start_y,end_x,end_y,x_obstacle,y_obstacle,bezier_path[:,0], bezier_path[:,1],bezier_angles,robot_radius)
    
    # x_to_interpolate, y_to_interpolate = zip(*new_path)
    # f,u = interpolate.splprep([x_to_interpolate,y_to_interpolate],s=0)
    # x_int , y _int = interpolate.splev(np.linspace(0,1,500),f)

if __name__ == "__main__":
    main()