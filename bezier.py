import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import pdb
import cv2
import csv
# find the a & b points
def constraint_bezier_coef(points):
    list_angles = calc_list_of_angles(points)
    A = []
    B = []
    for i in range (len(list_angles)-1):
        d = np.hypot(points[i][0]-points[i+1][0],points[i][1]-points[i+1][1])/3.0
        a=[points[i][0] + d * np.cos(list_angles[i]), points[i][1] + d * np.sin(list_angles[i])]
        b=[points[i+1][0] - d * np.cos(list_angles[i+1]), points[i+1][1] - d * np.sin(list_angles[i+1])]
        A.append(a)
        B.append(b)
    return np.array(A),np.array(B)

# returns the general Bezier cubic formula given 4 control points
def get_cubic(a, b, c, d):
    return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d

# return one cubic curve for each consecutive points
def get_bezier_cubic(points):
    A,B = constraint_bezier_coef(points)
    return [
        get_cubic(points[i], A[i], B[i], points[i + 1])
        for i in range(len(points) - 1)
    ]

# evalute each cubic curve on the range [0, 1] sliced in n points
def evaluate_bezier(points, n):
    curves = get_bezier_cubic(points)
    return np.array([fun(t) for fun in curves for t in np.linspace(0, 1, n)])


#check collision if line beetween a and b
def isBetween(a, b, c, epsilon = 20):
    crossproduct = (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1])

    #distance
    if abs(crossproduct) > epsilon:
        return False

    dotproduct = (c[0] - a[0]) * (b[0] - a[0]) + (c[1] - a[1])*(b[1] - a[1])
    if dotproduct < 0:
        return False

    squaredlengthba = (b[0] - a[0])*(b[0] - a[0]) + (b[1] - a[1])*(b[1] - a[1])
    if dotproduct > squaredlengthba:
        return False

    return True

#remove points from orginal path
def remove_points(path,obstacles,epsilon=20,points_between=5):
    # print(path[0])
    new_path = [path[0]]
    for i in range(1,len(path)):
        for obstacle in obstacles:
            # pdb.set_trace()
            itemindex = path.tolist().index([new_path[-1][0],new_path[-1][1]])
            # print(np.where(path == new_path[-1]))
            # print(i)
            # print(itemindex[0][0])
            if isBetween(path[i],new_path[-1],obstacle,epsilon) or abs(itemindex - i) > points_between:
                new_path.append(path[i-1])
    new_path.append(path[-1])
    return new_path

def calc_distance_and_angle(from_node, to_node):
        dx = to_node[0] - from_node[0]
        dy = to_node[1] - from_node[1]
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

def calc_list_of_angles(path):
    # print(path[0])
    list_of_angles = [0]
    for i in range(1,len(path)):
        _, theta = calc_distance_and_angle(path[i-1],path[i])
        list_of_angles.append(theta)
    return list_of_angles

if __name__ == "__main__":
        
    #read data from file with path from algorithms
    with open('prm_maze2.csv') as inf:
        x = []
        y = []
        for line in csv.reader(inf):
            tx, ty = line
            x.append(int(float(tx)))
            y.append(int(float(ty)))

    path = np.array([x,y])
    path = np.transpose(path)


    image = cv2.imread('maze2.jpg')
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

    obstacles = list(zip(x_obstacle,y_obstacle))

    new_path = remove_points(path, obstacles)
    new_path = np.array(new_path)
    # print(new_path)
    # new_path = np.unique(new_path, axis=0)
    indexes = np.unique(new_path, axis=0, return_index=True)[1]
    # print(indexes)
    new_path = np.array([new_path[index] for index in sorted(indexes)])
    # print(new_path)
    # new_path = np.array(new_path)
    # print(new_path)
    new_path = np.flip(new_path, axis=0)
    # print(new_path)


    # print('-----')
    # print(new_path)
    # print(new_path[0])
    # print(path[0])
    # print(x[0],y[0])
    # print('-----')
    x_p = [x for x,y in path]
    y_p = [y for x,y in path]
    x_n = [x for x,y in new_path]
    y_n = [y for x,y in new_path]


    points=new_path
    path = evaluate_bezier(points, 20)
    angles = calc_list_of_angles(path)
    # print(angles)
    # print("angles", angles)
    # print("len", len(angles))
    # extract x & y coordinates of points
    x1, y1 = points[:,0], points[:,1]
    px, py = path[:,0], path[:,1]
    # print(px,py)
    # print("len path", len(px))


    plt.figure()
    plt.plot(x_obstacle, y_obstacle, ".k")
    plt.plot(px, py, 'b-')
    plt.plot(x1, y1, 'ro')
    plt.plot(x,y, 'cx')
    plt.plot(x_p,y_p, 'y-')


    # from scipy import interpolate
    # arr=new_path
    # x2, y2 = zip(*arr)
    # #in this specific instance, append an endpoint to the starting point to create a closed shape
    # x2 = np.r_[x2, x2[0]]
    # y2 = np.r_[y2, y2[0]]
    # #create spline function
    # f, u = interpolate.splprep([x2, y2], s=0, per=True)
    # #create interpolated lists of points
    # xint, yint = interpolate.splev(np.linspace(0, 1, 500), f)
    # angles = calc_list_of_angles(list(zip(xint, yint)))

    patch = patches.Rectangle((x1[0], y1[0]), 5, 2, 0, fc='m')

    def init():
        # graph, = plt.plot([], [], 'og')

        plt.gca().add_patch(patch)
        return patch

    def cal_center_robot(x_out, y_out, angle):
        if angle > 0:
            return x_out + 2 / 2, y_out + 2 /2
        if angle <= 0 :
            return x_out - 2 / 2, y_out - 2 /2

    def animate(i):
        x_robot,y_robot =cal_center_robot(px[i],py[i],angles[i])
        # x_robot,y_robot =cal_center_robot(xint[i],yint[i],angles[i])
        patch.set_xy([x_robot,y_robot])
        # print(angles[i])
        patch.angle = np.rad2deg(angles[i])
        # plt.plot(x_out_path[i], y_out_path[i], "*", color='yellow')
        plt.plot(px[i], py[i], "o", color='red')
        # plt.plot(xint[i], yint[i], "o", color='red')
        return patch

    fig = plt.figure()
    plt.plot(x_obstacle, y_obstacle, ".k")
    ani = FuncAnimation(fig, animate,  init_func=init, frames=len(px), interval=220)
    plt.show()

    #interoplacja >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    from scipy import interpolate
    arr=new_path
    # arr = np.array([[0,0],[2,.5],[2.5, 1.25],[2.6,2.8],[1.3,1.1]])
    x2, y2 = zip(*arr)
    #in this specific instance, append an endpoint to the starting point to create a closed shape
    # x2 = np.r_[x2, x2[0]]
    # y2 = np.r_[y2, y2[0]]
    #create spline function
    f, u = interpolate.splprep([x2, y2], s=0)
    #create interpolated lists of points
    xint, yint = interpolate.splev(np.linspace(0, 1, 500), f)
    plt.figure()
    plt.scatter(x2, y2)
    plt.plot(xint, yint)
    plt.plot(x_obstacle, y_obstacle, ".k")
    plt.show()

    # bezier >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # plot
    # plt.figure()
    # plt.plot(x_obstacle, y_obstacle, ".k")
    # plt.plot(px, py, 'b-')
    # plt.plot(x1, y1, 'ro')
    # plt.plot(x,y, 'cx')
    # plt.show()

    #linie proste>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # plt.figure()
    # plt.plot(x_obstacle, y_obstacle, ".k")
    # plt.plot(x_p, y_p)
    # plt.plot(x_n, y_n)
    # # plt.show()

    # print(x_n, len(x_n))
    # print(y_n, len(y_n))
    # triangle metod >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # ys=smoothTriangle(y_n,2)
    # xs=smoothTriangle(x_n,2)
    # plt.figure()
    # plt.plot(xs,ys)
    # plt.plot(x_obstacle, y_obstacle, ".k")

    # plt.show()
