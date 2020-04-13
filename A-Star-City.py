#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from grid import create_grid
from planning import a_star

get_ipython().run_line_magic('matplotlib', 'inline')

from bresenham import bresenham

plt.rcParams['figure.figsize'] = 12, 12


get_ipython().run_line_magic('pinfo', 'create_grid')

filename = 'colliders.csv'
data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
print(data)

# Static drone altitude (meters)
drone_altitude = 5

# Minimum distance stay away from obstacle (meters)
safe_distance = 3

grid = create_grid(data, drone_altitude, safe_distance)

plt.imshow(grid, origin='lower') 
plt.xlabel('EAST')
plt.ylabel('NORTH')
plt.show()

start_ne = (25,  100)
goal_ne = (750., 370.)

def heuristic(position, goal_position):
    h = np.sqrt((position[0]-goal_position[0])**2+(position[1]-goal_position[1])**2)
    return h

path, cost = a_star(grid, heuristic, start_ne, goal_ne)
print(path, cost)

plt.imshow(grid, cmap='Greys', origin='lower')

# For the purposes of the visual the east coordinate lay along
# the x-axis and the north coordinates long the y-axis.
plt.plot(start_ne[1], start_ne[0], 'x')
plt.plot(goal_ne[1], goal_ne[0], 'x')

if path is not None:
    pp = np.array(path)
    plt.plot(pp[:, 1], pp[:, 0], 'g')

plt.xlabel('EAST')
plt.ylabel('NORTH')
plt.show()

#Path Pruning


def point(p):
    return np.array([p[0], p[1], 1.]).reshape(1, -1)

def collinearity_check(p1, p2, p3, epsilon=1e-6):   
    m = np.concatenate((p1, p2, p3), 0)
    det = np.linalg.det(m)
    return abs(det) < epsilon

def prune_path(path):
    if path is not None:
        pruned_path = [p for p in path]
        i = 0
        while i < len(pruned_path) - 2:
            p1 = pruned_path[i]
            p2 = pruned_path[i+1]
            p3 = pruned_path[i+2]
        
            cells = list(bresenham(int(p1[0]), int(p1[1]), int(p3[0]), int(p3[1])))
            free = 1
            for j in cells:
                if grid[j[0]][j[1]]==1:
                    free=0
                    break
            if free == 1:
                pruned_path.remove(pruned_path[i+1])
            else:
                i += 1
    else:
        pruned_path = path
        
    return pruned_path

pruned_path = prune_path(path)
print(pruned_path)


# Replot the path, it will be the same as before but the drone flight will be much smoother.

plt.imshow(grid, cmap='Greys', origin='lower')

plt.plot(start_ne[1], start_ne[0], 'x')
plt.plot(goal_ne[1], goal_ne[0], 'x')

if pruned_path is not None:
    pp = np.array(pruned_path)
    plt.plot(pp[:, 1], pp[:, 0], 'g')
    plt.scatter(pp[:, 1], pp[:, 0])

plt.xlabel('EAST')
plt.ylabel('NORTH')

plt.show()