import matplotlib.pyplot as plt
import numpy as np
import math
import random


fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
g = 9.8
x = []
y = []
z = []
# scale factor to calibrate
sx = 1.05
sy = 0.94
sz = 0.96
# bias factor to calibrate
bx = 3.5
by = -2.2
bz = 4.7

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

for phi in np.linspace(0,2*math.pi,25):
	for theta in np.linspace(-0.5*math.pi,0.5*math.pi,25):
		x.append(g*math.cos(theta)*math.cos(phi))
		y.append(g*math.cos(theta)*math.sin(phi))
		z.append(g*math.sin(theta))

# add some outlier

for phi in np.linspace(0,0.4*math.pi,3):
	for theta in np.linspace(-0.3*math.pi,0.2*math.pi,3):
		x.append(2.5*g*math.cos(theta)*math.cos(phi))
		y.append(3.4*g*math.cos(theta)*math.sin(phi))
		z.append(2.7*g*math.sin(theta))



N = len(x)
# artificial IMU (with noise) data generation
x = np.array(x)
y = np.array(y)
z = np.array(z)
print(type(x))
noise = 0.1*np.random.randn(N)
print(type(noise))
Ax = (1.0/sx)*x - bx + noise
Ay = (1.0/sy)*y - by + noise
Az = (1.0/sz)*z - bz + noise


ax.scatter(x,y,z, s=1, marker=',', color='b', label='Ideal data')
ax.scatter(Ax,Ay,Az, s=1, marker=',', color='r', label='Artificial data')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_xlim(-25,25)
ax.set_ylim(-25,25)
ax.set_zlim(-25,25)
set_axes_equal(ax)
plt.show()

# save generated data to .csv file
Data = np.transpose(np.asarray([Ax,Ay,Az]))
#Data = np.asarray([[1,2,3], [4,5,6], [7,8,9]])
print(np.shape(Ax))
print(np.shape(Data))
#Data.tofile('./imu_data.csv', sep=',', format='%6.3f')
np.savetxt('./imu_data.csv', Data, delimiter=',', fmt='%6.4f')










