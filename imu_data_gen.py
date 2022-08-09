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
scale = np.array([[0.98, -0.02, 0.05],
                  [-0.07, 0.97, 0.04],
                  [-0.02, 0.08, 1.02]])
# bias factor to calibrate
bias =np.array([[3.5],
                [-2.2],
                [4.7]])

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

# generation of accelerometer raw data (which contain scaling, bias and noise)
for phi in np.linspace(0,2*math.pi,10):
	for theta in np.linspace(-0.5*math.pi,0.5*math.pi,10):
		x.append(g*math.cos(theta)*math.cos(phi))
		y.append(g*math.cos(theta)*math.sin(phi))
		z.append(g*math.sin(theta))

# add some outlier in accelerometer raw data
for phi in np.linspace(0,0.4*math.pi,3):
	for theta in np.linspace(-0.3*math.pi,0.2*math.pi,3):
		x.append(2.5*g*math.cos(theta)*math.cos(phi))
		y.append(3.4*g*math.cos(theta)*math.sin(phi))
		z.append(4.7*g*math.sin(theta))


N = len(x)
x = np.array(x).reshape(1, N)
y = np.array(y).reshape(1, N)
z = np.array(z).reshape(1, N)
print(type(x))
Acc_cal = np.concatenate((x, y, z), axis = 0)
print(Acc_cal.shape)

noise_x = 0.2*np.random.randn(N).reshape(1,N)
noise_y = 0.2*np.random.randn(N).reshape(1,N)
noise_z = 0.2*np.random.randn(N).reshape(1,N)
noise = np.concatenate((noise_x, noise_y, noise_z), axis = 0)
# add bias to acc
Acc_cal[0] = Acc_cal[0]-bias[0][0]
Acc_cal[1] = Acc_cal[1]-bias[1][0]
Acc_cal[2] = Acc_cal[2]-bias[2][0]
# add scale to acc
Acc_raw = np.matmul(np.linalg.inv(scale), Acc_cal)
# add noise to accelerometer raw data
Acc_raw = np.add(Acc_raw, noise)
Ax = Acc_raw[0]
Ay = Acc_raw[1]
Az = Acc_raw[2]

ax.scatter(x,y,z, s=1, marker=',', color='b', label='Ideal data')
ax.scatter(Ax,Ay,Az, s=1, marker=',', color='r', label='Artificial data')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_xlim(-25,25)
ax.set_ylim(-25,25)
ax.set_zlim(-25,25)
set_axes_equal(ax)


# generation of gyroscope raw data (which contain bias and noise)
bais_gyro = np.array([[0.1],
                      [-0.12],
                      [0.15]])
# add bias to gyro
Wx = np.zeros((N,), dtype=float) - bais_gyro[0][0]
Wy = np.zeros((N,), dtype=float) - bais_gyro[1][0]
Wz = np.zeros((N,), dtype=float) - bais_gyro[2][0]
# add noise to gyroscope raw data
noise_x = 0.02*np.random.randn(N)
noise_y = 0.02*np.random.randn(N)
noise_z = 0.02*np.random.randn(N)
Wx = np.add(Wx, noise_x)
Wy = np.add(Wy, noise_y)
Wz = np.add(Wz, noise_z)
# add some outliers in gyro raw data 
Wx[10] = 3.1
Wy[10] = -4.5
Wz[10] = -3.6
Wx[20] = 5.6
Wy[20] = -2.1
Wz[20] = -1.6

# display gyroscope raw data
plt.figure()
plt.plot(Wx,'r', Wy, 'g', Wz, 'b')
plt.xlabel('Number of data')
plt.ylabel('rad/sec')
plt.legend(["Wx", "Wy", "Wz"],loc = "lower right")
plt.show()

print(np.shape(Ax))
print(np.shape(Wx))
# Save generated data to .csv file
Data = np.transpose(np.asarray([Ax,Ay,Az,Wx,Wy,Wz]))
print(np.shape(Data))
# Data.tofile('./imu_data.csv', sep=',', format='%6.3f')
np.savetxt('./imu_data.csv', Data, delimiter=',', fmt='%8.5f')




