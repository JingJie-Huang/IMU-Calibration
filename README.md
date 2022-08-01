IMU Calibration for Accelerometer and Gyroscope
=====

### Depenency
1. ceres solver
2. g2o
3. Python3

### Test Environment
Ubuntu 20.04

### How to Use
1. Run ceres_imu.cpp for IMU Calibration using ceres solver
2. Run g2o_imu.cpp for IMU Calibration using g2o solver

### Aritificial IMU data Generation
Execute imu_data_gen.py to genterate artificial IMU dat

### Example Aritificial IMU Data
In this example, add some outliers(polluted points) to test the robustness of the calibration program
![image](https://github.com/JingJie-Huang/IMU-Calibration/blob/main/raw_data.png)


