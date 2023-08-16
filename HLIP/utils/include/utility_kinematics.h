#ifndef utility_kinematics_h
#define utility_kinematics_h

#include <Eigen/Dense>

Eigen::Vector<double, 3> QuatWXYZToEulerZYX(Eigen::Vector<double, 4> quat_wxyz);

Eigen::Vector<double, 3> QuatXYZWToEulerZYX(Eigen::Vector<double, 4> quat_xyzw);

Eigen::Vector<double, 4> EulerZYXToQuatWXYZ(Eigen::Vector<double, 3> euler_zyx);

Eigen::Vector<double, 4> EulerZYXToQuatXYZW(Eigen::Vector<double, 3> euler_zyx);

Eigen::Matrix3d R_x(double x);

Eigen::Matrix3d R_y(double y);

Eigen::Matrix3d R_z(double z);

// Used to be called GetRotEulerZYX 
Eigen::Matrix3d REulerZYX(double roll, double pitch, double yaw);

Eigen::Matrix3d AngularToEulerZYXRates(Eigen::Vector<double, 3> omega, Eigen::Vector<double, 3> euler);


#endif