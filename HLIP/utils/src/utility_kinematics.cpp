#include "utility_kinematics.h"

Eigen::Vector<double, 3> QuatWXYZToEulerZYX(Eigen::Vector<double, 4> quat_wxyz) {
    double q_w = quat_wxyz(0);
    double q_x = quat_wxyz(1);
    double q_y = quat_wxyz(2);
    double q_z = quat_wxyz(3);

    Eigen::Vector<double, 3> euler_zyx;
    // Assuming the quaterinion is normalized
    // roll
    euler_zyx(0) = std::atan2(2*(q_w*q_x + q_y*q_z), 1 - 2*(q_x*q_x + q_y*q_y)); 
    // pitch
    euler_zyx(1) = -M_PI/2 + 2*std::atan2(std::sqrt(1 + 2*(q_w*q_y - q_x*q_z)), std::sqrt(1 - 2*(q_w*q_y - q_x*q_z)));
    // yaw
    euler_zyx(2) = std::atan2(2*(q_w*q_z + q_x*q_y), 1 - 2*(q_y*q_y + q_z*q_z));

    return euler_zyx;
}

Eigen::Vector<double, 3> QuatXYZWToEulerZYX(Eigen::Vector<double, 4> quat_xyzw) {
    Eigen::Vector<double, 4> quat_wxyz;
    quat_wxyz(0) = quat_xyzw(3);
    quat_wxyz(1) = quat_xyzw(0);
    quat_wxyz(2) = quat_xyzw(1);
    quat_wxyz(3) = quat_xyzw(2);

    Eigen::Vector<double, 3> euler_zyx = QuatWXYZToEulerZYX(quat_wxyz);

    return euler_zyx;
}

Eigen::Vector<double, 4> EulerZYXToQuatWXYZ(Eigen::Vector<double, 3> euler_zyx) {
    double roll = euler_zyx(0);
    double pitch = euler_zyx(1);
    double yaw = euler_zyx(2);

    double cr = std::cos(roll/2);
    double sr = std::sin(roll/2);
    double cp = std::cos(pitch/2);
    double sp = std::sin(pitch/2);
    double cy = std::cos(yaw/2);
    double sy = std::sin(yaw/2);

    double q_w, q_x, q_y, q_z;
    q_w = cr * cp * cy + sr * sp * sy;
    q_x = sr * cp * cy - cr * sp * sy;
    q_y = cr * sp * cy + sr * cp * sy;
    q_z = cr * cp * sy - sr * sp * cy;
    
    Eigen::Vector<double, 4> quat_wxyz;

    quat_wxyz(0) = q_w;
    quat_wxyz(1) = q_x;
    quat_wxyz(2) = q_y;
    quat_wxyz(3) = q_z;

    return quat_wxyz;
}

Eigen::Vector<double, 4> EulerZYXToQuatXYZW(Eigen::Vector<double, 3> euler_zyx) {
    Eigen::Vector<double, 4> quat_wxyz = EulerZYXToQuatWXYZ(euler_zyx);

    Eigen::Vector<double, 4> quat_xyzw;
    quat_xyzw(0) = quat_wxyz(1);
    quat_xyzw(1) = quat_wxyz(2);
    quat_xyzw(2) = quat_wxyz(3);
    quat_xyzw(3) = quat_wxyz(0);

    return quat_xyzw;
}

Eigen::Matrix<double, 3, 4> EAngVelToQuatRate(Eigen::Vector<double, 4> quat_xyzw) {
    double q_x = quat_xyzw(0);
    double q_y = quat_xyzw(1);
    double q_z = quat_xyzw(2);
    double q_w = quat_xyzw(3);

    Eigen::Matrix<double, 3, 4> H;

    H << -q_x,  q_w, -q_z,  q_y,
         -q_y,  q_z,  q_w, -q_x,
         -q_z, -q_y,  q_x,  q_w;
    
    return H;
}

Eigen::Matrix3d R_x(double x) {
    Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
    
    R(0, 0) = 1.0;
    R(1, 1) = cos(x);
    R(1, 2) = -sin(x);
    R(2, 1) = sin(x);
    R(2, 2) = cos(x);

    return R;
}

Eigen::Matrix3d R_y(double y) {
    Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
    
    R(0, 0) = cos(y);
    R(0, 2) = sin(y);
    R(1, 1) = 1.0;
    R(2, 0) = -sin(y);
    R(2, 2) = cos(y);

    return R;
}

Eigen::Matrix3d R_z(double z) {
    Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
    
    R(0, 0) = cos(z);
    R(0, 1) = -sin(z);
    R(1, 0) = sin(z);
    R(1, 1) = cos(z);
    R(2, 2) = 1.0;

    return R;
}

Eigen::Matrix3d REulerZYX(double roll, double pitch, double yaw) {
    Eigen::Matrix3d R = Eigen::Matrix3d::Zero();

    R = R_z(yaw) * R_y(pitch) * R_x(roll);

    return R;
}

Eigen::Matrix<double, 3, 3> AngularToEulerZYXRates(Eigen::Matrix<double, 3, 1> omega, Eigen::Matrix<double, 3, 1> euler) {
    double roll = euler(0);
    double pitch = euler(1);

    Eigen::Matrix<double, 3, 3> E;

    E(0, 0) = 1.0;
    E(0, 1) = sin(roll) * tan(pitch);
    E(0, 2) = cos(roll) * tan(pitch);

    E(1, 0) = 0.0;
    E(1, 1) = cos(roll);
    E(1, 2) = -sin(roll);

    E(2, 0) = 0.0;
    E(2, 1) = sin(roll) / cos(pitch);
    E(2, 2) = cos(roll) / cos(pitch);

    return E;
}

