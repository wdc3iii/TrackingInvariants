#ifndef utility_math_h
#define utility_math_h

#include <Eigen/Dense>

#include "stdio.h"
#include <iostream>

double WrapAngle(double angle);

Eigen::VectorXd WrapAngles(Eigen::VectorXd angles);

double CalculateCosineRuleAngle(double l_adjacent_1, double l_adjacent_2, double l_opposite);

double CalculateSineRuleAngle(double l_1, double ang_1, double l_2);

double sinh(double x);

double cosh(double x);

double tanh(double x);

double coth(double x);

double sech(double x);

int Factorial(int n);

double sign(double x);

double EvalPoly(Eigen::VectorXd p, double x, int d);

Eigen::VectorXd GetPolyCoeffs(Eigen::VectorXd X, Eigen::VectorXd Y, Eigen::VectorXd D);

namespace bezier_tools {

    double singleterm_bezier(int m, int k, double s);
    double bezier(const Eigen::VectorXd& coeff, double s);
    void bezier(const Eigen::MatrixXd& coeffs, double s, Eigen::VectorXd& out);
    double dbezier(const Eigen::VectorXd& coeff, double s);
    void dbezier(const Eigen::MatrixXd& coeffs, double s, Eigen::VectorXd& out);
    double d2bezier(const Eigen::VectorXd& coeff, double s);
    void d2bezier(const Eigen::MatrixXd& coeffs, double s, Eigen::VectorXd& out);

    // time derivatives 
    double dtime2Bezier(const Eigen::VectorXd& coeff, double s, double sdot);
    double dtimeBezier(const Eigen::VectorXd& coeff, double s, double sdot);

    Eigen::MatrixXd A_bezier(const Eigen::VectorXd& coeff, double s, double sdot);
    Eigen::MatrixXd dA_bezier(const Eigen::VectorXd& coeff, double s, double sdot);
    Eigen::MatrixXd d2A_bezier(const Eigen::VectorXd& coeff, double s, double sdot);

}

#endif