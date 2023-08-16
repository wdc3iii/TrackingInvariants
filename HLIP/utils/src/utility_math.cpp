#include "utility_math.h"

double WrapAngle(double angle) {
    while(angle > M_PI) {
        angle -= 2 * M_PI;
    }

    while(angle < -M_PI) {
        angle += 2 * M_PI;
    }

    return angle;
}

Eigen::VectorXd WrapAngles(Eigen::VectorXd angles) {
    int number_of_elements = angles.rows();

    for(int i = 0; i < number_of_elements; i++) {
        angles(i) = WrapAngle(angles(i));
    }

    return angles;
}

double sinh(double x) {
    return (exp(x) - exp(-x)) / 2.0;
}

double cosh(double x) {
    return (exp(x) + exp(-x)) / 2.0;
}

double tanh(double x) {
    return sinh(x) / cosh(x); 
}

double coth(double x) {
    return (exp(x) + exp(-x)) / (exp(x) - exp(-x));
}

double sech(double x) {
    return 2.0 / (exp(x) + exp(-x));
}

int Factorial(int n) {
    double res = 1.0;

    if (n < 0) {
        std::cout << "Error: Input must be bigger than zero";
    } else if (n > 0) {
        for (int i = 1; i < n + 1; i++) {
            res *= i;
        }
    }
    return res;
}

double sign(double x) {
    double y;
    if (x > 0) {
        y = 1.0;
    } else if (x < 0) {
        y = -1.0;
    } else {
        y = 0.0;
    }
    return y;
}

Eigen::VectorXd GetPolyCoeffs(Eigen::VectorXd X, Eigen::VectorXd Y, Eigen::VectorXd D) {
    const int n = X.rows();

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd b = Y;

    for(int i = 0; i < n; i++) {
        double d = D(i);
        double x = X(i);

        for(int k = d; k < n; k++) {
            A(i, k) = double(Factorial(k)) / double(Factorial(k - int(d))) * pow(x, k - d);

            double f_k = Factorial(k);
            double f_k_d = Factorial((k - int(d)));
        }
    }

    Eigen::VectorXd p = A.fullPivLu().solve(b);

    return p;
}

double EvalPoly(Eigen::VectorXd p, double x, int d)
{
    int n = p.rows();
    
    double res = 0.0;

    for(int i = d; i < n; i++) {
        res += p(i) * double(Factorial(i)) / double(Factorial(i - int(d))) * pow(x, i - d);
    }

    return res;
}

double CalculateCosineRuleAngle(double l_adjacent_1, double l_adjacent_2, double l_opposite)
{
    double arg = (l_adjacent_1 * l_adjacent_1 + l_adjacent_2 * l_adjacent_2 - l_opposite * l_opposite) / (2.0 * l_adjacent_1 * l_adjacent_2);

    double tol = 0.0000001;

    if (abs(arg - 1.0) < tol) {
        return 0; 
    } else if (abs(arg - (-1.0)) < tol) {
        return M_PI;
    } else {
        return acos(arg);
    }


}

double CalculateSineRuleAngle(double l_side, double ang_side, double l_opposite) { 
    return asin(l_opposite * sin(ang_side) / l_side);
}

namespace bezier_tools {

    double factorial(int n) {
        double out = 1.;
        for (int i = 2; i <= n; ++i) {
            out *= i;
        }
        return out;
    }

    double nchoosek(int n, int k) {
        return factorial(n) / (factorial(k) * factorial(n - k));
    }

    double singleterm_bezier(int m, int k, double s) {
        if (k == 0) {
            return nchoosek(m, k) * std::pow(1 - s, m - k);
        } else if (k == m) {
            return nchoosek(m, k) * std::pow(s, k);
        }
        return nchoosek(m, k) * std::pow(s, k) * std::pow(1 - s, m - k);   
    }

    double bezier(const Eigen::VectorXd& coeff, double s) {
        int m = coeff.size() - 1;
        double fcn = 0.;
        for (int k = 0; k <= m; ++k) {
            fcn += coeff(k) * singleterm_bezier(m, k, s);
        }
        return fcn;
    }

    void bezier(const Eigen::MatrixXd& coeffs, double s, Eigen::VectorXd& out) {
        for (int i = 0; i < coeffs.rows(); ++i) {
            out(i) = bezier(coeffs.row(i), s);
        }
    }

    void diff_coeff(const Eigen::VectorXd& coeff, Eigen::VectorXd& dcoeff) {
        int m = coeff.size() - 1;
        Eigen::MatrixXd A(m, m + 1);
        A.setZero();

        for (int i = 0; i < m; ++i) {
            A(i, i) = -(m - i) * nchoosek(m, i) / nchoosek(m - 1, i);
            A(i, i + 1) = (i + 1) * nchoosek(m, i + 1) / nchoosek(m - 1, i);
        }
        A(m - 1, m) = m * nchoosek(m, m);
        dcoeff = A * coeff;
    }

    double dbezier(const Eigen::VectorXd& coeff, double s) {
        Eigen::VectorXd dcoeff;
        dcoeff.resizeLike(coeff);
        diff_coeff(coeff, dcoeff);
        return bezier(dcoeff, s);
    }

    void dbezier(const Eigen::MatrixXd& coeffs, double s, Eigen::VectorXd& out) {
        for (int i = 0; i < coeffs.rows(); ++i) {
            out(i) = dbezier(coeffs.row(i), s);
        }
    }

    double d2bezier(const Eigen::VectorXd& coeff, double s) {
        Eigen::VectorXd dcoeff, d2coeff;
        dcoeff.resizeLike(coeff);
        d2coeff.resizeLike(coeff);
        diff_coeff(coeff, dcoeff);
        diff_coeff(dcoeff, d2coeff);
        return bezier(d2coeff, s);
    }

    void d2bezier(const Eigen::MatrixXd& coeffs, double s, Eigen::VectorXd& out) {
        for (int i = 0; i < coeffs.rows(); ++i) {
            out(i) = d2bezier(coeffs.row(i), s);
        }
    }


    double dtimeBezier(const Eigen::VectorXd& coeff, double s, double sdot) {
        double out;
        Eigen::VectorXd dcoeff;
        dcoeff.resizeLike(coeff);
        diff_coeff(coeff, dcoeff);
        out = bezier(dcoeff, s) * sdot;
        return out;
    }

    double dtime2Bezier(const Eigen::VectorXd& coeff, double s, double sdot) {
        double out;
        Eigen::VectorXd dcoeff, d2coeff;
        dcoeff.resizeLike(coeff);
        d2coeff.resizeLike(coeff);
        diff_coeff(coeff, dcoeff);
        diff_coeff(dcoeff, d2coeff);
        out = bezier(d2coeff, s) * pow(sdot, 2);
        return out;
    }

    Eigen::MatrixXd A_bezier(const Eigen::VectorXd& coeff, double s, double sdot) {
        Eigen::MatrixXd A;
        int ncoeff = coeff.size();
        A.resize(1, ncoeff);
        for (int j = 0; j < ncoeff; j++) {
            A(0, j) = nchoosek(ncoeff - 1, j) * pow(s, j) * pow(1 - s, ncoeff - 1 - j);
        }
        return A;
    };

    Eigen::MatrixXd dA_bezier(const Eigen::VectorXd& coeff, double s, double sdot) {
        Eigen::MatrixXd A, A_vec, A_mat;
        int ncoeff = coeff.size();
        A_vec.resize(1, ncoeff - 1);
        A_mat.resize(ncoeff - 1, ncoeff);
        for (int j = 0; j < ncoeff - 1; j++) {
            A_vec(0, j) = sdot * factorial(ncoeff - 1) / factorial(j) / factorial(ncoeff - j - 2) * pow(s, j) * pow(1 - s, ncoeff - 2 - j);
            A_mat(j, j) = -1.;
            A_mat(j, j + 1) = 1.;
        }
        A = A_vec * A_mat;
        return A;
    };

    Eigen::MatrixXd d2A_bezier(const Eigen::VectorXd& coeff, double s, double sdot) {
        Eigen::MatrixXd A, A_vec, A_mat;
        int ncoeff = coeff.size();
        A_vec.resize(1, ncoeff - 2);
        A_mat.resize(ncoeff - 2, ncoeff);
        for (int j = 0; j < ncoeff - 2; j++) {
            A_vec(0, j) = pow(sdot, 2) * factorial(ncoeff - 1) / factorial(j) / factorial(ncoeff - j - 3) * pow(s, j) * pow(1 - s, ncoeff - 3 - j);
            A_mat(j, j) = 1.;
            A_mat(j, j + 1) = -2.;
            A_mat(j, j + 2) = 1.;
        }
        A = A_vec * A_mat;
        return A;
    };
}
