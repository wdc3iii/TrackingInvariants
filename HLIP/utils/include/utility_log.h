#ifndef utility_log_h
#define utility_log_h

#include <Eigen/Dense>

#include "stdio.h"
#include <iostream>
#include <fstream>

#include<cstdlib>

class Logger {
    public: Logger(std::string file_name);

    public: ~Logger();

    private: std::ofstream file_id;

    public: void Write(Eigen::VectorXd data);

    public: void AddLabels(std::string labels);
};

#endif