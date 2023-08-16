#include "utility_log.h"

Logger::Logger(std::string file_name) {
    this->file_id.open(file_name);
}

Logger::~Logger() {
    this->file_id.close();
}

void Logger::Write(Eigen::VectorXd data) {
    int n = data.rows();

    for(int i = 0; i < n; i++) {
        this->file_id << data(i) << ",";
    }
    this->file_id << "\n";
}

void Logger::AddLabels(std::string labels) {
    this->file_id << labels << std::endl;
}