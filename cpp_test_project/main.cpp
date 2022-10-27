#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    std::cout << "cv_version: " << CV_VERSION << std::endl;
    cv::Mat M(2,2, CV_8UC3, cv::Scalar(0,0,255));
    std::cout << "M = " << std::endl << " " << M << std::endl << std::endl;
    std::cout << "Hello, World!" << std::endl;
    return 0;
}