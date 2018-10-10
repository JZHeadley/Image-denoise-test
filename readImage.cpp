#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>
using namespace cv;

int main(int argc, char *argv[])
{
    String imageName("noisy-man.png");

    Mat image = imread(imageName, IMREAD_COLOR); // read in the file
    namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", image);
    waitKey(0);
}