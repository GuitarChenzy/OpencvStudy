#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

void onMouse(int event, int x, int y, int flags, void *param);

int main(int, char **)
{
    ///2020.10.20
    string img_path = "/home/czy/Documents/MyCPP/data/test.jpg";
    Mat img = imread(img_path, 0);
    cv::setMouseCallback("Original Image", onMouse, reinterpret_cast<void *>(&img));
    circle(img, Point(100, 100), 65, 0, 3);
    imshow("test", img);
    cout << "image channels : " << img.channels() << endl;
    cout << "image is : " << img.rows << " x " << img.cols << endl;
    Mat result;
    flip(img, result, 1);
    imshow("hroizontal image", result);
    waitKey();
    imwrite("/home/czy/Documents/MyCPP/data/output.jpg", result);
    

    ///2020.10.21
    Mat img2(Size(320, 240), CV_8UC3, Scalar(0, 0, 255));
    imshow("test2", img2);
    Matx33d matrix(3.0, 2.0, 1.0,
                   2.0, 1.0, 3.0,
                   1.0, 2.0, 3.0);
    Matx31d vector(5.0, 1.0, 3.0);
    Matx31d result = matrix * vector;
    cout << result << endl;
    Mat image = imread("/home/czy/Documents/MyCPP/data/guitar.jpg", 0);
    Mat logo = imread("/home/czy/Documents/MyCPP/data/cake.jpg", 0);
    Mat maskMat(Size(100, 100), CV_8U, Scalar(255));
    Mat imageROI(image, Rect(image.cols / 2 - maskMat.cols / 2, image.rows / 2 - maskMat.rows / 2, maskMat.cols, maskMat.rows));
    //Mat imageROI = image.rowRange(0,400);
    Mat mask(maskMat);
    maskMat.copyTo(imageROI, mask);
    imshow("ROI Image", image);
    waitKey();
    return 0;
}

void onMouse(int event, int x, int y, int flags, void *param)
{
    Mat *img = reinterpret_cast<Mat *>(param);
    switch (event)
    {
    case EVENT_LBUTTONDOWN:
        cout << "at (" << x << "," << y << ") value is : " << static_cast<int>(img->at<uchar>(Point(x, y))) << endl;
        break;
    default:
        cout << "Nothing" << endl;
        break;
    }
}