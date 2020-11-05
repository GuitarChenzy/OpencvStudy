#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

Mat salt(const Mat &image, int n);

class LaplacianZC
{
private:
    Mat laplace;  //Laplace operator
    int aperture; //Laplace kernel size
public:
    LaplacianZC() : aperture(3) {}
    void setAperture(int a)
    {
        aperture = a;
    }
    Mat computeLaplacian(const Mat &image)
    {
        Laplacian(image, laplace, CV_32F, aperture);
        return laplace;
    }
    Mat getLaplacionImage(double scale = 1.0)
    {
        if (scale < 1.0)
        {
            double lapmin, lapmax;
            minMaxLoc(laplace, &lapmin, &lapmax);
            scale = 127 / max(-lapmin, lapmax);
        }
        Mat LaplaceImage;
        laplace.convertTo(LaplaceImage, CV_8U, scale, 128);
        return LaplaceImage;
    }
    Mat getZeroCrosssings(Mat laplace)
    {
        Mat signImage;
        threshold(laplace, signImage, 0, 255, THRESH_BINARY);
        Mat binary;
        signImage.convertTo(binary, CV_8U);
        Mat dilated;
        dilate(binary, dilated, Mat());
        return dilated;
    }
};

int main()
{
    ///2020.10.25
    Mat image_gary = imread("/home/czy/Documents/MyCPP/data/test.jpg", 0);
    Mat image_color = imread("/home/czy/Documents/MyCPP/data/test.jpg", 1);
    Mat result;
    blur(image_gary, result, Size(5, 5));
    imshow("blur image", result);
    waitKey();

    GaussianBlur(image_gary, result, Size(5, 5), 1.5); //the sigma more lagre ,the picture more blur
    imshow("gaussian blur image", result);
    waitKey();

    GaussianBlur(image_gary, result, Size(11, 11), 2.0);
    Mat reduced(image_gary.rows / 4, image_gary.cols / 4, CV_8U);
    for (int i = 0; i < reduced.rows; i++)
    {
        for (int j = 0; j < reduced.cols; j++)
        {
            reduced.at<uchar>(i, j) = image_gary.at<uchar>(i * 4, j * 4);
        }
    }
    imshow("downsampling image", reduced);
    waitKey();

    Mat reducedImage, upImage;
    pyrDown(image_gary, reducedImage);
    imshow("down image", reducedImage);
    pyrUp(image_gary, upImage);
    imshow("up image", upImage);
    waitKey();

    Mat resizeImage;
    resize(image_gary, resizeImage, Size(), 2, 2, INTER_LINEAR);
    imshow("inter_linear image", resizeImage);
    waitKey();

    result = salt(image_color, 3000);
    imshow("salt image", result);
    medianBlur(result, result, 5);
    imshow("medianBlur image", result);// this blur is no-linear
    waitKey();

    Mat sobelX, sobelY;
    Sobel(image_color, sobelX, CV_8U, 1, 0, 3, 0.4, 128);
    Sobel(image_color, sobelY, CV_8U, 0, 1, 3, 0.4, 128);
    imshow("sobelX image", sobelX);
    imshow("sobelY image", sobelY);
    waitKey();

    Sobel(image_color, sobelX, CV_16S, 1, 0);
    Sobel(image_color, sobelY, CV_16S, 0, 1);
    Mat sobel;
    sobel = abs(sobelX) + abs(sobelY);// L1 norm
    double sobmin, sobmax;
    minMaxLoc(sobel, &sobmin, &sobmax);
    Mat sobelImage;
    sobel.convertTo(sobelImage, CV_8U, -255. / sobmax, 255);
    imshow("sobel image", sobelImage);
    waitKey();

    Mat sobelThresholded;
    threshold(sobelImage,sobelThresholded,175,255,THRESH_BINARY);
    imshow("soble & threshold image",sobelThresholded);
    waitKey();

    Sobel(image_color, sobelX, CV_32F, 1, 0);
    Sobel(image_color, sobelY, CV_32F, 0, 1);
    Mat norm, dir;
    cartToPolar(sobelX, sobelY, norm, dir, true);
    cout << "norm : " << norm.at<float>(100,100) << " dir : " << dir.at<float>(100,100) << endl;

    Sobel(image_gary, sobelX, CV_16S, 1, 0, CV_SCHARR);
    imshow("Scharr image", sobelX);
    waitKey();

    LaplacianZC l;
    l.setAperture(3);
    Mat flap = l.computeLaplacian(image_gary);
    result = l.getLaplacionImage();
    imshow("laplace image", result);
    waitKey();

    Mat gauss20, gauss22, dog, zeros;
    GaussianBlur(image_gary, gauss20, Size(), 2.0);
    GaussianBlur(image_gary, gauss22, Size(), 2.2);
    subtract(gauss22, gauss20, dog, Mat(), CV_32F);
    zeros = l.getZeroCrosssings(dog);
    imshow("dog image", zeros);
    waitKey();

    return 0;
}

Mat salt(const Mat &image, int n)
{
    default_random_engine generator;
    uniform_int_distribution<int> randomRow(0, image.rows - 1);
    uniform_int_distribution<int> randomCol(0, image.cols - 1);
    Mat result = image.clone();
    int i, j;
    for (int k = 0; k < n; k++)
    {
        i = randomCol(generator);
        j = randomRow(generator);
        if (result.type() == CV_8UC1)
        {
            result.at<uchar>(j, i) = 255;
        }
        else if (result.type() == CV_8UC3)
        {
            result.at<Vec3b>(j, i) = Vec3b(255, 255, 255);
        }
    }
    return result;
}