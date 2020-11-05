#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

class Histogram1D
{
private:
    int histSize[1];
    float hranges[2];
    const float *ranges[1];
    int channels[1];

public:
    Histogram1D()
    {
        histSize[0] = 256;
        hranges[0] = 0.0;
        hranges[1] = 256.0;
        ranges[0] = hranges;
        channels[0] = 0;
    }
    Mat getHistogram(const Mat &image)
    {
        Mat hist;
        calcHist(&image, 1, channels, Mat(), hist, 1, histSize, ranges);
        return hist;
    }
    Mat getHistogramImage(const Mat &image, int zoom = 2)
    {
        Mat hist = getHistogram(image);
        return getImageOfHistogram(hist, zoom);
    }
    static Mat getImageOfHistogram(const Mat &hist, int zoom)
    {
        double maxVal = 0;
        double minVal = 0;
        minMaxLoc(hist, &minVal, &maxVal, 0, 0);
        int histsize = hist.rows;
        Mat histImg(histsize * zoom, histsize * zoom, CV_8U, Scalar(255));
        int hpt = saturate_cast<int>(0.9 * histsize);
        int tep = 0;
        int number = 0;
        for (int h = 0; h < histsize; h++)
        {
            float binVal = hist.at<float>(h);
            if (binVal > 0)
            {
                int intensity = static_cast<int>(binVal * hpt / maxVal);
                if (tep < intensity)
                {
                    tep = intensity;
                    number = h;
                }
                line(histImg, Point(h * zoom, histsize * zoom), Point(h * zoom, (histsize - intensity) * zoom), Scalar(0), zoom);
            }
        }
        cout << "Max hist number of value : " << number << endl;
        return histImg;
    }
    static Mat applyLookUp(const Mat &image, const Mat &lookUp)
    {
        Mat result;
        LUT(image, lookUp, result);
        return result;
    }
    void colorReduce(Mat &image, int div = 64)
    {
        Mat lookup(1, 256, CV_8U);
        for (int i = 0; i < 256; i++)
        {
            lookup.at<uchar>(i) = i / div * div + div / 2;
        }
        LUT(image, lookup, image);
    }
    Mat calcBackImage(Mat &image, Mat &imageROI)
    {
        Mat result;
        Mat hist = getHistogram(imageROI);
        normalize(hist, hist, 1.0);
        calcBackProject(&image, 1, channels, hist, result, ranges, 255.0);
        return result;
    }
};

int main()
{
    ///2020.10.23
    string img_path = "/home/czy/Documents/MyCPP/data/guitar.jpg";
    Mat image_gary = imread(img_path, 0);
    Mat image_color = imread(img_path, 1);
    Histogram1D h;
    Mat histoutput = h.getHistogramImage(image_gary);
    imshow("hist image", histoutput);
    waitKey();

    Mat thresholded;
    threshold(image_gary, thresholded, 20, 255, THRESH_BINARY);
    imshow("threshold image", thresholded);
    waitKey();

    /*Mat lookUp_image;
    Mat lookUp(1, 256, CV_8U);
    for (int i = 0; i < 256; i++)
    {
        lookUp_image.at<uchar>(i) = 255 - i;
    }
    lookUp_image = h.applyLookUp(image_gary, lookUp);
    imshow("lut image", lookUp_image);
    waitKey();*/

    Mat result = image_color.clone();
    h.colorReduce(result);
    imshow("lookup in reduce image", result);
    waitKey();

    Mat result2;
    equalizeHist(image_gary, result2);
    Mat histoutput2 = h.getHistogramImage(result2);
    imshow("equal hist image", histoutput2);
    waitKey();

    Mat image_magic = imread("/home/czy/Documents/MyCPP/data/magic.jpg", 0);
    Mat imageROI;
    Mat result3;
    imageROI = image_magic(Rect(375, 75, 100, 100));
    Histogram1D h;
    result3 = h.calcBackImage(image_magic, imageROI);
    threshold(result3, result3, 10, 255, THRESH_BINARY);
    imshow("back image", result3);
    waitKey();

    ///2020.10.24
    Mat image_book = imread("/home/czy/Documents/MyCPP/data/book_text.jpg", 0);
    Mat binaryFixed;
    threshold(image_book, binaryFixed, 125, 255, THRESH_BINARY);
    imshow("binary image", binaryFixed);
    waitKey();

    Mat iimage;
    //integral(image_book,iimage,CV_32S); //compute the integral image
    adaptiveThreshold(image_book, iimage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 21, 10);
    imshow("adapt binary image", iimage);
    waitKey();

    return 0;
}