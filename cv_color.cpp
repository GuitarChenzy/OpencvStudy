#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

class ColorDetector
{
private:
    int maxDist;
    Vec3b target;
    Mat result;
    Mat converted;

public:
    ColorDetector() : maxDist(100), target(0, 0, 0) {}
    void setTargetColor(uchar blue, uchar green, uchar red);
    void setTargetColor(Vec3b color) { target = color; }
    void setColorDistanceThreshold(int distance);

    int getColorDistanceThreshold() const { return maxDist; }
    Vec3b getTargetColor() const { return target; }
    int getDistanceToTargetColor(const Vec3b &color) const;
    int getColorDistance(const Vec3b &color1, const Vec3b &color2) const;
    Mat process(Mat &image);
};

void ColorDetector::setTargetColor(uchar blue, uchar green, uchar red)
{
    Mat temp(1, 1, CV_8UC3);
    temp.at<Vec3b>(0, 0) = Vec3b(blue, green, red);
    target = Vec3b(blue, green, red);
    cvtColor(temp, temp, CV_BGR2Lab);
    target = temp.at<Vec3b>(0, 0);
};

Mat ColorDetector::process(Mat &image)
{
    result.create(image.size(), CV_8U);
    cvtColor(image, converted, CV_BGR2Lab);
    Mat_<Vec3b>::iterator it = converted.begin<Vec3b>();
    Mat_<Vec3b>::iterator end = converted.end<Vec3b>();
    //Mat_<Vec3b>::const_iterator it = image.begin<Vec3b>();
    //Mat_<Vec3b>::const_iterator end = image.end<Vec3b>();
    Mat_<uchar>::iterator itout = result.begin<uchar>();
    for (; it != end; ++it, ++itout)
    {
        if (getDistanceToTargetColor(*it) <= maxDist)
        {
            *itout = 255;
        }
        else
        {
            *itout = 0;
        }
    }
    return result;
    /*Mat output;
    absdiff(image, Scalar(target), output);
    vector<Mat> images;
    split(output, images);
    output = images[0] + images[1] + images[2];
    threshold(output, output, maxDist, 255, THRESH_BINARY_INV);
    return output;*/
}

int ColorDetector::getDistanceToTargetColor(const Vec3b &color) const
{
    return getColorDistance(color, target);
}

int ColorDetector::getColorDistance(const Vec3b &color1, const Vec3b &color2) const
{
    return abs(color1[0] - color2[0]) + abs(color1[1] - color2[1]) + abs(color1[2] - color2[2]);
}

void ColorDetector::setColorDistanceThreshold(int distance)
{
    if (distance < 0)
        distance = 0;
    maxDist = distance;
}

int main()
{
    ///2020.10.22
    string img_path = "/home/czy/Documents/MyCPP/data/test.jpg";
    Mat image_gary = imread(img_path, 0);
    Mat image_color = imread(img_path, 1);
    imshow("ori image", image_color);
    waitKey();
    if (image_color.empty())
        return 0;
    ColorDetector cdetect;
    cdetect.setTargetColor(100, 100, 100);
    //cdetect.setColorDistanceThreshold(50);
    Mat result = cdetect.process(image_color);
    imshow("detect image", result);
    waitKey();

    Mat image = imread("/home/czy/Documents/MyCPP/data/guitar.jpg", 1);
    floodFill(image, Point(100, 50), Scalar(255, 255, 255), (Rect *)0, Scalar(35, 35, 35), Scalar(35, 35, 35), FLOODFILL_FIXED_RANGE);
    imshow("fill image", image);
    waitKey();

    Mat value_img;
    Mat saturate_img;
    Mat hue_img;
    Mat HSV_img;
    cvtColor(image_color, HSV_img, CV_BGR2HSV);
    vector<Mat> channels;
    split(HSV_img, channels);
    value_img = channels[0];
    saturate_img = channels[1];
    hue_img = channels[2];
    imshow("value_img", value_img);
    imshow("saturate_img", saturate_img);
    imshow("hue_img", hue_img);
    waitKey();
    
    /*Mat result2;
    Mat bgModel, fgModel;
    Rect rectangle(50, 50, 260, 120);
    grabCut(image_color, result2, rectangle, bgModel, fgModel, 5, GC_INIT_WITH_RECT);
    compare(result2, GC_PR_FGD, result, CMP_EQ);
    Mat foreground(image_color.size(), CV_8UC3, Scalar(255, 255, 255));
    image_color.copyTo(foreground, result2);
    imshow("grabCut image", result2);
    waitKey();*/

    ///2020.10.23
    Mat hs(128, 360, CV_8UC3);
    for (int h = 0; h < 360; h++)
    {
        for (int s = 0; s < 128; s++)
        {
            hs.at<Vec3b>(s, h)[0] = h / 2;
            hs.at<Vec3b>(s, h)[1] = 255 - s * 2;
            hs.at<Vec3b>(s, h)[2] = 255;
        }
    }
    imshow("hsv image", hs);
    waitKey();
    return 0;
}