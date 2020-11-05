#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

class WatershedSegmenter
{
private:
    Mat markers;

public:
    void setMakers(const Mat &markerImage)
    {
        markerImage.convertTo(markers, CV_32S);
    }
    Mat process(const Mat &image)
    {
        watershed(image, markers);
        return markers;
    }
    Mat getSegmentation()
    {
        Mat temp;
        markers.convertTo(temp, CV_8U);
        return temp;
    }
    Mat getWatersheds()
    {
        Mat temp;
        markers.convertTo(temp, CV_8U, 128, 255);
        return temp;
    }
};

int main()
{
    ///2020.10.24
    Mat image_book = imread("/home/czy/Documents/MyCPP/data/test.jpg", 0);
    Mat image_book_color = imread("/home/czy/Documents/MyCPP/data/test.jpg", 1);
    
    Mat eroded;
    //Mat element(7, 7, CV_8U, Scalar(1));
    erode(image_book, eroded, Mat());
    Mat dilated;
    dilate(image_book, dilated, Mat());
    imshow("erode image", eroded);
    waitKey();
    imshow("dilate image", dilated);
    waitKey();

    Mat element2(5, 5, CV_8U, Scalar(1));
    Mat closed;
    morphologyEx(image_book, closed, MORPH_CLOSE, element2);
    imshow("close image", closed);
    waitKey();

    Mat opened;
    morphologyEx(image_book, opened, MORPH_OPEN, element2);
    imshow("open image", opened);
    waitKey();

    Mat result;
    morphologyEx(image_book, result, MORPH_GRADIENT, Mat());
    imshow("grad image", result);
    waitKey();
    
    Mat result2;
    Mat element3(7, 7, CV_8U, Scalar(1));
    morphologyEx(image_book, result2, MORPH_BLACKHAT, element3);
    // black-hat more effective than the top-hat at this picture
    imshow("black-hat image", result2);
    waitKey();
   
    Mat fg;
    erode(image_book, fg, Mat(), Point(-1, -1), 1);
    imshow("erode mu image", fg);
    waitKey();
    Mat bg;
    dilate(image_book, bg, Mat(), Point(-1, -1), 1);
    threshold(bg, bg, 128, 255, THRESH_BINARY_INV);
    imshow("dilate mu image", bg);
    waitKey();
    Mat markers(image_book.size(), CV_8U, Scalar(0));
    markers = fg + bg;
    imshow("marker image", markers);
    waitKey();
    WatershedSegmenter w;
    w.setMakers(markers);
    //imshow("seg",w.getSegmentation());
    //imshow("wat",w.getWatersheds());
    Mat result3 = w.process(image_book_color);
    imshow("water image", result3);
    waitKey();
    

    Ptr<MSER> mser = MSER::create(5, 200, 2000);
    vector<vector<Point>> points;
    vector<Rect> rects;
    mser->detectRegions(image_book_color, points, rects);
    Mat output(image_book_color.size(), CV_8UC3);
    output = Scalar(255, 255, 255);
    RNG rng;
    for (vector<vector<Point>>::reverse_iterator it = points.rbegin(); it != points.rend(); ++it)
    {
        Vec3b c(rng.uniform(0, 254), rng.uniform(0, 254), rng.uniform(0, 254));
        for (vector<Point>::iterator itPts = it->begin(); itPts != it->end(); ++itPts)
        {
            if (output.at<Vec3b>(*itPts)[0] == 255)
            {
                output.at<Vec3b>(*itPts) = c;
            }
        }
    }
    imshow("MSER image", output);
    waitKey();

    return 0;
}