#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

class HarrisDetector
{
private:
    Mat cornerStrength; //32-bit float harris img
    Mat cornerTh;       // 32-bit float threshold harris img
    Mat localMax;       //local max img
    int neighborhood;   //the size of the domain of smooth derivatives
    int aperture;       // gatdient value;
    double k;           // Harris parameter
    double maxStrength; //max threshold
    double threshold_d;
    int nonMaxSize; //the domain size of non-maximum
    Mat kernel;     //the kernel of non-maximum
public:
    HarrisDetector() : neighborhood(3), aperture(3), k(0.01), maxStrength(0.0), threshold_d(0.01), nonMaxSize(3)
    {
        setLocalMaxWindowSize(nonMaxSize);
    }
    void setLocalMaxWindowSize(int nonMaxSize)
    {
        //kernel(nonMaxSize, nonMaxSize, CV_8U);
    }
    void detect(const Mat &image)
    {
        cornerHarris(image, cornerStrength, neighborhood, aperture, k);
        minMaxLoc(cornerStrength, 0, &maxStrength);
        Mat dilated;
        dilate(cornerStrength, dilated, Mat());
        compare(cornerStrength, dilated, localMax, CMP_EQ);
    }
    Mat getCornerMap(double qualityLevel)
    {
        Mat cornerMap;
        threshold_d = qualityLevel * maxStrength;
        threshold(cornerStrength, cornerTh, threshold_d, 255, THRESH_BINARY);
        cornerTh.convertTo(cornerMap, CV_8U);
        bitwise_and(cornerMap, localMax, cornerMap); // non-maximum
        return cornerMap;
    }
    void getCorners(vector<Point> &points, double qualityLevel)
    {
        Mat cornerMap = getCornerMap(qualityLevel);
        getCorners(points, cornerMap);
    }
    void getCorners(vector<Point> &points, const Mat &cornerMap)
    {
        for (int y = 0; y < cornerMap.rows; y++)
        {
            const uchar *rowPtr = cornerMap.ptr<uchar>(y);
            for (int x = 0; x < cornerMap.cols; x++)
            {
                if (rowPtr[x])
                {
                    points.push_back(Point(x, y));
                }
            }
        }
    }
    void drawOnImage(Mat &image, const vector<Point> &points, Scalar color = Scalar(255, 255, 255), int radius = 3, int thickness = 1)
    {
        vector<Point>::const_iterator it = points.begin();
        while (it != points.end())
        {
            circle(image, *it, radius, color, thickness);
            ++it;
        }
    }
};

int main()
{
    ///2020.10.26
    Mat image_g = imread("/home/czy/Documents/MyCPP/data/test.jpg", 0);
    Mat image_c = imread("/home/czy/Documents/MyCPP/data/test.jpg", 1);
    Mat cornerStrength;
    cornerHarris(image_g, cornerStrength, 3, 3, 0.01);
    Mat harrisCoeners;
    double threshold_d = 0.0001;
    threshold(cornerStrength, harrisCoeners, threshold_d, 255, THRESH_BINARY_INV);
    imshow("harris image", harrisCoeners);
    waitKey();

    HarrisDetector h;
     Mat result = image_c.clone();
    h.detect(image_g);
    vector<Point> pts;
    h.getCorners(pts, 0.02);
    h.drawOnImage(result, pts);
    imshow("harris det image", result);
    waitKey();

    vector<KeyPoint> keypoints;
    Mat result2;
    Ptr<GFTTDetector> ptrGFTT = GFTTDetector::create(500, 0.01, 20);
    ptrGFTT->detect(image_c, keypoints);
    drawKeypoints(image_c, keypoints, result2);
    imshow("GFTT image", result2);
    waitKey();

    vector<KeyPoint> keypoints2;
    Mat result3 = image_c.clone();
    Ptr<FastFeatureDetector> ptrFast = FastFeatureDetector::create(40);
    ptrFast->detect(image_c, keypoints2);
    cout << keypoints2.size() << endl;
    drawKeypoints(image_c, keypoints2, result3, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_OVER_OUTIMG);
    imshow("Fast image", result3);
    waitKey();

    vector<KeyPoint> keypoints3;
    Mat result4 = image_c.clone();
    Ptr<xfeatures2d::SurfFeatureDetector> ptrSurf = xfeatures2d::SurfFeatureDetector::create(2000.0);
    ptrSurf->detect(image_c, keypoints3);
    drawKeypoints(image_c, keypoints3, result4, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("Surf image", result4);
    waitKey();

    vector<KeyPoint> keypoints4;
    Mat result5 = image_c.clone();
    Ptr<xfeatures2d::SiftFeatureDetector> ptrSift = xfeatures2d::SiftFeatureDetector::create();
    ptrSift->detect(image_c, keypoints4);
    drawKeypoints(image_c, keypoints4, result5, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("Sift image", result5);
    waitKey();

    vector<KeyPoint> keypoints5;
    Mat result6 = image_c.clone();
    Ptr<BRISK> ptrBrisk = BRISK::create();
    ptrBrisk->detect(image_c, keypoints5);
    drawKeypoints(image_c, keypoints5, result6, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("Brisk image", result6);
    waitKey();

    vector<KeyPoint> keypoints6;
    Mat result7 = image_c.clone();
    Ptr<ORB> ptrOrb = ORB::create(75,1.2,8);
    ptrOrb->detect(image_c, keypoints6);
    drawKeypoints(image_c, keypoints6, result7, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("Orb image", result7);
    waitKey();

    return 0;
}