#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

class RobustMatcher
{
private:
    Ptr<FeatureDetector> ptrDetector;
    Ptr<DescriptorExtractor> ptrDesc;
    int normType;
    float ratio;
    bool refineF;
    bool refineM;
    double distance; // the min distance to pole
    double confidence;
    Mat image1, image2;

public:
    RobustMatcher(const Ptr<FeatureDetector> &detector,
                  const Ptr<DescriptorExtractor> &desc = Ptr<DescriptorExtractor>()) : ptrDetector(detector), ptrDesc(desc), normType(NORM_L2),
                                                                                       ratio(0.8f), refineF(true), refineM(true), confidence(0.98), distance(1.0)
    {
        if (!this->ptrDesc)
        {
            this->ptrDesc = this->ptrDetector;
        }
    }
    void setImage(Mat img1, Mat img2)
    {
        this->image1 = img1.clone();
        this->image2 = img2.clone();
    }
    Mat match(Mat &image1, Mat &image2, vector<DMatch> matches, vector<KeyPoint> &kp1, vector<KeyPoint> &kp2)
    {
        ptrDetector->detect(image1, kp1);
        ptrDetector->detect(image2, kp2);
        Mat desc1, desc2;
        ptrDesc->compute(image1, kp1, desc1);
        ptrDesc->compute(image2, kp2, desc2);
        BFMatcher matcher(normType, true);
        vector<DMatch> outputMatches;
        matcher.match(desc1, desc2, outputMatches);
        Mat fundamental = ransacTest(outputMatches, kp1, kp2, matches);
        return fundamental;
    }
    Mat ransacTest(vector<DMatch> &matches, vector<KeyPoint> &kp1, vector<KeyPoint> &kp2, vector<DMatch> &outMatches)
    {
        vector<Point2f> points1, points2;
        for (vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
        {
            points1.push_back(kp1[it->queryIdx].pt);
            points2.push_back(kp2[it->trainIdx].pt);
        }
        vector<uchar> inlires(points1.size(), 0);
        Mat fundamental = findFundamentalMat(points1, points2, inlires, FM_RANSAC, distance, confidence);
        vector<uchar>::const_iterator itIn = inlires.begin();
        vector<DMatch>::const_iterator itM = matches.begin();
        for (; itIn != inlires.end(); ++itIn, ++itM)
        {
            if (*itIn)
            {
                outMatches.push_back(*itM);
            }
        }
        /*Mat homography = findHomography(points1, points2, inlires, RANSAC, 1.);
        Mat result;
        warpPerspective(image1, result, homography, Size(2 * image1.cols, image1.rows));
        Mat half(result, Rect(0, 0, image2.cols, image2.rows));
        image2.copyTo(half);
        imshow("homography image", result);
        waitKey();*/

        return fundamental;
    }
};

class TargetMatcher
{
private:
    Ptr<FeatureDetector> ptrDetector;
    Ptr<DescriptorExtractor> ptrDesc;
    Mat target;
    int normType;
    double distance;
    int numberOfLevels; // Number of image pyramids
    double scaleFactor; // range between image pyramids;
    vector<Mat> pyramid;
    vector<vector<KeyPoint>> pyKeypoints;
    vector<Mat> pyDesc;

public:
    TargetMatcher(const Ptr<FeatureDetector> &detector,
                  const Ptr<DescriptorExtractor> &descriptor = Ptr<DescriptorExtractor>())
        : ptrDetector(detector), ptrDesc(descriptor), distance(1.0),
          numberOfLevels(8), scaleFactor(0.9)
    {
        if (!this->ptrDesc)
        {
            this->ptrDesc = this->ptrDetector;
        }
    }
    void setNormType(int normType)
    {
        this->normType = normType;
    }
    void setTarget(const Mat img)
    {
        this->target = img;
        createPyramid();
    }
    void createPyramid()
    {
        pyramid.clear();
        Mat layer(target);
        for (int i = 0; i < numberOfLevels; i++)
        {
            pyramid.push_back(target.clone());
            resize(target, target, Size(), scaleFactor, scaleFactor);
        }
        pyKeypoints.clear();
        pyDesc.clear();
        for (int i = 0; i < numberOfLevels; i++)
        {
            pyKeypoints.push_back(vector<KeyPoint>());
            ptrDetector->detect(pyramid[i], pyKeypoints[i]);
            pyDesc.push_back(Mat());
            ptrDesc->compute(pyramid[i], pyKeypoints[i], pyDesc[i]);
        }
    }
    Mat detectTarget(const Mat &image, vector<Point2f> &detectedCorners)
    {
        vector<KeyPoint> keypoints;
        ptrDetector->detect(image, keypoints);
        Mat desc;
        ptrDesc->compute(image, keypoints, desc);
        vector<DMatch> matches;
        Mat bestHomography;
        Size bestSize;
        int maxInliers = 0;
        Mat homography;
        BFMatcher matcher(normType);
        for (int i = 0; i < numberOfLevels; i++)
        {
            matches.clear();
            matcher.match(pyDesc[i], desc, matches);
            vector<DMatch> inliers;
            homography = ransacTest(matches, pyKeypoints[i], keypoints, inliers);
            if (inliers.size() > maxInliers)
            {
                maxInliers = inliers.size();
                bestHomography = homography;
                bestSize = pyramid[i].size();
            }
            Mat imageMatches;
            drawMatches(target, pyKeypoints[i], // 1st image and its keypoints
                        image, keypoints,       // 2nd image and its keypoints
                        inliers,                // the matches
                        imageMatches,           // the image produced
                        Scalar(255, 255, 255),  // color of the lines
                        Scalar(255, 255, 255),  // color of the keypoints
                        std::vector<char>(),
                        2);
            imshow("Target matches", imageMatches);
            waitKey();
        }
        if (maxInliers > 8)
        {
            vector<Point2f> corners;
            corners.push_back(Point2f(0, 0));
            corners.push_back(Point2f(bestSize.width - 1, 0));
            corners.push_back(Point2f(bestSize.width - 1, bestSize.height - 1));
            corners.push_back(Point2f(0, bestSize.height - 1));
            perspectiveTransform(corners, detectedCorners, bestHomography);
        }
        return bestHomography;
    }

    Mat ransacTest(vector<DMatch> &matches, vector<KeyPoint> &kp1, vector<KeyPoint> &kp2, vector<DMatch> &outMatches)
    {
        vector<Point2f> points1, points2;
        for (vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
        {
            points1.push_back(kp1[it->queryIdx].pt);
            points2.push_back(kp2[it->trainIdx].pt);
        }
        vector<uchar> inlire(points1.size(), 0);
        Mat homography = findHomography(points1, points2, inlire, RANSAC, distance);
        vector<uchar>::const_iterator itIn = inlire.begin();
        vector<DMatch>::const_iterator itM = matches.begin();
        cout << inlire.size() << "\t" << matches.size() << endl;
        for (; itIn != inlire.end(); ++itIn, ++itM)
        {
            if (*itIn)
            {
                outMatches.push_back(*itM);
            }
        }
        return homography;
    }
};

int main()
{
    ///2020.10.28
    Mat image1 = imread("/home/czy/Documents/MyCPP/data/test.jpg");
    Mat image2 = imread("/home/czy/Documents/MyCPP/data/output.jpg");
    Ptr<xfeatures2d::SIFT> ptrSift = xfeatures2d::SIFT::create(30);
    vector<KeyPoint> kp1, kp2;
    vector<Point2f> selPoints1, selPoints2;
    vector<int> pointIndex1, pointIndex2;
    ptrSift->detect(image1, kp1);
    ptrSift->detect(image2, kp2);
    KeyPoint::convert(kp1, selPoints1, pointIndex1);
    KeyPoint::convert(kp2, selPoints2, pointIndex2);
    Mat fundamental = findFundamentalMat(selPoints1, selPoints2, FM_7POINT);
    vector<Vec3f> lines;
    computeCorrespondEpilines(selPoints1, 1, fundamental, lines);
    Mat result = image2.clone();
    for (vector<Vec3f>::const_iterator it = lines.begin(); it != lines.end(); ++it)
    {
        line(result, Point(0, -(*it)[2] / (*it)[1]), Point(image2.cols, -((*it)[2] + (*it)[0] * image2.cols) / (*it)[1]), Scalar(255, 255, 255));
    }
    imshow("epilines image", result);
    waitKey();

    ///2020.10.31
    RobustMatcher r(xfeatures2d::SIFT::create(250));
    vector<DMatch> matches2;
    vector<KeyPoint> kps1, kps2, kpp1, kpp2;
    Mat fundamental2 = r.match(image1, image2, matches2, kps1, kps2);
    /*vector<Point2f> np1, np2;
    correctMatches(fundamental2, kps1, kps2, np1, np2);
    Mat result2;
    KeyPoint::convert(np1, kpp1);
    KeyPoint::convert(np2, kpp2);
    drawMatches(image1, kpp1, image2, kpp2, matches2, result2, Scalar(255, 255, 255));
    imshow("ransac image", result2);
    waitKey();*/

    TargetMatcher t(FastFeatureDetector::create(10), BRISK::create());
    t.setNormType(NORM_HAMMING);
    t.setTarget(image1);
    vector<Point2f> corners;
    t.detectTarget(image1, corners);
    if (corners.size() == 4)
    {
        line(image1, Point(corners[0]), Point(corners[1]), Scalar(0, 0, 255), 3);
        line(image1, Point(corners[1]), Point(corners[2]), Scalar(0, 0, 255), 3);
        line(image1, Point(corners[2]), Point(corners[3]), Scalar(0, 0, 255), 3);
        line(image1, Point(corners[3]), Point(corners[0]), Scalar(0, 0, 255), 3);
    }
    namedWindow("Target detection");
    imshow("Target detection", image1);
    waitKey();

    return 0;
}