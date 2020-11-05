#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

void matchHamming(const Mat &image1, const Mat &image2);

int main()
{
    ///2020.10.26
    Mat image1 = imread("/home/czy/Documents/MyCPP/data/test.jpg");
    Mat image2 = imread("/home/czy/Documents/MyCPP/data/output.jpg");
    Ptr<FeatureDetector> ptrDetector = ORB::create(100);
    vector<KeyPoint> keypoints1, keypoints2;
    ptrDetector->detect(image1, keypoints1);
    ptrDetector->detect(image2, keypoints2);
    const int nsize(11); // domain size
    Rect neighborhood(0, 0, nsize, nsize);
    Mat patch1, patch2;
    Mat result;
    vector<DMatch> matches;
    for (size_t i = 0; i < keypoints1.size(); i++)
    {
        neighborhood.x = keypoints1[i].pt.x - nsize / 2;
        neighborhood.y = keypoints1[i].pt.y - nsize / 2;
        if (neighborhood.x < 0 || neighborhood.y < 0 || neighborhood.x + nsize >= image1.cols || neighborhood.y + nsize >= image2.rows)
            continue;                  //out of the image,do next point
        patch1 = image1(neighborhood); // patch of ths first image
        DMatch bestMatch;
        for (size_t j = 0; j < keypoints2.size(); j++)
        {
            neighborhood.x = keypoints2[j].pt.x - nsize / 2;
            neighborhood.y = keypoints2[j].pt.y - nsize / 2;
            if (neighborhood.x < 0 || neighborhood.y < 0 || neighborhood.x + nsize >= image1.cols || neighborhood.y + nsize >= image2.rows)
                continue;
            patch2 = image2(neighborhood); // patch of ths second image
            matchTemplate(patch1, patch2, result, TM_SQDIFF_NORMED);
            if (result.at<float>(0, 0) < bestMatch.distance)
            {
                bestMatch.distance = result.at<float>(0, 0);
                bestMatch.queryIdx = i;
                bestMatch.trainIdx = j;
            }
        }
        result.at<float>(0, 0) = 0.0;
        matches.push_back(bestMatch);
    }
    nth_element(matches.begin(), matches.begin() + 25, matches.end());
    //nth_element(s,p,e): s < p,p < e ;p location not move and [s,p] , [p,e]  unsort
    matches.erase(matches.begin() + 25, matches.end());
    //remove [25,end] elements
    Mat mathImage;
    drawMatches(image1, keypoints1, image2, keypoints2, matches, mathImage, Scalar(255, 255, 255), Scalar::all(-1));
    imshow("mathes image", mathImage);
    waitKey();

    ///2020.10.27
    vector<KeyPoint> keypoints3, keypoints4;
    Ptr<Feature2D> ptrFeature2D = xfeatures2d::SURF::create(2000.0);
    ptrFeature2D->detect(image1, keypoints3);
    ptrFeature2D->detect(image2, keypoints4);
    Mat desc1, desc2;
    ptrFeature2D->compute(image1, keypoints3, desc1);
    ptrFeature2D->compute(image2, keypoints3, desc2);
    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches2;
    matcher.match(desc1, desc2, matches2);
    Mat mathImage2;
    drawMatches(image1, keypoints3, image2, keypoints4, matches2, mathImage2, Scalar(255, 255, 255));
    //ptrFeature2D->detectAndCompute(image1,noArray(),keypoints3,desc1);
    imshow("mathes desc image", mathImage2);
    waitKey();

    vector<vector<DMatch>> matches3;
    vector<DMatch> new_matches;
    matcher.knnMatch(desc1, desc2, matches3, 2); //before use this function ,please set BFMatcher with crossCheck = True
    double ratio = 0.85;
    vector<vector<DMatch>>::iterator it;
    for (it = matches3.begin(); it != matches3.end(); ++it)
    {
        if ((*it)[0].distance / (*it)[1].distance < ratio)
        {
            new_matches.push_back((*it)[0]);
        }
    }
    Mat mathImage3;
    drawMatches(image1, keypoints3, image2, keypoints4, new_matches, mathImage3, Scalar(255, 255, 255));
    imshow("mathes desc ration image", mathImage2);
    waitKey();

    float maxDist = 0.4;
    vector<vector<DMatch>> matches4;
    vector<DMatch> new_matches2;
    matcher.radiusMatch(desc1, desc2, matches4, maxDist);
    vector<vector<DMatch>>::iterator it2;
    for (it2 = matches4.begin(); it2 != matches4.end(); ++it2)
    {
        vector<DMatch> dm = (*it2);
        if (dm.empty())
            continue; // some DMatch is empty, must code this line.
        float dist_img = 10000.0;
        int inedx_dm = 0;
        for (size_t i = 0, len = dm.size(); i < len; i++)
        {
            if (dm[i].distance < dist_img && dm[i].distance != 0)
            {
                dist_img = dm[i].distance;
                inedx_dm = i;
            }
        }
        new_matches2.push_back((*it2)[inedx_dm]);
    }
    Mat mathImage4;
    drawMatches(image1, keypoints3, image2, keypoints4, new_matches2, mathImage4, Scalar(255, 255, 255));
    imshow("mathes desc threshold image", mathImage4);
    waitKey();

    matchHamming(image1, image2);

    return 0;
}

void matchHamming(const Mat &image1, const Mat &image2)
{
    vector<KeyPoint> kp1, kp2;
    Mat desc1, desc2, outImage;
    Ptr<Feature2D> ptrORB = ORB::create(100);
    ptrORB->detectAndCompute(image1, noArray(), kp1, desc1);
    ptrORB->detectAndCompute(image2, noArray(), kp2, desc2);
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(desc1, desc2, matches);
    drawMatches(image1, kp1, image2, kp2, matches, outImage, Scalar(255, 255, 255));
    imshow("ORB desc image", outImage);
    waitKey();
}