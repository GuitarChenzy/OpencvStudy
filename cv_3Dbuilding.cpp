#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

class CameraCalibrator
{
    vector<vector<Point3f>> objectPoints;
    vector<vector<Point2f>> imagePoints;
    Mat cameraMatrix;
    Mat distCoeffs;
    Mat map1, map2;
    int flag;
    bool mustInitUndistort;

public:
    CameraCalibrator() : flag(0), mustInitUndistort(true) {}
    int addChessboardPoints(const vector<string> &filelist, Size &boardSize);
    void addPoints(const vector<Point2f> &imageCorners, const vector<Point3f> &objectCorners);
    Mat remap(const Mat &image);
    double calibrate(Size &imageSize);
};

void CameraCalibrator::addPoints(const vector<Point2f> &imageCorners, const vector<Point3f> &objectCorners)
{
    imagePoints.push_back(imageCorners);
    objectPoints.push_back(objectCorners);
}

int CameraCalibrator::addChessboardPoints(const vector<string> &filelist, Size &boardSize)
{
    vector<Point2f> imageCorners;
    vector<Point3f> objectCorners;
    for (int i = 0; i < boardSize.height; i++)
    {
        for (int j = 0; j < boardSize.width; j++)
        {
            objectCorners.push_back(Point3f(i, j, 0.0f));
        }
    }
    Mat image;
    int success = 0;
    for (int i = 0; i < filelist.size(); i++)
    {
        image = imread(filelist[i], 0);
        bool found = findChessboardCorners(image, boardSize, imageCorners);
        if (found)
        {
            cornerSubPix(image, imageCorners, Size(5, 5), Size(-1, -1), TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 30, 0.1));
            if (imageCorners.size() == boardSize.area())
            {
                addPoints(imageCorners, objectCorners);
                success++;
            }
        }
        if (imageCorners.size() == boardSize.area())
        {
            drawChessboardCorners(image, boardSize, imageCorners, found);
            imshow("drwa chess image", image);
            waitKey(100);
        }
    }
    return success;
}

double CameraCalibrator::calibrate(Size &imageSize)
{
    mustInitUndistort = true;
    vector<Mat> rvecs, tvecs;
    return calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flag);
}

Mat CameraCalibrator::remap(const Mat &image)
{
    Mat undistorted;
    if (mustInitUndistort)
    {
        initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), Mat(), image.size(), CV_32FC1, map1, map2);
        mustInitUndistort = false;
    }
    cv::remap(image, undistorted, map1, map2, INTER_LINEAR);
    return undistorted;
};

Vec3d triangulate(const Mat &p1, const Mat &p2, const Vec2d &u1, const Vec2d &u2)
{
    // image=[u,v] , X=[x,y,z,1]
    Matx43d A(u1(0) * p1.at<double>(2, 0) - p1.at<double>(0, 0), u1(0) * p1.at<double>(2, 1) - p1.at<double>(0, 1), u1(0) * p1.at<double>(2, 2) - p1.at<double>(0, 2),
              u1(1) * p1.at<double>(2, 0) - p1.at<double>(1, 0), u1(1) * p1.at<double>(2, 1) - p1.at<double>(1, 1), u1(1) * p1.at<double>(2, 2) - p1.at<double>(1, 2),
              u2(0) * p2.at<double>(2, 0) - p2.at<double>(0, 0), u2(0) * p2.at<double>(2, 1) - p2.at<double>(0, 1), u2(0) * p2.at<double>(2, 2) - p2.at<double>(0, 2),
              u2(1) * p2.at<double>(2, 0) - p2.at<double>(1, 0), u2(1) * p2.at<double>(2, 1) - p2.at<double>(1, 1), u2(1) * p2.at<double>(2, 2) - p2.at<double>(1, 2));
    Matx41d B(p1.at<double>(0, 3) - u1(0) * p1.at<double>(2, 3),
              p1.at<double>(1, 3) - u1(1) * p1.at<double>(2, 3),
              p2.at<double>(0, 3) - u2(0) * p2.at<double>(2, 3),
              p2.at<double>(1, 3) - u2(1) * p2.at<double>(2, 3));
    Vec3d X;
    // solve AX=B
    solve(A, B, X, cv::DECOMP_SVD);
    return X;
}

void triangulate(const Mat &p1, const Mat &p2, const vector<Vec2d> &pts1, const vector<Vec2d> &pts2, vector<Vec3d> &pts3D)
{

    for (int i = 0; i < pts1.size(); i++)
    {

        pts3D.push_back(triangulate(p1, p2, pts1[i], pts2[i]));
    }
}

int main()
{
    /*
    ///2020.10.31
    Mat chess = imread("/home/czy/Documents/MyCPP/data/chessboard.png", 0);
    Mat chess2 = imread("/home/czy/Documents/MyCPP/data/left04.jpg");
    ///2020.11.01
    vector<Point2f> imageCorners;
    Size boardSize(8, 7);
    bool found = findChessboardCorners(chess, boardSize, imageCorners);
    if (found)
    {
        drawChessboardCorners(chess, boardSize, imageCorners, found);
        imshow("chess image", chess);
        waitKey();
    }

    //2020.11.02
    Mat image = imread("/home/czy/dataset/rgbd_dataset_freiburg1_room/rgb/1305031910.765238.png", 0);
    Mat cameraMatrix;
    Mat cameraDistCoeffs;
    FileStorage fs("/home/czy/Documents/MyCPP/calib.xml", cv::FileStorage::READ);
    fs["Intrinsic"] >> cameraMatrix;
    fs["Distortion"] >> cameraDistCoeffs;
    cout << " Camera intrinsic: " << cameraMatrix.rows << "x" << cameraMatrix.cols << std::endl;
    cout << cameraMatrix.at<double>(0, 0) << " " << cameraMatrix.at<double>(0, 1) << " " << cameraMatrix.at<double>(0, 2) << endl;
    cout << cameraMatrix.at<double>(1, 0) << " " << cameraMatrix.at<double>(1, 1) << " " << cameraMatrix.at<double>(1, 2) << endl;
    cout << cameraMatrix.at<double>(2, 0) << " " << cameraMatrix.at<double>(2, 1) << " " << cameraMatrix.at<double>(2, 2) << endl
         << endl;
    cv::Matx33d cMatrix(cameraMatrix);
    viz::Viz3d visualizer("Viz window");
    visualizer.setBackgroundColor(viz::Color::white());
    //Matx33d cMatrix(517.3, 0.0, 318.6, 0.0, 516.5, 255.3, 0.0, 0.0, 1.);
    viz::WCameraPosition cam(cMatrix, image, 30.0, viz::Color::black());
    viz::WCube plane1(Point3f(0.0, 45.0, 0.0), Point3f(242.5, 21.0, -9.0), true, viz::Color::blue());
    plane1.setRenderingProperty(viz::LINE_WIDTH, 4.0);
    viz::WCube plane2(Point3f(0.0, 9.0, -9.0), Point3f(242.5, 0.0, 44.5), true, viz::Color::blue());
    plane2.setRenderingProperty(viz::LINE_WIDTH, 4.0);
    visualizer.showWidget("top", plane1);
    visualizer.showWidget("bottom", plane2);
    visualizer.showWidget("Camera", cam);
    Mat rvec, tvec, rotation;
    vector<Point2f> imagePoints;
    imagePoints.push_back(cv::Point2f(136, 113));
    imagePoints.push_back(cv::Point2f(379, 114));
    imagePoints.push_back(cv::Point2f(379, 150));
    imagePoints.push_back(cv::Point2f(138, 135));
    imagePoints.push_back(cv::Point2f(143, 146));
    imagePoints.push_back(cv::Point2f(381, 166));
    imagePoints.push_back(cv::Point2f(345, 194));
    imagePoints.push_back(cv::Point2f(103, 161));
    vector<Point3f> objectPoints;
    objectPoints.push_back(cv::Point3f(0, 45, 0));
    objectPoints.push_back(cv::Point3f(242.5, 45, 0));
    objectPoints.push_back(cv::Point3f(242.5, 21, 0));
    objectPoints.push_back(cv::Point3f(0, 21, 0));
    objectPoints.push_back(cv::Point3f(0, 9, -9));
    objectPoints.push_back(cv::Point3f(242.5, 9, -9));
    objectPoints.push_back(cv::Point3f(242.5, 9, 44.5));
    objectPoints.push_back(cv::Point3f(0, 9, 44.5));
    solvePnP(objectPoints, imagePoints, cMatrix, cameraDistCoeffs, rvec, tvec);
    Rodrigues(rvec, rotation);
    Affine3d pose(rotation, tvec);
    visualizer.setWidgetPose("top", pose);
    visualizer.setWidgetPose("bottom", pose);
    while (waitKey(100) == -1 && !visualizer.wasStopped())
    {
        visualizer.spinOnce(2000, true);
    }
    */

    //Mat image_01 = imread("/home/czy/dataset/rgbd_dataset_freiburg1_room/rgb/1305031910.765238.png", 0);
    //Mat image_02 = imread("/home/czy/dataset/rgbd_dataset_freiburg1_room/rgb/1305031910.797230.png", 0);
    Mat image_01 = imread("/home/czy/dataset/00/image_0/000000.png", 0);
    Mat image_02 = imread("/home/czy/dataset/00/image_1/000000.png", 0);
    vector<KeyPoint> kp1, kp2;
    Mat desc1, desc2;
    Ptr<Feature2D> ptrFeature2D = xfeatures2d::SURF::create();
    ptrFeature2D->detectAndCompute(image_01, noArray(), kp1, desc1);
    ptrFeature2D->detectAndCompute(image_02, noArray(), kp2, desc2);
    BFMatcher matcher(NORM_L2, true);
    vector<DMatch> matches;
    matcher.match(desc1, desc2, matches);
    vector<Point2f> points1, points2;
    for (vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
    {
        float x = kp1[it->queryIdx].pt.x;
        float y = kp1[it->queryIdx].pt.y;
        points1.push_back(Point2f(x, y));
        x = kp2[it->queryIdx].pt.x;
        y = kp2[it->queryIdx].pt.y;
        points2.push_back(Point2f(x, y));
    }
    Matx33d cMatrix(517.3, 0.0, 318.6, 0.0, 516.5, 255.3, 0.0, 0.0, 1.);
    /*
    Mat inliers, result;
    Mat essential = findEssentialMat(points1, points2, cMatrix, RANSAC, 0.9, 1.0, inliers);
    drawMatches(image_01, kp1, image_02, kp2, matches, result, Scalar::all(-1), Scalar(255, 255, 255), inliers, 2);
    imshow("out", result);
    waitKey();
    Mat rotation, translation;
    recoverPose(essential, points1, points2, cMatrix, rotation, translation, inliers);
    cout << "R: " << rotation << endl
         << "t: " << translation << endl;
    Mat projection(3, 4, CV_64F);
    rotation.copyTo(projection(Rect(0, 0, 3, 3)));
    translation.copyTo(projection.colRange(3, 4));
    Mat projection_normal(3, 4, CV_64F, 0.);
    Mat diag(Mat::eye(3, 3, CV_64F));
    cout << "diag: " << diag << endl;
    diag.copyTo(projection_normal(Rect(0, 0, 3, 3)));
    vector<Vec2d> inliersPts1, inliersPts2;
    int j(0);
    for (int i = 0; i < inliers.rows; i++)
    {
        if (inliers.at<uchar>(i))
        {
            inliersPts1.push_back(Vec2d(points1[i].x, points1[i].y));
            inliersPts2.push_back(Vec2d(points2[i].x, points2[i].y));
        }
    }
    ///2020.11.03
    ///this should undistortPoints, but i did not do it because of the missing distcoeffs >.<
    vector<Vec3d> points3D;
    triangulate(projection_normal, projection, inliersPts1, inliersPts2, points3D);
    viz::Viz3d visualizer2("Viz triangulate window");
    visualizer2.setBackgroundColor(viz::Color::white());
    viz::WCameraPosition cam1(cMatrix, image_01, 1.0, viz::Color::black());
    viz::WCameraPosition cam2(cMatrix, image_02, 1.0, viz::Color::black());
    Vec3d testPoint = triangulate(projection_normal, projection, inliersPts1[124], inliersPts2[124]);
    viz::WSphere point3D(testPoint, 0.05, 10, viz::Color::red());
    double lenght(15.);
    viz::WLine line1(Point3d(0., 0., 0.), Point3d(lenght * inliersPts1[124](0), lenght * inliersPts1[124](1), lenght), viz::Color::green());
    viz::WLine line2(Point3d(0., 0., 0.), Point3d(lenght * inliersPts2[124](0), lenght * inliersPts2[124](1), lenght), viz::Color::green());
    viz::WCloud cloud(points3D, viz::Color::blue());
    cloud.setRenderingProperty(viz::POINT_SIZE, 3.);
    visualizer2.showWidget("Camera1", cam1);
    visualizer2.showWidget("Camera2", cam2);
    visualizer2.showWidget("Cloud", cloud);
    visualizer2.showWidget("Line1", line1);
    visualizer2.showWidget("Line2", line2);
    visualizer2.showWidget("Triangulated", point3D);
    Affine3d pose2(rotation, translation);
    visualizer2.setWidgetPose("Camera2", pose2);
    visualizer2.setWidgetPose("Line2", pose2);

    // visualization loop
    while (waitKey(1000) == -1 && !visualizer2.wasStopped())
    {
        visualizer2.spinOnce(1000,  // pause 1ms
                             true); // redraw
    }
    waitKey();
    */
    Mat inliers2, result2;
    Mat fundamental = findFundamentalMat(points1, points2, inliers2, FM_RANSAC, 1.0, 0.98);
    drawMatches(image_01, kp1, image_02, kp2, matches, result2, Scalar::all(-1), Scalar(255, 255, 255), inliers2, 2);
    imshow("out2", result2);
    waitKey();
    Mat h1, h2;
    stereoRectifyUncalibrated(points1, points2, fundamental, image_01.size(), h1, h2);
    Mat rectified1, rectified2;
    warpPerspective(image_01, rectified1, h1, image_01.size());
    warpPerspective(image_02, rectified2, h2, image_02.size());
    namedWindow("Left Rectified Image");
    imshow("Left Rectified Image", rectified1);
    namedWindow("Right Rectified Image");
    imshow("Right Rectified Image", rectified2);
    waitKey();

    vector<Vec3f> lines1;
    computeCorrespondEpilines(points1, 1, fundamental, lines1);

    for (vector<Vec3f>::const_iterator it = lines1.begin();
         it != lines1.end(); ++it)
    {

        line(image_02, Point(0, -(*it)[2] / (*it)[1]),
             Point(image_02.cols, -((*it)[2] + (*it)[0] * image_02.cols) / (*it)[1]),
             Scalar(255, 255, 255));
    }

    vector<Vec3f> lines2;
    computeCorrespondEpilines(points2, 2, fundamental, lines2);

    for (vector<Vec3f>::const_iterator it = lines2.begin();
         it != lines2.end(); ++it)
    {

        line(image_01, Point(0, -(*it)[2] / (*it)[1]),
             Point(image_01.cols, -((*it)[2] + (*it)[0] * image_01.cols) / (*it)[1]),
             Scalar(255, 255, 255));
    }

    // Display the images with epipolar lines
    namedWindow("Left Epilines");
    imshow("Left Epilines", image_01);
    namedWindow("Right Epilines");
    imshow("Right Epilines", image_02);
    waitKey();

    // draw the pair
    cv::drawMatches(image_01, kp1, // 1st image
                    image_02, kp2, // 2nd image
                    vector<DMatch>(),
                    result2, // the image produced
                    Scalar(255, 255, 255),
                    Scalar(255, 255, 255),
                    vector<char>(),
                    2);
    namedWindow("A Stereo pair");
    imshow("A Stereo pair", result2);

    // Compute disparity
    Mat disparity;
    Ptr<cv::StereoMatcher> pStereo = StereoSGBM::create(0,  // minimum disparity
                                                        32, // maximum disparity
                                                        5); // block size
    pStereo->compute(rectified1, rectified2, disparity);
    double minv, maxv;
    disparity = disparity * 64;
    minMaxLoc(disparity, &minv, &maxv);
    cout << minv << "+" << maxv << endl;
    namedWindow("Disparity Map");
    imshow("Disparity Map", disparity);
    waitKey();
    
    return 0;
}