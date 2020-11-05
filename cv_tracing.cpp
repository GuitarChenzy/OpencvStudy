#include <iostream>
#include <sys/types.h>
#include <dirent.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/tracking.hpp>

#include "cv_video.cpp"

using namespace std;
using namespace cv;

class FeatureTracker : public FrameProcessor
{
    Mat gray, gray_prev;
    vector<Point2f> points[2]; // the feature value from 0 to 1;
    vector<Point2f> inital;    // the original position of the featured points
    vector<Point2f> features;  // the featured points
    int max_count;
    double qlevel; // the quality level of the featured points
    double minDist;
    vector<uchar> status; // the status of the featured points
    vector<float> err;

public:
    FeatureTracker() : max_count(100), qlevel(0.1), minDist(10.) {}

    void process2(Mat &frame, Mat &output)
    {
        cvtColor(frame, gray, CV_BGR2GRAY);
        frame.copyTo(output);
        if (addNewPoints())
        {
            detectFeaturePoints();
            points[0].insert(points[0].end(), features.begin(), features.end());
            inital.insert(inital.end(), features.begin(), features.end());
        }
        if (gray_prev.empty())
            gray.copyTo(gray_prev);
        calcOpticalFlowPyrLK(gray_prev, gray, points[0], points[1], status, err);

        int k = 0;
        for (int i = 0; i < points[1].size(); i++)
        {
            if (acceptTrackedPoint(i))
            {
                inital[k] = inital[i];
                points[1][k++] = points[1][i];
            }
        }
        points[1].resize(k);
        inital.resize(k);
        handleTrackedPoints(frame, output);
        swap(points[1], points[0]);
        swap(gray_prev, gray);
    }

    bool addNewPoints()
    {
        return points[0].size() <= 10;
    }

    void detectFeaturePoints()
    {
        goodFeaturesToTrack(gray, features, max_count, qlevel, minDist);
    }

    bool acceptTrackedPoint(int i)
    {
        return status[i] && (abs(points[0][i].x - points[1][i].x) +
                                 (abs(points[0][i].y - points[1][i].y)) >
                             2);
    }

    void handleTrackedPoints(Mat &frame, Mat &output)
    {
        for (int i = 0; i < points[1].size(); i++)
        {
            line(output, inital[i], points[1][i], Scalar(255, 255, 255));
            circle(output, points[1][i], 3, Scalar::all(-1), 2);
        }
    }
};

void GetFileNames(string path, vector<string> &filenames)
{
    DIR *pDir;
    struct dirent *ptr;
    if (!(pDir = opendir(path.c_str())))
        return;
    while ((ptr = readdir(pDir)) != 0)
    {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
            filenames.push_back(path + "/" + ptr->d_name);
    }
    closedir(pDir);
}

void drawOpticalFlow(const Mat &oflow, Mat &flowImage, int stride, float scale, const Scalar &color)
{
    if (flowImage.size() != oflow.size())
    {
        flowImage.create(oflow.size(), CV_8UC3);
        flowImage = cv::Vec3i(255, 255, 255);
    }

    for (int y = 0; y < oflow.rows; y += stride)
    {
        for (int x = 0; x < oflow.cols; x += stride)
        {
            Point2f vector_p = oflow.at<Point2f>(y, x);
            line(flowImage, Point(x, y), Point(static_cast<int>(x + scale * vector_p.x + 0.5), static_cast<int>(y + scale * vector_p.y + 0.5)), color);
            circle(flowImage, Point(static_cast<int>(x + scale * vector_p.x + 0.5), static_cast<int>(y + scale * vector_p.y + 0.5)), 1, color, -1);
        }
    }
}

class VisualTracker : public FrameProcessor
{
    Ptr<Tracker> tracker;
    Rect2d box;
    bool reset;

public:
    VisualTracker(Ptr<Tracker> tracker) : reset(true), tracker(tracker) {}

    void setBoundingBox(const Rect2d &b)
    {
        box = b;
        reset = true;
    }

    void process2(Mat &frame, Mat &output)
    {
        if (reset)
        {
            reset = false;
            tracker->init(frame, box);
        }
        else
        {
            tracker->update(frame, box);
        }
        frame.copyTo(output);
        rectangle(output, box, Scalar(255, 255, 255), 2);
    }
};

int main()
{

    ///2020.11.05
    /*
    //vector<string> file_name;
    //string path = "/home/czy/dataset/rgbd_dataset_freiburg1_room/rgb/";
    //GetFileNames(path, file_name);
    VideoProcessor v("input image", "output image", "");
    FeatureTracker f;
    v.setInput("/home/czy/Documents/MyCPP/data/vtest.avi");
    //v.setInput(file_name);
    v.setFrameProcessor(&f);
    v.setDelay(1000/v.getFrameRate());
    v.callProcess(true);
    v.run();*/

    /*Ptr<DualTVL1OpticalFlow> tvl1 = createOptFlow_DualTVL1();
    Mat oflow;
    Mat f1, f2;
    f1 = imread("/home/czy/dataset/rgbd_dataset_freiburg1_room/rgb/1305031910.765238.png", 0);
    f2 = imread("/home/czy/dataset/rgbd_dataset_freiburg1_room/rgb/1305031910.797230.png", 0);
    Mat combined(f1.rows, f1.cols + f2.cols, CV_8U);
    f1.copyTo(combined.colRange(0, f1.cols));
    f2.copyTo(combined.colRange(f1.cols, f1.cols + f2.cols));
    cv::imshow("Frames", combined);
    tvl1->calc(f1, f2, oflow);
    Mat flowImage;
    drawOpticalFlow(oflow, flowImage, 8, 2, Scalar(0, 0, 0));
    imshow("flow image", flowImage);
    waitKey();
    */

    VideoProcessor v("input image", "output image", "");
    VisualTracker t(TrackerMedianFlow::create());
    v.setInput("/home/czy/Documents/MyCPP/data/vtest.avi");
    v.setFrameProcessor(&t);
    v.setDelay(1000 / v.getFrameRate());
    t.setBoundingBox(Rect(250, 225, 50, 65));
    v.run();

    return 0;
}