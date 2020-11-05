#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

double PI = 3.1415926;

class LineFinder
{

private:
    Mat image;
    vector<Vec4i> lines;
    double deltaRho;   //dist
    double deltaTheta; //angle
    int minVote;       //the min vote before confirm the line
    double minLength;  // the min length of line
    double maxGap;     // the allowed max gap of line
public:
    LineFinder() : deltaRho(1), deltaTheta(PI / 180), minVote(10), minLength(0.), maxGap(0.) {}
    void setAccResolution(double dRho, double dTheta)
    {
        deltaRho = dRho;
        deltaTheta = dTheta;
    }
    void setMinVote(int minV)
    {
        minVote = minV;
    }
    void setLineLengthAndGap(double length, double gap)
    {
        minLength = length;
        maxGap = gap;
    }
    vector<Vec4i> findLines(Mat &binary)
    {
        lines.clear();
        HoughLinesP(binary, lines, deltaRho, deltaTheta, minVote, minLength, maxGap);
        return lines;
    }
    void drawDetectedLines(Mat &image, Scalar color = Scalar(255, 255, 255))
    {
        vector<Vec4i>::const_iterator it = lines.begin();
        while (it != lines.end())
        {
            Point pt1((*it)[0], (*it)[1]);
            Point pt2((*it)[2], (*it)[3]);
            line(image, pt1, pt2, color, 2);
            ++it;
        }
    }
};

int main()
{
    ///2020.10.25
    Mat image_gary = imread("/home/czy/Documents/MyCPP/data/building.jpg", 0);
    Mat image_color = imread("/home/czy/Documents/MyCPP/data/building.jpg", 1);

    Mat contours;
    Canny(image_gary, contours, 125, 350);
    imshow("canny image", contours);
    waitKey();

    Mat result = image_gary.clone();
    vector<Vec2f> lines;
    HoughLines(contours, lines, 1, PI / 180, 60);
    vector<Vec2f>::const_iterator it = lines.begin();
    while (it != lines.end())
    {
        float rho = (*it)[0];   // distance
        float theta = (*it)[1]; //angle
        if (theta < PI / 4. || theta > 3. * PI / 4.)
        {
            Point pt1(rho / cos(theta), 0);
            Point pt2((rho - result.rows * sin(theta)) / cos(theta), result.rows);
            line(result, pt1, pt2, Scalar(255), 1);
        }
        else
        {
            Point pt1(0, rho / sin(theta));
            Point pt2(result.cols, (rho - result.cols * cos(theta)) / sin(theta));
            line(result, pt1, pt2, Scalar(255), 1);
        }
        ++it;
    }
    imshow("hough lines", result);
    waitKey();

    LineFinder l;
    Mat result = image_color.clone();
    l.setLineLengthAndGap(100, 20);
    l.setMinVote(50);
    vector<Vec4i> lines = l.findLines(contours);
    l.drawDetectedLines(result, Scalar(0, 0, 255));
    imshow("hough_P lines", result);
    waitKey();

    Mat image_magic = imread("/home/czy/Documents/MyCPP/data/board.jpg", 0);
    GaussianBlur(image_magic, image_magic, Size(5, 5), 1.5);
    vector<Vec3f> circles;
    HoughCircles(image_magic, circles, HOUGH_GRADIENT, 2, 50, 200, 100, 25, 100);
    vector<Vec3f>::const_iterator it2 = circles.begin();
    while (it2 != circles.end())
    {
        circle(image_magic, Point((*it2)[0], (*it2)[1]), (*it2)[2], Scalar(255), 2);
        ++it2;
    }
    imshow("hough circles", image_magic);
    waitKey();

    int n = 0;
    Mat online(contours.size(), CV_8U, Scalar(0));
    line(online, Point(lines[n][0], lines[n][1]), Point(lines[n][2], lines[n][3]), Scalar(255), 3);
    bitwise_and(contours, online, online);
    imshow("fit line & point", online);
    waitKey();

    vector<Point> points;
    Mat result2 = image_color.clone();
    for (int y = 0; y < online.rows; y++)
    {
        uchar *rowPtr = online.ptr<uchar>(y);
        for (int x = 0; x < online.cols; x++)
        {
            if (rowPtr[x])
            {
                points.push_back(Point(x, y));
            }
        }
    }
    Vec4f line;
    fitLine(points, line, DIST_L2, 0, 0.01, 0.01);
    int x0 = line[2];
    int y0 = line[3];
    int x1 = x0 + 100 * line[0];
    int y1 = y0 + 100 * line[1];
    cv::line(result2, Point(x0, y0), Point(x1, y1), Scalar(0, 0, 255), 2);
    imshow("fit line", result2);
    waitKey();

    Mat plate_g = imread("/home/czy/Documents/MyCPP/data/plate.jpg", 0);
    Mat plate_c = imread("/home/czy/Documents/MyCPP/data/plate.jpg", 1);
    //cout << plate.empty() <<endl;
    Mat result3;
    threshold(plate_g, result3, 80, 255, THRESH_BINARY);
    dilate(result3, result3, Mat(), Point(-1, -1), 1);
    //erode(result3, result3, Mat(), Point(-1, -1), 1);
    imshow("result3", result3);
    waitKey();

    vector<vector<Point>> contours2;
    findContours(result3, contours2, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    cout << "contours number : " << contours2.size() << endl;
    Mat result4(result3.size(), CV_8U, Scalar(255));
    drawContours(result4, contours2, -1, 0, 2);
    vector<vector<Point>>::iterator it3 = contours2.begin();
    while (it3 != contours2.end())
    {
        Rect r0 = boundingRect(*it3);
        if (r0.area() > 200)
        {
            cout << r0.width / r0.height <<endl;
            rectangle(plate_c, r0, Scalar(0, 0, 255), 2);
        }
        ++it3;
    }
    imshow("contours2 image", plate_c);
    waitKey();

    return 0;
}