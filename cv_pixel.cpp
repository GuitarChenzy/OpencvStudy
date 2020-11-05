#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void salt(Mat image, int n);
void colorReduce(const Mat &image, Mat &result, int div);
void sharpen(const Mat &image, Mat &result);
void wave(const Mat &image, Mat &result);

int main()
{
    ///2020.10.21
    string img_path = "/home/czy/Documents/MyCPP/data/test.jpg";
    Mat image_gary = imread(img_path, 0);
    Mat image_color = imread(img_path, 1);
    waitKey();
    cout << "color image fill : " << image_color.isContinuous() << endl;
    cout << "gary image fill : " << image_gary.isContinuous() << endl;

    Mat result;
    sharpen(image_color, result);
    imshow("sharpen image", result);
    waitKey();

    Mat add_img = imread("/home/czy/Documents/MyCPP/data/output.jpg", 1);
    Mat result2;
    addWeighted(image_color, 0.7, add_img, 0.9, 0.5, result2);
    imshow("add image", result2);
    waitKey();

    vector<Mat> planes;
    Mat result3;
    split(image_color, planes);
    planes[0] += image_gary;
    merge(planes, result3);
    imshow("split and merge image", result3);
    waitKey();

    Mat result4;
    wave(image_color, result4);
    imshow("reject image", result4);
    waitKey();
    
    Mat image = image_color.clone();
    salt(image_gary, 3000);
    salt(image_color, 3000);
    const int64 start = getTickCount();
    colorReduce(image, image, 64);
    double duration = (getTickCount() - start) / getTickFrequency();
    cout << "cost time : " << duration << endl;
    //Mat_<uchar> img(image_gary);
    //img(50,100) = 0;
    imshow("gray salt", image_gary);
    imshow("color salt", image_color);
    imshow("reduce image", image);
    cout << image.at<Vec3b>(100, 100) << endl;
    waitKey();

    return 0;
}

void salt(Mat image, int n)
{
    default_random_engine generator;
    uniform_int_distribution<int> randomRow(0, image.rows - 1);
    uniform_int_distribution<int> randomCol(0, image.cols - 1);
    int i, j;
    for (int k = 0; k < n; k++)
    {
        i = randomCol(generator);
        j = randomRow(generator);
        if (image.type() == CV_8UC1)
        {
            image.at<uchar>(j, i) = 255;
        }
        else if (image.type() == CV_8UC3)
        {
            image.at<Vec3b>(j, i) = Vec3b(255, 255, 255);
        }
    }
}

void colorReduce(const Mat &image, Mat &result, int div)
{
    int nl = image.rows;
    int nc = image.cols * image.channels();
    int n = static_cast<int>(log(static_cast<double>(div) / log(2.0) + 0.5));
    uchar mask = 0xFF << n;
    int div2 = div >> 1;
    cout << "sum of pixel at row : " << nc << endl;
    //Mat_<Vec3b>::iterator it = result.begin<Vec3b>();
    //Mat_<Vec3b>::iterator end = result.end<Vec3b>();
    for (int j = 0; j < nl; j++)
    {
        const uchar *data_in = image.ptr<uchar>(j);
        uchar *data_out = result.ptr<uchar>(j);
        for (int i = 0; i < nc; i++)
        {
            //data_out[i] = data_in[i] / div * div + div / 2;
            *data_out &= mask;
            *data_out++ |= div2;
        }
    }
}

void sharpen(const Mat &image, Mat &result)
{
    result.create(image.size(), image.type());
    /*int chan = image.channels();
    for (int j = 1; j < image.rows - 1; j++)
    {
        const uchar *pervious = image.ptr<const uchar>(j - 1);
        const uchar *current = image.ptr<const uchar>(j);
        const uchar *next = image.ptr<const uchar>(j + 1);
        uchar *output = result.ptr<uchar>(j);
        for (int i = chan; i < (image.cols - 1) * chan; i++)
        {
            *output++ = saturate_cast<uchar>(5 * current[i] - current[i - chan] - current[i + chan] - pervious[i] - next[i]);
            //*output++ = saturate_cast<uchar>(10 * current[i] - current[i - chan] - current[i + chan] - pervious[i] - pervious[i - chan] - pervious[i + chan] - next[i] - next[i - chan] - next[i + chan]);
        }
    }
    result.row(0).setTo(Scalar(0));
    result.row(result.rows - 1).setTo(Scalar(0));
    result.col(0).setTo(Scalar(0));
    result.col(result.cols - 1).setTo(Scalar(0));*/

    Mat kernel(3, 3, CV_32F, Scalar(0));
    kernel.at<float>(1, 1) = 5.0;
    kernel.at<float>(0, 1) = -1.0;
    kernel.at<float>(2, 1) = -1.0;
    kernel.at<float>(1, 0) = -1.0;
    kernel.at<float>(1, 2) = -1.0;
    filter2D(image, result, image.depth(), kernel);
    //cvFilter2D(image, result, kernel);
}

void wave(const Mat &image, Mat &result)
{
    Mat srcX(image.rows, image.cols, CV_32F);
    Mat srcY(image.rows, image.cols, CV_32F);
    size_t r = image.rows;
    size_t c = image.cols;
    for (size_t i = 0; i < r; i++)
    {
        for (size_t j = 0; j < c; j++)
        {
            srcX.at<float>(i, j) = j + 5;
            srcY.at<float>(i, j) = i + 5 * sin(j / 10.0);
        }
    }
    remap(image, result, srcX, srcY, INTER_LINEAR);
}