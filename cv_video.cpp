#include <iostream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/bgsegm.hpp>

using namespace std;
using namespace cv;

class FrameProcessor
{
public:
    virtual void process2(Mat &input, Mat &output) = 0;
};

class VideoProcessor
{
private:
    VideoCapture capture;
    VideoWriter vwriter;
    string outputFile;
    int currentIndex;
    int digits;
    string extension;

    FrameProcessor *frameProcessor;
    Mat prevFrame;
    void (*process)(Mat &, Mat &, Mat &, Mat &); // callback function
    bool callIt;                                 // mean whether callback do
    string winNameInput, winNameOutput, winNameFeature;
    int delay;
    long fnumber, frameToStop;
    bool stop;

    vector<string> images;
    vector<string>::const_iterator itImg;

public:
    VideoProcessor() {}
    VideoProcessor(string winInput, string winOutput, string winFeature) : winNameInput(winInput), winNameOutput(winOutput), winNameFeature(winFeature) {}


    void setFrameProcessor(void (*frameProcessingCallback)(Mat &, Mat &, Mat &, Mat &))
    {
        frameProcessor = 0;
        process = frameProcessingCallback;
    }

    void setFrameProcessor(FrameProcessor *frameProcessorPtr)
    {
        process = 0;
        frameProcessor = frameProcessorPtr;
        //callProcess(true);
    }

    bool setInput(string filename)
    {
        fnumber = 0;
        capture.release();
        if (capture.open(filename))
        {
            frameToStop = capture.get(CV_CAP_PROP_FRAME_COUNT);
            return true;
        }
        return false;
    }

    bool setInput(const vector<string> &imgs)
    {
        fnumber = 0;
        capture.release();
        images.clear();
        images = imgs;
        itImg = images.begin();
        frameToStop = images.size();
        return true;
    }

    bool isOpened()
    {
        return capture.isOpened() || !images.empty();
    }

    bool readNextFrame(Mat &frame)
    {
        if (images.size() == 0)
            return capture.read(frame);
        else
        {
            if (itImg != images.end())
            {
                frame = imread(*itImg);
                itImg++;
                return frame.data != 0;
            }
            else
            {
                return false;
            }
        }
    }

    void displayInput(string wn)
    {
        winNameInput = wn;
        namedWindow(winNameInput);
    }

    void displayOutput(string wn)
    {
        winNameOutput = wn;
        namedWindow(winNameOutput);
    }

    void displayFeature(string wn)
    {
        winNameFeature = wn;
        namedWindow(winNameFeature);
    }

    long getFrameNumber()
    {
        if (images.size() != 0)
            return images.size();
        long fnumber = static_cast<long>(capture.get(CV_CAP_PROP_POS_FRAMES));
        return fnumber;
    }

    bool setFrameNumber(long pos)
    {
        if (images.size() != 0)
        {

            itImg = images.begin() + pos;
            if (pos < images.size())
                return true;
            else
                return false;
        }
        else
        {
            return capture.set(cv::CAP_PROP_POS_FRAMES, pos);
        }
    }

    void setDelay(int d)
    {
        delay = d;
    }

    void callProcess(bool flag)
    {
        callIt = flag;
    }

    void stopAtFrame(long frame)
    {
        frameToStop = frame;
    }

    double getFrameRate()
    {
        return capture.get(CV_CAP_PROP_FPS);
    }

    Size getFrameSize()
    {
        if (images.size() == 0)
        {
            int w = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
            int h = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
            return Size(w, h);
        }
        else
        {
            Mat tmp = imread(images[0]);
            if (!tmp.data)
                return Size(0, 0);
            else
                return tmp.size();
        }
    }

    bool setOutput(const string &filename, int codec = 0, double framerate = 0.0, bool isColor = false)
    {
        outputFile = filename;
        extension.clear();
        if (framerate == 0.0)
            framerate = getFrameRate();
        char c[4];
        if (codec == 0)
            codec = getCodec(c);
        return vwriter.open(outputFile, codec, framerate, getFrameSize(), isColor);
    }

    int getCodec(char codec[4])
    {
        // undefined for vector of images
        if (images.size() != 0)
            return -1;

        union
        {
            int value;
            char code[4];
        } returned;

        returned.value = static_cast<int>(capture.get(cv::CAP_PROP_FOURCC));

        codec[0] = returned.code[0];
        codec[1] = returned.code[1];
        codec[2] = returned.code[2];
        codec[3] = returned.code[3];

        return returned.value;
    }

    void writeNextFrame(Mat &frame)
    {
        if (extension.length())
        {
            stringstream ss;
            ss << outputFile << setfill('0') << setw(digits) << currentIndex++ << extension;
            imwrite(ss.str(), frame);
        }
        else
        {
            vwriter.write(frame);
        }
    }

    void run()
    {
        Mat frame, output, outfea;
        if (!isOpened())
            return;
        stop = false;
        while (!stop)
        {
            if (!readNextFrame(frame))
                break;
            if (winNameInput.length() != 0)
                imshow(winNameInput, frame);
            if (callIt)
            {
                if (process)
                {
                    process(frame, output, outfea, prevFrame);
                    fnumber++;
                    //if (fnumber != 0)
                    //prevFrame = frame;
                }
                else if (frameProcessor)
                {
                    frameProcessor->process2(frame, output);
                    fnumber++;
                }
            }
            else
            {
                output = frame;
                outfea = frame;
            }
            if (outputFile.length() != 0)
                writeNextFrame(output);
            if (winNameOutput.length() != 0)
                imshow(winNameOutput, output);
            if (winNameFeature.length() != 0 && !outfea.empty())
                imshow(winNameFeature, outfea);
            if (delay >= 0 && waitKey(delay) >= 0)
                stop = true;
            if (frameToStop >= 0 && fnumber == frameToStop)
                stop = true;
        }
    }
};

void processFrame(Mat &img, Mat &out, Mat &fea, Mat &prev)
{
    if (img.channels() == 3)
    {
        cvtColor(img, out, COLOR_BGR2GRAY);
        cvtColor(img, fea, COLOR_BGR2GRAY);
    }
    Canny(out, out, 100, 200);
    threshold(out, out, 128, 255, THRESH_BINARY);

    if (!fea.empty())
    {
        Ptr<Feature2D> ptrDetector = xfeatures2d::SIFT::create(50);
        //Ptr<Feature2D> ptrDetector = ORB::create();
        //Ptr<Feature2D> ptrDetector = xfeatures2d::SURF::create(100);
        vector<KeyPoint> kp1, kp2;
        Mat desc1, desc2;
        if (prev.empty())
        {
            ptrDetector->detect(fea, kp1);
            drawKeypoints(fea, kp1, fea);
        }
        else
        {
            GaussianBlur(img, img, Size(5, 5), 1.5);
            GaussianBlur(prev, prev, Size(5, 5), 1.5);
            ptrDetector->detectAndCompute(img, noArray(), kp1, desc1);
            ptrDetector->detectAndCompute(prev, noArray(), kp2, desc2);
            BFMatcher matcher(NORM_L2, true);
            vector<DMatch> matches;
            matcher.match(desc1, desc2, matches);
            drawMatches(img, kp1, prev, kp2, matches, fea, Scalar::all(-1), Scalar(255, 255, 255));
        }
    }
}

class BGFGSegmentor : public FrameProcessor
{
private:
    Mat gray, background, backImage, foreground;
    double learningRate;
    int threshold_v;

public:
    void setThreshold(int thres)
    {
        threshold_v = thres;
    }

    void process2(Mat &frame, Mat &output)
    {
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        if (background.empty())
        {
            gray.convertTo(background, CV_32F);
        }
        background.convertTo(backImage, CV_8U);
        absdiff(backImage, gray, foreground);
        threshold(foreground, output, threshold_v, 255, THRESH_BINARY_INV);
        accumulateWeighted(gray, background, learningRate, output);
    }
};

/*int main()
{
    ///2020.11.03
    /*VideoCapture capture("/home/czy/Documents/MyCPP/data/Megamind.avi");
    if (!capture.isOpened())
        return 1;
    double rate = capture.get(CV_CAP_PROP_FPS);
    long t = static_cast<long>(capture.get(CV_CAP_PROP_FRAME_COUNT));
    cout << "the number of frame: " << t << endl;
    bool stop(false);
    Mat frame;
    namedWindow("Extracted Frame");
    int delay = 1000 / rate;
    while (!stop)
    {
        if (!capture.read(frame))
            break;
        imshow("Extracted Frame", frame);
        if (waitKey(delay) >= 0)
            stop = true;
    }
    capture.release();*/

///2020.11.04
/*VideoProcessor v("input image", "output image", "feature image");
    BGFGSegmentor b;
    b.setThreshold(25);
    v.setInput("/home/czy/Documents/MyCPP/data/Megamind.avi");
    v.setDelay(1000. / v.getFrameRate());
    v.setFrameProcessor(&b);
    //v.setFrameProcessor(f);
    v.callProcess(true);
    //v.setOutput("/home/czy/Documents/MyCPP/data/Megamind_output.avi");
    v.run();*/

///2020.11.05
/*VideoCapture capture("/home/czy/Documents/MyCPP/data/vtest.avi");
    if (!capture.isOpened())
        return 0;
    double rate = capture.get(CV_CAP_PROP_FPS);
    int delay = 1000 / rate;
    Mat frame, foreground, background;
    namedWindow("Extract Foreground");
    Ptr<BackgroundSubtractor> ptrMOG = bgsegm::createBackgroundSubtractorMOG();
    bool stop(false);
    while (!stop)
    {
        if (!capture.read(frame))
            break;
        ptrMOG->apply(frame, foreground, 0.01);
        threshold(foreground, foreground, 128, 255, THRESH_BINARY_INV);
        imshow("Original Frame", frame);
        imshow("Extract Foreground", foreground);
        if (waitKey(delay) >= 0)
        {
            stop = true;
        }
    }
    return 0;
}
*/