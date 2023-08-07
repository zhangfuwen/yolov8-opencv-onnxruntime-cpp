#include <math.h>
#include <time.h>
#include <unistd.h>

#include <iostream>
#include <opencv2/opencv.hpp>

#include "yolov8.h"
#include "yolov8_onnx.h"
#include "yolov8_seg.h"
#include "yolov8_seg_onnx.h"

using namespace std;
using namespace cv;
using namespace dnn;

template <typename _Tp>
int yolov8(_Tp& cls, Mat& img, string& model_path) {
    Net net;
    cout << "read model from " << model_path << endl;
    if (cls.ReadModel(net, model_path, false)) {
        cout << "read net ok!" << endl;
    } else {
        return -1;
    }
    // 生成随机颜色
    vector<Scalar> color;
    srand(time(0));
    for (int i = 0; i < 80; i++) {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(Scalar(b, g, r));
    }
    vector<OutputSeg> result;

    if (cls.Detect(img, net, result)) {
        for (const auto& seg : result) {
            cout << "seg " << seg.id << ", " << seg.confidence << endl;
        }
        DrawPred(img, result, cls._className, color);
    } else {
        cout << "Detect Failed!" << endl;
    }
    system("pause");
    return 0;
}

template <typename _Tp>
int yolov8_onnx(_Tp& cls, Mat& img, string& model_path) {
    cout << "read model from " << model_path << endl;
    if (cls.ReadModel(model_path, false)) {
        cout << "read net ok!" << endl;
    } else {
        return -1;
    }
    // 生成随机颜色
    vector<Scalar> color;
    srand(time(0));
    for (int i = 0; i < 80; i++) {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(Scalar(b, g, r));
    }
    vector<OutputSeg> result;
    if (cls.OnnxDetect(img, result)) {
        for (const auto& seg : result) {
            cout << "seg " << seg.id << ", " << seg.confidence << endl;
        }
        DrawPred(img, result, cls._className, color);
    } else {
        cout << "Detect Failed!" << endl;
    }
    system("pause");
    return 0;
}

int main(int argc, char* argv[]) {
    string img_path = "./images/1.jpg";
    string detect_model_path = "./models/best.onnx";

    int o = 0;
    while ((o = getopt(argc, argv, "m:f:")) != -1) {
        switch (o) {
            case 'm':
                detect_model_path = optarg;
                break;
            case 'f':
                img_path = optarg;
                break;
            case '?':
                std::cout << "Usage: Yolov8 -m path/to/model -f path/to/file" << std::endl;
                return 0;
        }
    }

    std::cout << "image path: " << img_path << std::endl;
    std::cout << "model path: " << detect_model_path << std::endl;

    //	string seg_model_path = "./models/yolov8s-seg.onnx";
    //	string detect_model_path = "./models/yolov8n.onnx";
    Mat img = imread(img_path);

    //	Yolov8 task_detect;
    //	Yolov8Seg task_segment;
    Yolov8Onnx task_detect_onnx;
    //	Yolov8SegOnnx task_segment_onnx;

    //	yolov8(task_detect,img,detect_model_path);    //Opencv detect
    //	yolov8(task_segment,img,seg_model_path);   //opencv segment
    yolov8_onnx(task_detect_onnx, img, detect_model_path); // onnxruntime detect
    //	yolov8_onnx(task_segment_onnx,img,seg_model_path); //onnxruntime segment

    return 0;
}
