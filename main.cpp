#include <math.h>
#include <time.h>
#include <unistd.h>

#include <fstream>
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

std::string trim(const std::string& str) {
    std::size_t first = str.find_first_not_of(' ');
    std::size_t last = str.find_last_not_of(' ');

    if (first == std::string::npos || last == std::string::npos) {
        return "";
    }

    return str.substr(first, last - first + 1);
}

int main(int argc, char* argv[]) {
    string img_path = "./images/1.jpg";
    string detect_model_path = "./models/best.onnx";

    // a classes file, for example: "./model/classes.txt";
    // in the file, on classes are listed line after line
    // like: 
    // ```
    //      car
    //      motercycle
    // ```
    string classes = ""; 

    int o = 0;
    while ((o = getopt(argc, argv, "m:f:c:")) != -1) {
        switch (o) {
            case 'c':
                classes = optarg;
                break;
            case 'm':
                detect_model_path = optarg;
                break;
            case 'f':
                img_path = optarg;
                break;
            case '?':
                std::cout << "Usage: Yolov8 " << std::endl;
                std::cout << "-m path/to/model" << std::endl;
                std::cout << "-f path/to/file" << std::endl;
                std::cout << "-c path/to/classes .txt" << std::endl;
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
    std::ifstream ifs(classes);
    if (!classes.empty() && ifs.is_open()) {
        task_detect_onnx._className.clear();
        std::string line;
        while (std::getline(ifs, line)) {
            line = trim(line);
            if (!line.empty()) {
                task_detect_onnx._className.push_back(line);
            }
        }
        ifs.close();
    }
    //	Yolov8SegOnnx task_segment_onnx;

    //	yolov8(task_detect,img,detect_model_path);    //Opencv detect
    //	yolov8(task_segment,img,seg_model_path);   //opencv segment
    yolov8_onnx(task_detect_onnx, img, detect_model_path); // onnxruntime detect
    //	yolov8_onnx(task_segment_onnx,img,seg_model_path); //onnxruntime segment

    return 0;
}
