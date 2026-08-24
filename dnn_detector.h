// dnn_detector.h
// New header for YOLOv8 DNN detection with auto GPU backend selection

#pragma once

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>

struct Detection {
    cv::Rect box;           // Bounding box
    float confidence;       // Confidence score
    int classId;            // Class ID (0 = person, etc.)
};

class DnnDetector {
public:
    DnnDetector();
    ~DnnDetector() = default;

    // Load ONNX model, auto-select best backend (CUDA > OpenCL > CPU)
    bool loadModel(const std::string& modelPath, 
                   float confThreshold = 0.5, 
                   float nmsThreshold = 0.4);

    // Run inference and return detections
    std::vector<Detection> detect(const cv::Mat& frame);

    bool isLoaded() const { return loaded; }

    // Manually override backend/target (optional)
    void setBackend(int backend, int target);

private:
    cv::dnn::Net net;
    float confThreshold;
    float nmsThreshold;
    bool loaded = false;

    // Post-processing for YOLOv8 output
    void postprocess(const cv::Mat& frame,
                     const std::vector<cv::Mat>& outputs,
                     std::vector<Detection>& detections);
};