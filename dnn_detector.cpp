// dnn_detector.cpp
// Implementation of DnnDetector with automatic GPU/CPU selection

#include "dnn_detector.h"

DnnDetector::DnnDetector() 
    : confThreshold(0.5), nmsThreshold(0.4), loaded(false) {}

// Auto backend selector
bool DnnDetector::loadModel(const std::string& modelPath, float confTh, float nmsTh) {
    confThreshold = confTh;
    nmsThreshold = nmsTh;

    try {
        net = cv::dnn::readNetFromONNX(modelPath);
    } catch (const cv::Exception& e) {
        std::cerr << "[DNN] Failed to load ONNX model: " << e.what() << std::endl;
        return false;
    }

    if (net.empty()) {
        std::cerr << "[DNN] Network is empty, check model path." << std::endl;
        return false;
    }

    // Auto Backend Selection
    // Try CUDA 
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        std::cout << "[DNN] CUDA device found. Using CUDA backend." << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    // Try OpenCL (AMD / Intel / NVIDIA fallback)
    else {
        std::cout << "[DNN] No CUDA. Trying OpenCL (for AMD/Intel iGPU)..." << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
        // Note: You can also try DNN_BACKEND_INFERENCE_ENGINE with DNN_TARGET_MYRIAD for Intel VPU.
    }

    // Optional: If you have Intel OpenVINO, uncomment below to try:
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
    // net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); // or DNN_TARGET_OPENCL

    loaded = true;
    std::cout << "[DNN] Model loaded successfully." << std::endl;
    return true;
}

void DnnDetector::setBackend(int backend, int target) {
    if (loaded) {
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
    }
}

std::vector<Detection> DnnDetector::detect(const cv::Mat& frame) {
    std::vector<Detection> detections;
    if (!loaded || frame.empty()) return detections;

    // Preprocess
    // YOLOv8 expects 640x640, BGR, normalized to [0,1]
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0/255.0, cv::Size(640, 640),
                                           cv::Scalar(), true, false);
    net.setInput(blob);

    // Inference
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // Postprocess
    postprocess(frame, outputs, detections);
    return detections;
}

//  YOLOv8 post-processing
void DnnDetector::postprocess(const cv::Mat& frame,
                              const std::vector<cv::Mat>& outputs,
                              std::vector<Detection>& detections) {
    // YOLOv8 output shape: [1, 84, 8400] where 84 = 4 box coords + 80 class scores
    // We need to convert this to bounding boxes.

    cv::Mat output = outputs[0]; // shape: [batch, num_classes+4, num_boxes]
    // Reshape to [num_boxes, num_classes+4]
    int numBoxes = output.size[2];
    int numClasses = output.size[1] - 4; // typically 80

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    // Iterate over each detection
    for (int i = 0; i < numBoxes; ++i) {
        float* ptr = (float*)output.data + i * (numClasses + 4);
        
        // YOLOv8 format: cx, cy, w, h (normalized relative to 640x640)
        float cx = ptr[0];
        float cy = ptr[1];
        float w = ptr[2];
        float h = ptr[3];

        // Get max confidence and class ID from class scores
        float maxConf = 0.0f;
        int maxClassId = -1;
        for (int j = 4; j < numClasses + 4; ++j) {
            if (ptr[j] > maxConf) {
                maxConf = ptr[j];
                maxClassId = j - 4;
            }
        }

        // Filter by confidence threshold
        if (maxConf > confThreshold) {
            // Convert normalized coords to frame dimensions (scale)
            float scaleX = frame.cols / 640.0f;
            float scaleY = frame.rows / 640.0f;

            int x = static_cast<int>((cx - w/2) * scaleX);
            int y = static_cast<int>((cy - h/2) * scaleY);
            int width = static_cast<int>(w * scaleX);
            int height = static_cast<int>(h * scaleY);

            // Clamp to image bounds
            x = std::max(0, std::min(x, frame.cols - 1));
            y = std::max(0, std::min(y, frame.rows - 1));
            width = std::min(width, frame.cols - x);
            height = std::min(height, frame.rows - y);

            if (width > 0 && height > 0) {
                boxes.push_back(cv::Rect(x, y, width, height));
                confidences.push_back(maxConf);
                classIds.push_back(maxClassId);
            }
        }
    }

    // Non-Maximum Suppression (NMS)
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    for (int idx : indices) {
        Detection det;
        det.box = boxes[idx];
        det.confidence = confidences[idx];
        det.classId = classIds[idx];
        detections.push_back(det);
    }
}