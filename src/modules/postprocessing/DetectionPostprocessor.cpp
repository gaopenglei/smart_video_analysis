/**
 * @file DetectionPostprocessor.cpp
 * @brief 目标检测后处理模块实现文件
 */

#include "modules/postprocessing/DetectionPostprocessor.hpp"
#include "core/Logger.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace smart_video_analysis {
namespace modules {
namespace postprocessing {

// ============================================================================
// Detection 方法实现
// ============================================================================

float Detection::iou(const Detection& other) const {
    float inter_x1 = std::max(x1, other.x1);
    float inter_y1 = std::max(y1, other.y1);
    float inter_x2 = std::min(x2, other.x2);
    float inter_y2 = std::min(y2, other.y2);
    
    float inter_area = std::max(0.0f, inter_x2 - inter_x1) * 
                       std::max(0.0f, inter_y2 - inter_y1);
    
    float union_area = area() + other.area() - inter_area;
    
    return union_area > 0 ? inter_area / union_area : 0.0f;
}

// ============================================================================
// DetectionResult 方法实现
// ============================================================================

void DetectionResult::sortByConfidence() {
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });
}

void DetectionResult::sortByClassId() {
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.class_id < b.class_id;
              });
}

std::vector<Detection> DetectionResult::getDetectionsByClass(int class_id) const {
    std::vector<Detection> result;
    for (const auto& det : detections) {
        if (det.class_id == class_id) {
            result.push_back(det);
        }
    }
    return result;
}

std::vector<Detection> DetectionResult::getDetectionsAboveConfidence(float threshold) const {
    std::vector<Detection> result;
    for (const auto& det : detections) {
        if (det.confidence >= threshold) {
            result.push_back(det);
        }
    }
    return result;
}

// ============================================================================
// DetectionPostprocessor 实现
// ============================================================================

DetectionPostprocessor::DetectionPostprocessor(const core::PostprocessConfig& config)
    : config_(config) {
    LOG_DEBUG("DetectionPostprocessor created: conf_threshold=%.2f, nms_threshold=%.2f",
              config_.confidence_threshold, config_.nms_threshold);
}

bool DetectionPostprocessor::process(const std::vector<std::vector<float>>& output_data,
                                      DetectionResult& result) {
    // 基类默认实现
    LOG_WARN("Base process() called, should be overridden");
    return false;
}

void DetectionPostprocessor::setConfidenceThreshold(float threshold) {
    config_.confidence_threshold = threshold;
}

void DetectionPostprocessor::setNMSThreshold(float threshold) {
    config_.nms_threshold = threshold;
}

void DetectionPostprocessor::setClassNames(const std::vector<std::string>& names) {
    class_names_ = names;
}

const std::vector<std::string>& DetectionPostprocessor::getClassNames() const {
    return class_names_;
}

void DetectionPostprocessor::setOriginalSize(int width, int height) {
    original_width_ = width;
    original_height_ = height;
}

void DetectionPostprocessor::setScaleParams(float scale_x, float scale_y, int pad_x, int pad_y) {
    scale_x_ = scale_x;
    scale_y_ = scale_y;
    pad_x_ = pad_x;
    pad_y_ = pad_y;
}

std::vector<int> DetectionPostprocessor::nms(const std::vector<cv::Rect>& boxes,
                                              const std::vector<float>& scores,
                                              float nms_threshold) {
    std::vector<int> indices;
    if (boxes.empty()) return indices;
    
    // 按分数排序
    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&scores](int a, int b) { return scores[a] > scores[b]; });
    
    std::vector<bool> suppressed(boxes.size(), false);
    
    for (size_t i = 0; i < order.size(); ++i) {
        int idx = order[i];
        if (suppressed[idx]) continue;
        
        indices.push_back(idx);
        
        for (size_t j = i + 1; j < order.size(); ++j) {
            int next_idx = order[j];
            if (suppressed[next_idx]) continue;
            
            // 计算IoU（防止除零）
            cv::Rect inter = boxes[idx] & boxes[next_idx];
            float inter_area = static_cast<float>(inter.area());
            float union_area = static_cast<float>(boxes[idx].area() + boxes[next_idx].area())
                               - inter_area;
            float iou = union_area > 0.0f ? inter_area / union_area : 0.0f;
            
            if (iou > nms_threshold) {
                suppressed[next_idx] = true;
            }
        }
    }
    
    return indices;
}

void DetectionPostprocessor::mapToOriginalSpace(Detection& det) {
    // 从模型空间映射回原始图像空间
    det.x1 = (det.x1 - pad_x_) / scale_x_;
    det.y1 = (det.y1 - pad_y_) / scale_y_;
    det.x2 = (det.x2 - pad_x_) / scale_x_;
    det.y2 = (det.y2 - pad_y_) / scale_y_;
    
    // 确保坐标在图像范围内
    det.x1 = std::max(0.0f, std::min(det.x1, static_cast<float>(original_width_)));
    det.y1 = std::max(0.0f, std::min(det.y1, static_cast<float>(original_height_)));
    det.x2 = std::max(0.0f, std::min(det.x2, static_cast<float>(original_width_)));
    det.y2 = std::max(0.0f, std::min(det.y2, static_cast<float>(original_height_)));
    
    // 更新边界框
    det.bbox.x = static_cast<int>(det.x1);
    det.bbox.y = static_cast<int>(det.y1);
    det.bbox.width = static_cast<int>(det.x2 - det.x1);
    det.bbox.height = static_cast<int>(det.y2 - det.y1);
}

void DetectionPostprocessor::filterByConfidence(std::vector<Detection>& detections,
                                                 float threshold) {
    detections.erase(
        std::remove_if(detections.begin(), detections.end(),
                       [threshold](const Detection& d) { return d.confidence < threshold; }),
        detections.end());
}

// ============================================================================
// YoloV8Postprocessor 实现
// ============================================================================

YoloV8Postprocessor::YoloV8Postprocessor(const core::PostprocessConfig& config)
    : DetectionPostprocessor(config) {
    LOG_DEBUG("YoloV8Postprocessor created");
}

bool YoloV8Postprocessor::process(const std::vector<std::vector<float>>& output_data,
                                   DetectionResult& result) {
    if (output_data.empty()) {
        LOG_ERROR("Empty output data");
        return false;
    }
    
    // YOLOv8输出形状: [1, 84, 8400] 或 [batch, 4+num_classes, num_anchors]
    // 84 = 4 (bbox) + 80 (classes for COCO)
    const auto& output = output_data[0];
    
    // 假设输出形状为 [1, 84, 8400]
    int num_features = 84;  // 4 + 80 classes
    int num_anchors = 8400;
    
    // 根据实际输出调整
    if (output.size() == 84 * 8400) {
        num_features = 84;
        num_anchors = 8400;
    } else if (output.size() == 5 * 8400) {
        // 单类别检测
        num_features = 5;
        num_anchors = 8400;
    } else {
        // 尝试推断形状
        num_anchors = output.size() / num_features;
    }
    
    int num_classes = num_features - 4;
    
    std::vector<Detection> detections;
    if (!parseOutput(output, num_classes, num_anchors, detections)) {
        LOG_ERROR("Failed to parse YOLOv8 output");
        return false;
    }
    
    // 过滤低置信度检测
    filterByConfidence(detections, config_.confidence_threshold);
    
    // 应用NMS
    if (!detections.empty()) {
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        
        for (const auto& det : detections) {
            boxes.push_back(det.bbox);
            scores.push_back(det.confidence);
        }
        
        std::vector<int> keep_indices = nms(boxes, scores, config_.nms_threshold);
        
        std::vector<Detection> final_detections;
        for (int idx : keep_indices) {
            Detection det = detections[idx];
            mapToOriginalSpace(det);
            final_detections.push_back(det);
        }
        
        result.detections = std::move(final_detections);
    }
    
    result.original_width = original_width_;
    result.original_height = original_height_;
    
    LOG_DEBUG("YOLOv8 postprocess: %zu detections", result.detections.size());
    return true;
}

bool YoloV8Postprocessor::parseOutput(const std::vector<float>& output,
                                       int num_classes,
                                       int num_anchors,
                                       std::vector<Detection>& detections) {
    detections.clear();
    
    for (int anchor = 0; anchor < num_anchors; ++anchor) {
        // 找到最大类别置信度
        float max_class_score = 0.0f;
        int max_class_id = 0;
        
        for (int c = 0; c < num_classes; ++c) {
            float score = output[(4 + c) * num_anchors + anchor];
            if (score > max_class_score) {
                max_class_score = score;
                max_class_id = c;
            }
        }
        
        if (max_class_score < config_.confidence_threshold) {
            continue;
        }
        
        // 解析边界框 (center_x, center_y, width, height)
        float cx = output[0 * num_anchors + anchor];
        float cy = output[1 * num_anchors + anchor];
        float w = output[2 * num_anchors + anchor];
        float h = output[3 * num_anchors + anchor];
        
        // 转换为角点坐标
        Detection det;
        det.class_id = max_class_id;
        det.confidence = max_class_score;
        det.x1 = cx - w / 2.0f;
        det.y1 = cy - h / 2.0f;
        det.x2 = cx + w / 2.0f;
        det.y2 = cy + h / 2.0f;
        det.bbox = cv::Rect(static_cast<int>(det.x1), static_cast<int>(det.y1),
                            static_cast<int>(w), static_cast<int>(h));
        
        // 设置类别名称
        if (max_class_id >= 0 && max_class_id < static_cast<int>(class_names_.size())) {
            det.class_name = class_names_[max_class_id];
        }
        
        detections.push_back(det);
    }
    
    return true;
}

// ============================================================================
// YoloV5Postprocessor 实现
// ============================================================================

YoloV5Postprocessor::YoloV5Postprocessor(const core::PostprocessConfig& config)
    : DetectionPostprocessor(config) {
    LOG_DEBUG("YoloV5Postprocessor created");
}

bool YoloV5Postprocessor::process(const std::vector<std::vector<float>>& output_data,
                                   DetectionResult& result) {
    if (output_data.empty()) {
        LOG_ERROR("Empty output data");
        return false;
    }
    
    // YOLOv5输出形状: [1, num_anchors, 85] 或 [1, num_anchors, 5+num_classes]
    const auto& output = output_data[0];
    
    // 假设输出形状
    int num_features = 85;  // 4 + 1 (obj) + 80 classes
    int num_anchors = output.size() / num_features;
    
    std::vector<Detection> detections;
    
    for (int i = 0; i < num_anchors; ++i) {
        const float* row = output.data() + i * num_features;
        
        // 获取目标置信度
        float obj_conf = row[4];
        if (obj_conf < config_.confidence_threshold) {
            continue;
        }
        
        // 找到最大类别置信度
        float max_class_score = 0.0f;
        int max_class_id = 0;
        
        for (int c = 5; c < num_features; ++c) {
            float score = row[c] * obj_conf;
            if (score > max_class_score) {
                max_class_score = score;
                max_class_id = c - 5;
            }
        }
        
        if (max_class_score < config_.confidence_threshold) {
            continue;
        }
        
        // 解析边界框
        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];
        
        Detection det;
        det.class_id = max_class_id;
        det.confidence = max_class_score;
        det.x1 = cx - w / 2.0f;
        det.y1 = cy - h / 2.0f;
        det.x2 = cx + w / 2.0f;
        det.y2 = cy + h / 2.0f;
        det.bbox = cv::Rect(static_cast<int>(det.x1), static_cast<int>(det.y1),
                            static_cast<int>(w), static_cast<int>(h));
        
        if (max_class_id >= 0 && max_class_id < static_cast<int>(class_names_.size())) {
            det.class_name = class_names_[max_class_id];
        }
        
        detections.push_back(det);
    }
    
    // 应用NMS
    if (!detections.empty()) {
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        
        for (const auto& det : detections) {
            boxes.push_back(det.bbox);
            scores.push_back(det.confidence);
        }
        
        std::vector<int> keep_indices = nms(boxes, scores, config_.nms_threshold);
        
        std::vector<Detection> final_detections;
        for (int idx : keep_indices) {
            Detection det = detections[idx];
            mapToOriginalSpace(det);
            final_detections.push_back(det);
        }
        
        result.detections = std::move(final_detections);
    }
    
    result.original_width = original_width_;
    result.original_height = original_height_;
    
    return true;
}

// ============================================================================
// YoloV11Postprocessor 实现
// ============================================================================

YoloV11Postprocessor::YoloV11Postprocessor(const core::PostprocessConfig& config)
    : YoloV8Postprocessor(config) {
    LOG_DEBUG("YoloV11Postprocessor created");
}
// process() 直接继承 YoloV8Postprocessor::process()，无需重写。

// ============================================================================
// ResNetPostprocessor 实现
// ============================================================================

ResNetPostprocessor::ResNetPostprocessor(const core::PostprocessConfig& config)
    : DetectionPostprocessor(config) {
    LOG_DEBUG("ResNetPostprocessor created");
}

bool ResNetPostprocessor::process(const std::vector<std::vector<float>>& output_data,
                                   DetectionResult& result) {
    if (output_data.empty()) {
        LOG_ERROR("Empty output data");
        return false;
    }
    
    const auto& output = output_data[0];
    
    // 应用Softmax
    std::vector<float> probs = softmax(output);
    
    // 获取Top-K结果
    int k = 5;
    auto top_k = getTopK(probs, k);
    
    // 将分类结果转换为检测格式（全图检测）
    for (const auto& [class_id, confidence] : top_k) {
        Detection det;
        det.class_id = class_id;
        det.confidence = confidence;
        det.x1 = 0;
        det.y1 = 0;
        det.x2 = original_width_;
        det.y2 = original_height_;
        det.bbox = cv::Rect(0, 0, original_width_, original_height_);
        
        if (class_id >= 0 && class_id < static_cast<int>(class_names_.size())) {
            det.class_name = class_names_[class_id];
        }
        
        result.detections.push_back(det);
    }
    
    result.original_width = original_width_;
    result.original_height = original_height_;
    
    return true;
}

std::vector<std::pair<int, float>> ResNetPostprocessor::getTopK(
    const std::vector<float>& probs, int k) {
    
    std::vector<std::pair<int, float>> indexed_probs;
    for (size_t i = 0; i < probs.size(); ++i) {
        indexed_probs.emplace_back(static_cast<int>(i), probs[i]);
    }
    
    std::partial_sort(indexed_probs.begin(), indexed_probs.begin() + k,
                      indexed_probs.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
    
    indexed_probs.resize(k);
    return indexed_probs;
}

std::vector<float> ResNetPostprocessor::softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());
    
    // 找到最大值（数值稳定性）
    float max_val = *std::max_element(logits.begin(), logits.end());
    
    // 计算exp和sum
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }
    
    // 归一化
    for (float& p : probs) {
        p /= sum;
    }
    
    return probs;
}

// ============================================================================
// PostprocessorFactory 实现
// ============================================================================

std::unique_ptr<IDetectionPostprocessor> PostprocessorFactory::create(
    const std::string& model_type,
    const core::PostprocessConfig& config) {
    
    std::string type = model_type;
    std::transform(type.begin(), type.end(), type.begin(), ::tolower);
    
    if (type.find("yolov8") != std::string::npos) {
        return std::make_unique<YoloV8Postprocessor>(config);
    }
    if (type.find("yolov5") != std::string::npos) {
        return std::make_unique<YoloV5Postprocessor>(config);
    }
    if (type.find("yolov11") != std::string::npos) {
        return std::make_unique<YoloV11Postprocessor>(config);
    }
    if (type.find("resnet") != std::string::npos) {
        return std::make_unique<ResNetPostprocessor>(config);
    }
    
    // 默认使用YOLOv8后处理器
    LOG_INFO("Using default YOLOv8 postprocessor for model type: %s", model_type.c_str());
    return std::make_unique<YoloV8Postprocessor>(config);
}

} // namespace postprocessing
} // namespace modules
} // namespace smart_video_analysis
