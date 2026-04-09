/**
 * @file DetectionPostprocessor.hpp
 * @brief 目标检测后处理模块头文件
 * 
 * 提供目标检测结果的后处理功能，包括YOLOv8/v11等模型的输出解析。
 */

#ifndef MODULES_POSTPROCESSING_DETECTION_POSTPROCESSOR_HPP
#define MODULES_POSTPROCESSING_DETECTION_POSTPROCESSOR_HPP

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "core/Config.hpp"

namespace smart_video_analysis {
namespace modules {
namespace postprocessing {

/**
 * @brief 检测结果结构体
 */
struct Detection {
    int class_id;               ///< 类别ID
    std::string class_name;     ///< 类别名称
    float confidence;           ///< 置信度
    cv::Rect bbox;              ///< 边界框 (x, y, width, height)
    float x1, y1, x2, y2;       ///< 边界框坐标
    
    // 可选的分割掩码
    cv::Mat mask;               ///< 分割掩码
    
    // 可选的关键点
    std::vector<cv::Point2f> keypoints;  ///< 关键点
    std::vector<float> keypoint_confidences;  ///< 关键点置信度
    
    Detection() : class_id(-1), confidence(0.0f), x1(0), y1(0), x2(0), y2(0) {}
    
    /**
     * @brief 获取边界框面积
     */
    float area() const {
        return (x2 - x1) * (y2 - y1);
    }
    
    /**
     * @brief 计算与另一个检测框的IoU
     */
    float iou(const Detection& other) const;
};

/**
 * @brief 检测结果集合
 */
struct DetectionResult {
    std::vector<Detection> detections;   ///< 检测结果列表
    int64_t frame_id;                    ///< 帧ID
    int64_t timestamp_ms;                ///< 时间戳
    double inference_time_ms;            ///< 推理耗时
    int original_width;                  ///< 原始图像宽度
    int original_height;                 ///< 原始图像高度
    
    DetectionResult() : frame_id(-1), timestamp_ms(0), 
                        inference_time_ms(0), original_width(0), original_height(0) {}
    
    /**
     * @brief 按置信度排序
     */
    void sortByConfidence();
    
    /**
     * @brief 按类别ID排序
     */
    void sortByClassId();
    
    /**
     * @brief 获取指定类别的检测
     */
    std::vector<Detection> getDetectionsByClass(int class_id) const;
    
    /**
     * @brief 获取指定置信度以上的检测
     */
    std::vector<Detection> getDetectionsAboveConfidence(float threshold) const;
};

/**
 * @brief 后处理抽象基类
 */
class IDetectionPostprocessor {
public:
    virtual ~IDetectionPostprocessor() = default;
    
    /**
     * @brief 处理推理输出
     * @param output_data 推理输出数据
     * @param result 检测结果
     * @return 成功返回true
     */
    virtual bool process(const std::vector<std::vector<float>>& output_data,
                         DetectionResult& result) = 0;
    
    /**
     * @brief 设置置信度阈值
     */
    virtual void setConfidenceThreshold(float threshold) = 0;
    
    /**
     * @brief 设置NMS阈值
     */
    virtual void setNMSThreshold(float threshold) = 0;
};

/**
 * @brief 目标检测后处理基类
 */
class DetectionPostprocessor : public IDetectionPostprocessor {
public:
    explicit DetectionPostprocessor(const core::PostprocessConfig& config);
    ~DetectionPostprocessor() override = default;
    
    bool process(const std::vector<std::vector<float>>& output_data,
                 DetectionResult& result) override;
    
    void setConfidenceThreshold(float threshold) override;
    void setNMSThreshold(float threshold) override;
    
    /**
     * @brief 设置类别名称
     */
    void setClassNames(const std::vector<std::string>& names);
    
    /**
     * @brief 获取类别名称
     */
    const std::vector<std::string>& getClassNames() const;
    
    /**
     * @brief 设置原始图像尺寸
     */
    void setOriginalSize(int width, int height);
    
    /**
     * @brief 设置预处理缩放参数
     */
    void setScaleParams(float scale_x, float scale_y, int pad_x, int pad_y);

protected:
    /**
     * @brief 非极大值抑制 (NMS)
     */
    std::vector<int> nms(const std::vector<cv::Rect>& boxes,
                         const std::vector<float>& scores,
                         float nms_threshold);
    
    /**
     * @brief 将坐标从模型空间映射回原始图像空间
     */
    void mapToOriginalSpace(Detection& det);
    
    /**
     * @brief 过滤低置信度检测
     */
    void filterByConfidence(std::vector<Detection>& detections, float threshold);

    // 成员变量
    core::PostprocessConfig config_;
    std::vector<std::string> class_names_;
    int original_width_ = 0;
    int original_height_ = 0;
    float scale_x_ = 1.0f;
    float scale_y_ = 1.0f;
    int pad_x_ = 0;
    int pad_y_ = 0;
};

/**
 * @brief YOLOv8后处理器
 */
class YoloV8Postprocessor : public DetectionPostprocessor {
public:
    explicit YoloV8Postprocessor(const core::PostprocessConfig& config);
    ~YoloV8Postprocessor() override = default;
    
    bool process(const std::vector<std::vector<float>>& output_data,
                 DetectionResult& result) override;

private:
    /**
     * @brief 解析YOLOv8输出
     */
    bool parseOutput(const std::vector<float>& output,
                     int num_classes,
                     int num_anchors,
                     std::vector<Detection>& detections);
};

/**
 * @brief YOLOv5后处理器
 */
class YoloV5Postprocessor : public DetectionPostprocessor {
public:
    explicit YoloV5Postprocessor(const core::PostprocessConfig& config);
    ~YoloV5Postprocessor() override = default;
    
    bool process(const std::vector<std::vector<float>>& output_data,
                 DetectionResult& result) override;

private:
    /**
     * @brief 解析YOLOv5输出
     */
    bool parseOutput(const std::vector<float>& output,
                     const std::vector<int64_t>& shape,
                     std::vector<Detection>& detections);
};

/**
 * @brief YOLOv11后处理器
 */
class YoloV11Postprocessor : public DetectionPostprocessor {
public:
    explicit YoloV11Postprocessor(const core::PostprocessConfig& config);
    ~YoloV11Postprocessor() override = default;
    
    bool process(const std::vector<std::vector<float>>& output_data,
                 DetectionResult& result) override;

private:
    bool parseOutput(const std::vector<float>& output,
                     int num_classes,
                     int num_anchors,
                     std::vector<Detection>& detections);
};

/**
 * @brief ResNet分类后处理器
 */
class ResNetPostprocessor : public DetectionPostprocessor {
public:
    explicit ResNetPostprocessor(const core::PostprocessConfig& config);
    ~ResNetPostprocessor() override = default;
    
    bool process(const std::vector<std::vector<float>>& output_data,
                 DetectionResult& result) override;
    
    /**
     * @brief 获取Top-K分类结果
     */
    std::vector<std::pair<int, float>> getTopK(const std::vector<float>& probs, int k);

private:
    /**
     * @brief Softmax函数
     */
    std::vector<float> softmax(const std::vector<float>& logits);
};

/**
 * @brief 后处理器工厂类
 */
class PostprocessorFactory {
public:
    static std::unique_ptr<IDetectionPostprocessor> create(
        const std::string& model_type,
        const core::PostprocessConfig& config);
};

} // namespace postprocessing
} // namespace modules
} // namespace smart_video_analysis

#endif // MODULES_POSTPROCESSING_DETECTION_POSTPROCESSOR_HPP
