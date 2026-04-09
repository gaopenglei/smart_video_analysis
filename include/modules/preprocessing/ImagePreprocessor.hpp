/**
 * @file ImagePreprocessor.hpp
 * @brief 图像预处理模块头文件
 * 
 * 提供图像预处理功能，包括尺寸调整、归一化、颜色空间转换等。
 */

#ifndef MODULES_PREPROCESSING_IMAGE_PREPROCESSOR_HPP
#define MODULES_PREPROCESSING_IMAGE_PREPROCESSOR_HPP

#include <string>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include "core/Config.hpp"

namespace smart_video_analysis {
namespace modules {
namespace preprocessing {

/**
 * @brief 预处理结果结构体
 */
struct PreprocessResult {
    cv::Mat processed_image;            ///< 处理后的图像
    std::vector<float> input_tensor;    ///< 输入张量数据
    float scale_x;                      ///< X方向缩放比例
    float scale_y;                      ///< Y方向缩放比例
    int pad_x;                          ///< X方向填充
    int pad_y;                          ///< Y方向填充
    int original_width;                 ///< 原始宽度
    int original_height;                ///< 原始高度
    
    PreprocessResult() : scale_x(1.0f), scale_y(1.0f), pad_x(0), pad_y(0),
                         original_width(0), original_height(0) {}
};

/**
 * @brief 图像预处理抽象基类
 */
class IImagePreprocessor {
public:
    virtual ~IImagePreprocessor() = default;
    
    /**
     * @brief 预处理图像
     * @param input 输入图像
     * @param result 输出预处理结果
     * @return 成功返回true，失败返回false
     */
    virtual bool process(const cv::Mat& input, PreprocessResult& result) = 0;
    
    /**
     * @brief 获取输入张量大小
     * @return 张量元素数量
     */
    virtual size_t getInputTensorSize() const = 0;
    
    /**
     * @brief 获取输入形状
     * @return 形状向量 {batch, channels, height, width}
     */
    virtual std::vector<int64_t> getInputShape() const = 0;
};

/**
 * @brief 图像预处理基类
 * 
 * 提供通用的图像预处理功能。
 */
class ImagePreprocessor : public IImagePreprocessor {
public:
    /**
     * @brief 构造函数
     * @param config 预处理配置
     */
    explicit ImagePreprocessor(const core::PreprocessConfig& config);
    
    ~ImagePreprocessor() override = default;
    
    bool process(const cv::Mat& input, PreprocessResult& result) override;
    
    size_t getInputTensorSize() const override;
    std::vector<int64_t> getInputShape() const override;
    
    /**
     * @brief 设置目标尺寸
     */
    void setTargetSize(int width, int height);
    
    /**
     * @brief 设置归一化参数
     */
    void setNormalization(float mean_r, float mean_g, float mean_b,
                          float std_r, float std_g, float std_b);
    
    /**
     * @brief 设置缩放因子
     */
    void setScale(float scale);
    
    /**
     * @brief 设置是否交换R和B通道
     */
    void setSwapRB(bool swap);
    
    /**
     * @brief 设置缩放模式
     * @param mode "letterbox", "stretch", "crop"
     */
    void setResizeMode(const std::string& mode);

protected:
    /**
     * @brief 调整图像尺寸
     */
    virtual bool resize(const cv::Mat& input, cv::Mat& output, 
                        PreprocessResult& result);
    
    /**
     * @brief Letterbox缩放
     */
    bool letterboxResize(const cv::Mat& input, cv::Mat& output,
                         PreprocessResult& result);
    
    /**
     * @brief 拉伸缩放
     */
    bool stretchResize(const cv::Mat& input, cv::Mat& output,
                       PreprocessResult& result);
    
    /**
     * @brief 裁剪缩放
     */
    bool cropResize(const cv::Mat& input, cv::Mat& output,
                    PreprocessResult& result);
    
    /**
     * @brief 颜色空间转换
     */
    bool convertColor(const cv::Mat& input, cv::Mat& output);
    
    /**
     * @brief 归一化处理
     */
    bool normalize(const cv::Mat& input, std::vector<float>& output);
    
    /**
     * @brief 转换为张量格式 (CHW)
     */
    bool toTensor(const cv::Mat& input, std::vector<float>& output);

    // 成员变量
    core::PreprocessConfig config_;
};

/**
 * @brief YOLO系列模型预处理器
 * 
 * 专门针对YOLOv8/v11等模型的预处理实现。
 */
class YoloPreprocessor : public ImagePreprocessor {
public:
    /**
     * @brief 构造函数
     * @param config 预处理配置
     */
    explicit YoloPreprocessor(const core::PreprocessConfig& config);
    
    ~YoloPreprocessor() override = default;
    
    bool process(const cv::Mat& input, PreprocessResult& result) override;
    
    /**
     * @brief 设置YOLO版本
     * @param version "yolov5", "yolov8", "yolov11"
     */
    void setYoloVersion(const std::string& version);

private:
    std::string yolo_version_ = "yolov8";
};

/**
 * @brief ResNet模型预处理器
 * 
 * 专门针对ResNet系列模型的预处理实现。
 */
class ResNetPreprocessor : public ImagePreprocessor {
public:
    explicit ResNetPreprocessor(const core::PreprocessConfig& config);
    ~ResNetPreprocessor() override = default;
    
    bool process(const cv::Mat& input, PreprocessResult& result) override;
    
    /**
     * @brief 使用ImageNet标准归一化
     */
    void useImageNetNormalization();

private:
    bool use_imagenet_norm_ = false;
};

/**
 * @brief MobileNet模型预处理器
 * 
 * 专门针对MobileNet系列模型的预处理实现。
 */
class MobileNetPreprocessor : public ImagePreprocessor {
public:
    explicit MobileNetPreprocessor(const core::PreprocessConfig& config);
    ~MobileNetPreprocessor() override = default;
    
    bool process(const cv::Mat& input, PreprocessResult& result) override;
    
    /**
     * @brief 使用MobileNet标准归一化
     */
    void useMobileNetNormalization();

private:
    bool use_mobilenet_norm_ = false;
};

/**
 * @brief 预处理器工厂类
 */
class PreprocessorFactory {
public:
    /**
     * @brief 创建预处理器
     * @param model_type 模型类型
     * @param config 预处理配置
     * @return 预处理器实例
     */
    static std::unique_ptr<IImagePreprocessor> create(
        const std::string& model_type,
        const core::PreprocessConfig& config);
};

} // namespace preprocessing
} // namespace modules
} // namespace smart_video_analysis

#endif // MODULES_PREPROCESSING_IMAGE_PREPROCESSOR_HPP
