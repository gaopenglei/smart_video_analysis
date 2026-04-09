/**
 * @file ImagePreprocessor.cpp
 * @brief 图像预处理模块实现文件
 */

#include "modules/preprocessing/ImagePreprocessor.hpp"
#include "core/Logger.hpp"
#include <algorithm>
#include <cmath>

namespace smart_video_analysis {
namespace modules {
namespace preprocessing {

// ============================================================================
// ImagePreprocessor 实现
// ============================================================================

ImagePreprocessor::ImagePreprocessor(const core::PreprocessConfig& config)
    : config_(config) {
    LOG_DEBUG("ImagePreprocessor created: target=%dx%d, normalize=%d",
              config_.target_width, config_.target_height, config_.normalize);
}

bool ImagePreprocessor::process(const cv::Mat& input, PreprocessResult& result) {
    if (input.empty()) {
        LOG_ERROR("Input image is empty");
        return false;
    }
    
    // 保存原始尺寸
    result.original_width = input.cols;
    result.original_height = input.rows;
    
    // 步骤1: 调整尺寸
    cv::Mat resized;
    if (!resize(input, resized, result)) {
        LOG_ERROR("Failed to resize image");
        return false;
    }
    
    // 步骤2: 颜色空间转换
    cv::Mat converted;
    if (!convertColor(resized, converted)) {
        LOG_ERROR("Failed to convert color space");
        return false;
    }
    
    // 步骤3: 归一化并转换为张量
    if (!normalize(converted, result.input_tensor)) {
        LOG_ERROR("Failed to normalize image");
        return false;
    }
    
    result.processed_image = converted;
    
    return true;
}

size_t ImagePreprocessor::getInputTensorSize() const {
    return static_cast<size_t>(config_.target_width) * 
           static_cast<size_t>(config_.target_height) * 
           static_cast<size_t>(config_.input_channels);
}

std::vector<int64_t> ImagePreprocessor::getInputShape() const {
    return {1, config_.input_channels, config_.target_height, config_.target_width};
}

void ImagePreprocessor::setTargetSize(int width, int height) {
    config_.target_width = width;
    config_.target_height = height;
}

void ImagePreprocessor::setNormalization(float mean_r, float mean_g, float mean_b,
                                          float std_r, float std_g, float std_b) {
    config_.mean_r = mean_r;
    config_.mean_g = mean_g;
    config_.mean_b = mean_b;
    config_.std_r = std_r;
    config_.std_g = std_g;
    config_.std_b = std_b;
    config_.normalize = true;
}

void ImagePreprocessor::setScale(float scale) {
    config_.scale = scale;
}

void ImagePreprocessor::setSwapRB(bool swap) {
    config_.swap_rb = swap;
}

void ImagePreprocessor::setResizeMode(const std::string& mode) {
    config_.resize_mode = mode;
}

bool ImagePreprocessor::resize(const cv::Mat& input, cv::Mat& output,
                                PreprocessResult& result) {
    if (config_.resize_mode == "letterbox") {
        return letterboxResize(input, output, result);
    } else if (config_.resize_mode == "stretch") {
        return stretchResize(input, output, result);
    } else if (config_.resize_mode == "crop") {
        return cropResize(input, output, result);
    } else {
        LOG_WARN("Unknown resize mode: %s, using letterbox", config_.resize_mode.c_str());
        return letterboxResize(input, output, result);
    }
}

bool ImagePreprocessor::letterboxResize(const cv::Mat& input, cv::Mat& output,
                                         PreprocessResult& result) {
    int target_w = config_.target_width;
    int target_h = config_.target_height;
    int orig_w = input.cols;
    int orig_h = input.rows;
    
    // 计算缩放比例，保持宽高比
    float scale = std::min(static_cast<float>(target_w) / orig_w,
                           static_cast<float>(target_h) / orig_h);
    
    int new_w = static_cast<int>(orig_w * scale);
    int new_h = static_cast<int>(orig_h * scale);
    
    // 计算填充
    result.pad_x = (target_w - new_w) / 2;
    result.pad_y = (target_h - new_h) / 2;
    result.scale_x = scale;
    result.scale_y = scale;
    
    // 缩放图像
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    
    // 创建目标图像并填充
    output = cv::Mat(target_h, target_w, input.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(output(cv::Rect(result.pad_x, result.pad_y, new_w, new_h)));
    
    return true;
}

bool ImagePreprocessor::stretchResize(const cv::Mat& input, cv::Mat& output,
                                       PreprocessResult& result) {
    int target_w = config_.target_width;
    int target_h = config_.target_height;
    
    result.scale_x = static_cast<float>(target_w) / input.cols;
    result.scale_y = static_cast<float>(target_h) / input.rows;
    result.pad_x = 0;
    result.pad_y = 0;
    
    cv::resize(input, output, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);
    
    return true;
}

bool ImagePreprocessor::cropResize(const cv::Mat& input, cv::Mat& output,
                                    PreprocessResult& result) {
    int target_w = config_.target_width;
    int target_h = config_.target_height;
    int orig_w = input.cols;
    int orig_h = input.rows;
    
    // 计算裁剪区域，保持宽高比
    float scale = std::max(static_cast<float>(target_w) / orig_w,
                           static_cast<float>(target_h) / orig_h);
    
    int new_w = static_cast<int>(target_w / scale);
    int new_h = static_cast<int>(target_h / scale);
    
    int crop_x = (orig_w - new_w) / 2;
    int crop_y = (orig_h - new_h) / 2;
    
    result.scale_x = scale;
    result.scale_y = scale;
    result.pad_x = 0;
    result.pad_y = 0;
    
    // 裁剪并缩放
    cv::Rect crop_rect(crop_x, crop_y, new_w, new_h);
    cv::Mat cropped = input(crop_rect);
    cv::resize(cropped, output, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);
    
    return true;
}

bool ImagePreprocessor::convertColor(const cv::Mat& input, cv::Mat& output) {
    if (config_.swap_rb) {
        cv::cvtColor(input, output, cv::COLOR_BGR2RGB);
    } else {
        output = input.clone();
    }
    return true;
}

bool ImagePreprocessor::normalize(const cv::Mat& input, std::vector<float>& output) {
    output.clear();
    output.reserve(getInputTensorSize());
    
    // 转换为浮点型
    cv::Mat float_img;
    input.convertTo(float_img, CV_32F, config_.scale);
    
    // 归一化
    if (config_.normalize) {
        std::vector<cv::Mat> channels(3);
        cv::split(float_img, channels);
        
        // R通道
        channels[0] = (channels[0] - config_.mean_r) / config_.std_r;
        // G通道
        channels[1] = (channels[1] - config_.mean_g) / config_.std_g;
        // B通道
        channels[2] = (channels[2] - config_.mean_b) / config_.std_b;
        
        cv::merge(channels, float_img);
    }
    
    // 转换为CHW格式
    return toTensor(float_img, output);
}

bool ImagePreprocessor::toTensor(const cv::Mat& input, std::vector<float>& output) {
    int channels = input.channels();
    int height = input.rows;
    int width = input.cols;
    
    output.resize(static_cast<size_t>(channels) * height * width);
    
    // OpenCV存储格式为HWC，需要转换为CHW
    std::vector<cv::Mat> channel_images(channels);
    cv::split(input, channel_images);
    
    size_t offset = 0;
    for (int c = 0; c < channels; ++c) {
        if (channel_images[c].isContinuous()) {
            memcpy(output.data() + offset, channel_images[c].data, 
                   static_cast<size_t>(height) * width * sizeof(float));
            offset += static_cast<size_t>(height) * width;
        } else {
            for (int h = 0; h < height; ++h) {
                const float* row_ptr = channel_images[c].ptr<float>(h);
                memcpy(output.data() + offset, row_ptr, width * sizeof(float));
                offset += width;
            }
        }
    }
    
    return true;
}

// ============================================================================
// YoloPreprocessor 实现
// ============================================================================

YoloPreprocessor::YoloPreprocessor(const core::PreprocessConfig& config)
    : ImagePreprocessor(config) {
    // YOLO默认使用letterbox缩放
    setResizeMode("letterbox");
    // YOLO默认归一化参数
    setScale(1.0f / 255.0f);
    setNormalization(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
    setSwapRB(true);
    
    LOG_DEBUG("YoloPreprocessor created for %s", yolo_version_.c_str());
}

bool YoloPreprocessor::process(const cv::Mat& input, PreprocessResult& result) {
    // YOLO预处理流程
    return ImagePreprocessor::process(input, result);
}

void YoloPreprocessor::setYoloVersion(const std::string& version) {
    yolo_version_ = version;
    
    // 不同版本的YOLO可能有不同的预处理参数
    if (version == "yolov5" || version == "yolov8" || version == "yolov11") {
        setScale(1.0f / 255.0f);
        setNormalization(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
    }
}

// ============================================================================
// ResNetPreprocessor 实现
// ============================================================================

ResNetPreprocessor::ResNetPreprocessor(const core::PreprocessConfig& config)
    : ImagePreprocessor(config) {
    // ResNet默认使用拉伸缩放
    setResizeMode("stretch");
    setSwapRB(true);
}

bool ResNetPreprocessor::process(const cv::Mat& input, PreprocessResult& result) {
    if (use_imagenet_norm_) {
        // ImageNet标准归一化
        setScale(1.0f / 255.0f);
        setNormalization(0.485f, 0.456f, 0.406f, 0.229f, 0.224f, 0.225f);
    }
    
    return ImagePreprocessor::process(input, result);
}

void ResNetPreprocessor::useImageNetNormalization() {
    use_imagenet_norm_ = true;
}

// ============================================================================
// MobileNetPreprocessor 实现
// ============================================================================

MobileNetPreprocessor::MobileNetPreprocessor(const core::PreprocessConfig& config)
    : ImagePreprocessor(config) {
    setResizeMode("stretch");
    setSwapRB(true);
}

bool MobileNetPreprocessor::process(const cv::Mat& input, PreprocessResult& result) {
    if (use_mobilenet_norm_) {
        // MobileNet标准归一化
        setScale(1.0f / 127.5f);
        setNormalization(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    }
    
    return ImagePreprocessor::process(input, result);
}

void MobileNetPreprocessor::useMobileNetNormalization() {
    use_mobilenet_norm_ = true;
}

// ============================================================================
// PreprocessorFactory 实现
// ============================================================================

std::unique_ptr<IImagePreprocessor> PreprocessorFactory::create(
    const std::string& model_type,
    const core::PreprocessConfig& config) {
    
    std::string type = model_type;
    std::transform(type.begin(), type.end(), type.begin(), ::tolower);
    
    if (type.find("yolo") != std::string::npos) {
        auto preprocessor = std::make_unique<YoloPreprocessor>(config);
        if (type.find("v5") != std::string::npos) {
            preprocessor->setYoloVersion("yolov5");
        } else if (type.find("v8") != std::string::npos) {
            preprocessor->setYoloVersion("yolov8");
        } else if (type.find("v11") != std::string::npos) {
            preprocessor->setYoloVersion("yolov11");
        }
        return preprocessor;
    }
    
    if (type.find("resnet") != std::string::npos) {
        auto preprocessor = std::make_unique<ResNetPreprocessor>(config);
        preprocessor->useImageNetNormalization();
        return preprocessor;
    }
    
    if (type.find("mobilenet") != std::string::npos) {
        auto preprocessor = std::make_unique<MobileNetPreprocessor>(config);
        preprocessor->useMobileNetNormalization();
        return preprocessor;
    }
    
    // 默认使用通用预处理器
    LOG_INFO("Using generic preprocessor for model type: %s", model_type.c_str());
    return std::make_unique<ImagePreprocessor>(config);
}

} // namespace preprocessing
} // namespace modules
} // namespace smart_video_analysis
