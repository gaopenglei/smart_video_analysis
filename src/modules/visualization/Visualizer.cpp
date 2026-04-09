/**
 * @file Visualizer.cpp
 * @brief 结果可视化模块实现文件
 */

#include "modules/visualization/Visualizer.hpp"
#include "core/Logger.hpp"
#include <chrono>
#include <iomanip>
#include <sstream>

namespace smart_video_analysis {
namespace modules {
namespace visualization {

// 预定义颜色列表
const std::vector<cv::Scalar> DetectionVisualizer::kDefaultColors = {
    cv::Scalar(255, 0, 0),     // 蓝色
    cv::Scalar(0, 255, 0),     // 绿色
    cv::Scalar(0, 0, 255),     // 红色
    cv::Scalar(255, 255, 0),   // 青色
    cv::Scalar(255, 0, 255),   // 紫色
    cv::Scalar(0, 255, 255),   // 黄色
    cv::Scalar(128, 0, 255),   // 粉色
    cv::Scalar(255, 128, 0),   // 天蓝色
    cv::Scalar(128, 255, 0),   // 春绿色
    cv::Scalar(0, 128, 255),   // 橙色
    cv::Scalar(255, 0, 128),   // 玫瑰色
    cv::Scalar(0, 255, 128),   // 青绿色
    cv::Scalar(128, 0, 0),     // 深蓝色
    cv::Scalar(0, 128, 0),     // 深绿色
    cv::Scalar(0, 0, 128),     // 深红色
    cv::Scalar(128, 128, 0),   // 深青色
    cv::Scalar(128, 0, 128),   // 深紫色
    cv::Scalar(0, 128, 128),   // 深黄色
    cv::Scalar(192, 192, 192), // 银色
    cv::Scalar(128, 128, 128), // 灰色
};

// ============================================================================
// DetectionVisualizer 实现
// ============================================================================

DetectionVisualizer::DetectionVisualizer() {
    // 初始化默认类别颜色
    for (size_t i = 0; i < kDefaultColors.size(); ++i) {
        class_colors_[static_cast<int>(i)] = kDefaultColors[i];
    }
}

DetectionVisualizer::DetectionVisualizer(const VisualizerConfig& config)
    : config_(config) {
    // 初始化默认类别颜色
    for (size_t i = 0; i < kDefaultColors.size(); ++i) {
        class_colors_[static_cast<int>(i)] = kDefaultColors[i];
    }
}

bool DetectionVisualizer::visualize(
    const cv::Mat& input,
    const postprocessing::DetectionResult& detections,
    VisualizationResult& result) {
    
    if (input.empty()) {
        LOG_ERROR("Input image is empty");
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 复制输入图像
    result.output_image = input.clone();
    
    // 绘制每个检测框
    for (const auto& det : detections.detections) {
        drawDetection(result.output_image, det);
    }
    
    // 绘制FPS
    if (config_.show_fps && current_fps_ > 0) {
        drawFPS(result.output_image);
    }
    
    // 绘制时间戳
    if (config_.show_timestamp && detections.timestamp_ms > 0) {
        drawTimestamp(result.output_image, detections.timestamp_ms);
    }
    
    // 绘制统计信息
    if (config_.show_statistics && !statistics_.empty()) {
        drawStatistics(result.output_image);
    }
    
    // 执行自定义回调
    for (const auto& callback : callbacks_) {
        callback(result.output_image, detections);
    }
    
    result.frame_id = detections.frame_id;
    result.timestamp_ms = detections.timestamp_ms;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.render_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();
    
    return true;
}

void DetectionVisualizer::setConfig(const VisualizerConfig& config) {
    config_ = config;
}

VisualizerConfig DetectionVisualizer::getConfig() const {
    return config_;
}

void DetectionVisualizer::setClassColors(const std::map<int, cv::Scalar>& colors) {
    class_colors_ = colors;
}

void DetectionVisualizer::setClassNames(const std::map<int, std::string>& names) {
    class_names_ = names;
}

void DetectionVisualizer::addCustomDrawCallback(
    std::function<void(cv::Mat&, const postprocessing::DetectionResult&)> callback) {
    callbacks_.push_back(callback);
}

void DetectionVisualizer::setFPS(double fps) {
    current_fps_ = fps;
}

void DetectionVisualizer::setStatistics(const std::map<std::string, std::string>& stats) {
    statistics_ = stats;
}

void DetectionVisualizer::drawDetection(cv::Mat& image, 
                                         const postprocessing::Detection& det) {
    cv::Scalar color = getClassColor(det.class_id);
    
    // 绘制边界框
    if (config_.show_bbox) {
        drawBoundingBox(image, det, color);
    }
    
    // 绘制标签
    if (config_.show_label || config_.show_confidence) {
        drawLabel(image, det, color);
    }
}

void DetectionVisualizer::drawBoundingBox(cv::Mat& image,
                                           const postprocessing::Detection& det,
                                           const cv::Scalar& color) {
    cv::Rect bbox(static_cast<int>(det.x1), static_cast<int>(det.y1),
                  static_cast<int>(det.x2 - det.x1),
                  static_cast<int>(det.y2 - det.y1));
    
    // 确保边界框在图像范围内
    bbox &= cv::Rect(0, 0, image.cols, image.rows);
    
    if (bbox.width > 0 && bbox.height > 0) {
        cv::rectangle(image, bbox, color, config_.box_thickness);
    }
}

void DetectionVisualizer::drawLabel(cv::Mat& image,
                                     const postprocessing::Detection& det,
                                     const cv::Scalar& color) {
    // 构建标签文本
    std::string label;
    
    if (config_.show_label) {
        label = getClassName(det.class_id);
        if (label.empty()) {
            label = "Class_" + std::to_string(det.class_id);
        }
    }
    
    if (config_.show_confidence) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << (det.confidence * 100) << "%";
        
        if (!label.empty()) {
            label += " ";
        }
        label += oss.str();
    }
    
    if (label.empty()) {
        return;
    }
    
    // 计算文本大小
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                          config_.font_scale, config_.font_thickness,
                                          &baseline);
    
    // 计算文本位置
    int text_x = static_cast<int>(det.x1);
    int text_y = static_cast<int>(det.y1) - 5;
    
    if (text_y < text_size.height + 5) {
        text_y = static_cast<int>(det.y1) + text_size.height + 5;
    }
    
    // 绘制文本背景
    cv::Rect text_bg(text_x - 2, text_y - text_size.height - 2,
                     text_size.width + 4, text_size.height + 4);
    text_bg &= cv::Rect(0, 0, image.cols, image.rows);
    
    if (text_bg.width > 0 && text_bg.height > 0) {
        cv::rectangle(image, text_bg, color, cv::FILLED);
    }
    
    // 绘制文本
    cv::putText(image, label, cv::Point(text_x, text_y),
                cv::FONT_HERSHEY_SIMPLEX, config_.font_scale,
                config_.text_color, config_.font_thickness);
}

void DetectionVisualizer::drawFPS(cv::Mat& image) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << "FPS: " << current_fps_;
    std::string fps_text = oss.str();
    
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(fps_text, cv::FONT_HERSHEY_SIMPLEX,
                                          config_.font_scale, config_.font_thickness,
                                          &baseline);
    
    // 绘制在右上角
    int text_x = image.cols - text_size.width - 10;
    int text_y = text_size.height + 10;
    
    cv::rectangle(image, cv::Point(text_x - 5, text_y - text_size.height - 5),
                  cv::Point(text_x + text_size.width + 5, text_y + 5),
                  config_.text_bg_color, cv::FILLED);
    
    cv::putText(image, fps_text, cv::Point(text_x, text_y),
                cv::FONT_HERSHEY_SIMPLEX, config_.font_scale,
                cv::Scalar(0, 255, 0), config_.font_thickness);
}

void DetectionVisualizer::drawTimestamp(cv::Mat& image, int64_t timestamp_ms) {
    // 转换为可读时间格式
    time_t seconds = timestamp_ms / 1000;
    int milliseconds = timestamp_ms % 1000;
    
    struct tm* time_info = localtime(&seconds);
    char time_str[32];
    strftime(time_str, sizeof(time_str), "%H:%M:%S", time_info);
    
    std::ostringstream oss;
    oss << time_str << "." << std::setfill('0') << std::setw(3) << milliseconds;
    std::string timestamp_text = oss.str();
    
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(timestamp_text, cv::FONT_HERSHEY_SIMPLEX,
                                          config_.font_scale * 0.8, config_.font_thickness,
                                          &baseline);
    
    // 绘制在左下角
    int text_x = 10;
    int text_y = image.rows - 10;
    
    cv::rectangle(image, cv::Point(text_x - 5, text_y - text_size.height - 5),
                  cv::Point(text_x + text_size.width + 5, text_y + 5),
                  config_.text_bg_color, cv::FILLED);
    
    cv::putText(image, timestamp_text, cv::Point(text_x, text_y),
                cv::FONT_HERSHEY_SIMPLEX, config_.font_scale * 0.8,
                config_.text_color, config_.font_thickness);
}

void DetectionVisualizer::drawStatistics(cv::Mat& image) {
    int y_offset = 30;
    int line_height = 20;
    
    for (const auto& [key, value] : statistics_) {
        std::string stat_text = key + ": " + value;
        
        cv::putText(image, stat_text, cv::Point(10, y_offset),
                    cv::FONT_HERSHEY_SIMPLEX, config_.font_scale * 0.7,
                    cv::Scalar(255, 255, 255), config_.font_thickness);
        
        y_offset += line_height;
    }
}

cv::Scalar DetectionVisualizer::getClassColor(int class_id) {
    auto it = class_colors_.find(class_id);
    if (it != class_colors_.end()) {
        return it->second;
    }
    
    // 使用默认颜色
    int color_idx = class_id % static_cast<int>(kDefaultColors.size());
    return kDefaultColors[color_idx];
}

std::string DetectionVisualizer::getClassName(int class_id) {
    auto it = class_names_.find(class_id);
    if (it != class_names_.end()) {
        return it->second;
    }
    return "";
}

// ============================================================================
// VideoOutputWriter 实现
// ============================================================================

VideoOutputWriter::VideoOutputWriter() = default;

VideoOutputWriter::~VideoOutputWriter() {
    close();
}

bool VideoOutputWriter::open(const std::string& output_path, int width, int height, double fps) {
    output_path_ = output_path;
    
    // 使用H.264编码
    int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
    
    writer_.open(output_path, fourcc, fps, cv::Size(width, height));
    
    if (!writer_.isOpened()) {
        // 尝试使用MJPG编码
        fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        output_path_ = output_path;
        std::string mjpg_path = output_path.substr(0, output_path.find_last_of('.')) + ".avi";
        writer_.open(mjpg_path, fourcc, fps, cv::Size(width, height));
        
        if (!writer_.isOpened()) {
            LOG_ERROR("Failed to open video writer for: %s", output_path.c_str());
            return false;
        }
        output_path_ = mjpg_path;
    }
    
    LOG_INFO("Video writer opened: %s (%dx%d @ %.1f fps)",
             output_path_.c_str(), width, height, fps);
    return true;
}

bool VideoOutputWriter::write(const cv::Mat& frame) {
    if (!writer_.isOpened()) {
        LOG_ERROR("Video writer is not opened");
        return false;
    }
    
    writer_.write(frame);
    frame_count_++;
    return true;
}

void VideoOutputWriter::close() {
    if (writer_.isOpened()) {
        writer_.release();
        LOG_INFO("Video writer closed: %s (%ld frames written)",
                 output_path_.c_str(), frame_count_);
    }
}

bool VideoOutputWriter::isOpened() const {
    return writer_.isOpened();
}

int64_t VideoOutputWriter::getFrameCount() const {
    return frame_count_;
}

// ============================================================================
// VisualizationManager 实现
// ============================================================================

VisualizationManager::VisualizationManager()
    : visualizer_(std::make_unique<DetectionVisualizer>()) {
}

bool VisualizationManager::initialize(const core::VisualizationConfig& config) {
    config_ = config;
    
    VisualizerConfig viz_config;
    viz_config.show_bbox = config_.show_bbox;
    viz_config.show_label = config_.show_label;
    viz_config.show_confidence = config_.show_confidence;
    viz_config.show_fps = config_.show_fps;
    viz_config.box_thickness = config_.box_thickness;
    viz_config.font_scale = config_.font_scale;
    
    visualizer_->setConfig(viz_config);
    
    LOG_INFO("VisualizationManager initialized");
    return true;
}

bool VisualizationManager::visualize(const cv::Mat& input,
                                      const postprocessing::DetectionResult& detections,
                                      cv::Mat& output) {
    VisualizationResult result;
    bool success = visualizer_->visualize(input, detections, result);
    
    if (success) {
        output = result.output_image;
    }
    
    return success;
}

void VisualizationManager::show(const cv::Mat& image, const std::string& window_name) {
    cv::imshow(window_name, image);
    cv::waitKey(1);
}

bool VisualizationManager::save(const cv::Mat& image, const std::string& output_path) {
    bool success = cv::imwrite(output_path, image);
    if (success) {
        LOG_DEBUG("Image saved: %s", output_path.c_str());
    } else {
        LOG_ERROR("Failed to save image: %s", output_path.c_str());
    }
    return success;
}

void VisualizationManager::setClassNames(const std::vector<std::string>& names) {
    for (size_t i = 0; i < names.size(); ++i) {
        class_names_[static_cast<int>(i)] = names[i];
    }
    visualizer_->setClassNames(class_names_);
}

void VisualizationManager::updateFPS(double fps) {
    visualizer_->setFPS(fps);
}

void VisualizationManager::updateStatistics(const std::string& key, const std::string& value) {
    std::map<std::string, std::string> stats;
    stats[key] = value;
    visualizer_->setStatistics(stats);
}

} // namespace visualization
} // namespace modules
} // namespace smart_video_analysis
