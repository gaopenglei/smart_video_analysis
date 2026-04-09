/**
 * @file Visualizer.hpp
 * @brief 结果可视化模块头文件
 * 
 * 提供检测结果的可视化功能，包括绘制边界框、标签等。
 */

#ifndef MODULES_VISUALIZATION_VISUALIZER_HPP
#define MODULES_VISUALIZATION_VISUALIZER_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include "modules/postprocessing/DetectionPostprocessor.hpp"
#include "core/Config.hpp"

namespace smart_video_analysis {
namespace modules {
namespace visualization {

/**
 * @brief 可视化配置结构体
 */
struct VisualizerConfig {
    bool show_bbox = true;              ///< 是否显示边界框
    bool show_label = true;             ///< 是否显示标签
    bool show_confidence = true;        ///< 是否显示置信度
    bool show_fps = true;               ///< 是否显示FPS
    bool show_timestamp = true;         ///< 是否显示时间戳
    bool show_statistics = true;        ///< 是否显示统计信息
    int box_thickness = 2;              ///< 边界框线宽
    int font_scale = 0.6;               ///< 字体大小
    int font_thickness = 1;             ///< 字体线宽
    double alpha = 0.6;                 ///< 透明度
    cv::Scalar default_color = cv::Scalar(0, 255, 0);  ///< 默认颜色 (BGR)
    cv::Scalar text_bg_color = cv::Scalar(0, 0, 0);    ///< 文字背景色
    cv::Scalar text_color = cv::Scalar(255, 255, 255); ///< 文字颜色
};

/**
 * @brief 可视化结果结构体
 */
struct VisualizationResult {
    cv::Mat output_image;               ///< 输出图像
    int64_t frame_id;                   ///< 帧ID
    int64_t timestamp_ms;               ///< 时间戳
    double render_time_ms;              ///< 渲染耗时
    
    VisualizationResult() : frame_id(-1), timestamp_ms(0), render_time_ms(0) {}
};

/**
 * @brief 可视化器抽象基类
 */
class IVisualizer {
public:
    virtual ~IVisualizer() = default;
    
    /**
     * @brief 可视化检测结果
     * @param input 输入图像
     * @param detections 检测结果
     * @param result 可视化结果
     * @return 成功返回true
     */
    virtual bool visualize(
        const cv::Mat& input,
        const postprocessing::DetectionResult& detections,
        VisualizationResult& result) = 0;
    
    /**
     * @brief 设置配置
     */
    virtual void setConfig(const VisualizerConfig& config) = 0;
    
    /**
     * @brief 获取配置
     */
    virtual VisualizerConfig getConfig() const = 0;
};

/**
 * @brief 目标检测可视化器
 */
class DetectionVisualizer : public IVisualizer {
public:
    DetectionVisualizer();
    explicit DetectionVisualizer(const VisualizerConfig& config);
    ~DetectionVisualizer() override = default;
    
    bool visualize(
        const cv::Mat& input,
        const postprocessing::DetectionResult& detections,
        VisualizationResult& result) override;
    
    void setConfig(const VisualizerConfig& config) override;
    VisualizerConfig getConfig() const override;
    
    /**
     * @brief 设置类别颜色映射
     */
    void setClassColors(const std::map<int, cv::Scalar>& colors);
    
    /**
     * @brief 设置类别名称映射
     */
    void setClassNames(const std::map<int, std::string>& names);
    
    /**
     * @brief 添加自定义绘制回调
     */
    void addCustomDrawCallback(
        std::function<void(cv::Mat&, const postprocessing::DetectionResult&)> callback);
    
    /**
     * @brief 设置FPS显示
     */
    void setFPS(double fps);
    
    /**
     * @brief 设置统计信息
     */
    void setStatistics(const std::map<std::string, std::string>& stats);

private:
    /**
     * @brief 绘制单个检测框
     */
    void drawDetection(cv::Mat& image, const postprocessing::Detection& det);
    
    /**
     * @brief 绘制边界框
     */
    void drawBoundingBox(cv::Mat& image, const postprocessing::Detection& det,
                         const cv::Scalar& color);
    
    /**
     * @brief 绘制标签
     */
    void drawLabel(cv::Mat& image, const postprocessing::Detection& det,
                   const cv::Scalar& color);
    
    /**
     * @brief 绘制FPS
     */
    void drawFPS(cv::Mat& image);
    
    /**
     * @brief 绘制时间戳
     */
    void drawTimestamp(cv::Mat& image, int64_t timestamp_ms);
    
    /**
     * @brief 绘制统计信息
     */
    void drawStatistics(cv::Mat& image);
    
    /**
     * @brief 获取类别颜色
     */
    cv::Scalar getClassColor(int class_id);
    
    /**
     * @brief 获取类别名称
     */
    std::string getClassName(int class_id);

    // 成员变量
    VisualizerConfig config_;
    std::map<int, cv::Scalar> class_colors_;
    std::map<int, std::string> class_names_;
    std::vector<std::function<void(cv::Mat&, const postprocessing::DetectionResult&)>> callbacks_;
    double current_fps_ = 0.0;
    std::map<std::string, std::string> statistics_;
    
    // 预定义颜色
    static const std::vector<cv::Scalar> kDefaultColors;
};

/**
 * @brief 视频输出器
 */
class VideoOutputWriter {
public:
    VideoOutputWriter();
    ~VideoOutputWriter();
    
    /**
     * @brief 打开输出视频文件
     */
    bool open(const std::string& output_path, int width, int height, double fps);
    
    /**
     * @brief 写入帧
     */
    bool write(const cv::Mat& frame);
    
    /**
     * @brief 关闭输出
     */
    void close();
    
    /**
     * @brief 是否已打开
     */
    bool isOpened() const;
    
    /**
     * @brief 获取已写入帧数
     */
    int64_t getFrameCount() const;

private:
    cv::VideoWriter writer_;
    std::string output_path_;
    int64_t frame_count_ = 0;
};

/**
 * @brief 可视化管理器
 */
class VisualizationManager {
public:
    VisualizationManager();
    ~VisualizationManager() = default;
    
    /**
     * @brief 初始化
     */
    bool initialize(const core::VisualizationConfig& config);
    
    /**
     * @brief 可视化检测结果
     */
    bool visualize(const cv::Mat& input,
                   const postprocessing::DetectionResult& detections,
                   cv::Mat& output);
    
    /**
     * @brief 显示到窗口
     */
    void show(const cv::Mat& image, const std::string& window_name = "Detection Result");
    
    /**
     * @brief 保存到文件
     */
    bool save(const cv::Mat& image, const std::string& output_path);
    
    /**
     * @brief 设置类别名称
     */
    void setClassNames(const std::vector<std::string>& names);
    
    /**
     * @brief 更新FPS
     */
    void updateFPS(double fps);
    
    /**
     * @brief 更新统计信息
     */
    void updateStatistics(const std::string& key, const std::string& value);

private:
    std::unique_ptr<DetectionVisualizer> visualizer_;
    core::VisualizationConfig config_;
    std::map<int, std::string> class_names_;
};

} // namespace visualization
} // namespace modules
} // namespace smart_video_analysis

#endif // MODULES_VISUALIZATION_VISUALIZER_HPP
