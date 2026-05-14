/**
 * @file SmartVideoAnalysisSystem.hpp
 * @brief 智能视频分析系统集成头文件
 * 
 * 整合所有模块，提供统一的视频分析系统接口。
 */

#ifndef SMART_VIDEO_ANALYSIS_SYSTEM_HPP
#define SMART_VIDEO_ANALYSIS_SYSTEM_HPP

#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>

#include "core/Logger.hpp"
#include "core/Config.hpp"
#include "core/ErrorHandling.hpp"
#include "modules/video_input/VideoInput.hpp"
#include "modules/preprocessing/ImagePreprocessor.hpp"
#include "modules/inference/InferenceEngine.hpp"
#include "modules/postprocessing/DetectionPostprocessor.hpp"
#include "modules/visualization/Visualizer.hpp"
#include "modules/operator_adapter/OperatorAdapter.hpp"

namespace smart_video_analysis {

/**
 * @brief 系统状态枚举
 */
enum class SystemState {
    UNINITIALIZED,      ///< 未初始化
    INITIALIZED,        ///< 已初始化
    RUNNING,            ///< 运行中
    PAUSED,             ///< 已暂停
    STOPPED,            ///< 已停止
    ERROR               ///< 错误状态
};

/**
 * @brief 系统统计信息结构体
 */
struct SystemStats {
    int64_t total_frames_processed;     ///< 总处理帧数
    int64_t total_frames_dropped;       ///< 丢弃帧数
    double average_fps;                 ///< 平均帧率
    double average_inference_time_ms;   ///< 平均推理时间
    double average_total_time_ms;       ///< 平均总处理时间
    double peak_memory_mb;              ///< 峰值内存使用
    double current_memory_mb;           ///< 当前内存使用
    int64_t start_time_ms;              ///< 启动时间
    int64_t elapsed_time_ms;            ///< 运行时间
    
    SystemStats() : total_frames_processed(0), total_frames_dropped(0),
                    average_fps(0.0), average_inference_time_ms(0.0),
                    average_total_time_ms(0.0), peak_memory_mb(0.0),
                    current_memory_mb(0.0), start_time_ms(0), elapsed_time_ms(0) {}
};

/**
 * @brief 回调函数类型定义
 */
using DetectionCallback = std::function<void(
    const postprocessing::DetectionResult& result,
    const cv::Mat& visualized_frame)>;

using ErrorCallback = std::function<void(
    core::ErrorCode error_code,
    const std::string& error_message)>;

/**
 * @brief 智能视频分析系统主类
 */
class SmartVideoAnalysisSystem {
public:
    SmartVideoAnalysisSystem();
    ~SmartVideoAnalysisSystem();
    
    // 禁止拷贝
    SmartVideoAnalysisSystem(const SmartVideoAnalysisSystem&) = delete;
    SmartVideoAnalysisSystem& operator=(const SmartVideoAnalysisSystem&) = delete;
    
    /**
     * @brief 初始化系统
     * @param config_file 配置文件路径
     * @return 成功返回ErrorCode::SUCCESS
     */
    core::ErrorCode initialize(const std::string& config_file);
    
    /**
     * @brief 初始化系统（使用默认配置）
     * @return 成功返回ErrorCode::SUCCESS
     */
    core::ErrorCode initialize();
    
    /**
     * @brief 启动系统
     * @return 成功返回ErrorCode::SUCCESS
     */
    core::ErrorCode start();
    
    /**
     * @brief 停止系统
     * @return 成功返回ErrorCode::SUCCESS
     */
    core::ErrorCode stop();
    
    /**
     * @brief 暂停系统
     * @return 成功返回ErrorCode::SUCCESS
     */
    core::ErrorCode pause();
    
    /**
     * @brief 恢复系统
     * @return 成功返回ErrorCode::SUCCESS
     */
    core::ErrorCode resume();
    
    /**
     * @brief 关闭系统
     */
    void shutdown();
    
    /**
     * @brief 获取系统状态
     */
    SystemState getState() const;
    
    /**
     * @brief 获取系统统计信息
     */
    SystemStats getStats() const;
    
    /**
     * @brief 设置检测回调
     */
    void setDetectionCallback(DetectionCallback callback);
    
    /**
     * @brief 设置错误回调
     */
    void setErrorCallback(ErrorCallback callback);
    
    /**
     * @brief 处理单帧图像
     * @param frame 输入图像
     * @param result 检测结果
     * @return 成功返回ErrorCode::SUCCESS
     */
    core::ErrorCode processFrame(const cv::Mat& frame,
                                  postprocessing::DetectionResult& result);
    
    /**
     * @brief 处理视频文件
     * @param video_path 视频文件路径
     * @param output_path 输出路径（可选）
     * @return 成功返回ErrorCode::SUCCESS
     */
    core::ErrorCode processVideo(const std::string& video_path,
                                  const std::string& output_path = "");
    
    /**
     * @brief 设置模型
     * @param model_path 模型路径
     * @param model_type 模型类型
     * @return 成功返回ErrorCode::SUCCESS
     */
    core::ErrorCode setModel(const std::string& model_path,
                              const std::string& model_type);
    
    /**
     * @brief 设置视频源
     * @param source_type 源类型 ("camera", "file", "rtsp")
     * @param source_path 源路径
     * @return 成功返回ErrorCode::SUCCESS
     */
    core::ErrorCode setVideoSource(const std::string& source_type,
                                    const std::string& source_path);
    
    /**
     * @brief 设置置信度阈值
     */
    void setConfidenceThreshold(float threshold);
    
    /**
     * @brief 设置NMS阈值
     */
    void setNMSThreshold(float threshold);
    
    /**
     * @brief 设置类别名称
     */
    void setClassNames(const std::vector<std::string>& names);
    
    /**
     * @brief 获取配置
     */
    const core::SystemConfig& getConfig() const;
    
    /**
     * @brief 重新加载配置
     */
    core::ErrorCode reloadConfig();

private:
    /**
     * @brief 主处理循环
     */
    void processingLoop();
    
    /**
     * @brief 处理单帧
     * @param frame 已读取的视频帧
     */
    bool processSingleFrame(modules::video_input::VideoFrame& frame);
    
    /**
     * @brief 更新统计信息
     */
    void updateStats(double inference_time, double total_time);
    
    /**
     * @brief 检查算子支持性
     */
    bool checkOperatorSupport(const std::string& model_path);
    
    /**
     * @brief 初始化模块
     */
    bool initializeModules();
    
    /**
     * @brief 初始化类别名称
     */
    void initializeClassNames();

    // 核心模块
    std::unique_ptr<modules::video_input::VideoInputManager> video_input_;
    std::unique_ptr<modules::preprocessing::IImagePreprocessor> preprocessor_;
    std::unique_ptr<modules::inference::OnnxInferenceEngine> inference_engine_;
    std::unique_ptr<modules::postprocessing::IDetectionPostprocessor> postprocessor_;
    std::unique_ptr<modules::visualization::VisualizationManager> visualizer_;
    std::unique_ptr<modules::operator_adapter::IOperatorAdapter> operator_adapter_;
    
    // 配置
    core::SystemConfig config_;
    std::string config_file_path_;
    
    // 状态
    std::atomic<SystemState> state_;
    std::atomic<bool> stop_requested_;
    std::atomic<bool> pause_requested_;
    
    // 线程
    std::thread processing_thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    
    // 统计
    mutable std::mutex stats_mutex_;
    SystemStats stats_;
    
    // 回调
    DetectionCallback detection_callback_;
    ErrorCallback error_callback_;
    
    // 类别名称
    std::vector<std::string> class_names_;
    
    // 帧缓冲
    std::queue<modules::video_input::VideoFrame> frame_buffer_;
    size_t max_buffer_size_ = 10;
};

/**
 * @brief 系统工厂函数
 */
std::unique_ptr<SmartVideoAnalysisSystem> createSystem();

/**
 * @brief 系统工厂函数（带配置文件）
 */
std::unique_ptr<SmartVideoAnalysisSystem> createSystem(const std::string& config_file);

} // namespace smart_video_analysis

#endif // SMART_VIDEO_ANALYSIS_SYSTEM_HPP
