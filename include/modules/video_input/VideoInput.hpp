/**
 * @file VideoInput.hpp
 * @brief 视频输入模块头文件
 * 
 * 提供统一的视频输入接口，支持摄像头、视频文件和网络流输入。
 */

#ifndef MODULES_VIDEO_INPUT_VIDEO_INPUT_HPP
#define MODULES_VIDEO_INPUT_VIDEO_INPUT_HPP

#include <string>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <queue>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include "core/Config.hpp"
#include "core/ErrorHandling.hpp"

namespace smart_video_analysis {
namespace modules {
namespace video_input {

/**
 * @brief 视频帧结构体
 */
struct VideoFrame {
    cv::Mat frame;                      ///< 帧数据
    int64_t frame_id;                   ///< 帧ID
    int64_t timestamp_ms;               ///< 时间戳（毫秒）
    int width;                          ///< 帧宽度
    int height;                         ///< 帧高度
    int channels;                       ///< 通道数
    double fps;                         ///< 当前帧率
    
    VideoFrame() : frame_id(-1), timestamp_ms(0), width(0), height(0), channels(0), fps(0.0) {}
    
    bool isValid() const { return !frame.empty() && frame_id >= 0; }
};

/**
 * @brief 视频输入源类型枚举
 */
enum class InputSourceType {
    CAMERA,         ///< 摄像头输入
    FILE,           ///< 视频文件输入
    RTSP_STREAM,    ///< RTSP网络流输入
    HTTP_STREAM,    ///< HTTP网络流输入
    UNKNOWN         ///< 未知类型
};

/**
 * @brief 视频输入抽象基类
 * 
 * 定义视频输入的统一接口，所有具体的视频输入实现都继承此类。
 */
class IVideoInput {
public:
    virtual ~IVideoInput() = default;
    
    /**
     * @brief 打开视频输入源
     * @return 成功返回true，失败返回false
     */
    virtual bool open() = 0;
    
    /**
     * @brief 关闭视频输入源
     */
    virtual void close() = 0;
    
    /**
     * @brief 读取下一帧
     * @param frame 输出帧数据
     * @return 成功返回true，失败或到达末尾返回false
     */
    virtual bool readFrame(VideoFrame& frame) = 0;
    
    /**
     * @brief 检查输入源是否已打开
     * @return 已打开返回true，否则返回false
     */
    virtual bool isOpened() const = 0;
    
    /**
     * @brief 获取视频宽度
     * @return 视频宽度
     */
    virtual int getWidth() const = 0;
    
    /**
     * @brief 获取视频高度
     * @return 视频高度
     */
    virtual int getHeight() const = 0;
    
    /**
     * @brief 获取帧率
     * @return 帧率
     */
    virtual double getFPS() const = 0;
    
    /**
     * @brief 获取总帧数（仅对视频文件有效）
     * @return 总帧数，未知返回-1
     */
    virtual int64_t getTotalFrames() const = 0;
    
    /**
     * @brief 获取当前帧位置
     * @return 当前帧位置
     */
    virtual int64_t getCurrentFrame() const = 0;
    
    /**
     * @brief 设置帧位置（仅对视频文件有效）
     * @param frame_id 目标帧ID
     * @return 成功返回true，失败返回false
     */
    virtual bool setFramePosition(int64_t frame_id) = 0;
    
    /**
     * @brief 获取输入源类型
     * @return 输入源类型
     */
    virtual InputSourceType getSourceType() const = 0;
    
    /**
     * @brief 获取输入源路径
     * @return 输入源路径
     */
    virtual std::string getSourcePath() const = 0;
};

/**
 * @brief 视频输入管理器类
 * 
 * 管理视频输入源，提供帧缓冲和异步读取功能。
 */
class VideoInputManager : public IVideoInput {
public:
    /**
     * @brief 构造函数
     * @param config 视频输入配置
     */
    explicit VideoInputManager(const core::VideoInputConfig& config);
    
    /**
     * @brief 析构函数
     */
    ~VideoInputManager() override;
    
    // 实现IVideoInput接口
    bool open() override;
    void close() override;
    bool readFrame(VideoFrame& frame) override;
    bool isOpened() const override;
    int getWidth() const override;
    int getHeight() const override;
    double getFPS() const override;
    int64_t getTotalFrames() const override;
    int64_t getCurrentFrame() const override;
    bool setFramePosition(int64_t frame_id) override;
    InputSourceType getSourceType() const override;
    std::string getSourcePath() const override;
    
    /**
     * @brief 启动异步读取模式
     * @param buffer_size 缓冲区大小
     * @return 成功返回true，失败返回false
     */
    bool startAsyncRead(size_t buffer_size = 10);
    
    /**
     * @brief 停止异步读取模式
     */
    void stopAsyncRead();
    
    /**
     * @brief 从缓冲区获取帧（异步模式）
     * @param frame 输出帧数据
     * @param timeout_ms 超时时间（毫秒），-1表示无限等待
     * @return 成功返回true，超时或失败返回false
     */
    bool getFrameFromBuffer(VideoFrame& frame, int timeout_ms = -1);
    
    /**
     * @brief 获取缓冲区当前帧数
     * @return 缓冲区帧数
     */
    size_t getBufferSize() const;
    
    /**
     * @brief 检查是否到达视频末尾
     * @return 到达末尾返回true，否则返回false
     */
    bool isEndOfStream() const;
    
    /**
     * @brief 获取统计信息
     */
    struct Statistics {
        int64_t total_frames_read;      ///< 总读取帧数
        int64_t frames_dropped;         ///< 丢弃帧数
        double average_fps;             ///< 平均帧率
        double current_fps;             ///< 当前帧率
    };
    
    Statistics getStatistics() const;
    void resetStatistics();

private:
    /**
     * @brief 创建具体的视频输入实现
     */
    std::unique_ptr<IVideoInput> createInput();
    
    /**
     * @brief 异步读取线程函数
     */
    void asyncReadThread();
    
    // 成员变量
    core::VideoInputConfig config_;
    std::unique_ptr<IVideoInput> input_impl_;
    
    // 异步读取相关
    std::thread async_thread_;
    std::queue<VideoFrame> frame_buffer_;
    std::mutex buffer_mutex_;
    std::condition_variable buffer_cv_;
    std::atomic<bool> async_running_{false};
    std::atomic<bool> end_of_stream_{false};
    size_t max_buffer_size_ = 10;
    
    // 统计信息
    mutable std::mutex stats_mutex_;
    Statistics stats_;
    int64_t last_frame_time_ = 0;
};

/**
 * @brief 摄像头输入类
 */
class CameraInput : public IVideoInput {
public:
    explicit CameraInput(int camera_index, int width = 1920, int height = 1080);
    ~CameraInput() override;
    
    bool open() override;
    void close() override;
    bool readFrame(VideoFrame& frame) override;
    bool isOpened() const override;
    int getWidth() const override;
    int getHeight() const override;
    double getFPS() const override;
    int64_t getTotalFrames() const override { return -1; }
    int64_t getCurrentFrame() const override;
    bool setFramePosition(int64_t frame_id) override { return false; }
    InputSourceType getSourceType() const override { return InputSourceType::CAMERA; }
    std::string getSourcePath() const override;
    
private:
    int camera_index_;
    int target_width_;
    int target_height_;
    cv::VideoCapture capture_;
    int64_t frame_counter_ = 0;
};

/**
 * @brief 视频文件输入类
 */
class FileInput : public IVideoInput {
public:
    explicit FileInput(const std::string& file_path);
    ~FileInput() override;
    
    bool open() override;
    void close() override;
    bool readFrame(VideoFrame& frame) override;
    bool isOpened() const override;
    int getWidth() const override;
    int getHeight() const override;
    double getFPS() const override;
    int64_t getTotalFrames() const override;
    int64_t getCurrentFrame() const override;
    bool setFramePosition(int64_t frame_id) override;
    InputSourceType getSourceType() const override { return InputSourceType::FILE; }
    std::string getSourcePath() const override { return file_path_; }
    
private:
    std::string file_path_;
    cv::VideoCapture capture_;
    int64_t frame_counter_ = 0;
};

/**
 * @brief 网络流输入类
 */
class NetworkStreamInput : public IVideoInput {
public:
    explicit NetworkStreamInput(const std::string& stream_url, 
                                const std::string& protocol = "rtsp");
    ~NetworkStreamInput() override;
    
    bool open() override;
    void close() override;
    bool readFrame(VideoFrame& frame) override;
    bool isOpened() const override;
    int getWidth() const override;
    int getHeight() const override;
    double getFPS() const override;
    int64_t getTotalFrames() const override { return -1; }
    int64_t getCurrentFrame() const override;
    bool setFramePosition(int64_t frame_id) override { return false; }
    InputSourceType getSourceType() const override;
    std::string getSourcePath() const override { return stream_url_; }
    
    /**
     * @brief 设置重连参数
     */
    void setReconnectParams(int max_retries = 5, int retry_interval_ms = 1000);
    
    /**
     * @brief 设置缓冲区大小
     */
    void setBufferSize(int buffer_size);

private:
    bool reconnect();
    
    std::string stream_url_;
    std::string protocol_;
    cv::VideoCapture capture_;
    int64_t frame_counter_ = 0;
    int max_retries_ = 5;
    int retry_interval_ms_ = 1000;
    int buffer_size_ = 1024;
};

} // namespace video_input
} // namespace modules
} // namespace smart_video_analysis

#endif // MODULES_VIDEO_INPUT_VIDEO_INPUT_HPP
