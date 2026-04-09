/**
 * @file VideoInput.cpp
 * @brief 视频输入模块实现文件
 */

#include "modules/video_input/VideoInput.hpp"
#include "core/Logger.hpp"
#include <chrono>

namespace smart_video_analysis {
namespace modules {
namespace video_input {

// ============================================================================
// VideoInputManager 实现
// ============================================================================

VideoInputManager::VideoInputManager(const core::VideoInputConfig& config)
    : config_(config) {
    resetStatistics();
}

VideoInputManager::~VideoInputManager() {
    close();
}

std::unique_ptr<IVideoInput> VideoInputManager::createInput() {
    std::string source_type = config_.source_type;
    std::transform(source_type.begin(), source_type.end(), 
                   source_type.begin(), ::tolower);
    
    if (source_type == "camera" || source_type == "webcam") {
        return std::make_unique<CameraInput>(config_.camera_index,
                                             config_.frame_width,
                                             config_.frame_height);
    } else if (source_type == "file" || source_type == "video") {
        return std::make_unique<FileInput>(config_.source_path);
    } else if (source_type == "rtsp" || source_type == "rtmp" || source_type == "http") {
        return std::make_unique<NetworkStreamInput>(config_.source_path, source_type);
    } else {
        LOG_ERROR("Unknown video source type: %s", config_.source_type.c_str());
        return nullptr;
    }
}

bool VideoInputManager::open() {
    if (isOpened()) {
        LOG_WARN("Video input already opened");
        return true;
    }
    
    input_impl_ = createInput();
    if (!input_impl_) {
        return false;
    }
    
    if (!input_impl_->open()) {
        LOG_ERROR("Failed to open video input");
        input_impl_.reset();
        return false;
    }
    
    LOG_INFO("Video input opened successfully");
    LOG_INFO("  Source: %s", config_.source_path.c_str());
    LOG_INFO("  Resolution: %dx%d", getWidth(), getHeight());
    LOG_INFO("  FPS: %.2f", getFPS());
    
    return true;
}

void VideoInputManager::close() {
    stopAsyncRead();
    
    if (input_impl_) {
        input_impl_->close();
        input_impl_.reset();
    }
    
    LOG_INFO("Video input closed");
}

bool VideoInputManager::readFrame(VideoFrame& frame) {
    if (!isOpened()) {
        LOG_ERROR("Video input not opened");
        return false;
    }
    
    bool result = input_impl_->readFrame(frame);
    
    if (result) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.total_frames_read++;
        
        // 计算帧率
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        if (last_frame_time_ > 0) {
            double frame_time = static_cast<double>(now - last_frame_time_);
            if (frame_time > 0) {
                stats_.current_fps = 1000.0 / frame_time;
                // 使用指数移动平均计算平均帧率
                stats_.average_fps = 0.9 * stats_.average_fps + 0.1 * stats_.current_fps;
            }
        }
        last_frame_time_ = now;
    }
    
    return result;
}

bool VideoInputManager::isOpened() const {
    return input_impl_ && input_impl_->isOpened();
}

int VideoInputManager::getWidth() const {
    return input_impl_ ? input_impl_->getWidth() : 0;
}

int VideoInputManager::getHeight() const {
    return input_impl_ ? input_impl_->getHeight() : 0;
}

double VideoInputManager::getFPS() const {
    return input_impl_ ? input_impl_->getFPS() : 0.0;
}

int64_t VideoInputManager::getTotalFrames() const {
    return input_impl_ ? input_impl_->getTotalFrames() : -1;
}

int64_t VideoInputManager::getCurrentFrame() const {
    return input_impl_ ? input_impl_->getCurrentFrame() : -1;
}

bool VideoInputManager::setFramePosition(int64_t frame_id) {
    return input_impl_ ? input_impl_->setFramePosition(frame_id) : false;
}

InputSourceType VideoInputManager::getSourceType() const {
    return input_impl_ ? input_impl_->getSourceType() : InputSourceType::UNKNOWN;
}

std::string VideoInputManager::getSourcePath() const {
    return input_impl_ ? input_impl_->getSourcePath() : "";
}

bool VideoInputManager::startAsyncRead(size_t buffer_size) {
    if (async_running_) {
        LOG_WARN("Async read already running");
        return true;
    }
    
    if (!isOpened()) {
        LOG_ERROR("Cannot start async read: video input not opened");
        return false;
    }
    
    max_buffer_size_ = buffer_size;
    async_running_ = true;
    end_of_stream_ = false;
    
    async_thread_ = std::thread(&VideoInputManager::asyncReadThread, this);
    
    LOG_INFO("Async read started with buffer size: %zu", buffer_size);
    return true;
}

void VideoInputManager::stopAsyncRead() {
    if (!async_running_) {
        return;
    }
    
    async_running_ = false;
    buffer_cv_.notify_all();
    
    if (async_thread_.joinable()) {
        async_thread_.join();
    }
    
    // 清空缓冲区
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    while (!frame_buffer_.empty()) {
        frame_buffer_.pop();
    }
    
    LOG_INFO("Async read stopped");
}

void VideoInputManager::asyncReadThread() {
    LOG_DEBUG("Async read thread started");
    
    while (async_running_) {
        VideoFrame frame;
        bool success = readFrame(frame);
        
        if (!success) {
            end_of_stream_ = true;
            LOG_DEBUG("End of stream reached in async mode");
            break;
        }
        
        {
            std::unique_lock<std::mutex> lock(buffer_mutex_);
            
            // 等待缓冲区有空间
            buffer_cv_.wait(lock, [this] {
                return frame_buffer_.size() < max_buffer_size_ || !async_running_;
            });
            
            if (!async_running_) {
                break;
            }
            
            // 如果缓冲区已满，丢弃最旧的帧
            if (frame_buffer_.size() >= max_buffer_size_) {
                frame_buffer_.pop();
                std::lock_guard<std::mutex> stats_lock(stats_mutex_);
                stats_.frames_dropped++;
            }
            
            frame_buffer_.push(frame);
        }
        
        buffer_cv_.notify_one();
    }
    
    LOG_DEBUG("Async read thread ended");
}

bool VideoInputManager::getFrameFromBuffer(VideoFrame& frame, int timeout_ms) {
    std::unique_lock<std::mutex> lock(buffer_mutex_);
    
    if (timeout_ms < 0) {
        buffer_cv_.wait(lock, [this] {
            return !frame_buffer_.empty() || !async_running_ || end_of_stream_;
        });
    } else {
        if (!buffer_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                 [this] { return !frame_buffer_.empty() || !async_running_ || end_of_stream_; })) {
            return false;  // 超时
        }
    }
    
    if (frame_buffer_.empty()) {
        return false;
    }
    
    frame = frame_buffer_.front();
    frame_buffer_.pop();
    buffer_cv_.notify_one();
    
    return true;
}

size_t VideoInputManager::getBufferSize() const {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    return frame_buffer_.size();
}

bool VideoInputManager::isEndOfStream() const {
    return end_of_stream_;
}

VideoInputManager::Statistics VideoInputManager::getStatistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void VideoInputManager::resetStatistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.total_frames_read = 0;
    stats_.frames_dropped = 0;
    stats_.average_fps = 0.0;
    stats_.current_fps = 0.0;
    last_frame_time_ = 0;
}

// ============================================================================
// CameraInput 实现
// ============================================================================

CameraInput::CameraInput(int camera_index, int width, int height)
    : camera_index_(camera_index)
    , target_width_(width)
    , target_height_(height) {
}

CameraInput::~CameraInput() {
    close();
}

bool CameraInput::open() {
    if (capture_.isOpened()) {
        return true;
    }
    
    capture_.open(camera_index_);
    if (!capture_.isOpened()) {
        LOG_ERROR("Failed to open camera %d", camera_index_);
        return false;
    }
    
    // 设置摄像头参数
    capture_.set(cv::CAP_PROP_FRAME_WIDTH, target_width_);
    capture_.set(cv::CAP_PROP_FRAME_HEIGHT, target_height_);
    capture_.set(cv::CAP_PROP_FPS, 30);
    capture_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    
    frame_counter_ = 0;
    
    LOG_INFO("Camera %d opened: %dx%d", camera_index_, getWidth(), getHeight());
    return true;
}

void CameraInput::close() {
    if (capture_.isOpened()) {
        capture_.release();
    }
    frame_counter_ = 0;
}

bool CameraInput::readFrame(VideoFrame& frame) {
    if (!capture_.isOpened()) {
        return false;
    }
    
    cv::Mat mat;
    if (!capture_.read(mat)) {
        return false;
    }
    
    frame.frame = mat;
    frame.frame_id = frame_counter_++;
    frame.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    frame.width = mat.cols;
    frame.height = mat.rows;
    frame.channels = mat.channels();
    frame.fps = getFPS();
    
    return true;
}

bool CameraInput::isOpened() const {
    return capture_.isOpened();
}

int CameraInput::getWidth() const {
    return static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
}

int CameraInput::getHeight() const {
    return static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
}

double CameraInput::getFPS() const {
    double fps = capture_.get(cv::CAP_PROP_FPS);
    return fps > 0 ? fps : 30.0;
}

int64_t CameraInput::getCurrentFrame() const {
    return frame_counter_;
}

std::string CameraInput::getSourcePath() const {
    return "camera:" + std::to_string(camera_index_);
}

// ============================================================================
// FileInput 实现
// ============================================================================

FileInput::FileInput(const std::string& file_path)
    : file_path_(file_path) {
}

FileInput::~FileInput() {
    close();
}

bool FileInput::open() {
    if (capture_.isOpened()) {
        return true;
    }
    
    capture_.open(file_path_);
    if (!capture_.isOpened()) {
        LOG_ERROR("Failed to open video file: %s", file_path_.c_str());
        return false;
    }
    
    frame_counter_ = 0;
    
    LOG_INFO("Video file opened: %s", file_path_.c_str());
    LOG_INFO("  Resolution: %dx%d", getWidth(), getHeight());
    LOG_INFO("  FPS: %.2f", getFPS());
    LOG_INFO("  Total frames: %ld", getTotalFrames());
    
    return true;
}

void FileInput::close() {
    if (capture_.isOpened()) {
        capture_.release();
    }
    frame_counter_ = 0;
}

bool FileInput::readFrame(VideoFrame& frame) {
    if (!capture_.isOpened()) {
        return false;
    }
    
    cv::Mat mat;
    if (!capture_.read(mat)) {
        return false;
    }
    
    frame.frame = mat;
    frame.frame_id = frame_counter_++;
    frame.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    frame.width = mat.cols;
    frame.height = mat.rows;
    frame.channels = mat.channels();
    frame.fps = getFPS();
    
    return true;
}

bool FileInput::isOpened() const {
    return capture_.isOpened();
}

int FileInput::getWidth() const {
    return static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
}

int FileInput::getHeight() const {
    return static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
}

double FileInput::getFPS() const {
    double fps = capture_.get(cv::CAP_PROP_FPS);
    return fps > 0 ? fps : 25.0;
}

int64_t FileInput::getTotalFrames() const {
    return static_cast<int64_t>(capture_.get(cv::CAP_PROP_FRAME_COUNT));
}

int64_t FileInput::getCurrentFrame() const {
    return static_cast<int64_t>(capture_.get(cv::CAP_PROP_POS_FRAMES));
}

bool FileInput::setFramePosition(int64_t frame_id) {
    if (!capture_.isOpened()) {
        return false;
    }
    capture_.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(frame_id));
    frame_counter_ = frame_id;
    return true;
}

// ============================================================================
// NetworkStreamInput 实现
// ============================================================================

NetworkStreamInput::NetworkStreamInput(const std::string& stream_url, const std::string& protocol)
    : stream_url_(stream_url)
    , protocol_(protocol) {
}

NetworkStreamInput::~NetworkStreamInput() {
    close();
}

bool NetworkStreamInput::open() {
    if (capture_.isOpened()) {
        return true;
    }
    
    // 设置网络流参数
    if (protocol_ == "rtsp") {
        // RTSP特定设置
        capture_.set(cv::CAP_PROP_FORMAT, CV_8UC3);
    }
    
    capture_.open(stream_url_);
    if (!capture_.isOpened()) {
        LOG_ERROR("Failed to open network stream: %s", stream_url_.c_str());
        return false;
    }
    
    frame_counter_ = 0;
    
    LOG_INFO("Network stream opened: %s", stream_url_.c_str());
    LOG_INFO("  Resolution: %dx%d", getWidth(), getHeight());
    
    return true;
}

void NetworkStreamInput::close() {
    if (capture_.isOpened()) {
        capture_.release();
    }
    frame_counter_ = 0;
}

bool NetworkStreamInput::readFrame(VideoFrame& frame) {
    if (!capture_.isOpened()) {
        // 尝试重连
        if (!reconnect()) {
            return false;
        }
    }
    
    cv::Mat mat;
    if (!capture_.read(mat)) {
        // 读取失败，尝试重连
        if (reconnect()) {
            return readFrame(frame);
        }
        return false;
    }
    
    frame.frame = mat;
    frame.frame_id = frame_counter_++;
    frame.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    frame.width = mat.cols;
    frame.height = mat.rows;
    frame.channels = mat.channels();
    frame.fps = getFPS();
    
    return true;
}

bool NetworkStreamInput::isOpened() const {
    return capture_.isOpened();
}

int NetworkStreamInput::getWidth() const {
    return static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
}

int NetworkStreamInput::getHeight() const {
    return static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
}

double NetworkStreamInput::getFPS() const {
    double fps = capture_.get(cv::CAP_PROP_FPS);
    return fps > 0 ? fps : 25.0;
}

int64_t NetworkStreamInput::getCurrentFrame() const {
    return frame_counter_;
}

InputSourceType NetworkStreamInput::getSourceType() const {
    if (protocol_ == "rtsp") return InputSourceType::RTSP_STREAM;
    if (protocol_ == "http" || protocol_ == "https") return InputSourceType::HTTP_STREAM;
    return InputSourceType::UNKNOWN;
}

void NetworkStreamInput::setReconnectParams(int max_retries, int retry_interval_ms) {
    max_retries_ = max_retries;
    retry_interval_ms_ = retry_interval_ms;
}

void NetworkStreamInput::setBufferSize(int buffer_size) {
    buffer_size_ = buffer_size;
}

bool NetworkStreamInput::reconnect() {
    LOG_WARN("Attempting to reconnect to stream: %s", stream_url_.c_str());
    
    for (int i = 0; i < max_retries_; ++i) {
        capture_.release();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(retry_interval_ms_));
        
        capture_.open(stream_url_);
        if (capture_.isOpened()) {
            LOG_INFO("Reconnected to stream successfully");
            return true;
        }
        
        LOG_WARN("Reconnect attempt %d/%d failed", i + 1, max_retries_);
    }
    
    LOG_ERROR("Failed to reconnect after %d attempts", max_retries_);
    return false;
}

} // namespace video_input
} // namespace modules
} // namespace smart_video_analysis
