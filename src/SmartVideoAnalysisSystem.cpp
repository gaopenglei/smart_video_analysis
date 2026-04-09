/**
 * @file SmartVideoAnalysisSystem.cpp
 * @brief 智能视频分析系统实现文件
 */

#include "SmartVideoAnalysisSystem.hpp"
#include <chrono>
#include <fstream>
#include <sstream>

namespace smart_video_analysis {

// ============================================================================
// SmartVideoAnalysisSystem 实现
// ============================================================================

SmartVideoAnalysisSystem::SmartVideoAnalysisSystem()
    : state_(SystemState::UNINITIALIZED),
      stop_requested_(false),
      pause_requested_(false) {
    LOG_DEBUG("SmartVideoAnalysisSystem created");
}

SmartVideoAnalysisSystem::~SmartVideoAnalysisSystem() {
    shutdown();
    LOG_DEBUG("SmartVideoAnalysisSystem destroyed");
}

core::ErrorCode SmartVideoAnalysisSystem::initialize(const std::string& config_file) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (state_ != SystemState::UNINITIALIZED) {
        LOG_WARN("System already initialized");
        return core::ErrorCode::SUCCESS;
    }
    
    config_file_path_ = config_file;
    
    // 加载配置
    auto& config_manager = core::ConfigManager::getInstance();
    if (!config_manager.loadFromFile(config_file)) {
        LOG_ERROR("Failed to load config file: %s", config_file.c_str());
        return core::ErrorCode::INVALID_CONFIG;
    }
    
    config_ = config_manager.getSystemConfig();
    
    // 初始化模块
    if (!initializeModules()) {
        LOG_ERROR("Failed to initialize modules");
        return core::ErrorCode::MODEL_ERROR;
    }
    
    // 初始化类别名称
    initializeClassNames();
    
    state_ = SystemState::INITIALIZED;
    stats_.start_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    LOG_INFO("SmartVideoAnalysisSystem initialized successfully");
    return core::ErrorCode::SUCCESS;
}

core::ErrorCode SmartVideoAnalysisSystem::initialize() {
    // 使用默认配置
    auto& config_manager = core::ConfigManager::getInstance();
    config_manager.reset();
    config_ = config_manager.getSystemConfig();
    
    if (!initializeModules()) {
        LOG_ERROR("Failed to initialize modules");
        return core::ErrorCode::MODEL_ERROR;
    }
    
    initializeClassNames();
    
    state_ = SystemState::INITIALIZED;
    stats_.start_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    LOG_INFO("SmartVideoAnalysisSystem initialized with default config");
    return core::ErrorCode::SUCCESS;
}

bool SmartVideoAnalysisSystem::initializeModules() {
    // 初始化视频输入模块
    video_input_ = std::make_unique<modules::video_input::VideoInputManager>(
        config_.video_input);
    
    // 初始化预处理模块
    preprocessor_ = modules::preprocessing::PreprocessorFactory::create(
        config_.model.model_type, config_.preprocess);
    
    if (!preprocessor_) {
        LOG_ERROR("Failed to create preprocessor");
        return false;
    }
    
    // 初始化推理引擎
    inference_engine_ = modules::inference::InferenceEngineFactory::create(
        config_.inference);
    
    if (!inference_engine_) {
        LOG_ERROR("Failed to create inference engine");
        return false;
    }
    
    // 加载模型
    if (!inference_engine_->loadModel(config_.model.model_path)) {
        LOG_ERROR("Failed to load model: %s", config_.model.model_path.c_str());
        return false;
    }
    
    // 检查算子支持性
    if (!checkOperatorSupport(config_.model.model_path)) {
        LOG_WARN("Some operators may not be supported on target NPU");
    }
    
    // 初始化后处理模块
    postprocessor_ = modules::postprocessing::PostprocessorFactory::create(
        config_.model.model_type, config_.postprocess);
    
    if (!postprocessor_) {
        LOG_ERROR("Failed to create postprocessor");
        return false;
    }
    
    // 初始化可视化模块
    visualizer_ = std::make_unique<modules::visualization::VisualizationManager>();
    if (!visualizer_->initialize(config_.visualization)) {
        LOG_ERROR("Failed to initialize visualizer");
        return false;
    }
    
    // 初始化算子适配器
    operator_adapter_ = modules::operator_adapter::OperatorAdapterFactory::create(
        modules::operator_adapter::NPUType::RK3588);
    
    LOG_INFO("All modules initialized successfully");
    return true;
}

void SmartVideoAnalysisSystem::initializeClassNames() {
    // COCO数据集类别名称
    class_names_ = {
        "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird",
        "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "couch", "potted plant", "bed",
        "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    
    if (visualizer_) {
        visualizer_->setClassNames(class_names_);
    }
}

core::ErrorCode SmartVideoAnalysisSystem::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (state_ != SystemState::INITIALIZED && state_ != SystemState::STOPPED) {
        LOG_ERROR("Cannot start: invalid state");
        return core::ErrorCode::INVALID_PARAMETER;
    }
    
    // 打开视频输入
    if (!video_input_->open()) {
        LOG_ERROR("Failed to open video input");
        return core::ErrorCode::VIDEO_INPUT_ERROR;
    }
    
    stop_requested_ = false;
    pause_requested_ = false;
    
    // 启动处理线程
    processing_thread_ = std::thread(&SmartVideoAnalysisSystem::processingLoop, this);
    
    state_ = SystemState::RUNNING;
    LOG_INFO("System started");
    
    return core::ErrorCode::SUCCESS;
}

core::ErrorCode SmartVideoAnalysisSystem::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (state_ != SystemState::RUNNING && state_ != SystemState::PAUSED) {
            return core::ErrorCode::SUCCESS;
        }
        
        stop_requested_ = true;
        cv_.notify_all();
    }
    
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
    
    video_input_->close();
    state_ = SystemState::STOPPED;
    
    LOG_INFO("System stopped");
    return core::ErrorCode::SUCCESS;
}

core::ErrorCode SmartVideoAnalysisSystem::pause() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (state_ != SystemState::RUNNING) {
        return core::ErrorCode::INVALID_PARAMETER;
    }
    
    pause_requested_ = true;
    state_ = SystemState::PAUSED;
    
    LOG_INFO("System paused");
    return core::ErrorCode::SUCCESS;
}

core::ErrorCode SmartVideoAnalysisSystem::resume() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (state_ != SystemState::PAUSED) {
        return core::ErrorCode::INVALID_PARAMETER;
    }
    
    pause_requested_ = false;
    cv_.notify_all();
    state_ = SystemState::RUNNING;
    
    LOG_INFO("System resumed");
    return core::ErrorCode::SUCCESS;
}

void SmartVideoAnalysisSystem::shutdown() {
    stop();
    
    // 释放模块
    video_input_.reset();
    preprocessor_.reset();
    inference_engine_.reset();
    postprocessor_.reset();
    visualizer_.reset();
    operator_adapter_.reset();
    
    state_ = SystemState::UNINITIALIZED;
    LOG_INFO("System shutdown complete");
}

SystemState SmartVideoAnalysisSystem::getState() const {
    return state_;
}

SystemStats SmartVideoAnalysisSystem::getStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void SmartVideoAnalysisSystem::setDetectionCallback(DetectionCallback callback) {
    detection_callback_ = callback;
}

void SmartVideoAnalysisSystem::setErrorCallback(ErrorCallback callback) {
    error_callback_ = callback;
}

core::ErrorCode SmartVideoAnalysisSystem::processFrame(
    const cv::Mat& frame,
    postprocessing::DetectionResult& result) {
    
    if (frame.empty()) {
        return core::ErrorCode::INVALID_PARAMETER;
    }
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // 预处理
    modules::preprocessing::PreprocessResult preprocess_result;
    if (!preprocessor_->process(frame, preprocess_result)) {
        LOG_ERROR("Preprocessing failed");
        return core::ErrorCode::PREPROCESS_ERROR;
    }
    
    // 推理
    modules::inference::InferenceResult inference_result;
    if (!inference_engine_->infer(preprocess_result.input_tensor, inference_result)) {
        LOG_ERROR("Inference failed");
        return core::ErrorCode::INFERENCE_ERROR;
    }
    
    // 后处理
    result.original_width = frame.cols;
    result.original_height = frame.rows;
    
    if (!postprocessor_->process(inference_result.outputs, result)) {
        LOG_ERROR("Postprocessing failed");
        return core::ErrorCode::POSTPROCESS_ERROR;
    }
    
    // 更新统计
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(
        total_end - total_start).count();
    
    updateStats(inference_result.inference_time_ms, total_time);
    
    return core::ErrorCode::SUCCESS;
}

core::ErrorCode SmartVideoAnalysisSystem::processVideo(
    const std::string& video_path,
    const std::string& output_path) {
    
    // 设置视频源
    auto ret = setVideoSource("file", video_path);
    if (ret != core::ErrorCode::SUCCESS) {
        return ret;
    }
    
    // 打开视频输入
    if (!video_input_->open()) {
        return core::ErrorCode::VIDEO_FILE_OPEN_FAILED;
    }
    
    // 如果需要输出视频
    modules::visualization::VideoOutputWriter writer;
    if (!output_path.empty()) {
        int width = video_input_->getWidth();
        int height = video_input_->getHeight();
        double fps = video_input_->getFPS();
        
        if (!writer.open(output_path, width, height, fps)) {
            LOG_WARN("Failed to open output video writer");
        }
    }
    
    // 处理所有帧
    modules::video_input::VideoFrame video_frame;
    while (video_input_->readFrame(video_frame)) {
        // 处理帧
        postprocessing::DetectionResult result;
        auto ret = processFrame(video_frame.frame, result);
        
        if (ret != core::ErrorCode::SUCCESS) {
            LOG_WARN("Failed to process frame %ld", video_frame.frame_id);
            continue;
        }
        
        // 可视化
        cv::Mat visualized;
        visualizer_->visualize(video_frame.frame, result, visualized);
        
        // 写入输出
        if (writer.isOpened()) {
            writer.write(visualized);
        }
        
        // 调用回调
        if (detection_callback_) {
            detection_callback_(result, visualized);
        }
    }
    
    video_input_->close();
    writer.close();
    
    LOG_INFO("Video processing completed: %s", video_path.c_str());
    return core::ErrorCode::SUCCESS;
}

core::ErrorCode SmartVideoAnalysisSystem::setModel(
    const std::string& model_path,
    const std::string& model_type) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (state_ == SystemState::RUNNING) {
        LOG_ERROR("Cannot change model while running");
        return core::ErrorCode::INVALID_PARAMETER;
    }
    
    config_.model.model_path = model_path;
    config_.model.model_type = model_type;
    
    // 重新创建预处理器和后处理器
    preprocessor_ = modules::preprocessing::PreprocessorFactory::create(
        model_type, config_.preprocess);
    
    postprocessor_ = modules::postprocessing::PostprocessorFactory::create(
        model_type, config_.postprocess);
    
    // 加载新模型
    if (inference_engine_ && !inference_engine_->loadModel(model_path)) {
        LOG_ERROR("Failed to load model: %s", model_path.c_str());
        return core::ErrorCode::MODEL_LOAD_FAILED;
    }
    
    LOG_INFO("Model changed: %s (type: %s)", model_path.c_str(), model_type.c_str());
    return core::ErrorCode::SUCCESS;
}

core::ErrorCode SmartVideoAnalysisSystem::setVideoSource(
    const std::string& source_type,
    const std::string& source_path) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (state_ == SystemState::RUNNING) {
        LOG_ERROR("Cannot change video source while running");
        return core::ErrorCode::INVALID_PARAMETER;
    }
    
    config_.video_input.source_type = source_type;
    config_.video_input.source_path = source_path;
    
    video_input_ = std::make_unique<modules::video_input::VideoInputManager>(
        config_.video_input);
    
    LOG_INFO("Video source changed: %s (%s)", source_path.c_str(), source_type.c_str());
    return core::ErrorCode::SUCCESS;
}

void SmartVideoAnalysisSystem::setConfidenceThreshold(float threshold) {
    config_.postprocess.confidence_threshold = threshold;
    if (postprocessor_) {
        postprocessor_->setConfidenceThreshold(threshold);
    }
}

void SmartVideoAnalysisSystem::setNMSThreshold(float threshold) {
    config_.postprocess.nms_threshold = threshold;
    if (postprocessor_) {
        postprocessor_->setNMSThreshold(threshold);
    }
}

void SmartVideoAnalysisSystem::setClassNames(const std::vector<std::string>& names) {
    class_names_ = names;
    if (visualizer_) {
        visualizer_->setClassNames(names);
    }
}

const core::SystemConfig& SmartVideoAnalysisSystem::getConfig() const {
    return config_;
}

core::ErrorCode SmartVideoAnalysisSystem::reloadConfig() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (state_ == SystemState::RUNNING) {
        LOG_ERROR("Cannot reload config while running");
        return core::ErrorCode::INVALID_PARAMETER;
    }
    
    auto& config_manager = core::ConfigManager::getInstance();
    if (!config_manager.loadFromFile(config_file_path_)) {
        return core::ErrorCode::INVALID_CONFIG;
    }
    
    config_ = config_manager.getSystemConfig();
    
    LOG_INFO("Configuration reloaded");
    return core::ErrorCode::SUCCESS;
}

void SmartVideoAnalysisSystem::processingLoop() {
    LOG_INFO("Processing loop started");
    
    modules::video_input::VideoFrame video_frame;
    
    while (!stop_requested_) {
        // 检查暂停
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { 
                return !pause_requested_ || stop_requested_; 
            });
        }
        
        if (stop_requested_) {
            break;
        }
        
        // 读取帧
        if (!video_input_->readFrame(video_frame)) {
            LOG_WARN("Failed to read frame, retrying...");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        // 处理帧
        processSingleFrame();
    }
    
    LOG_INFO("Processing loop ended");
}

bool SmartVideoAnalysisSystem::processSingleFrame() {
    modules::video_input::VideoFrame video_frame;
    
    if (!video_input_->readFrame(video_frame)) {
        return false;
    }
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // 预处理
    modules::preprocessing::PreprocessResult preprocess_result;
    if (!preprocessor_->process(video_frame.frame, preprocess_result)) {
        LOG_ERROR("Preprocessing failed");
        return false;
    }
    
    // 推理
    modules::inference::InferenceResult inference_result;
    if (!inference_engine_->infer(preprocess_result.input_tensor, inference_result)) {
        LOG_ERROR("Inference failed");
        return false;
    }
    
    // 后处理
    postprocessing::DetectionResult result;
    result.frame_id = video_frame.frame_id;
    result.timestamp_ms = video_frame.timestamp_ms;
    result.original_width = video_frame.width;
    result.original_height = video_frame.height;
    result.inference_time_ms = inference_result.inference_time_ms;
    
    if (!postprocessor_->process(inference_result.outputs, result)) {
        LOG_ERROR("Postprocessing failed");
        return false;
    }
    
    // 可视化
    cv::Mat visualized;
    visualizer_->visualize(video_frame.frame, result, visualized);
    visualizer_->updateFPS(video_frame.fps);
    
    // 更新统计
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(
        total_end - total_start).count();
    
    updateStats(inference_result.inference_time_ms, total_time);
    
    // 调用回调
    if (detection_callback_) {
        detection_callback_(result, visualized);
    }
    
    return true;
}

void SmartVideoAnalysisSystem::updateStats(double inference_time, double total_time) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    stats_.total_frames_processed++;
    
    // 计算移动平均
    double alpha = 0.1;  // 平滑因子
    stats_.average_inference_time_ms = 
        alpha * inference_time + (1 - alpha) * stats_.average_inference_time_ms;
    stats_.average_total_time_ms = 
        alpha * total_time + (1 - alpha) * stats_.average_total_time_ms;
    
    // 计算FPS
    if (stats_.average_total_time_ms > 0) {
        stats_.average_fps = 1000.0 / stats_.average_total_time_ms;
    }
    
    // 更新运行时间
    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    stats_.elapsed_time_ms = now - stats_.start_time_ms;
}

bool SmartVideoAnalysisSystem::checkOperatorSupport(const std::string& model_path) {
    if (!operator_adapter_) {
        return true;  // 没有适配器，假设支持
    }
    
    modules::operator_adapter::AdapterResult result;
    auto ret = operator_adapter_->checkOperatorSupport(model_path, result);
    
    if (ret != core::ErrorCode::SUCCESS) {
        LOG_WARN("Operator support check failed");
        return false;
    }
    
    if (!result.success) {
        LOG_WARN("Model has %zu unsupported operators", result.unsupported_ops.size());
        for (const auto& op : result.unsupported_ops) {
            LOG_WARN("  - %s (%s)", op.name.c_str(), op.op_type.c_str());
        }
        return false;
    }
    
    LOG_INFO("All operators are supported");
    return true;
}

// ============================================================================
// 工厂函数实现
// ============================================================================

std::unique_ptr<SmartVideoAnalysisSystem> createSystem() {
    return std::make_unique<SmartVideoAnalysisSystem>();
}

std::unique_ptr<SmartVideoAnalysisSystem> createSystem(const std::string& config_file) {
    auto system = std::make_unique<SmartVideoAnalysisSystem>();
    if (system->initialize(config_file) != core::ErrorCode::SUCCESS) {
        return nullptr;
    }
    return system;
}

} // namespace smart_video_analysis
