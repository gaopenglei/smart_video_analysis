/**
 * @file main.cpp
 * @brief 智能视频分析系统主程序入口
 * 
 * 提供命令行接口，支持视频文件处理和实时摄像头分析。
 */

#include <iostream>
#include <string>
#include <csignal>
#include <atomic>
#include <chrono>

#include "SmartVideoAnalysisSystem.hpp"
#include "core/Logger.hpp"
#include "core/Config.hpp"

using namespace smart_video_analysis;

// 全局系统实例
std::atomic<bool> g_running(true);
SmartVideoAnalysisSystem* g_system = nullptr;

/**
 * @brief 信号处理函数
 */
void signalHandler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    g_running = false;
    
    if (g_system) {
        g_system->stop();
    }
}

/**
 * @brief 打印使用说明
 */
void printUsage(const char* program_name) {
    std::cout << "Smart Video Analysis System v" << PROJECT_VERSION << std::endl;
    std::cout << "================================" << std::endl;
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help              Show this help message" << std::endl;
    std::cout << "  -c, --config <file>     Configuration file path" << std::endl;
    std::cout << "  -m, --model <file>      ONNX model file path" << std::endl;
    std::cout << "  -i, --input <source>    Input source (camera index, video file, or RTSP URL)" << std::endl;
    std::cout << "  -o, --output <file>     Output video file path" << std::endl;
    std::cout << "  -t, --type <type>       Model type (yolov8, yolov5, resnet, mobilenet)" << std::endl;
    std::cout << "  --confidence <value>    Confidence threshold (0.0-1.0)" << std::endl;
    std::cout << "  --nms <value>           NMS threshold (0.0-1.0)" << std::endl;
    std::cout << "  --no-display            Disable display window" << std::endl;
    std::cout << "  --verbose               Enable verbose output" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " -m model.onnx -i video.mp4" << std::endl;
    std::cout << "  " << program_name << " -m model.onnx -i 0 -t yolov8" << std::endl;
    std::cout << "  " << program_name << " -c config.yaml -i rtsp://localhost/stream" << std::endl;
}

/**
 * @brief 命令行参数结构体
 */
struct CommandLineArgs {
    std::string config_file;
    std::string model_path;
    std::string input_source;
    std::string output_path;
    std::string model_type = "yolov8";
    float confidence_threshold = 0.25f;
    float nms_threshold = 0.45f;
    bool no_display = false;
    bool verbose = false;
    bool show_help = false;
};

/**
 * @brief 解析命令行参数
 */
bool parseCommandLine(int argc, char* argv[], CommandLineArgs& args) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            args.show_help = true;
            return true;
        }
        else if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            args.config_file = argv[++i];
        }
        else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            args.model_path = argv[++i];
        }
        else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            args.input_source = argv[++i];
        }
        else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            args.output_path = argv[++i];
        }
        else if ((arg == "-t" || arg == "--type") && i + 1 < argc) {
            args.model_type = argv[++i];
        }
        else if (arg == "--confidence" && i + 1 < argc) {
            args.confidence_threshold = std::stof(argv[++i]);
        }
        else if (arg == "--nms" && i + 1 < argc) {
            args.nms_threshold = std::stof(argv[++i]);
        }
        else if (arg == "--no-display") {
            args.no_display = true;
        }
        else if (arg == "--verbose") {
            args.verbose = true;
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return false;
        }
    }
    
    return true;
}

/**
 * @brief 检测回调函数
 */
void onDetectionResult(const postprocessing::DetectionResult& result,
                       const cv::Mat& visualized_frame) {
    // 打印检测信息
    if (!result.detections.empty()) {
        LOG_INFO("Frame %ld: Detected %zu objects", 
                 result.frame_id, result.detections.size());
        
        for (const auto& det : result.detections) {
            LOG_DEBUG("  - %s: %.2f (%.1f, %.1f, %.1f, %.1f)",
                      det.class_name.c_str(), det.confidence,
                      det.x1, det.y1, det.x2, det.y2);
        }
    }
}

/**
 * @brief 错误回调函数
 */
void onError(core::ErrorCode error_code, const std::string& error_message) {
    LOG_ERROR("Error [%d]: %s", static_cast<int>(error_code), error_message.c_str());
}

/**
 * @brief 主函数
 */
int main(int argc, char* argv[]) {
    // 解析命令行参数
    CommandLineArgs args;
    if (!parseCommandLine(argc, argv, args)) {
        printUsage(argv[0]);
        return 1;
    }
    
    if (args.show_help) {
        printUsage(argv[0]);
        return 0;
    }
    
    // 初始化日志系统
    core::LogConfig log_config;
    log_config.min_level = args.verbose ? core::LogLevel::DEBUG : core::LogLevel::INFO;
    log_config.console_output = true;
    log_config.file_output = true;
    log_config.log_file_path = "./logs/smart_video_analysis.log";
    
    auto& logger = core::Logger::getInstance();
    if (!logger.initialize(log_config)) {
        std::cerr << "Failed to initialize logger" << std::endl;
        return 1;
    }
    
    LOG_INFO("========================================");
    LOG_INFO("Smart Video Analysis System v%s", PROJECT_VERSION);
    LOG_INFO("========================================");
    
    // 注册信号处理
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    // 创建系统实例
    auto system = createSystem();
    if (!system) {
        LOG_ERROR("Failed to create system");
        return 1;
    }
    
    g_system = system.get();
    
    // 初始化系统
    core::ErrorCode ret;
    if (!args.config_file.empty()) {
        ret = system->initialize(args.config_file);
    } else {
        ret = system->initialize();
    }
    
    if (ret != core::ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to initialize system: %d", static_cast<int>(ret));
        return 1;
    }
    
    // 设置模型
    if (!args.model_path.empty()) {
        ret = system->setModel(args.model_path, args.model_type);
        if (ret != core::ErrorCode::SUCCESS) {
            LOG_ERROR("Failed to set model: %d", static_cast<int>(ret));
            return 1;
        }
    }
    
    // 设置阈值
    system->setConfidenceThreshold(args.confidence_threshold);
    system->setNMSThreshold(args.nms_threshold);
    
    // 设置回调
    system->setDetectionCallback(onDetectionResult);
    system->setErrorCallback(onError);
    
    // 确定输入源类型
    std::string source_type = "camera";
    std::string source_path = args.input_source;
    
    if (!source_path.empty()) {
        // 检查是否是数字（摄像头索引）
        bool is_camera = true;
        for (char c : source_path) {
            if (!std::isdigit(c)) {
                is_camera = false;
                break;
            }
        }
        
        if (!is_camera) {
            // 检查是否是网络流
            if (source_path.find("rtsp://") == 0 || 
                source_path.find("rtmp://") == 0 ||
                source_path.find("http://") == 0) {
                source_type = "rtsp";
            } else {
                source_type = "file";
            }
        }
        
        ret = system->setVideoSource(source_type, source_path);
        if (ret != core::ErrorCode::SUCCESS) {
            LOG_ERROR("Failed to set video source: %d", static_cast<int>(ret));
            return 1;
        }
    }
    
    // 打印配置信息
    LOG_INFO("Configuration:");
    LOG_INFO("  Model: %s", args.model_path.c_str());
    LOG_INFO("  Model Type: %s", args.model_type.c_str());
    LOG_INFO("  Input Source: %s (%s)", source_path.c_str(), source_type.c_str());
    LOG_INFO("  Confidence Threshold: %.2f", args.confidence_threshold);
    LOG_INFO("  NMS Threshold: %.2f", args.nms_threshold);
    LOG_INFO("  Display: %s", args.no_display ? "disabled" : "enabled");
    
    // 启动系统
    ret = system->start();
    if (ret != core::ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to start system: %d", static_cast<int>(ret));
        return 1;
    }
    
    LOG_INFO("System started. Press Ctrl+C to stop.");
    
    // 主循环
    while (g_running && system->getState() == SystemState::RUNNING) {
        // 获取统计信息
        auto stats = system->getStats();
        
        // 打印统计信息
        if (args.verbose && stats.total_frames_processed % 30 == 0) {
            LOG_INFO("Stats: FPS=%.1f, Frames=%ld, Avg Inference=%.1fms",
                     stats.average_fps,
                     stats.total_frames_processed,
                     stats.average_inference_time_ms);
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // 停止系统
    LOG_INFO("Stopping system...");
    system->stop();
    
    // 打印最终统计
    auto final_stats = system->getStats();
    LOG_INFO("Final Statistics:");
    LOG_INFO("  Total Frames Processed: %ld", final_stats.total_frames_processed);
    LOG_INFO("  Average FPS: %.1f", final_stats.average_fps);
    LOG_INFO("  Average Inference Time: %.1f ms", final_stats.average_inference_time_ms);
    LOG_INFO("  Total Runtime: %.1f seconds", final_stats.elapsed_time_ms / 1000.0);
    
    // 关闭系统
    system->shutdown();
    logger.shutdown();
    
    LOG_INFO("System shutdown complete. Goodbye!");
    
    return 0;
}
