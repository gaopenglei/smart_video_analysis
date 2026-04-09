/**
 * @file Logger.cpp
 * @brief 日志记录模块实现文件
 */

#include "core/Logger.hpp"
#include <iostream>
#include <cstdarg>
#include <cstdio>
#include <sys/stat.h>
#include <cstring>

namespace smart_video_analysis {
namespace core {

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

Logger::~Logger() {
    shutdown();
}

bool Logger::initialize(const LogConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) {
        return true;
    }
    
    config_ = config;
    
    // 如果需要文件输出，打开日志文件
    if (config_.file_output) {
        // 创建日志目录
        size_t pos = config_.log_file_path.find_last_of("/\\");
        if (pos != std::string::npos) {
            std::string dir = config_.log_file_path.substr(0, pos);
            mkdir(dir.c_str(), 0755);
        }
        
        log_file_.open(config_.log_file_path, std::ios::app);
        if (!log_file_.is_open()) {
            std::cerr << "Failed to open log file: " << config_.log_file_path << std::endl;
            return false;
        }
    }
    
    initialized_ = true;
    return true;
}

void Logger::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        return;
    }
    
    if (log_file_.is_open()) {
        log_file_.flush();
        log_file_.close();
    }
    
    initialized_ = false;
}

void Logger::log(LogLevel level, const char* file, int line,
                 const char* function, const char* format, ...) {
    if (!initialized_ || level < config_.min_level) {
        return;
    }
    
    // 格式化消息
    char buffer[4096];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    
    // 创建日志条目
    LogEntry entry;
    entry.level = level;
    entry.message = buffer;
    entry.file = file;
    entry.line = line;
    entry.function = function;
    entry.timestamp = std::chrono::system_clock::now();
    entry.thread_id = std::this_thread::get_id();
    
    // 加锁输出
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (config_.console_output) {
        outputToConsole(entry);
    }
    
    if (config_.file_output) {
        checkFileRotation();
        outputToFile(entry);
    }
}

std::string Logger::levelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO";
        case LogLevel::WARN:  return "WARN";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::FATAL: return "FATAL";
        default: return "UNKNOWN";
    }
}

LogLevel Logger::stringToLevel(const std::string& level_str) {
    if (level_str == "DEBUG") return LogLevel::DEBUG;
    if (level_str == "INFO")  return LogLevel::INFO;
    if (level_str == "WARN")  return LogLevel::WARN;
    if (level_str == "ERROR") return LogLevel::ERROR;
    if (level_str == "FATAL") return LogLevel::FATAL;
    return LogLevel::INFO;
}

void Logger::setMinLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.min_level = level;
}

LogLevel Logger::getMinLevel() const {
    return config_.min_level;
}

void Logger::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (log_file_.is_open()) {
        log_file_.flush();
    }
    std::cout.flush();
    std::cerr.flush();
}

std::string Logger::formatEntry(const LogEntry& entry) {
    std::ostringstream oss;
    
    // 时间戳
    if (config_.include_timestamp) {
        auto time_t_val = std::chrono::system_clock::to_time_t(entry.timestamp);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            entry.timestamp.time_since_epoch()) % 1000;
        
        std::tm tm_val;
        localtime_r(&time_t_val, &tm_val);
        
        oss << std::put_time(&tm_val, "%Y-%m-%d %H:%M:%S");
        oss << "." << std::setfill('0') << std::setw(3) << ms.count();
        oss << " ";
    }
    
    // 日志级别
    oss << "[" << std::setw(5) << levelToString(entry.level) << "] ";
    
    // 线程ID
    if (config_.include_thread_id) {
        oss << "[" << entry.thread_id << "] ";
    }
    
    // 文件信息
    if (config_.include_file_info) {
        // 提取文件名（不含路径）
        std::string filename = entry.file;
        size_t pos = filename.find_last_of("/\\");
        if (pos != std::string::npos) {
            filename = filename.substr(pos + 1);
        }
        oss << "[" << filename << ":" << entry.line << "][" << entry.function << "] ";
    }
    
    // 消息内容
    oss << entry.message;
    
    return oss.str();
}

void Logger::outputToConsole(const LogEntry& entry) {
    std::string formatted = formatEntry(entry);
    
    // 根据日志级别选择输出流和颜色
    std::string color = getLevelColor(entry.level);
    std::string reset = "\033[0m";
    
    if (entry.level >= LogLevel::ERROR) {
        std::cerr << color << formatted << reset << std::endl;
    } else {
        std::cout << color << formatted << reset << std::endl;
    }
}

void Logger::outputToFile(const LogEntry& entry) {
    if (!log_file_.is_open()) {
        return;
    }
    
    std::string formatted = formatEntry(entry);
    log_file_ << formatted << std::endl;
}

void Logger::checkFileRotation() {
    if (!log_file_.is_open()) {
        return;
    }
    
    // 获取当前文件大小
    log_file_.seekp(0, std::ios::end);
    std::streampos file_size = log_file_.tellp();
    
    if (static_cast<size_t>(file_size) >= config_.max_file_size) {
        rotateLogFile();
    }
}

void Logger::rotateLogFile() {
    // 关闭当前文件
    log_file_.close();
    
    // 删除最旧的备份文件
    std::string oldest_backup = config_.log_file_path + "." + 
                                std::to_string(config_.max_backup_files);
    remove(oldest_backup.c_str());
    
    // 重命名现有备份文件
    for (int i = config_.max_backup_files - 1; i >= 1; --i) {
        std::string old_name = config_.log_file_path + "." + std::to_string(i);
        std::string new_name = config_.log_file_path + "." + std::to_string(i + 1);
        rename(old_name.c_str(), new_name.c_str());
    }
    
    // 重命名当前日志文件
    std::string backup_name = config_.log_file_path + ".1";
    rename(config_.log_file_path.c_str(), backup_name.c_str());
    
    // 打开新的日志文件
    log_file_.open(config_.log_file_path, std::ios::app);
}

std::string Logger::getCurrentTimeString() {
    auto now = std::chrono::system_clock::now();
    auto time_t_val = std::chrono::system_clock::to_time_t(now);
    
    std::tm tm_val;
    localtime_r(&time_t_val, &tm_val);
    
    std::ostringstream oss;
    oss << std::put_time(&tm_val, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

std::string Logger::getLevelColor(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "\033[36m";  // 青色
        case LogLevel::INFO:  return "\033[32m";  // 绿色
        case LogLevel::WARN:  return "\033[33m";  // 黄色
        case LogLevel::ERROR: return "\033[31m";  // 红色
        case LogLevel::FATAL: return "\033[35m";  // 紫色
        default: return "\033[0m";
    }
}

} // namespace core
} // namespace smart_video_analysis
