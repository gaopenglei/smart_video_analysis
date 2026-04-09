/**
 * @file Logger.hpp
 * @brief 日志记录模块头文件
 * @author Smart Video Analysis Team
 * @version 1.0.0
 * 
 * 提供多级别日志记录功能，支持控制台和文件输出，
 * 包含调试、信息、警告、错误四个级别。
 */

#ifndef CORE_LOGGER_HPP
#define CORE_LOGGER_HPP

#include <string>
#include <fstream>
#include <mutex>
#include <memory>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <vector>

namespace smart_video_analysis {
namespace core {

/**
 * @brief 日志级别枚举
 */
enum class LogLevel {
    DEBUG = 0,      ///< 调试级别，用于详细的调试信息
    INFO = 1,       ///< 信息级别，用于一般运行信息
    WARN = 2,       ///< 警告级别，用于潜在问题提示
    ERROR = 3,      ///< 错误级别，用于错误报告
    FATAL = 4       ///< 致命错误级别，用于严重错误
};

/**
 * @brief 日志配置结构体
 */
struct LogConfig {
    LogLevel min_level = LogLevel::INFO;    ///< 最小日志级别
    bool console_output = true;              ///< 是否输出到控制台
    bool file_output = true;                 ///< 是否输出到文件
    std::string log_file_path = "./logs/smart_video_analysis.log";  ///< 日志文件路径
    size_t max_file_size = 10 * 1024 * 1024; ///< 最大文件大小 (10MB)
    int max_backup_files = 5;                ///< 最大备份文件数量
    bool include_timestamp = true;           ///< 是否包含时间戳
    bool include_file_info = true;           ///< 是否包含文件和行号信息
    bool include_thread_id = true;           ///< 是否包含线程ID
};

/**
 * @brief 日志记录条目结构体
 */
struct LogEntry {
    LogLevel level;                          ///< 日志级别
    std::string message;                     ///< 日志消息
    std::string file;                        ///< 源文件名
    int line;                                ///< 源文件行号
    std::string function;                    ///< 函数名
    std::chrono::system_clock::time_point timestamp;  ///< 时间戳
    std::thread::id thread_id;               ///< 线程ID
};

/**
 * @brief 日志记录器类
 * 
 * 实现线程安全的日志记录功能，支持多级别日志、
 * 控制台和文件输出、日志文件轮转等功能。
 * 
 * @example
 * @code
 * auto& logger = Logger::getInstance();
 * logger.initialize(LogConfig{LogLevel::DEBUG, true, true, "./logs/app.log"});
 * LOG_INFO("Application started");
 * LOG_ERROR("Failed to load model: {}", model_path);
 * @endcode
 */
class Logger {
public:
    /**
     * @brief 获取单例实例
     * @return Logger单例引用
     */
    static Logger& getInstance();

    /**
     * @brief 初始化日志记录器
     * @param config 日志配置
     * @return 成功返回true，失败返回false
     */
    bool initialize(const LogConfig& config);

    /**
     * @brief 关闭日志记录器
     */
    void shutdown();

    /**
     * @brief 记录日志
     * @param level 日志级别
     * @param file 源文件名
     * @param line 源文件行号
     * @param function 函数名
     * @param format 格式化字符串
     * @param ... 格式化参数
     */
    void log(LogLevel level, const char* file, int line, 
             const char* function, const char* format, ...);

    /**
     * @brief 记录日志（模板版本）
     * @tparam Args 参数类型
     * @param level 日志级别
     * @param file 源文件名
     * @param line 源文件行号
     * @param function 函数名
     * @param message 日志消息
     */
    template<typename... Args>
    void log(LogLevel level, const char* file, int line,
             const char* function, const std::string& message);

    /**
     * @brief 设置最小日志级别
     * @param level 最小日志级别
     */
    void setMinLevel(LogLevel level);

    /**
     * @brief 获取当前最小日志级别
     * @return 当前最小日志级别
     */
    LogLevel getMinLevel() const;

    /**
     * @brief 刷新日志缓冲区
     */
    void flush();

    /**
     * @brief 将日志级别转换为字符串
     * @param level 日志级别
     * @return 日志级别字符串
     */
    static std::string levelToString(LogLevel level);

    /**
     * @brief 将字符串转换为日志级别
     * @param level_str 日志级别字符串
     * @return 日志级别
     */
    static LogLevel stringToLevel(const std::string& level_str);

private:
    // 私有构造函数（单例模式）
    Logger() = default;
    ~Logger();

    // 禁止拷贝和赋值
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    /**
     * @brief 格式化日志条目
     * @param entry 日志条目
     * @return 格式化后的字符串
     */
    std::string formatEntry(const LogEntry& entry);

    /**
     * @brief 输出到控制台
     * @param entry 日志条目
     */
    void outputToConsole(const LogEntry& entry);

    /**
     * @brief 输出到文件
     * @param entry 日志条目
     */
    void outputToFile(const LogEntry& entry);

    /**
     * @brief 检查并执行日志文件轮转
     */
    void checkFileRotation();

    /**
     * @brief 执行日志文件轮转
     */
    void rotateLogFile();

    /**
     * @brief 获取当前时间字符串
     * @return 格式化的时间字符串
     */
    static std::string getCurrentTimeString();

    /**
     * @brief 获取日志级别颜色代码（用于控制台输出）
     * @param level 日志级别
     * @return ANSI颜色代码
     */
    static std::string getLevelColor(LogLevel level);

    // 成员变量
    LogConfig config_;                      ///< 日志配置
    std::ofstream log_file_;                ///< 日志文件流
    std::mutex mutex_;                      ///< 互斥锁
    bool initialized_ = false;              ///< 是否已初始化
    std::string log_buffer_;                ///< 日志缓冲区
};

} // namespace core
} // namespace smart_video_analysis

// ============================================================================
// 日志宏定义
// ============================================================================

/**
 * @brief 获取Logger实例的宏
 */
#define GET_LOGGER() smart_video_analysis::core::Logger::getInstance()

/**
 * @brief 通用日志记录宏
 */
#define LOG_LEVEL(level, fmt, ...) \
    GET_LOGGER().log(level, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)

/**
 * @brief 调试级别日志宏
 */
#define LOG_DEBUG(fmt, ...) \
    LOG_LEVEL(smart_video_analysis::core::LogLevel::DEBUG, fmt, ##__VA_ARGS__)

/**
 * @brief 信息级别日志宏
 */
#define LOG_INFO(fmt, ...) \
    LOG_LEVEL(smart_video_analysis::core::LogLevel::INFO, fmt, ##__VA_ARGS__)

/**
 * @brief 警告级别日志宏
 */
#define LOG_WARN(fmt, ...) \
    LOG_LEVEL(smart_video_analysis::core::LogLevel::WARN, fmt, ##__VA_ARGS__)

/**
 * @brief 错误级别日志宏
 */
#define LOG_ERROR(fmt, ...) \
    LOG_LEVEL(smart_video_analysis::core::LogLevel::ERROR, fmt, ##__VA_ARGS__)

/**
 * @brief 致命错误级别日志宏
 */
#define LOG_FATAL(fmt, ...) \
    LOG_LEVEL(smart_video_analysis::core::LogLevel::FATAL, fmt, ##__VA_ARGS__)

/**
 * @brief 条件日志宏
 */
#define LOG_IF(condition, level, fmt, ...) \
    if (condition) LOG_LEVEL(level, fmt, ##__VA_ARGS__)

/**
 * @brief 函数进入日志宏
 */
#define LOG_FUNCTION_ENTRY() \
    LOG_DEBUG("Entering function: %s", __func__)

/**
 * @brief 函数退出日志宏
 */
#define LOG_FUNCTION_EXIT() \
    LOG_DEBUG("Exiting function: %s", __func__)

/**
 * @brief 性能计时日志宏
 */
#define LOG_PERFORMANCE_START(timer_name) \
    auto timer_name##_start = std::chrono::high_resolution_clock::now()

#define LOG_PERFORMANCE_END(timer_name, operation) \
    do { \
        auto timer_name##_end = std::chrono::high_resolution_clock::now(); \
        auto timer_name##_duration = std::chrono::duration_cast<std::chrono::milliseconds>( \
            timer_name##_end - timer_name##_start).count(); \
        LOG_DEBUG("%s took %ld ms", operation, timer_name##_duration); \
    } while(0)

#endif // CORE_LOGGER_HPP
