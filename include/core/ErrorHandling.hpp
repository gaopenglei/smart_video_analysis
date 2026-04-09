/**
 * @file ErrorHandling.hpp
 * @brief 错误处理模块头文件
 * 
 * 提供统一的错误码定义、错误处理和异常管理功能。
 */

#ifndef CORE_ERROR_HANDLING_HPP
#define CORE_ERROR_HANDLING_HPP

#include <string>
#include <stdexcept>
#include <memory>
#include <map>
#include <vector>

namespace smart_video_analysis {
namespace core {

/**
 * @brief 错误码枚举
 * 
 * 定义系统中所有可能的错误类型，便于错误识别和处理。
 */
enum class ErrorCode : int {
    // 成功
    SUCCESS = 0,
    
    // 通用错误 (1-99)
    UNKNOWN_ERROR = 1,
    INVALID_PARAMETER = 2,
    NULL_POINTER = 3,
    OUT_OF_MEMORY = 4,
    FILE_NOT_FOUND = 5,
    FILE_READ_ERROR = 6,
    FILE_WRITE_ERROR = 7,
    INVALID_CONFIG = 8,
    
    // 视频输入模块错误 (100-199)
    VIDEO_INPUT_ERROR = 100,
    CAMERA_OPEN_FAILED = 101,
    VIDEO_FILE_OPEN_FAILED = 102,
    NETWORK_STREAM_ERROR = 103,
    FRAME_READ_ERROR = 104,
    UNSUPPORTED_FORMAT = 105,
    HARDWARE_ACCELERATION_ERROR = 106,
    
    // 预处理模块错误 (200-299)
    PREPROCESS_ERROR = 200,
    RESIZE_ERROR = 201,
    COLOR_CONVERT_ERROR = 202,
    NORMALIZATION_ERROR = 203,
    INVALID_IMAGE_FORMAT = 204,
    
    // 模型处理模块错误 (300-399)
    MODEL_ERROR = 300,
    MODEL_LOAD_FAILED = 301,
    MODEL_EXPORT_FAILED = 302,
    MODEL_CONVERSION_FAILED = 303,
    INVALID_MODEL_FORMAT = 304,
    WEIGHT_LOAD_FAILED = 305,
    MODEL_NOT_INITIALIZED = 306,
    
    // 算子适配模块错误 (400-499)
    OPERATOR_ERROR = 400,
    UNSUPPORTED_OPERATOR = 401,
    OPERATOR_FUSION_FAILED = 402,
    OPERATOR_FALLBACK_FAILED = 403,
    NPU_COMPATIBILITY_ERROR = 404,
    
    // 推理模块错误 (500-599)
    INFERENCE_ERROR = 500,
    ONNX_RUNTIME_ERROR = 501,
    SESSION_CREATE_FAILED = 502,
    INFERENCE_EXECUTION_FAILED = 503,
    INPUT_SHAPE_MISMATCH = 504,
    OUTPUT_PARSE_ERROR = 505,
    DEVICE_NOT_AVAILABLE = 506,
    MEMORY_ALLOCATION_FAILED = 507,
    
    // 后处理模块错误 (600-699)
    POSTPROCESS_ERROR = 600,
    NMS_ERROR = 601,
    DETECTION_PARSE_ERROR = 602,
    INVALID_OUTPUT_FORMAT = 603,
    
    // 可视化模块错误 (700-799)
    VISUALIZATION_ERROR = 700,
    DRAW_ERROR = 701,
    OUTPUT_WRITE_ERROR = 702,
    DISPLAY_ERROR = 703,
    
    // 系统错误 (800-899)
    SYSTEM_ERROR = 800,
    THREAD_ERROR = 801,
    RESOURCE_UNAVAILABLE = 802,
    TIMEOUT = 803,
    PERMISSION_DENIED = 804
};

/**
 * @brief 错误信息结构体
 */
struct ErrorInfo {
    ErrorCode code;                 ///< 错误码
    std::string message;            ///< 错误消息
    std::string file;               ///< 源文件名
    int line;                       ///< 源文件行号
    std::string function;           ///< 函数名
    std::string context;            ///< 上下文信息
    
    ErrorInfo() : code(ErrorCode::SUCCESS), line(0) {}
    
    ErrorInfo(ErrorCode c, const std::string& msg, 
              const std::string& f = "", int l = 0,
              const std::string& func = "", const std::string& ctx = "")
        : code(c), message(msg), file(f), line(l), function(func), context(ctx) {}
};

/**
 * @brief 智能视频分析异常类
 */
class SmartVideoAnalysisException : public std::runtime_error {
public:
    explicit SmartVideoAnalysisException(const ErrorInfo& error_info)
        : std::runtime_error(error_info.message), error_info_(error_info) {}
    
    SmartVideoAnalysisException(ErrorCode code, const std::string& message)
        : std::runtime_error(message), error_info_() {
        error_info_.code = code;
        error_info_.message = message;
    }
    
    const ErrorInfo& getErrorInfo() const { return error_info_; }
    ErrorCode getErrorCode() const { return error_info_.code; }
    
private:
    ErrorInfo error_info_;
};

/**
 * @brief 错误处理管理器类
 * 
 * 提供错误码到错误消息的映射、错误记录和错误处理功能。
 */
class ErrorManager {
public:
    /**
     * @brief 获取单例实例
     */
    static ErrorManager& getInstance();
    
    /**
     * @brief 获取错误码对应的错误消息
     * @param code 错误码
     * @return 错误消息字符串
     */
    std::string getErrorMessage(ErrorCode code) const;
    
    /**
     * @brief 注册自定义错误消息
     * @param code 错误码
     * @param message 错误消息
     */
    void registerErrorMessage(ErrorCode code, const std::string& message);
    
    /**
     * @brief 创建错误信息
     * @param code 错误码
     * @param file 源文件名
     * @param line 源文件行号
     * @param function 函数名
     * @param context 上下文信息
     * @return 错误信息结构体
     */
    ErrorInfo createError(ErrorCode code, const char* file, int line,
                          const char* function, const std::string& context = "") const;
    
    /**
     * @brief 抛出异常
     * @param error_info 错误信息
     */
    [[noreturn]] void throwError(const ErrorInfo& error_info) const;
    
    /**
     * @brief 检查条件，如果为false则抛出异常
     * @param condition 条件
     * @param code 错误码
     * @param file 源文件名
     * @param line 源文件行号
     * @param function 函数名
     * @param context 上下文信息
     */
    void checkCondition(bool condition, ErrorCode code, const char* file, int line,
                        const char* function, const std::string& context = "") const;
    
    /**
     * @brief 获取最后的错误信息
     * @return 最后的错误信息
     */
    ErrorInfo getLastEror() const { return last_error_; }
    
    /**
     * @brief 设置最后的错误信息
     * @param error 错误信息
     */
    void setLastError(const ErrorInfo& error) { last_error_ = error; }
    
private:
    ErrorManager();
    ~ErrorManager() = default;
    
    ErrorManager(const ErrorManager&) = delete;
    ErrorManager& operator=(const ErrorManager&) = delete;
    
    void initializeDefaultMessages();
    
    std::map<ErrorCode, std::string> error_messages_;
    ErrorInfo last_error_;
};

// ============================================================================
// 错误处理宏定义
// ============================================================================

/**
 * @brief 获取ErrorManager实例
 */
#define GET_ERROR_MANAGER() smart_video_analysis::core::ErrorManager::getInstance()

/**
 * @brief 创建错误信息
 */
#define CREATE_ERROR(code, context) \
    GET_ERROR_MANAGER().createError(code, __FILE__, __LINE__, __func__, context)

/**
 * @brief 抛出异常
 */
#define THROW_ERROR(code, context) \
    GET_ERROR_MANAGER().throwError(CREATE_ERROR(code, context))

/**
 * @brief 条件检查
 */
#define CHECK_CONDITION(condition, code, context) \
    GET_ERROR_MANAGER().checkCondition(condition, code, __FILE__, __LINE__, __func__, context)

/**
 * @brief 检查指针非空
 */
#define CHECK_NOT_NULL(ptr, context) \
    CHECK_CONDITION((ptr) != nullptr, smart_video_analysis::core::ErrorCode::NULL_POINTER, context)

/**
 * @brief 检查函数返回值
 */
#define CHECK_RETURN(ret, code, context) \
    if ((ret) != 0) { THROW_ERROR(code, context); }

/**
 * @brief 设置最后错误
 */
#define SET_LAST_ERROR(code, context) \
    GET_ERROR_MANAGER().setLastError(CREATE_ERROR(code, context))

/**
 * @brief 返回错误码
 */
#define RETURN_IF_ERROR(expr) \
    do { \
        auto __ret = (expr); \
        if (__ret != smart_video_analysis::core::ErrorCode::SUCCESS) { \
            return __ret; \
        } \
    } while(0)

} // namespace core
} // namespace smart_video_analysis

#endif // CORE_ERROR_HANDLING_HPP
