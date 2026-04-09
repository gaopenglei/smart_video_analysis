/**
 * @file ErrorHandling.cpp
 * @brief 错误处理模块实现文件
 */

#include "core/ErrorHandling.hpp"
#include "core/Logger.hpp"

namespace smart_video_analysis {
namespace core {

ErrorManager::ErrorManager() {
    initializeDefaultMessages();
}

ErrorManager& ErrorManager::getInstance() {
    static ErrorManager instance;
    return instance;
}

void ErrorManager::initializeDefaultMessages() {
    // 通用错误
    error_messages_[ErrorCode::SUCCESS] = "Success";
    error_messages_[ErrorCode::UNKNOWN_ERROR] = "Unknown error occurred";
    error_messages_[ErrorCode::INVALID_PARAMETER] = "Invalid parameter provided";
    error_messages_[ErrorCode::NULL_POINTER] = "Null pointer encountered";
    error_messages_[ErrorCode::OUT_OF_MEMORY] = "Out of memory";
    error_messages_[ErrorCode::FILE_NOT_FOUND] = "File not found";
    error_messages_[ErrorCode::FILE_READ_ERROR] = "Failed to read file";
    error_messages_[ErrorCode::FILE_WRITE_ERROR] = "Failed to write file";
    error_messages_[ErrorCode::INVALID_CONFIG] = "Invalid configuration";
    
    // 视频输入模块错误
    error_messages_[ErrorCode::VIDEO_INPUT_ERROR] = "Video input error";
    error_messages_[ErrorCode::CAMERA_OPEN_FAILED] = "Failed to open camera device";
    error_messages_[ErrorCode::VIDEO_FILE_OPEN_FAILED] = "Failed to open video file";
    error_messages_[ErrorCode::NETWORK_STREAM_ERROR] = "Network stream error";
    error_messages_[ErrorCode::FRAME_READ_ERROR] = "Failed to read video frame";
    error_messages_[ErrorCode::UNSUPPORTED_FORMAT] = "Unsupported video format";
    error_messages_[ErrorCode::HARDWARE_ACCELERATION_ERROR] = "Hardware acceleration error";
    
    // 预处理模块错误
    error_messages_[ErrorCode::PREPROCESS_ERROR] = "Preprocessing error";
    error_messages_[ErrorCode::RESIZE_ERROR] = "Failed to resize image";
    error_messages_[ErrorCode::COLOR_CONVERT_ERROR] = "Color space conversion error";
    error_messages_[ErrorCode::NORMALIZATION_ERROR] = "Normalization error";
    error_messages_[ErrorCode::INVALID_IMAGE_FORMAT] = "Invalid image format";
    
    // 模型处理模块错误
    error_messages_[ErrorCode::MODEL_ERROR] = "Model processing error";
    error_messages_[ErrorCode::MODEL_LOAD_FAILED] = "Failed to load model";
    error_messages_[ErrorCode::MODEL_EXPORT_FAILED] = "Failed to export model";
    error_messages_[ErrorCode::MODEL_CONVERSION_FAILED] = "Model conversion failed";
    error_messages_[ErrorCode::INVALID_MODEL_FORMAT] = "Invalid model format";
    error_messages_[ErrorCode::WEIGHT_LOAD_FAILED] = "Failed to load model weights";
    error_messages_[ErrorCode::MODEL_NOT_INITIALIZED] = "Model not initialized";
    
    // 算子适配模块错误
    error_messages_[ErrorCode::OPERATOR_ERROR] = "Operator adapter error";
    error_messages_[ErrorCode::UNSUPPORTED_OPERATOR] = "Unsupported operator detected";
    error_messages_[ErrorCode::OPERATOR_FUSION_FAILED] = "Operator fusion failed";
    error_messages_[ErrorCode::OPERATOR_FALLBACK_FAILED] = "Operator fallback failed";
    error_messages_[ErrorCode::NPU_COMPATIBILITY_ERROR] = "NPU compatibility error";
    
    // 推理模块错误
    error_messages_[ErrorCode::INFERENCE_ERROR] = "Inference error";
    error_messages_[ErrorCode::ONNX_RUNTIME_ERROR] = "ONNX Runtime error";
    error_messages_[ErrorCode::SESSION_CREATE_FAILED] = "Failed to create inference session";
    error_messages_[ErrorCode::INFERENCE_EXECUTION_FAILED] = "Inference execution failed";
    error_messages_[ErrorCode::INPUT_SHAPE_MISMATCH] = "Input shape mismatch";
    error_messages_[ErrorCode::OUTPUT_PARSE_ERROR] = "Failed to parse inference output";
    error_messages_[ErrorCode::DEVICE_NOT_AVAILABLE] = "Requested device not available";
    error_messages_[ErrorCode::MEMORY_ALLOCATION_FAILED] = "Memory allocation failed";
    
    // 后处理模块错误
    error_messages_[ErrorCode::POSTPROCESS_ERROR] = "Postprocessing error";
    error_messages_[ErrorCode::NMS_ERROR] = "NMS processing error";
    error_messages_[ErrorCode::DETECTION_PARSE_ERROR] = "Detection result parse error";
    error_messages_[ErrorCode::INVALID_OUTPUT_FORMAT] = "Invalid output format";
    
    // 可视化模块错误
    error_messages_[ErrorCode::VISUALIZATION_ERROR] = "Visualization error";
    error_messages_[ErrorCode::DRAW_ERROR] = "Drawing error";
    error_messages_[ErrorCode::OUTPUT_WRITE_ERROR] = "Failed to write output";
    error_messages_[ErrorCode::DISPLAY_ERROR] = "Display error";
    
    // 系统错误
    error_messages_[ErrorCode::SYSTEM_ERROR] = "System error";
    error_messages_[ErrorCode::THREAD_ERROR] = "Thread error";
    error_messages_[ErrorCode::RESOURCE_UNAVAILABLE] = "Resource unavailable";
    error_messages_[ErrorCode::TIMEOUT] = "Operation timeout";
    error_messages_[ErrorCode::PERMISSION_DENIED] = "Permission denied";
}

std::string ErrorManager::getErrorMessage(ErrorCode code) const {
    auto it = error_messages_.find(code);
    if (it != error_messages_.end()) {
        return it->second;
    }
    return "Unknown error code: " + std::to_string(static_cast<int>(code));
}

void ErrorManager::registerErrorMessage(ErrorCode code, const std::string& message) {
    error_messages_[code] = message;
}

ErrorInfo ErrorManager::createError(ErrorCode code, const char* file, int line,
                                    const char* function, const std::string& context) const {
    ErrorInfo error;
    error.code = code;
    error.message = getErrorMessage(code);
    error.file = file;
    error.line = line;
    error.function = function;
    error.context = context;
    return error;
}

void ErrorManager::throwError(const ErrorInfo& error_info) const {
    // 记录错误日志
    LOG_ERROR("[%s:%d] %s: %s - %s", 
              error_info.file.c_str(), 
              error_info.line,
              error_info.function.c_str(),
              error_info.message.c_str(),
              error_info.context.c_str());
    
    throw SmartVideoAnalysisException(error_info);
}

void ErrorManager::checkCondition(bool condition, ErrorCode code, const char* file, int line,
                                  const char* function, const std::string& context) const {
    if (!condition) {
        throwError(createError(code, file, line, function, context));
    }
}

} // namespace core
} // namespace smart_video_analysis
