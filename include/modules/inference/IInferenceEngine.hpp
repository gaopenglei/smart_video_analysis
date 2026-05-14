/**
 * @file IInferenceEngine.hpp
 * @brief 推理引擎抽象接口
 *
 * 定义与具体推理后端（ONNX Runtime / TensorRT / RKNN 等）无关的纯虚接口。
 * 上层模块（SmartVideoAnalysisSystem 等）只依赖此头文件，
 * 切换推理后端时上层代码零修改。
 *
 * 实现一个新后端的步骤：
 *   1. 继承 IInferenceEngine
 *   2. 实现所有纯虚函数
 *   3. 在 InferenceEngineFactory 中注册新类型
 *   4. 在 CMakeLists.txt 中添加对应库的链接
 */

#ifndef MODULES_INFERENCE_I_INFERENCE_ENGINE_HPP
#define MODULES_INFERENCE_I_INFERENCE_ENGINE_HPP

#include <string>
#include <vector>
#include <memory>
#include <future>
#include <map>
#include <stdexcept>
#include "core/ErrorHandling.hpp"

namespace smart_video_analysis {
namespace modules {
namespace inference {

// ============================================================================
// 后端无关的数据类型定义
// ============================================================================

/**
 * @brief 通用张量元素类型枚举（不依赖任何推理库）
 */
enum class TensorDataType {
    FLOAT32,    ///< 32位浮点
    FLOAT16,    ///< 16位浮点
    INT8,       ///< 8位有符号整型
    UINT8,      ///< 8位无符号整型
    INT16,      ///< 16位有符号整型
    INT32,      ///< 32位有符号整型
    INT64,      ///< 64位有符号整型
    BOOL,       ///< 布尔型
    STRING,     ///< 字符串型
    UNKNOWN     ///< 未知类型
};

/**
 * @brief 后端无关的张量信息结构体
 */
struct TensorInfo {
    std::string name;               ///< 张量名称
    std::vector<int64_t> shape;     ///< 张量形状（-1 表示动态维度）
    TensorDataType type;            ///< 数据类型
    size_t element_count;           ///< 元素数量（动态形状时为 0）
    size_t byte_size;               ///< 字节大小（动态形状时为 0）

    TensorInfo()
        : type(TensorDataType::FLOAT32), element_count(0), byte_size(0) {}
};

/**
 * @brief 推理结果结构体
 */
struct InferenceResult {
    std::vector<std::vector<float>> outputs;  ///< 各输出张量数据（float32）
    std::vector<TensorInfo>  output_infos;    ///< 各输出张量元信息
    double   inference_time_ms;               ///< 本次推理耗时（毫秒）
    double   preprocess_time_ms;              ///< 预处理耗时（由调用方填写）
    double   postprocess_time_ms;             ///< 后处理耗时（由调用方填写）
    int64_t  timestamp_ms;                    ///< 推理完成时间戳

    InferenceResult()
        : inference_time_ms(0), preprocess_time_ms(0),
          postprocess_time_ms(0), timestamp_ms(0) {}
};

/**
 * @brief 推理性能统计
 */
struct InferenceStats {
    int64_t total_inferences = 0;
    double  total_time_ms    = 0.0;
    double  min_time_ms      = 1e9;
    double  max_time_ms      = 0.0;
    double  avg_time_ms      = 0.0;
    double  fps              = 0.0;

    void update(double time_ms) {
        ++total_inferences;
        total_time_ms += time_ms;
        if (time_ms < min_time_ms) min_time_ms = time_ms;
        if (time_ms > max_time_ms) max_time_ms = time_ms;
        avg_time_ms = total_time_ms / total_inferences;
        fps = avg_time_ms > 0.0 ? 1000.0 / avg_time_ms : 0.0;
    }
};

// ============================================================================
// 推理引擎抽象接口
// ============================================================================

/**
 * @brief 推理引擎纯虚接口
 *
 * 所有推理后端实现类均需继承此接口。
 * 上层代码通过 std::unique_ptr<IInferenceEngine> 操作推理引擎，
 * 与具体后端完全解耦。
 */
class IInferenceEngine {
public:
    virtual ~IInferenceEngine() = default;

    // ----------------------------------------------------------------
    // 生命周期管理
    // ----------------------------------------------------------------

    /**
     * @brief 加载模型文件
     * @param model_path 模型文件路径（.onnx / .trt / .rknn 等，由实现类决定）
     * @return 成功返回 ErrorCode::SUCCESS
     */
    virtual core::ErrorCode loadModel(const std::string& model_path) = 0;

    /**
     * @brief 卸载已加载的模型，释放后端资源
     */
    virtual void unloadModel() = 0;

    /**
     * @brief 模型是否已成功加载
     */
    virtual bool isModelLoaded() const = 0;

    // ----------------------------------------------------------------
    // 推理接口
    // ----------------------------------------------------------------

    /**
     * @brief 单输入同步推理
     * @param input_data 展平后的输入张量数据（float32，NCHW 顺序）
     * @param result     推理结果（输出由实现类填写）
     * @return 成功返回 ErrorCode::SUCCESS
     */
    virtual core::ErrorCode infer(const std::vector<float>& input_data,
                                  InferenceResult& result) = 0;

    /**
     * @brief 多输入同步推理
     * @param input_data 各输入张量数据（float32）
     * @param result     推理结果
     * @return 成功返回 ErrorCode::SUCCESS
     */
    virtual core::ErrorCode infer(
        const std::vector<std::vector<float>>& input_data,
        InferenceResult& result) = 0;

    /**
     * @brief 异步推理
     * @param input_data 输入张量数据
     * @return std::future，get() 时抛出异常表示推理失败
     */
    virtual std::future<InferenceResult> inferAsync(
        const std::vector<float>& input_data) = 0;

    // ----------------------------------------------------------------
    // 模型元信息
    // ----------------------------------------------------------------

    /**
     * @brief 获取各输入张量信息
     */
    virtual const std::vector<TensorInfo>& getInputInfos() const = 0;

    /**
     * @brief 获取各输出张量信息
     */
    virtual const std::vector<TensorInfo>& getOutputInfos() const = 0;

    /**
     * @brief 获取指定输入的形状
     */
    virtual std::vector<int64_t> getInputShape(int index = 0) const = 0;

    /**
     * @brief 获取指定输出的形状
     */
    virtual std::vector<int64_t> getOutputShape(int index = 0) const = 0;

    /**
     * @brief 获取模型元数据键值对
     */
    virtual std::map<std::string, std::string> getModelMetadata() const = 0;

    // ----------------------------------------------------------------
    // 性能与调试
    // ----------------------------------------------------------------

    /**
     * @brief 获取推理性能统计
     */
    virtual const InferenceStats& getStats() const = 0;

    /**
     * @brief 重置性能统计
     */
    virtual void resetStats() = 0;

    /**
     * @brief 打印模型结构摘要（调试用）
     */
    virtual void printModelInfo() const = 0;

    /**
     * @brief 设置是否输出详细日志
     */
    virtual void setVerbose(bool verbose) = 0;

    /**
     * @brief 返回当前后端名称，如 "OnnxRuntime"、"TensorRT"、"RKNN"
     */
    virtual std::string getBackendName() const = 0;
};

} // namespace inference
} // namespace modules
} // namespace smart_video_analysis

#endif // MODULES_INFERENCE_I_INFERENCE_ENGINE_HPP
