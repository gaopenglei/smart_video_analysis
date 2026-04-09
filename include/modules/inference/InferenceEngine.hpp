/**
 * @file InferenceEngine.hpp
 * @brief ONNX Runtime推理引擎头文件
 * 
 * 提供基于ONNX Runtime的高效推理功能，支持多种执行提供者。
 */

#ifndef MODULES_INFERENCE_INFERENCE_ENGINE_HPP
#define MODULES_INFERENCE_INFERENCE_ENGINE_HPP

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <map>
#include <chrono>
#include <onnxruntime_cxx_api.h>
#include "core/Config.hpp"
#include "core/ErrorHandling.hpp"

namespace smart_video_analysis {
namespace modules {
namespace inference {

/**
 * @brief 张量信息结构体
 */
struct TensorInfo {
    std::string name;                   ///< 张量名称
    std::vector<int64_t> shape;         ///< 张量形状
    ONNXTensorElementDataType type;     ///< 数据类型
    size_t element_count;               ///< 元素数量
    size_t byte_size;                   ///< 字节大小
    
    TensorInfo() : type(ONNX_TENSOR_ELEMENT_DATA_FLOAT), 
                   element_count(0), byte_size(0) {}
};

/**
 * @brief 推理结果结构体
 */
struct InferenceResult {
    std::vector<std::vector<float>> outputs;    ///< 输出张量数据
    std::vector<TensorInfo> output_infos;       ///< 输出张量信息
    double inference_time_ms;                   ///< 推理耗时（毫秒）
    double preprocess_time_ms;                  ///< 预处理耗时（毫秒）
    double postprocess_time_ms;                 ///< 后处理耗时（毫秒）
    int64_t timestamp_ms;                       ///< 时间戳
    
    InferenceResult() : inference_time_ms(0), preprocess_time_ms(0),
                        postprocess_time_ms(0), timestamp_ms(0) {}
};

/**
 * @brief 推理性能统计结构体
 */
struct InferenceStats {
    int64_t total_inferences;           ///< 总推理次数
    double total_time_ms;               ///< 总耗时
    double min_time_ms;                 ///< 最小耗时
    double max_time_ms;                 ///< 最大耗时
    double avg_time_ms;                 ///< 平均耗时
    double fps;                         ///< 帧率
    
    InferenceStats() : total_inferences(0), total_time_ms(0), min_time_ms(1e9),
                       max_time_ms(0), avg_time_ms(0), fps(0) {}
    
    void update(double time_ms) {
        total_inferences++;
        total_time_ms += time_ms;
        min_time_ms = std::min(min_time_ms, time_ms);
        max_time_ms = std::max(max_time_ms, time_ms);
        avg_time_ms = total_time_ms / total_inferences;
        fps = 1000.0 / avg_time_ms;
    }
};

/**
 * @brief 执行提供者枚举
 */
enum class ExecutionProvider {
    CPU,            ///< CPU执行
    CUDA,           ///< NVIDIA CUDA
    TensorRT,       ///< NVIDIA TensorRT
    OpenVINO,       ///< Intel OpenVINO
    NNAPI,          ///< Android NNAPI
    CoreML,         ///< Apple CoreML
    DNNL,           ///< Intel DNNL
    XNNPACK,        ///< XNNPACK
    VitisAI,        ///< AMD Vitis AI
    RockchipNPU     ///< 瑞芯微NPU (RK3588)
};

/**
 * @brief ONNX Runtime推理引擎类
 * 
 * 封装ONNX Runtime API，提供高效的模型推理功能。
 */
class OnnxInferenceEngine {
public:
    /**
     * @brief 构造函数
     * @param config 推理配置
     */
    explicit OnnxInferenceEngine(const core::InferenceConfig& config);
    
    /**
     * @brief 析构函数
     */
    ~OnnxInferenceEngine();
    
    /**
     * @brief 加载模型
     * @param model_path 模型路径
     * @return 成功返回ErrorCode::SUCCESS
     */
    core::ErrorCode loadModel(const std::string& model_path);
    
    /**
     * @brief 卸载模型
     */
    void unloadModel();
    
    /**
     * @brief 执行推理
     * @param input_data 输入数据
     * @param result 输出结果
     * @return 成功返回ErrorCode::SUCCESS
     */
    core::ErrorCode infer(const std::vector<float>& input_data, 
                          InferenceResult& result);
    
    /**
     * @brief 执行推理（多输入）
     * @param input_data 多个输入数据
     * @param result 输出结果
     * @return 成功返回ErrorCode::SUCCESS
     */
    core::ErrorCode infer(const std::vector<std::vector<float>>& input_data,
                          InferenceResult& result);
    
    /**
     * @brief 异步推理
     * @param input_data 输入数据
     * @param callback 回调函数
     * @return 成功返回ErrorCode::SUCCESS
     */
    core::ErrorCode inferAsync(const std::vector<float>& input_data,
                               std::function<void(InferenceResult)> callback);
    
    /**
     * @brief 检查模型是否已加载
     */
    bool isModelLoaded() const;
    
    /**
     * @brief 获取输入张量信息
     */
    const std::vector<TensorInfo>& getInputInfos() const;
    
    /**
     * @brief 获取输出张量信息
     */
    const std::vector<TensorInfo>& getOutputInfos() const;
    
    /**
     * @brief 获取输入形状
     */
    std::vector<int64_t> getInputShape(int index = 0) const;
    
    /**
     * @brief 获取输出形状
     */
    std::vector<int64_t> getOutputShape(int index = 0) const;
    
    /**
     * @brief 获取性能统计
     */
    const InferenceStats& getStats() const;
    
    /**
     * @brief 重置性能统计
     */
    void resetStats();
    
    /**
     * @brief 设置执行提供者
     */
    void setExecutionProvider(ExecutionProvider provider);
    
    /**
     * @brief 获取当前执行提供者
     */
    ExecutionProvider getExecutionProvider() const;
    
    /**
     * @brief 设置线程数
     */
    void setNumThreads(int num_threads);
    
    /**
     * @brief 设置GPU设备ID
     */
    void setGpuDeviceId(int device_id);
    
    /**
     * @brief 启用/禁用详细日志
     */
    void setVerbose(bool verbose);
    
    /**
     * @brief 获取模型元数据
     */
    std::map<std::string, std::string> getModelMetadata() const;
    
    /**
     * @brief 打印模型信息
     */
    void printModelInfo() const;

private:
    /**
     * @brief 初始化ONNX Runtime环境
     */
    bool initializeEnvironment();
    
    /**
     * @brief 创建会话选项
     */
    Ort::SessionOptions createSessionOptions();
    
    /**
     * @brief 配置执行提供者
     */
    void configureExecutionProvider(Ort::SessionOptions& options);
    
    /**
     * @brief 提取模型输入输出信息
     */
    void extractModelInfo();
    
    /**
     * @brief 创建输入张量
     */
    Ort::Value createInputTensor(const std::vector<float>& data, int index);
    
    /**
     * @brief 创建输入张量（多输入）
     */
    std::vector<Ort::Value> createInputTensors(
        const std::vector<std::vector<float>>& input_data);

    // 成员变量
    core::InferenceConfig config_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;
    
    std::vector<TensorInfo> input_infos_;
    std::vector<TensorInfo> output_infos_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_name_ptrs_;
    std::vector<const char*> output_name_ptrs_;
    
    ExecutionProvider execution_provider_;
    InferenceStats stats_;
    std::mutex mutex_;
    bool model_loaded_ = false;
    bool verbose_ = false;
};

/**
 * @brief 推理引擎工厂类
 */
class InferenceEngineFactory {
public:
    /**
     * @brief 创建推理引擎
     * @param config 推理配置
     * @return 推理引擎实例
     */
    static std::unique_ptr<OnnxInferenceEngine> create(
        const core::InferenceConfig& config);
    
    /**
     * @brief 获取可用的执行提供者列表
     */
    static std::vector<ExecutionProvider> getAvailableProviders();
    
    /**
     * @brief 检查执行提供者是否可用
     */
    static bool isProviderAvailable(ExecutionProvider provider);
    
    /**
     * @brief 获取执行提供者名称
     */
    static std::string getProviderName(ExecutionProvider provider);
};

} // namespace inference
} // namespace modules
} // namespace smart_video_analysis

#endif // MODULES_INFERENCE_INFERENCE_ENGINE_HPP
