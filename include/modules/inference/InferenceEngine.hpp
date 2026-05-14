/**
 * @file InferenceEngine.hpp
 * @brief ONNX Runtime 推理引擎实现类头文件
 *
 * 仅包含 OnnxInferenceEngine 的声明。
 * 上层代码应使用 IInferenceEngine.hpp 中定义的抽象接口，
 * 不要在非推理模块中直接包含本头文件，以保持后端解耦。
 */

#ifndef MODULES_INFERENCE_INFERENCE_ENGINE_HPP
#define MODULES_INFERENCE_INFERENCE_ENGINE_HPP

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <map>
#include <future>
#include "modules/inference/IInferenceEngine.hpp"
#include "core/Config.hpp"

// ORT 头文件仅在 .cpp 中真正使用，这里前向声明其核心类型以减少编译依赖。
// 具体的 ORT 类型定义在 InferenceEngine.cpp 中通过 #include 引入。
namespace Ort {
    class Env;
    class Session;
    class SessionOptions;
    class Value;
    template<typename T> class AllocatorWithDefaultOptions_;
    using AllocatorWithDefaultOptions = AllocatorWithDefaultOptions_<void>;
}

namespace smart_video_analysis {
namespace modules {
namespace inference {

/**
 * @brief 基于 ONNX Runtime 的推理引擎实现
 *
 * 实现 IInferenceEngine 接口，所有 ORT 相关细节封装在此类中。
 * 如需接入 TensorRT 或 RKNN，只需新建对应实现类，无需改动上层代码。
 */
class OnnxInferenceEngine : public IInferenceEngine {
public:
    explicit OnnxInferenceEngine(const core::InferenceConfig& config);
    ~OnnxInferenceEngine() override;

    // -- IInferenceEngine 接口实现 --
    core::ErrorCode loadModel(const std::string& model_path) override;
    void unloadModel() override;
    bool isModelLoaded() const override;

    core::ErrorCode infer(const std::vector<float>& input_data,
                          InferenceResult& result) override;
    core::ErrorCode infer(const std::vector<std::vector<float>>& input_data,
                          InferenceResult& result) override;
    std::future<InferenceResult> inferAsync(
        const std::vector<float>& input_data) override;

    const std::vector<TensorInfo>& getInputInfos()  const override;
    const std::vector<TensorInfo>& getOutputInfos() const override;
    std::vector<int64_t> getInputShape(int index = 0)  const override;
    std::vector<int64_t> getOutputShape(int index = 0) const override;
    std::map<std::string, std::string> getModelMetadata() const override;

    const InferenceStats& getStats() const override;
    void resetStats() override;
    void printModelInfo() const override;
    void setVerbose(bool verbose) override;
    std::string getBackendName() const override { return "OnnxRuntime"; }

    // -- ORT 专属接口（上层代码不应调用，供单元测试/调试使用）--
    void setNumThreads(int num_threads);
    void setGpuDeviceId(int device_id);

private:
    bool initializeEnvironment();

    // 使用 pImpl 模式将 ORT 具体类型完全隔离在 .cpp 中
    struct Impl;
    std::unique_ptr<Impl> impl_;

    core::InferenceConfig config_;
    InferenceStats stats_;
    mutable std::mutex mutex_;
    bool model_loaded_ = false;
    bool verbose_      = false;
};

// ============================================================================
// 推理引擎工厂
// ============================================================================

/**
 * @brief 推理后端类型枚举
 *
 * 通过配置或编译宏选择要使用的推理后端，无需修改任何上层代码。
 */
enum class InferenceBackend {
    OnnxRuntime,    ///< ONNX Runtime（跨平台，支持 CPU/CUDA/OpenVINO/NNAPI）
    TensorRT,       ///< NVIDIA TensorRT（高性能 GPU 推理）
    RKNN,           ///< 瑞芯微 RKNN（RK3588 NPU 原生推理）
    OpenVINO,       ///< Intel OpenVINO（独立后端，非 ORT EP）
    CoreML,         ///< Apple CoreML（iOS/macOS）
    Auto            ///< 根据平台和配置自动选择
};

/**
 * @brief 推理引擎工厂
 *
 * 统一创建入口，返回 IInferenceEngine 接口指针。
 * 上层代码与具体实现类完全解耦。
 */
class InferenceEngineFactory {
public:
    /**
     * @brief 根据配置自动选择并创建推理引擎
     * @param config 推理配置（包含 execution_provider / backend 字段）
     * @return IInferenceEngine 接口实例
     */
    static std::unique_ptr<IInferenceEngine> create(
        const core::InferenceConfig& config);

    /**
     * @brief 显式指定后端类型创建推理引擎
     * @param backend 后端类型
     * @param config  推理配置
     * @return IInferenceEngine 接口实例
     */
    static std::unique_ptr<IInferenceEngine> create(
        InferenceBackend backend,
        const core::InferenceConfig& config);

    /**
     * @brief 检查指定后端在当前平台是否可用
     */
    static bool isBackendAvailable(InferenceBackend backend);

    /**
     * @brief 获取后端名称字符串
     */
    static std::string getBackendName(InferenceBackend backend);

    /**
     * @brief 将字符串（如 "OnnxRuntime"、"RKNN"）解析为枚举值
     */
    static InferenceBackend parseBackend(const std::string& name);
};

} // namespace inference
} // namespace modules
} // namespace smart_video_analysis

#endif // MODULES_INFERENCE_INFERENCE_ENGINE_HPP
