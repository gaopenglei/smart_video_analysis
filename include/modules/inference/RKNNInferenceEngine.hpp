/**
 * @file RKNNInferenceEngine.hpp
 * @brief 瑞芯微 RKNN 推理引擎实现类头文件
 *
 * 实现 IInferenceEngine 接口，封装 RKNN Runtime API（rknn_api.h）。
 * 支持 RK3588 三核 NPU，可通过 core_mask 指定单核/多核模式。
 *
 * 使用前提：
 *   - 目标设备：RK3588（或 RK3568/RK3566）
 *   - 模型格式：.rknn（通过 RKNN-Toolkit2 在 PC 端从 ONNX/PyTorch 转换而来）
 *   - 依赖库：librknnrt.so（RKNN Runtime，随板子 BSP 或 SDK 提供）
 *
 * 不要在非推理模块中直接包含本头文件。
 * 上层代码通过 IInferenceEngine 接口操作，与 RKNN 完全解耦。
 */

#ifndef MODULES_INFERENCE_RKNN_INFERENCE_ENGINE_HPP
#define MODULES_INFERENCE_RKNN_INFERENCE_ENGINE_HPP

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <map>
#include <future>
#include "modules/inference/IInferenceEngine.hpp"
#include "core/Config.hpp"

namespace smart_video_analysis {
namespace modules {
namespace inference {

/**
 * @brief 基于 RKNN Runtime 的推理引擎实现
 *
 * RKNN 推理流程：
 *   rknn_init() → rknn_inputs_set() → rknn_run() → rknn_outputs_get()
 *
 * 与 OnnxInferenceEngine 一样实现 IInferenceEngine，
 * 上层调用者无需感知使用的是哪个后端。
 */
class RKNNInferenceEngine : public IInferenceEngine {
public:
    explicit RKNNInferenceEngine(const core::InferenceConfig& config);
    ~RKNNInferenceEngine() override;

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
    std::vector<int64_t> getInputShape(int index = 0) const override;
    std::vector<int64_t> getOutputShape(int index = 0) const override;
    std::map<std::string, std::string> getModelMetadata() const override;

    const InferenceStats& getStats() const override;
    void resetStats() override;
    void printModelInfo() const override;
    void setVerbose(bool verbose) override;
    std::string getBackendName() const override { return "RKNN"; }

private:
    // pImpl：将所有 rknn_api.h 具体类型隐藏在 .cpp 中
    struct Impl;
    std::unique_ptr<Impl> impl_;

    core::InferenceConfig config_;
    InferenceStats        stats_;
    mutable std::mutex    mutex_;
    bool model_loaded_ = false;
    bool verbose_      = false;
};

} // namespace inference
} // namespace modules
} // namespace smart_video_analysis

#endif // MODULES_INFERENCE_RKNN_INFERENCE_ENGINE_HPP
