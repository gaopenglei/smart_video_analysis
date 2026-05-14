/**
 * @file InferenceEngine.cpp
 * @brief ONNX Runtime 推理引擎实现
 *
 * 所有 ORT 头文件和 ORT 具体类型只在此文件中出现，不会泄漏到上层。
 * 切换推理后端时，只需替换此文件（及对应头文件中的实现类）。
 */

// ORT 头文件仅在此 .cpp 中引入，彻底隔离于上层代码
#include <onnxruntime_cxx_api.h>

#include "modules/inference/InferenceEngine.hpp"
#include "modules/inference/RKNNInferenceEngine.hpp"
#include "core/Logger.hpp"
#include <algorithm>
#include <chrono>
#include <future>
#include <numeric>
#include <cstring>
#include <stdexcept>

namespace smart_video_analysis {
namespace modules {
namespace inference {

// ============================================================================
// pImpl：将所有 ORT 具体类型隐藏在此结构体中
// ============================================================================

struct OnnxInferenceEngine::Impl {
    std::unique_ptr<Ort::Env>     env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator;

    std::vector<TensorInfo>   input_infos;
    std::vector<TensorInfo>   output_infos;
    std::vector<std::string>  input_names;
    std::vector<std::string>  output_names;
    std::vector<const char*>  input_name_ptrs;
    std::vector<const char*>  output_name_ptrs;

    // ----------------------------------------------------------------
    // ORT 数据类型 → 通用 TensorDataType 的转换
    // ----------------------------------------------------------------
    static TensorDataType fromOrtType(ONNXTensorElementDataType t) {
        switch (t) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return TensorDataType::FLOAT32;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return TensorDataType::FLOAT16;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return TensorDataType::INT8;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return TensorDataType::UINT8;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   return TensorDataType::INT16;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return TensorDataType::INT32;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return TensorDataType::INT64;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return TensorDataType::BOOL;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:  return TensorDataType::STRING;
            default:                                     return TensorDataType::UNKNOWN;
        }
    }

    // ----------------------------------------------------------------
    // 提取模型输入输出元信息
    // ----------------------------------------------------------------
    void extractModelInfo() {
        size_t num_inputs = session->GetInputCount();
        input_infos.resize(num_inputs);
        input_names.resize(num_inputs);
        input_name_ptrs.resize(num_inputs);

        for (size_t i = 0; i < num_inputs; ++i) {
            auto name_ptr = session->GetInputNameAllocated(i, *allocator);
            input_names[i] = name_ptr.get();
            input_name_ptrs[i] = input_names[i].c_str();

            auto type_info  = session->GetInputTypeInfo(i);
            auto tensor_inf = type_info.GetTensorTypeAndShapeInfo();

            input_infos[i].name          = input_names[i];
            input_infos[i].type          = fromOrtType(tensor_inf.GetElementType());
            input_infos[i].shape         = tensor_inf.GetShape();
            input_infos[i].element_count = tensor_inf.GetElementCount();
            input_infos[i].byte_size     = input_infos[i].element_count * sizeof(float);
        }

        size_t num_outputs = session->GetOutputCount();
        output_infos.resize(num_outputs);
        output_names.resize(num_outputs);
        output_name_ptrs.resize(num_outputs);

        for (size_t i = 0; i < num_outputs; ++i) {
            auto name_ptr = session->GetOutputNameAllocated(i, *allocator);
            output_names[i] = name_ptr.get();
            output_name_ptrs[i] = output_names[i].c_str();

            auto type_info  = session->GetOutputTypeInfo(i);
            auto tensor_inf = type_info.GetTensorTypeAndShapeInfo();

            output_infos[i].name          = output_names[i];
            output_infos[i].type          = fromOrtType(tensor_inf.GetElementType());
            output_infos[i].shape         = tensor_inf.GetShape();
            output_infos[i].element_count = tensor_inf.GetElementCount();
            output_infos[i].byte_size     = output_infos[i].element_count * sizeof(float);
        }
    }

    // ----------------------------------------------------------------
    // 创建单个输入张量
    // ----------------------------------------------------------------
    Ort::Value createInputTensor(const std::vector<float>& data, int index) {
        auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        return Ort::Value::CreateTensor<float>(
            mem_info,
            const_cast<float*>(data.data()),
            data.size(),
            input_infos[index].shape.data(),
            input_infos[index].shape.size());
    }
};

// ============================================================================
// OnnxInferenceEngine 实现
// ============================================================================

OnnxInferenceEngine::OnnxInferenceEngine(const core::InferenceConfig& config)
    : impl_(std::make_unique<Impl>()), config_(config) {
    LOG_DEBUG("OnnxInferenceEngine created (backend: OnnxRuntime)");
}

OnnxInferenceEngine::~OnnxInferenceEngine() {
    unloadModel();
}

bool OnnxInferenceEngine::initializeEnvironment() {
    try {
        OrtLoggingLevel log_level =
            verbose_ ? ORT_LOGGING_LEVEL_VERBOSE : ORT_LOGGING_LEVEL_WARNING;
        impl_->env       = std::make_unique<Ort::Env>(log_level, "SmartVideoAnalysis");
        impl_->allocator = std::make_unique<Ort::AllocatorWithDefaultOptions>();
        return true;
    } catch (const Ort::Exception& e) {
        LOG_ERROR("Failed to initialize ORT environment: %s", e.what());
        return false;
    }
}

// ----------------------------------------------------------------
// 图优化级别映射
// ----------------------------------------------------------------
static OrtGraphOptimizationLevel parseOptLevel(const std::string& level) {
    if (level == "all")      return ORT_ENABLE_ALL;
    if (level == "extended") return ORT_ENABLE_EXTENDED;
    if (level == "basic")    return ORT_ENABLE_BASIC;
    return ORT_DISABLE_ALL;
}

// ----------------------------------------------------------------
// 执行提供者配置（仅在 ORT 后端内部使用）
// ----------------------------------------------------------------
static void configureProvider(Ort::SessionOptions& opts,
                              const std::string& provider,
                              const core::InferenceConfig& cfg) {
    std::string p = provider;
    std::transform(p.begin(), p.end(), p.begin(), ::tolower);

    if (p == "cuda") {
#ifdef USE_CUDA
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id = cfg.gpu_device_id;
        cuda_opts.gpu_mem_limit = static_cast<size_t>(2) * 1024 * 1024 * 1024;
        opts.AppendExecutionProvider_CUDA(cuda_opts);
        LOG_INFO("ORT CUDA provider configured (device=%d)", cfg.gpu_device_id);
#else
        LOG_WARN("ORT CUDA provider not compiled in, falling back to CPU");
#endif
    } else if (p == "tensorrt") {
#ifdef USE_TENSORRT
        OrtTensorRTProviderOptions trt_opts{};
        trt_opts.device_id = cfg.gpu_device_id;
        trt_opts.trt_max_workspace_size = 1ULL * 1024 * 1024 * 1024;
        trt_opts.trt_fp16_enable = 1;
        opts.AppendExecutionProvider_TensorRT(trt_opts);
        LOG_INFO("ORT TensorRT provider configured");
#else
        LOG_WARN("ORT TensorRT provider not compiled in, falling back to CPU");
#endif
    } else if (p == "openvino") {
#ifdef USE_OPENVINO
        OrtOpenVINOProviderOptions ov_opts{};
        ov_opts.device_type = "CPU";
        opts.AppendExecutionProvider_OpenVINO(ov_opts);
        LOG_INFO("ORT OpenVINO provider configured");
#else
        LOG_WARN("ORT OpenVINO provider not compiled in, falling back to CPU");
#endif
    } else if (p == "nnapi") {
#ifdef USE_NNAPI
        opts.AppendExecutionProvider_NNAPI(0);
        LOG_INFO("ORT NNAPI provider configured");
#else
        LOG_WARN("ORT NNAPI provider not compiled in, falling back to CPU");
#endif
    } else {
        LOG_INFO("ORT CPU provider, threads=%d", cfg.num_threads);
    }
}

core::ErrorCode OnnxInferenceEngine::loadModel(const std::string& model_path) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (model_loaded_) {
        LOG_WARN("Model already loaded, unloading previous model first");
        unloadModel();
    }

    LOG_INFO("Loading ONNX model: %s", model_path.c_str());

    if (!initializeEnvironment()) {
        return core::ErrorCode::MODEL_LOAD_FAILED;
    }

    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(config_.num_threads);
        opts.SetInterOpNumThreads(config_.num_threads);
        opts.SetGraphOptimizationLevel(parseOptLevel(config_.optimization_level));

        if (config_.enable_memory_pattern) {
            opts.EnableMemPattern();
        }

        configureProvider(opts, config_.execution_provider, config_);

        impl_->session = std::make_unique<Ort::Session>(
            *impl_->env, model_path.c_str(), opts);

        impl_->extractModelInfo();
        model_loaded_ = true;

        LOG_INFO("ONNX model loaded successfully");
        printModelInfo();
        return core::ErrorCode::SUCCESS;

    } catch (const Ort::Exception& e) {
        LOG_ERROR("Failed to load model: %s", e.what());
        return core::ErrorCode::MODEL_LOAD_FAILED;
    }
}

void OnnxInferenceEngine::unloadModel() {
    impl_->session.reset();
    impl_->allocator.reset();
    impl_->env.reset();
    impl_->input_infos.clear();
    impl_->output_infos.clear();
    impl_->input_names.clear();
    impl_->output_names.clear();
    impl_->input_name_ptrs.clear();
    impl_->output_name_ptrs.clear();
    model_loaded_ = false;
    LOG_DEBUG("ONNX model unloaded");
}

bool OnnxInferenceEngine::isModelLoaded() const {
    return model_loaded_;
}

// ----------------------------------------------------------------
// 同步推理（单输入）
// ----------------------------------------------------------------
core::ErrorCode OnnxInferenceEngine::infer(const std::vector<float>& input_data,
                                            InferenceResult& result) {
    return infer(std::vector<std::vector<float>>{input_data}, result);
}

// ----------------------------------------------------------------
// 同步推理（多输入）
// ----------------------------------------------------------------
core::ErrorCode OnnxInferenceEngine::infer(
    const std::vector<std::vector<float>>& input_data,
    InferenceResult& result) {

    if (!model_loaded_) {
        LOG_ERROR("Model not loaded");
        return core::ErrorCode::MODEL_NOT_INITIALIZED;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto start = std::chrono::high_resolution_clock::now();

    try {
        // 构建输入张量
        std::vector<Ort::Value> input_tensors;
        input_tensors.reserve(input_data.size());
        for (size_t i = 0; i < input_data.size(); ++i) {
            input_tensors.push_back(
                impl_->createInputTensor(input_data[i], static_cast<int>(i)));
        }

        // 执行推理
        auto output_tensors = impl_->session->Run(
            Ort::RunOptions{nullptr},
            impl_->input_name_ptrs.data(),
            input_tensors.data(),
            input_tensors.size(),
            impl_->output_name_ptrs.data(),
            impl_->output_name_ptrs.size());

        // 提取输出（统一转换为 float32 向量）
        result.outputs.resize(output_tensors.size());
        result.output_infos = impl_->output_infos;

        for (size_t i = 0; i < output_tensors.size(); ++i) {
            auto info   = output_tensors[i].GetTensorTypeAndShapeInfo();
            size_t count = info.GetElementCount();
            const float* data = output_tensors[i].GetTensorData<float>();
            result.outputs[i].assign(data, data + count);
        }

        auto end = std::chrono::high_resolution_clock::now();
        result.inference_time_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        result.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        stats_.update(result.inference_time_ms);
        return core::ErrorCode::SUCCESS;

    } catch (const Ort::Exception& e) {
        LOG_ERROR("Inference failed: %s", e.what());
        return core::ErrorCode::INFERENCE_ERROR;
    }
}

// ----------------------------------------------------------------
// 异步推理
// ----------------------------------------------------------------
std::future<InferenceResult> OnnxInferenceEngine::inferAsync(
    const std::vector<float>& input_data) {

    return std::async(std::launch::async, [this, input_data]() -> InferenceResult {
        InferenceResult result;
        core::ErrorCode ret = this->infer(input_data, result);
        if (ret != core::ErrorCode::SUCCESS) {
            throw std::runtime_error(
                "Async inference failed, error code: " +
                std::to_string(static_cast<int>(ret)));
        }
        return result;
    });
}

// ----------------------------------------------------------------
// 元信息访问
// ----------------------------------------------------------------
const std::vector<TensorInfo>& OnnxInferenceEngine::getInputInfos() const {
    return impl_->input_infos;
}
const std::vector<TensorInfo>& OnnxInferenceEngine::getOutputInfos() const {
    return impl_->output_infos;
}
std::vector<int64_t> OnnxInferenceEngine::getInputShape(int index) const {
    if (index >= 0 && index < static_cast<int>(impl_->input_infos.size())) {
        return impl_->input_infos[index].shape;
    }
    return {};
}
std::vector<int64_t> OnnxInferenceEngine::getOutputShape(int index) const {
    if (index >= 0 && index < static_cast<int>(impl_->output_infos.size())) {
        return impl_->output_infos[index].shape;
    }
    return {};
}

std::map<std::string, std::string> OnnxInferenceEngine::getModelMetadata() const {
    std::map<std::string, std::string> meta;
    if (!model_loaded_) return meta;
    try {
        auto m = impl_->session->GetModelMetadata();
        meta["producer_name"] = m.GetProducerNameAllocated(*impl_->allocator).get();
        meta["graph_name"]    = m.GetGraphNameAllocated(*impl_->allocator).get();
        meta["domain"]        = m.GetDomainAllocated(*impl_->allocator).get();
        meta["description"]   = m.GetDescriptionAllocated(*impl_->allocator).get();
    } catch (const Ort::Exception& e) {
        LOG_WARN("Failed to get model metadata: %s", e.what());
    }
    return meta;
}

// ----------------------------------------------------------------
// 性能统计与调试
// ----------------------------------------------------------------
const InferenceStats& OnnxInferenceEngine::getStats() const { return stats_; }
void OnnxInferenceEngine::resetStats() { stats_ = InferenceStats{}; }
void OnnxInferenceEngine::setVerbose(bool verbose) { verbose_ = verbose; }
void OnnxInferenceEngine::setNumThreads(int n) { config_.num_threads = n; }
void OnnxInferenceEngine::setGpuDeviceId(int id) { config_.gpu_device_id = id; }

void OnnxInferenceEngine::printModelInfo() const {
    LOG_INFO("===== OnnxRuntime Model Info =====");
    for (const auto& info : impl_->input_infos) {
        std::string shape;
        for (size_t i = 0; i < info.shape.size(); ++i) {
            shape += std::to_string(info.shape[i]);
            if (i + 1 < info.shape.size()) shape += "x";
        }
        LOG_INFO("  Input  %s: [%s]", info.name.c_str(), shape.c_str());
    }
    for (const auto& info : impl_->output_infos) {
        std::string shape;
        for (size_t i = 0; i < info.shape.size(); ++i) {
            shape += std::to_string(info.shape[i]);
            if (i + 1 < info.shape.size()) shape += "x";
        }
        LOG_INFO("  Output %s: [%s]", info.name.c_str(), shape.c_str());
    }
    LOG_INFO("  Provider: %s, Threads: %d, OptLevel: %s",
             config_.execution_provider.c_str(),
             config_.num_threads,
             config_.optimization_level.c_str());
    LOG_INFO("==================================");
}

// ============================================================================
// InferenceEngineFactory 实现
// ============================================================================

InferenceBackend InferenceEngineFactory::parseBackend(const std::string& name) {
    std::string n = name;
    std::transform(n.begin(), n.end(), n.begin(), ::tolower);
    if (n == "onnxruntime" || n == "ort") return InferenceBackend::OnnxRuntime;
    if (n == "tensorrt"    || n == "trt") return InferenceBackend::TensorRT;
    if (n == "rknn")                      return InferenceBackend::RKNN;
    if (n == "openvino")                  return InferenceBackend::OpenVINO;
    if (n == "coreml")                    return InferenceBackend::CoreML;
    return InferenceBackend::Auto;
}

std::string InferenceEngineFactory::getBackendName(InferenceBackend backend) {
    switch (backend) {
        case InferenceBackend::OnnxRuntime: return "OnnxRuntime";
        case InferenceBackend::TensorRT:    return "TensorRT";
        case InferenceBackend::RKNN:        return "RKNN";
        case InferenceBackend::OpenVINO:    return "OpenVINO";
        case InferenceBackend::CoreML:      return "CoreML";
        default:                            return "Auto";
    }
}

bool InferenceEngineFactory::isBackendAvailable(InferenceBackend backend) {
    switch (backend) {
        case InferenceBackend::OnnxRuntime: return true;  // 始终编译
#ifdef USE_TENSORRT
        case InferenceBackend::TensorRT:    return true;
#endif
#ifdef USE_RKNN
        case InferenceBackend::RKNN:        return true;
#endif
#ifdef USE_OPENVINO
        case InferenceBackend::OpenVINO:    return true;
#endif
        default:                            return false;
    }
}

std::unique_ptr<IInferenceEngine> InferenceEngineFactory::create(
    InferenceBackend backend,
    const core::InferenceConfig& config) {

    switch (backend) {
        case InferenceBackend::OnnxRuntime:
            LOG_INFO("Creating OnnxRuntime inference engine");
            return std::make_unique<OnnxInferenceEngine>(config);

        case InferenceBackend::RKNN:
            LOG_INFO("Creating RKNN inference engine");
            return std::make_unique<RKNNInferenceEngine>(config);

        case InferenceBackend::TensorRT:
            LOG_ERROR("TensorRT backend not yet implemented.");
            return nullptr;

        case InferenceBackend::Auto:
        default:
            // Auto：根据平台自动选择最优后端
            LOG_INFO("Backend=Auto, selecting OnnxRuntime as default");
            return std::make_unique<OnnxInferenceEngine>(config);
    }
}

std::unique_ptr<IInferenceEngine> InferenceEngineFactory::create(
    const core::InferenceConfig& config) {

    // 优先根据配置中的 backend 字段选择后端
    InferenceBackend backend = parseBackend(config.backend);
    if (backend == InferenceBackend::Auto) {
        // 未指定 backend 时，根据 execution_provider 向后兼容推断
        backend = parseBackend(config.execution_provider);
        if (backend == InferenceBackend::Auto) {
            backend = InferenceBackend::OnnxRuntime;
        }
    }
    return create(backend, config);
}

} // namespace inference
} // namespace modules
} // namespace smart_video_analysis
