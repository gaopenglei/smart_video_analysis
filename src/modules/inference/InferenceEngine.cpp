/**
 * @file InferenceEngine.cpp
 * @brief ONNX Runtime推理引擎实现文件
 */

#include "modules/inference/InferenceEngine.hpp"
#include "core/Logger.hpp"
#include <algorithm>
#include <future>
#include <numeric>
#include <cstring>
#include <stdexcept>

namespace smart_video_analysis {
namespace modules {
namespace inference {

// ============================================================================
// OnnxInferenceEngine 实现
// ============================================================================

OnnxInferenceEngine::OnnxInferenceEngine(const core::InferenceConfig& config)
    : config_(config), execution_provider_(ExecutionProvider::CPU) {
    
    // 根据配置设置执行提供者
    std::string provider = config_.execution_provider;
    std::transform(provider.begin(), provider.end(), provider.begin(), ::tolower);
    
    if (provider == "cuda") {
        execution_provider_ = ExecutionProvider::CUDA;
    } else if (provider == "tensorrt") {
        execution_provider_ = ExecutionProvider::TensorRT;
    } else if (provider == "openvino") {
        execution_provider_ = ExecutionProvider::OpenVINO;
    } else if (provider == "nnapi") {
        execution_provider_ = ExecutionProvider::NNAPI;
    } else if (provider == "coreml") {
        execution_provider_ = ExecutionProvider::CoreML;
    } else if (provider == "rockchip" || provider == "rk3588" || provider == "npu") {
        execution_provider_ = ExecutionProvider::RockchipNPU;
    } else {
        execution_provider_ = ExecutionProvider::CPU;
    }
    
    LOG_DEBUG("OnnxInferenceEngine created with provider: %s", 
              InferenceEngineFactory::getProviderName(execution_provider_).c_str());
}

OnnxInferenceEngine::~OnnxInferenceEngine() {
    unloadModel();
}

bool OnnxInferenceEngine::initializeEnvironment() {
    try {
        // 创建ONNX Runtime环境
        OrtLoggingLevel log_level = verbose_ ? ORT_LOGGING_LEVEL_VERBOSE : ORT_LOGGING_LEVEL_WARNING;
        env_ = std::make_unique<Ort::Env>(log_level, "SmartVideoAnalysis");
        
        // 创建默认分配器
        allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();
        
        return true;
    } catch (const Ort::Exception& e) {
        LOG_ERROR("Failed to initialize ONNX Runtime environment: %s", e.what());
        return false;
    }
}

Ort::SessionOptions OnnxInferenceEngine::createSessionOptions() {
    Ort::SessionOptions options;
    
    // 设置线程数
    options.SetIntraOpNumThreads(config_.num_threads);
    options.SetInterOpNumThreads(config_.num_threads);
    
    // 设置图优化级别
    OrtGraphOptimizationLevel opt_level = ORT_DISABLE_ALL;
    if (config_.optimization_level == "all") {
        opt_level = ORT_ENABLE_ALL;
    } else if (config_.optimization_level == "basic") {
        opt_level = ORT_ENABLE_BASIC;
    } else if (config_.optimization_level == "extended") {
        opt_level = ORT_ENABLE_EXTENDED;
    }
    options.SetGraphOptimizationLevel(opt_level);
    
    // 配置执行提供者
    configureExecutionProvider(options);
    
    return options;
}

void OnnxInferenceEngine::configureExecutionProvider(Ort::SessionOptions& options) {
    switch (execution_provider_) {
        case ExecutionProvider::CUDA: {
#ifdef USE_CUDA
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = config_.gpu_device_id;
            cuda_options.arena_extend_strategy = 0;
            cuda_options.gpu_mem_limit = 2 * 1024 * 1024 * 1024; // 2GB
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
            cuda_options.do_copy_in_default_stream = true;
            
            options.AppendExecutionProvider_CUDA(cuda_options);
            LOG_INFO("CUDA execution provider configured (device_id=%d)", config_.gpu_device_id);
#else
            LOG_WARN("CUDA provider not available, falling back to CPU");
            execution_provider_ = ExecutionProvider::CPU;
#endif
            break;
        }
        
        case ExecutionProvider::TensorRT: {
#ifdef USE_TENSORRT
            OrtTensorRTProviderOptions trt_options;
            trt_options.device_id = config_.gpu_device_id;
            trt_options.trt_max_workspace_size = 1 * 1024 * 1024 * 1024; // 1GB
            trt_options.trt_fp16_enable = true;
            
            options.AppendExecutionProvider_TensorRT(trt_options);
            LOG_INFO("TensorRT execution provider configured");
#else
            LOG_WARN("TensorRT provider not available, falling back to CPU");
            execution_provider_ = ExecutionProvider::CPU;
#endif
            break;
        }
        
        case ExecutionProvider::OpenVINO: {
#ifdef USE_OPENVINO
            OrtOpenVINOProviderOptions ov_options;
            ov_options.device_type = "CPU"; // or "GPU", "MYRIAD"
            
            options.AppendExecutionProvider_OpenVINO(ov_options);
            LOG_INFO("OpenVINO execution provider configured");
#else
            LOG_WARN("OpenVINO provider not available, falling back to CPU");
            execution_provider_ = ExecutionProvider::CPU;
#endif
            break;
        }
        
        case ExecutionProvider::NNAPI: {
#ifdef USE_NNAPI
            OrtNNAPIProviderOptions nnapi_options;
            options.AppendExecutionProvider_NNAPI(nnapi_options);
            LOG_INFO("NNAPI execution provider configured");
#else
            LOG_WARN("NNAPI provider not available, falling back to CPU");
            execution_provider_ = ExecutionProvider::CPU;
#endif
            break;
        }
        
        case ExecutionProvider::RockchipNPU: {
            // 瑞芯微RK3588 NPU通常通过RKNN或自定义后端支持
            // 这里需要根据实际的RKNN-ONNX Runtime集成来配置
            LOG_INFO("Rockchip NPU execution provider - using custom configuration");
            // 可能需要使用RKNN Toolkit进行模型转换
            break;
        }
        
        default:
            // CPU执行提供者是默认的，不需要额外配置
            LOG_INFO("Using CPU execution provider with %d threads", config_.num_threads);
            break;
    }
}

core::ErrorCode OnnxInferenceEngine::loadModel(const std::string& model_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (model_loaded_) {
        LOG_WARN("Model already loaded, unloading previous model");
        unloadModel();
    }
    
    LOG_INFO("Loading ONNX model: %s", model_path.c_str());
    
    // 初始化环境
    if (!initializeEnvironment()) {
        return core::ErrorCode::MODEL_LOAD_FAILED;
    }
    
    try {
        // 创建会话选项
        Ort::SessionOptions options = createSessionOptions();
        
        // 创建会话
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), options);
        
        // 提取模型信息
        extractModelInfo();
        
        model_loaded_ = true;
        
        LOG_INFO("Model loaded successfully");
        printModelInfo();
        
        return core::ErrorCode::SUCCESS;
        
    } catch (const Ort::Exception& e) {
        LOG_ERROR("Failed to load model: %s", e.what());
        return core::ErrorCode::MODEL_LOAD_FAILED;
    }
}

void OnnxInferenceEngine::unloadModel() {
    session_.reset();
    allocator_.reset();
    env_.reset();
    
    input_infos_.clear();
    output_infos_.clear();
    input_names_.clear();
    output_names_.clear();
    input_name_ptrs_.clear();
    output_name_ptrs_.clear();
    
    model_loaded_ = false;
    
    LOG_DEBUG("Model unloaded");
}

void OnnxInferenceEngine::extractModelInfo() {
    // 获取输入信息
    size_t num_inputs = session_->GetInputCount();
    input_infos_.resize(num_inputs);
    input_names_.resize(num_inputs);
    input_name_ptrs_.resize(num_inputs);
    
    for (size_t i = 0; i < num_inputs; ++i) {
        // 获取输入名称
        Ort::AllocatedStringPtr name = session_->GetInputNameAllocated(i, *allocator_);
        input_names_[i] = name.get();
        input_name_ptrs_[i] = input_names_[i].c_str();
        
        // 获取输入类型和形状
        Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        
        input_infos_[i].name = input_names_[i];
        input_infos_[i].type = tensor_info.GetElementType();
        input_infos_[i].shape = tensor_info.GetShape();
        input_infos_[i].element_count = tensor_info.GetElementCount();
        input_infos_[i].byte_size = tensor_info.GetElementCount() * sizeof(float);
    }
    
    // 获取输出信息
    size_t num_outputs = session_->GetOutputCount();
    output_infos_.resize(num_outputs);
    output_names_.resize(num_outputs);
    output_name_ptrs_.resize(num_outputs);
    
    for (size_t i = 0; i < num_outputs; ++i) {
        // 获取输出名称
        Ort::AllocatedStringPtr name = session_->GetOutputNameAllocated(i, *allocator_);
        output_names_[i] = name.get();
        output_name_ptrs_[i] = output_names_[i].c_str();
        
        // 获取输出类型和形状
        Ort::TypeInfo type_info = session_->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        
        output_infos_[i].name = output_names_[i];
        output_infos_[i].type = tensor_info.GetElementType();
        output_infos_[i].shape = tensor_info.GetShape();
        output_infos_[i].element_count = tensor_info.GetElementCount();
        output_infos_[i].byte_size = tensor_info.GetElementCount() * sizeof(float);
    }
}

Ort::Value OnnxInferenceEngine::createInputTensor(const std::vector<float>& data, int index) {
    const auto& info = input_infos_[index];
    
    // 创建张量
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    
    return Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(data.data()),
        data.size(),
        info.shape.data(),
        info.shape.size()
    );
}

std::vector<Ort::Value> OnnxInferenceEngine::createInputTensors(
    const std::vector<std::vector<float>>& input_data) {
    
    std::vector<Ort::Value> tensors;
    tensors.reserve(input_data.size());
    
    for (size_t i = 0; i < input_data.size(); ++i) {
        tensors.push_back(createInputTensor(input_data[i], static_cast<int>(i)));
    }
    
    return tensors;
}

core::ErrorCode OnnxInferenceEngine::infer(const std::vector<float>& input_data,
                                            InferenceResult& result) {
    return infer(std::vector<std::vector<float>>{input_data}, result);
}

core::ErrorCode OnnxInferenceEngine::infer(
    const std::vector<std::vector<float>>& input_data,
    InferenceResult& result) {
    
    if (!model_loaded_) {
        LOG_ERROR("Model not loaded");
        return core::ErrorCode::MODEL_NOT_INITIALIZED;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // 创建输入张量
        std::vector<Ort::Value> input_tensors = createInputTensors(input_data);
        
        // 执行推理
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_name_ptrs_.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_name_ptrs_.data(),
            output_name_ptrs_.size()
        );
        
        // 提取输出数据
        result.outputs.resize(output_tensors.size());
        result.output_infos = output_infos_;
        
        for (size_t i = 0; i < output_tensors.size(); ++i) {
            const auto& tensor = output_tensors[i];
            auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
            size_t element_count = tensor_info.GetElementCount();
            
            float* data = const_cast<float*>(tensor.GetTensorData<float>());
            result.outputs[i].assign(data, data + element_count);
        }
        
        // 计算推理时间
        auto end_time = std::chrono::high_resolution_clock::now();
        result.inference_time_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();
        result.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        // 更新统计
        stats_.update(result.inference_time_ms);
        
        return core::ErrorCode::SUCCESS;
        
    } catch (const Ort::Exception& e) {
        LOG_ERROR("Inference failed: %s", e.what());
        return core::ErrorCode::INFERENCE_ERROR;
    }
}

std::future<InferenceResult> OnnxInferenceEngine::inferAsync(
    const std::vector<float>& input_data) {

    return std::async(std::launch::async, [this, input_data]() -> InferenceResult {
        InferenceResult result;
        core::ErrorCode ret = this->infer(input_data, result);
        if (ret != core::ErrorCode::SUCCESS) {
            throw std::runtime_error("Async inference failed with error code: " +
                                     std::to_string(static_cast<int>(ret)));
        }
        return result;
    });
}

bool OnnxInferenceEngine::isModelLoaded() const {
    return model_loaded_;
}

const std::vector<TensorInfo>& OnnxInferenceEngine::getInputInfos() const {
    return input_infos_;
}

const std::vector<TensorInfo>& OnnxInferenceEngine::getOutputInfos() const {
    return output_infos_;
}

std::vector<int64_t> OnnxInferenceEngine::getInputShape(int index) const {
    if (index >= 0 && index < static_cast<int>(input_infos_.size())) {
        return input_infos_[index].shape;
    }
    return {};
}

std::vector<int64_t> OnnxInferenceEngine::getOutputShape(int index) const {
    if (index >= 0 && index < static_cast<int>(output_infos_.size())) {
        return output_infos_[index].shape;
    }
    return {};
}

const InferenceStats& OnnxInferenceEngine::getStats() const {
    return stats_;
}

void OnnxInferenceEngine::resetStats() {
    stats_ = InferenceStats();
}

void OnnxInferenceEngine::setExecutionProvider(ExecutionProvider provider) {
    execution_provider_ = provider;
}

ExecutionProvider OnnxInferenceEngine::getExecutionProvider() const {
    return execution_provider_;
}

void OnnxInferenceEngine::setNumThreads(int num_threads) {
    config_.num_threads = num_threads;
}

void OnnxInferenceEngine::setGpuDeviceId(int device_id) {
    config_.gpu_device_id = device_id;
}

void OnnxInferenceEngine::setVerbose(bool verbose) {
    verbose_ = verbose;
}

std::map<std::string, std::string> OnnxInferenceEngine::getModelMetadata() const {
    std::map<std::string, std::string> metadata;
    
    if (!model_loaded_) {
        return metadata;
    }
    
    try {
        Ort::ModelMetadata model_metadata = session_->GetModelMetadata();
        
        // 获取生产者名称
        auto producer_name = model_metadata.GetProducerNameAllocated(*allocator_);
        metadata["producer_name"] = producer_name.get();
        
        // 获取图名称
        auto graph_name = model_metadata.GetGraphNameAllocated(*allocator_);
        metadata["graph_name"] = graph_name.get();
        
        // 获取域
        auto domain = model_metadata.GetDomainAllocated(*allocator_);
        metadata["domain"] = domain.get();
        
        // 获取描述
        auto description = model_metadata.GetDescriptionAllocated(*allocator_);
        metadata["description"] = description.get();
        
    } catch (const Ort::Exception& e) {
        LOG_WARN("Failed to get model metadata: %s", e.what());
    }
    
    return metadata;
}

void OnnxInferenceEngine::printModelInfo() const {
    LOG_INFO("========== Model Information ==========");
    
    // 输入信息
    LOG_INFO("Inputs:");
    for (const auto& info : input_infos_) {
        std::string shape_str;
        for (size_t i = 0; i < info.shape.size(); ++i) {
            shape_str += std::to_string(info.shape[i]);
            if (i < info.shape.size() - 1) shape_str += "x";
        }
        LOG_INFO("  %s: [%s], elements=%zu", 
                 info.name.c_str(), shape_str.c_str(), info.element_count);
    }
    
    // 输出信息
    LOG_INFO("Outputs:");
    for (const auto& info : output_infos_) {
        std::string shape_str;
        for (size_t i = 0; i < info.shape.size(); ++i) {
            shape_str += std::to_string(info.shape[i]);
            if (i < info.shape.size() - 1) shape_str += "x";
        }
        LOG_INFO("  %s: [%s], elements=%zu", 
                 info.name.c_str(), shape_str.c_str(), info.element_count);
    }
    
    LOG_INFO("Execution Provider: %s", 
             InferenceEngineFactory::getProviderName(execution_provider_).c_str());
    LOG_INFO("========================================");
}

// ============================================================================
// InferenceEngineFactory 实现
// ============================================================================

std::unique_ptr<OnnxInferenceEngine> InferenceEngineFactory::create(
    const core::InferenceConfig& config) {
    return std::make_unique<OnnxInferenceEngine>(config);
}

std::vector<ExecutionProvider> InferenceEngineFactory::getAvailableProviders() {
    std::vector<ExecutionProvider> providers;
    
    // CPU始终可用
    providers.push_back(ExecutionProvider::CPU);
    
#ifdef USE_CUDA
    providers.push_back(ExecutionProvider::CUDA);
#endif
#ifdef USE_TENSORRT
    providers.push_back(ExecutionProvider::TensorRT);
#endif
#ifdef USE_OPENVINO
    providers.push_back(ExecutionProvider::OpenVINO);
#endif
#ifdef USE_NNAPI
    providers.push_back(ExecutionProvider::NNAPI);
#endif
    
    return providers;
}

bool InferenceEngineFactory::isProviderAvailable(ExecutionProvider provider) {
    auto providers = getAvailableProviders();
    return std::find(providers.begin(), providers.end(), provider) != providers.end();
}

std::string InferenceEngineFactory::getProviderName(ExecutionProvider provider) {
    switch (provider) {
        case ExecutionProvider::CPU: return "CPU";
        case ExecutionProvider::CUDA: return "CUDA";
        case ExecutionProvider::TensorRT: return "TensorRT";
        case ExecutionProvider::OpenVINO: return "OpenVINO";
        case ExecutionProvider::NNAPI: return "NNAPI";
        case ExecutionProvider::CoreML: return "CoreML";
        case ExecutionProvider::DNNL: return "DNNL";
        case ExecutionProvider::XNNPACK: return "XNNPACK";
        case ExecutionProvider::VitisAI: return "VitisAI";
        case ExecutionProvider::RockchipNPU: return "RockchipNPU";
        default: return "Unknown";
    }
}

} // namespace inference
} // namespace modules
} // namespace smart_video_analysis
