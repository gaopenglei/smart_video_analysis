/**
 * @file RKNNInferenceEngine.cpp
 * @brief 瑞芯微 RKNN 推理引擎实现
 *
 * 所有 rknn_api.h 具体类型只在此文件中出现，完全隔离于上层代码。
 * 切换到其他推理库时，上层（SmartVideoAnalysisSystem 等）无需任何修改。
 *
 * RKNN 推理流程：
 *   PC 端（离线）: PyTorch/ONNX → rknn-toolkit2 → .rknn 模型文件
 *   设备端（在线): rknn_init → rknn_inputs_set → rknn_run → rknn_outputs_get
 *
 * 编译条件：需要定义 USE_RKNN 宏并链接 librknnrt.so
 */

#ifdef USE_RKNN

// RKNN Runtime 头文件（仅在此 .cpp 中引入）
#include <rknn_api.h>

#include "modules/inference/RKNNInferenceEngine.hpp"
#include "core/Logger.hpp"
#include <chrono>
#include <cstring>
#include <future>
#include <stdexcept>
#include <fstream>

namespace smart_video_analysis {
namespace modules {
namespace inference {

// ============================================================================
// pImpl：将所有 RKNN 具体类型隐藏于此结构体
// ============================================================================

struct RKNNInferenceEngine::Impl {
    rknn_context ctx = 0;           ///< RKNN 上下文句柄
    rknn_input_output_num io_num{}; ///< 输入/输出张量数量

    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;
    std::vector<TensorInfo>       input_infos;
    std::vector<TensorInfo>       output_infos;

    // 模型文件数据（rknn_init 需要整块内存）
    std::vector<unsigned char> model_data;

    // ----------------------------------------------------------------
    // RKNN 数据类型 → 通用 TensorDataType
    // ----------------------------------------------------------------
    static TensorDataType fromRKNNType(rknn_tensor_type t) {
        switch (t) {
            case RKNN_TENSOR_FLOAT32: return TensorDataType::FLOAT32;
            case RKNN_TENSOR_FLOAT16: return TensorDataType::FLOAT16;
            case RKNN_TENSOR_INT8:    return TensorDataType::INT8;
            case RKNN_TENSOR_UINT8:   return TensorDataType::UINT8;
            case RKNN_TENSOR_INT16:   return TensorDataType::INT16;
            case RKNN_TENSOR_INT32:   return TensorDataType::INT32;
            case RKNN_TENSOR_INT64:   return TensorDataType::INT64;
            default:                  return TensorDataType::UNKNOWN;
        }
    }

    // ----------------------------------------------------------------
    // RKNN 格式（NCHW/NHWC）→ 标准 shape [N,C,H,W]
    // ----------------------------------------------------------------
    static std::vector<int64_t> toShape(const rknn_tensor_attr& attr) {
        std::vector<int64_t> shape(attr.n_dims);
        for (uint32_t i = 0; i < attr.n_dims; ++i) {
            shape[i] = static_cast<int64_t>(attr.dims[i]);
        }
        // RKNN 的 dims 按 [N,C,H,W] 顺序存储，与 IInferenceEngine 约定一致
        return shape;
    }

    // ----------------------------------------------------------------
    // 填充 TensorInfo
    // ----------------------------------------------------------------
    static TensorInfo toTensorInfo(const rknn_tensor_attr& attr) {
        TensorInfo info;
        info.name  = attr.name;
        info.shape = toShape(attr);
        info.type  = fromRKNNType(attr.type);
        info.element_count = static_cast<size_t>(attr.n_elems);
        info.byte_size     = static_cast<size_t>(attr.size);
        return info;
    }
};

// ============================================================================
// RKNNInferenceEngine 实现
// ============================================================================

RKNNInferenceEngine::RKNNInferenceEngine(const core::InferenceConfig& config)
    : impl_(std::make_unique<Impl>()), config_(config) {
    LOG_DEBUG("RKNNInferenceEngine created (backend: RKNN)");
}

RKNNInferenceEngine::~RKNNInferenceEngine() {
    unloadModel();
}

// ----------------------------------------------------------------
// loadModel：读取 .rknn 文件并初始化上下文
// ----------------------------------------------------------------
core::ErrorCode RKNNInferenceEngine::loadModel(const std::string& model_path) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (model_loaded_) {
        LOG_WARN("RKNN model already loaded, unloading first");
        unloadModel();
    }

    LOG_INFO("Loading RKNN model: %s", model_path.c_str());

    // 读取模型文件到内存
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_ERROR("Cannot open RKNN model file: %s", model_path.c_str());
        return core::ErrorCode::MODEL_LOAD_FAILED;
    }
    size_t model_size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    impl_->model_data.resize(model_size);
    if (!file.read(reinterpret_cast<char*>(impl_->model_data.data()), model_size)) {
        LOG_ERROR("Failed to read RKNN model file: %s", model_path.c_str());
        return core::ErrorCode::MODEL_LOAD_FAILED;
    }
    file.close();

    // 初始化 RKNN 上下文
    int ret = rknn_init(&impl_->ctx,
                        impl_->model_data.data(),
                        static_cast<uint32_t>(model_size),
                        0,    // flag
                        nullptr);
    if (ret != RKNN_SUCC) {
        LOG_ERROR("rknn_init failed, ret=%d", ret);
        return core::ErrorCode::MODEL_LOAD_FAILED;
    }

    // RK3588 多核 NPU 设置（core_mask: 0=自动, 0x01/0x02/0x04=指定核心）
    if (config_.rknn_core_mask != 0) {
        ret = rknn_set_core_mask(impl_->ctx,
                                 static_cast<rknn_core_mask>(config_.rknn_core_mask));
        if (ret != RKNN_SUCC) {
            LOG_WARN("rknn_set_core_mask failed, ret=%d, using default core", ret);
        } else {
            LOG_INFO("RKNN core mask set to 0x%02X", config_.rknn_core_mask);
        }
    }

    // 查询输入/输出数量
    ret = rknn_query(impl_->ctx, RKNN_QUERY_IN_OUT_NUM,
                     &impl_->io_num, sizeof(impl_->io_num));
    if (ret != RKNN_SUCC) {
        LOG_ERROR("rknn_query IN_OUT_NUM failed, ret=%d", ret);
        rknn_destroy(impl_->ctx);
        impl_->ctx = 0;
        return core::ErrorCode::MODEL_LOAD_FAILED;
    }

    // 查询各输入张量属性
    impl_->input_attrs.resize(impl_->io_num.n_input);
    impl_->input_infos.resize(impl_->io_num.n_input);
    for (uint32_t i = 0; i < impl_->io_num.n_input; ++i) {
        memset(&impl_->input_attrs[i], 0, sizeof(rknn_tensor_attr));
        impl_->input_attrs[i].index = i;
        ret = rknn_query(impl_->ctx, RKNN_QUERY_INPUT_ATTR,
                         &impl_->input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            LOG_ERROR("rknn_query INPUT_ATTR[%u] failed, ret=%d", i, ret);
            rknn_destroy(impl_->ctx);
            impl_->ctx = 0;
            return core::ErrorCode::MODEL_LOAD_FAILED;
        }
        impl_->input_infos[i] = Impl::toTensorInfo(impl_->input_attrs[i]);
    }

    // 查询各输出张量属性
    impl_->output_attrs.resize(impl_->io_num.n_output);
    impl_->output_infos.resize(impl_->io_num.n_output);
    for (uint32_t i = 0; i < impl_->io_num.n_output; ++i) {
        memset(&impl_->output_attrs[i], 0, sizeof(rknn_tensor_attr));
        impl_->output_attrs[i].index = i;
        ret = rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR,
                         &impl_->output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            LOG_ERROR("rknn_query OUTPUT_ATTR[%u] failed, ret=%d", i, ret);
            rknn_destroy(impl_->ctx);
            impl_->ctx = 0;
            return core::ErrorCode::MODEL_LOAD_FAILED;
        }
        impl_->output_infos[i] = Impl::toTensorInfo(impl_->output_attrs[i]);
    }

    model_loaded_ = true;
    LOG_INFO("RKNN model loaded: %u input(s), %u output(s)",
             impl_->io_num.n_input, impl_->io_num.n_output);
    printModelInfo();
    return core::ErrorCode::SUCCESS;
}

// ----------------------------------------------------------------
// unloadModel：销毁 RKNN 上下文，释放 NPU 资源
// ----------------------------------------------------------------
void RKNNInferenceEngine::unloadModel() {
    if (impl_->ctx != 0) {
        rknn_destroy(impl_->ctx);
        impl_->ctx = 0;
    }
    impl_->model_data.clear();
    impl_->input_attrs.clear();
    impl_->output_attrs.clear();
    impl_->input_infos.clear();
    impl_->output_infos.clear();
    model_loaded_ = false;
    LOG_DEBUG("RKNN model unloaded");
}

bool RKNNInferenceEngine::isModelLoaded() const {
    return model_loaded_;
}

// ----------------------------------------------------------------
// infer（单输入）
// ----------------------------------------------------------------
core::ErrorCode RKNNInferenceEngine::infer(const std::vector<float>& input_data,
                                            InferenceResult& result) {
    return infer(std::vector<std::vector<float>>{input_data}, result);
}

// ----------------------------------------------------------------
// infer（多输入）—— 核心推理函数
// ----------------------------------------------------------------
core::ErrorCode RKNNInferenceEngine::infer(
    const std::vector<std::vector<float>>& input_data,
    InferenceResult& result) {

    if (!model_loaded_) {
        LOG_ERROR("RKNN model not loaded");
        return core::ErrorCode::MODEL_NOT_INITIALIZED;
    }

    if (input_data.size() != impl_->io_num.n_input) {
        LOG_ERROR("Input count mismatch: expected %u, got %zu",
                  impl_->io_num.n_input, input_data.size());
        return core::ErrorCode::INVALID_PARAMETER;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto start = std::chrono::high_resolution_clock::now();

    // ---- 设置输入 ----
    std::vector<rknn_input> inputs(impl_->io_num.n_input);
    for (uint32_t i = 0; i < impl_->io_num.n_input; ++i) {
        memset(&inputs[i], 0, sizeof(rknn_input));
        inputs[i].index        = i;
        inputs[i].type         = RKNN_TENSOR_FLOAT32;
        inputs[i].fmt          = RKNN_TENSOR_NCHW;
        inputs[i].buf          = const_cast<float*>(input_data[i].data());
        inputs[i].size         = static_cast<uint32_t>(
                                     input_data[i].size() * sizeof(float));
        inputs[i].pass_through = 0; // 由 RKNN Runtime 做量化/格式转换
    }

    int ret = rknn_inputs_set(impl_->ctx,
                              impl_->io_num.n_input,
                              inputs.data());
    if (ret != RKNN_SUCC) {
        LOG_ERROR("rknn_inputs_set failed, ret=%d", ret);
        return core::ErrorCode::INFERENCE_ERROR;
    }

    // ---- 执行推理 ----
    ret = rknn_run(impl_->ctx, nullptr);
    if (ret != RKNN_SUCC) {
        LOG_ERROR("rknn_run failed, ret=%d", ret);
        return core::ErrorCode::INFERENCE_ERROR;
    }

    // ---- 获取输出（统一以 float32 返回，方便上层统一处理）----
    std::vector<rknn_output> outputs(impl_->io_num.n_output);
    for (uint32_t i = 0; i < impl_->io_num.n_output; ++i) {
        memset(&outputs[i], 0, sizeof(rknn_output));
        outputs[i].want_float = 1;  // 自动将量化输出反量化为 float32
        outputs[i].index      = i;
        outputs[i].is_prealloc = 0; // 由 RKNN Runtime 分配缓冲区
    }

    ret = rknn_outputs_get(impl_->ctx,
                           impl_->io_num.n_output,
                           outputs.data(),
                           nullptr);
    if (ret != RKNN_SUCC) {
        LOG_ERROR("rknn_outputs_get failed, ret=%d", ret);
        return core::ErrorCode::INFERENCE_ERROR;
    }

    // ---- 将 RKNN 输出拷贝到 InferenceResult ----
    result.outputs.resize(impl_->io_num.n_output);
    result.output_infos = impl_->output_infos;

    for (uint32_t i = 0; i < impl_->io_num.n_output; ++i) {
        size_t n_elem = outputs[i].size / sizeof(float);
        const float* data = static_cast<const float*>(outputs[i].buf);
        result.outputs[i].assign(data, data + n_elem);
    }

    // 释放 RKNN 输出缓冲区
    rknn_outputs_release(impl_->ctx, impl_->io_num.n_output, outputs.data());

    // ---- 记录推理耗时 ----
    auto end = std::chrono::high_resolution_clock::now();
    result.inference_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    result.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    stats_.update(result.inference_time_ms);
    return core::ErrorCode::SUCCESS;
}

// ----------------------------------------------------------------
// 异步推理
// ----------------------------------------------------------------
std::future<InferenceResult> RKNNInferenceEngine::inferAsync(
    const std::vector<float>& input_data) {

    return std::async(std::launch::async, [this, input_data]() -> InferenceResult {
        InferenceResult result;
        core::ErrorCode ret = this->infer(input_data, result);
        if (ret != core::ErrorCode::SUCCESS) {
            throw std::runtime_error(
                "RKNN async inference failed, error code: " +
                std::to_string(static_cast<int>(ret)));
        }
        return result;
    });
}

// ----------------------------------------------------------------
// 元信息访问
// ----------------------------------------------------------------
const std::vector<TensorInfo>& RKNNInferenceEngine::getInputInfos() const {
    return impl_->input_infos;
}
const std::vector<TensorInfo>& RKNNInferenceEngine::getOutputInfos() const {
    return impl_->output_infos;
}
std::vector<int64_t> RKNNInferenceEngine::getInputShape(int index) const {
    if (index >= 0 && index < static_cast<int>(impl_->input_infos.size()))
        return impl_->input_infos[index].shape;
    return {};
}
std::vector<int64_t> RKNNInferenceEngine::getOutputShape(int index) const {
    if (index >= 0 && index < static_cast<int>(impl_->output_infos.size()))
        return impl_->output_infos[index].shape;
    return {};
}

std::map<std::string, std::string> RKNNInferenceEngine::getModelMetadata() const {
    if (!model_loaded_) return {};

    rknn_sdk_version ver{};
    int ret = rknn_query(impl_->ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver));

    std::map<std::string, std::string> meta;
    if (ret == RKNN_SUCC) {
        meta["rknn_api_version"] = ver.api_version;
        meta["rknn_drv_version"] = ver.drv_version;
    }
    meta["backend"] = "RKNN";
    meta["n_inputs"]  = std::to_string(impl_->io_num.n_input);
    meta["n_outputs"] = std::to_string(impl_->io_num.n_output);
    return meta;
}

// ----------------------------------------------------------------
// 性能统计与调试
// ----------------------------------------------------------------
const InferenceStats& RKNNInferenceEngine::getStats() const { return stats_; }
void RKNNInferenceEngine::resetStats()       { stats_ = InferenceStats{}; }
void RKNNInferenceEngine::setVerbose(bool v) { verbose_ = v; }

void RKNNInferenceEngine::printModelInfo() const {
    LOG_INFO("===== RKNN Model Info =====");
    LOG_INFO("  Inputs : %u", impl_->io_num.n_input);
    for (const auto& info : impl_->input_infos) {
        std::string shape;
        for (size_t i = 0; i < info.shape.size(); ++i) {
            shape += std::to_string(info.shape[i]);
            if (i + 1 < info.shape.size()) shape += "x";
        }
        LOG_INFO("    %s: [%s]", info.name.c_str(), shape.c_str());
    }
    LOG_INFO("  Outputs: %u", impl_->io_num.n_output);
    for (const auto& info : impl_->output_infos) {
        std::string shape;
        for (size_t i = 0; i < info.shape.size(); ++i) {
            shape += std::to_string(info.shape[i]);
            if (i + 1 < info.shape.size()) shape += "x";
        }
        LOG_INFO("    %s: [%s]", info.name.c_str(), shape.c_str());
    }
    LOG_INFO("  NPU core_mask: 0x%02X", config_.rknn_core_mask);
    LOG_INFO("===========================");

    if (verbose_ && impl_->ctx != 0) {
        rknn_perf_detail perf{};
        if (rknn_query(impl_->ctx, RKNN_QUERY_PERF_DETAIL,
                       &perf, sizeof(perf)) == RKNN_SUCC) {
            LOG_DEBUG("RKNN perf detail: %s", perf.perf_data);
        }
    }
}

} // namespace inference
} // namespace modules
} // namespace smart_video_analysis

#else // !USE_RKNN

// 未启用 RKNN 时提供空实现，防止链接报错
#include "modules/inference/RKNNInferenceEngine.hpp"
#include "core/Logger.hpp"
#include <stdexcept>

namespace smart_video_analysis {
namespace modules {
namespace inference {

struct RKNNInferenceEngine::Impl {};

RKNNInferenceEngine::RKNNInferenceEngine(const core::InferenceConfig&)
    : impl_(std::make_unique<Impl>()) {
    LOG_ERROR("RKNNInferenceEngine is not available: "
              "recompile with -DUSE_RKNN=ON and link librknnrt.so");
}
RKNNInferenceEngine::~RKNNInferenceEngine() = default;

core::ErrorCode RKNNInferenceEngine::loadModel(const std::string&) {
    return core::ErrorCode::MODEL_LOAD_FAILED;
}
void RKNNInferenceEngine::unloadModel() {}
bool RKNNInferenceEngine::isModelLoaded() const { return false; }

core::ErrorCode RKNNInferenceEngine::infer(const std::vector<float>&, InferenceResult&) {
    return core::ErrorCode::INFERENCE_ERROR;
}
core::ErrorCode RKNNInferenceEngine::infer(const std::vector<std::vector<float>>&, InferenceResult&) {
    return core::ErrorCode::INFERENCE_ERROR;
}
std::future<InferenceResult> RKNNInferenceEngine::inferAsync(const std::vector<float>&) {
    return std::async(std::launch::deferred, []() -> InferenceResult {
        throw std::runtime_error("RKNN backend not compiled in");
    });
}

const std::vector<TensorInfo>& RKNNInferenceEngine::getInputInfos() const {
    static std::vector<TensorInfo> empty;
    return empty;
}
const std::vector<TensorInfo>& RKNNInferenceEngine::getOutputInfos() const {
    static std::vector<TensorInfo> empty;
    return empty;
}
std::vector<int64_t> RKNNInferenceEngine::getInputShape(int)  const { return {}; }
std::vector<int64_t> RKNNInferenceEngine::getOutputShape(int) const { return {}; }
std::map<std::string, std::string> RKNNInferenceEngine::getModelMetadata() const { return {}; }
const InferenceStats& RKNNInferenceEngine::getStats() const {
    static InferenceStats s;
    return s;
}
void RKNNInferenceEngine::resetStats() {}
void RKNNInferenceEngine::printModelInfo() const {
    LOG_WARN("RKNN backend not compiled in");
}
void RKNNInferenceEngine::setVerbose(bool) {}

} // namespace inference
} // namespace modules
} // namespace smart_video_analysis

#endif // USE_RKNN
