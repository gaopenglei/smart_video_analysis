/**
 * @file ModelProcessor.cpp
 * @brief ONNX模型处理模块实现文件
 */

#include "modules/model_processor/ModelProcessor.hpp"
#include "core/Logger.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <chrono>

// ONNX ProtoBuf 头文件
#include <onnx/onnx_pb.h>
#include <onnx/shape_inference/implementation.h>
#include <onnxoptimizer/optimize.h>

namespace smart_video_analysis {
namespace modules {
namespace model_processor {

// ============================================================================
// TensorShape 方法实现
// ============================================================================

bool TensorShape::isDynamic() const {
    for (const auto& dim : dims) {
        if (dim < 0) return true;
    }
    for (const auto& param : dim_params) {
        if (!param.empty()) return true;
    }
    return false;
}

int64_t TensorShape::elementCount() const {
    if (isDynamic()) return -1;
    int64_t count = 1;
    for (const auto& dim : dims) {
        count *= dim;
    }
    return count;
}

int64_t TensorShape::byteSize() const {
    int64_t count = elementCount();
    if (count < 0) return -1;
    
    // 根据数据类型计算大小
    if (data_type == "float" || data_type == "float32") {
        return count * 4;
    } else if (data_type == "double" || data_type == "float64") {
        return count * 8;
    } else if (data_type == "int32") {
        return count * 4;
    } else if (data_type == "int64") {
        return count * 8;
    } else if (data_type == "int16") {
        return count * 2;
    } else if (data_type == "int8" || data_type == "uint8") {
        return count;
    }
    return count * 4; // 默认float
}

std::string TensorShape::toString() const {
    std::ostringstream oss;
    oss << name << ": [";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) oss << ", ";
        if (dims[i] < 0 && i < dim_params.size() && !dim_params[i].empty()) {
            oss << dim_params[i];
        } else {
            oss << dims[i];
        }
    }
    oss << "] (" << data_type << ")";
    return oss.str();
}

// ============================================================================
// OperatorNode 方法实现
// ============================================================================

std::string OperatorNode::toString() const {
    std::ostringstream oss;
    oss << op_type << "('" << name << "')";
    if (!inputs.empty()) {
        oss << " inputs=[";
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << inputs[i];
        }
        oss << "]";
    }
    if (!outputs.empty()) {
        oss << " outputs=[";
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << outputs[i];
        }
        oss << "]";
    }
    return oss.str();
}

// ============================================================================
// ModelInfo 方法实现
// ============================================================================

void ModelInfo::print() const {
    LOG_INFO("========== Model Information ==========");
    LOG_INFO("Path: %s", model_path.c_str());
    LOG_INFO("IR Version: %s", ir_version.c_str());
    LOG_INFO("Producer: %s (%s)", producer_name.c_str(), producer_version.c_str());
    LOG_INFO("Model Version: %s", model_version.c_str());
    LOG_INFO("");
    
    LOG_INFO("Inputs:");
    for (const auto& input : inputs) {
        LOG_INFO("  %s", input.toString().c_str());
    }
    LOG_INFO("");
    
    LOG_INFO("Outputs:");
    for (const auto& output : outputs) {
        LOG_INFO("  %s", output.toString().c_str());
    }
    LOG_INFO("");
    
    LOG_INFO("Statistics:");
    LOG_INFO("  Total Nodes: %ld", total_nodes);
    LOG_INFO("  Total Parameters: %s", model_utils::formatParams(total_params).c_str());
    LOG_INFO("  Estimated FLOPs: %s", model_utils::formatFlops(total_flops).c_str());
    LOG_INFO("  File Size: %s", model_utils::formatFileSize(file_size).c_str());
    LOG_INFO("");
    
    LOG_INFO("Operator Types (%zu):", op_types.size());
    for (const auto& op_type : op_types) {
        int64_t count = 0;
        auto it = op_counts.find(op_type);
        if (it != op_counts.end()) {
            count = it->second;
        }
        LOG_INFO("  - %s: %ld", op_type.c_str(), count);
    }
    LOG_INFO("========================================");
}

std::string ModelInfo::getSummary() const {
    std::ostringstream oss;
    oss << "Model: " << model_path << "\n";
    oss << "Inputs: " << inputs.size() << ", Outputs: " << outputs.size() << "\n";
    oss << "Nodes: " << total_nodes << ", Ops: " << op_types.size() << "\n";
    oss << "Params: " << model_utils::formatParams(total_params) << "\n";
    oss << "FLOPs: " << model_utils::formatFlops(total_flops) << "\n";
    oss << "Size: " << model_utils::formatFileSize(file_size);
    return oss.str();
}

// ============================================================================
// OnnxModelProcessor 实现
// ============================================================================

OnnxModelProcessor::OnnxModelProcessor() 
    : model_(std::make_unique<onnx::ModelProto>()) {
    LOG_DEBUG("OnnxModelProcessor created");
}

OnnxModelProcessor::~OnnxModelProcessor() {
    LOG_DEBUG("OnnxModelProcessor destroyed");
}

core::ErrorCode OnnxModelProcessor::loadModel(const std::string& model_path) {
    LOG_INFO("Loading ONNX model: %s", model_path.c_str());
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 检查文件是否存在
    std::ifstream ifs(model_path, std::ios::binary);
    if (!ifs.is_open()) {
        LOG_ERROR("Failed to open model file: %s", model_path.c_str());
        return core::ErrorCode::FILE_NOT_FOUND;
    }
    
    // 解析ONNX模型
    if (!model_->ParseFromIstream(&ifs)) {
        LOG_ERROR("Failed to parse ONNX model: %s", model_path.c_str());
        ifs.close();
        return core::ErrorCode::INVALID_MODEL_FORMAT;
    }
    ifs.close();
    
    model_path_ = model_path;
    model_loaded_ = true;
    
    // 解析模型信息
    parseModelInfo();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    LOG_INFO("Model loaded successfully in %ld ms", duration);
    LOG_INFO("Model summary: %s", model_info_.getSummary().c_str());
    
    return core::ErrorCode::SUCCESS;
}

void OnnxModelProcessor::parseModelInfo() {
    model_info_.model_path = model_path_;
    
    // IR版本
    model_info_.ir_version = std::to_string(model_->ir_version());
    
    // 生产者信息
    model_info_.producer_name = model_->producer_name();
    model_info_.producer_version = model_->producer_version();
    model_info_.model_version = std::to_string(model_->model_version());
    model_info_.doc_string = model_->doc_string();
    
    // 文件大小
    std::ifstream file(model_path_, std::ios::binary | std::ios::ate);
    if (file.is_open()) {
        model_info_.file_size = file.tellg();
        file.close();
    }
    
    // 解析图
    parseInputsOutputs();
    parseNodes();
    calculateStatistics();
}

void OnnxModelProcessor::parseInputsOutputs() {
    const auto& graph = model_->graph();
    
    // 解析输入
    for (const auto& input : graph.input()) {
        TensorShape shape;
        shape.name = input.name();
        
        if (input.has_type() && input.type().has_tensor_type()) {
            const auto& tensor_type = input.type().tensor_type();
            shape.data_type = model_utils::getDataTypeString(tensor_type.elem_type());
            
            if (tensor_type.has_shape()) {
                for (const auto& dim : tensor_type.shape().dim()) {
                    if (dim.has_dim_value()) {
                        shape.dims.push_back(dim.dim_value());
                    } else if (dim.has_dim_param()) {
                        shape.dims.push_back(-1);
                        shape.dim_params.push_back(dim.dim_param());
                    } else {
                        shape.dims.push_back(-1);
                    }
                }
            }
        }
        
        model_info_.inputs.push_back(shape);
    }
    
    // 解析输出
    for (const auto& output : graph.output()) {
        TensorShape shape;
        shape.name = output.name();
        
        if (output.has_type() && output.type().has_tensor_type()) {
            const auto& tensor_type = output.type().tensor_type();
            shape.data_type = model_utils::getDataTypeString(tensor_type.elem_type());
            
            if (tensor_type.has_shape()) {
                for (const auto& dim : tensor_type.shape().dim()) {
                    if (dim.has_dim_value()) {
                        shape.dims.push_back(dim.dim_value());
                    } else if (dim.has_dim_param()) {
                        shape.dims.push_back(-1);
                        shape.dim_params.push_back(dim.dim_param());
                    } else {
                        shape.dims.push_back(-1);
                    }
                }
            }
        }
        
        model_info_.outputs.push_back(shape);
    }
}

void OnnxModelProcessor::parseNodes() {
    const auto& graph = model_->graph();
    
    model_info_.nodes.clear();
    model_info_.op_types.clear();
    model_info_.op_counts.clear();
    
    for (const auto& node : graph.node()) {
        OperatorNode op_node;
        op_node.name = node.name();
        op_node.op_type = node.op_type();
        
        for (const auto& input : node.input()) {
            op_node.inputs.push_back(input);
        }
        for (const auto& output : node.output()) {
            op_node.outputs.push_back(output);
        }
        
        // 解析属性
        for (const auto& attr : node.attribute()) {
            std::string value;
            if (attr.type() == onnx::AttributeProto::INT) {
                value = std::to_string(attr.i());
            } else if (attr.type() == onnx::AttributeProto::FLOAT) {
                value = std::to_string(attr.f());
            } else if (attr.type() == onnx::AttributeProto::STRING) {
                value = attr.s();
            } else if (attr.type() == onnx::AttributeProto::INTS) {
                std::ostringstream oss;
                for (int i = 0; i < attr.ints_size(); ++i) {
                    if (i > 0) oss << ",";
                    oss << attr.ints(i);
                }
                value = oss.str();
            }
            op_node.attributes[attr.name()] = value;
        }
        
        // 估算FLOPS
        op_node.flops = estimateNodeFlops(op_node);
        
        model_info_.nodes.push_back(op_node);
        model_info_.op_types.insert(op_node.op_type);
        model_info_.op_counts[op_node.op_type]++;
    }
    
    model_info_.total_nodes = model_info_.nodes.size();
}

void OnnxModelProcessor::calculateStatistics() {
    model_info_.total_flops = 0;
    model_info_.total_params = 0;
    
    const auto& graph = model_->graph();
    
    // 计算参数量（从初始化器中获取）
    for (const auto& init : graph.initializer()) {
        int64_t count = 1;
        for (const auto& dim : init.dims()) {
            count *= dim;
        }
        model_info_.total_params += count;
    }
    
    // 计算总FLOPS
    for (const auto& node : model_info_.nodes) {
        model_info_.total_flops += node.flops;
    }
}

int64_t OnnxModelProcessor::estimateNodeFlops(const OperatorNode& node) const {
    // 简化的FLOPS估算
    std::string op_type = node.op_type;
    
    if (op_type == "Conv") {
        // 对于卷积，需要获取权重形状
        // 这里使用简化估算
        return 0; // 需要更详细的分析
    } else if (op_type == "MatMul") {
        return 0; // 需要输入形状信息
    } else if (op_type == "Gemm") {
        return 0; // 需要属性信息
    }
    
    return 0;
}

core::ErrorCode OnnxModelProcessor::getModelInfo(ModelInfo& info) const {
    if (!model_loaded_) {
        LOG_ERROR("Model not loaded");
        return core::ErrorCode::MODEL_NOT_INITIALIZED;
    }
    info = model_info_;
    return core::ErrorCode::SUCCESS;
}

bool OnnxModelProcessor::validate() const {
    if (!model_loaded_) {
        return false;
    }
    
    // 使用ONNX检查器验证模型
    std::string serialized;
    if (!model_->SerializeToString(&serialized)) {
        LOG_ERROR("Failed to serialize model for validation");
        return false;
    }
    
    // ONNX内置验证
    // onnx::checker::check_model(*model_);
    
    LOG_DEBUG("Model validation passed");
    return true;
}

core::ErrorCode OnnxModelProcessor::optimize(
    const OptimizationOptions& options,
    const std::string& output_path,
    ProcessResult& result) {
    
    if (!model_loaded_) {
        LOG_ERROR("Model not loaded");
        return core::ErrorCode::MODEL_NOT_INITIALIZED;
    }
    
    LOG_INFO("Optimizing model: %s", model_path_.c_str());
    auto start_time = std::chrono::high_resolution_clock::now();
    
    result.original_info = model_info_;
    result.original_size = model_info_.file_size;
    
    try {
        // 构建优化pass列表
        std::vector<std::string> passes;
        
        if (options.eliminate_identity) {
            passes.push_back("eliminate_identity");
        }
        if (options.eliminate_nop_transpose) {
            passes.push_back("eliminate_nop_transpose");
        }
        if (options.eliminate_nop_pad) {
            passes.push_back("eliminate_nop_pad");
        }
        if (options.eliminate_unused_nodes) {
            passes.push_back("eliminate_unused_initializer");
        }
        if (options.fuse_consecutive_transposes) {
            passes.push_back("fuse_consecutive_transposes");
        }
        if (options.fuse_consecutive_squeezes) {
            passes.push_back("fuse_consecutive_squeezes");
        }
        if (options.fuse_consecutive_reshapes) {
            passes.push_back("fuse_consecutive_reshapes");
        }
        if (options.fuse_add_bias_into_conv) {
            passes.push_back("fuse_add_bias_into_conv");
        }
        if (options.fuse_bn_into_conv) {
            passes.push_back("fuse_bn_into_conv");
        }
        
        // 使用onnxoptimizer进行优化
        onnx::ModelProto optimized_model = onnx::optimization::Optimize(
            *model_, passes);
        
        // 保存优化后的模型
        std::ofstream ofs(output_path, std::ios::binary);
        if (!optimized_model.SerializeToOstream(&ofs)) {
            LOG_ERROR("Failed to save optimized model");
            return core::ErrorCode::FILE_WRITE_ERROR;
        }
        ofs.close();
        
        // 更新结果
        result.success = true;
        result.output_path = output_path;
        result.message = "Model optimized successfully";
        
        // 获取优化后文件大小
        std::ifstream file(output_path, std::ios::binary | std::ios::ate);
        if (file.is_open()) {
            result.processed_size = file.tellg();
            file.close();
        }
        
        result.compression_ratio = static_cast<double>(result.original_size) / 
                                   result.processed_size;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Optimization failed: %s", e.what());
        result.success = false;
        result.message = e.what();
        return core::ErrorCode::MODEL_CONVERSION_FAILED;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.process_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    LOG_INFO("Model optimized in %ld ms, compression ratio: %.2fx",
             result.process_time_ms, result.compression_ratio);
    
    return core::ErrorCode::SUCCESS;
}

core::ErrorCode OnnxModelProcessor::simplify(
    const std::string& output_path,
    ProcessResult& result) {
    
    if (!model_loaded_) {
        LOG_ERROR("Model not loaded");
        return core::ErrorCode::MODEL_NOT_INITIALIZED;
    }
    
    LOG_INFO("Simplifying model: %s", model_path_.c_str());
    
    // 使用优化选项进行简化
    OptimizationOptions options;
    options.eliminate_identity = true;
    options.eliminate_nop_transpose = true;
    options.eliminate_nop_pad = true;
    options.eliminate_unused_nodes = true;
    options.fuse_consecutive_transposes = true;
    options.fuse_consecutive_squeezes = true;
    options.fuse_consecutive_reshapes = true;
    
    return optimize(options, output_path, result);
}

core::ErrorCode OnnxModelProcessor::fixDynamicDimensions(
    int64_t batch_size,
    const std::string& output_path,
    ProcessResult& result) {
    
    if (!model_loaded_) {
        LOG_ERROR("Model not loaded");
        return core::ErrorCode::MODEL_NOT_INITIALIZED;
    }
    
    LOG_INFO("Fixing dynamic dimensions with batch_size=%ld", batch_size);
    auto start_time = std::chrono::high_resolution_clock::now();
    
    result.original_info = model_info_;
    result.original_size = model_info_.file_size;
    
    // 复制模型
    onnx::ModelProto fixed_model = *model_;
    auto* graph = fixed_model.mutable_graph();
    
    // 修复输入形状
    for (auto& input : *graph->mutable_input()) {
        if (input.has_type() && input.type().has_tensor_type()) {
            auto* shape = input.mutable_type()->mutable_tensor_type()->mutable_shape();
            for (auto& dim : *shape->mutable_dim()) {
                if (dim.has_dim_param() && !dim.has_dim_value()) {
                    // 将动态维度替换为固定值
                    dim.clear_dim_param();
                    dim.set_dim_value(batch_size);
                    // 对于非batch维度，通常保持原值或使用配置
                }
            }
        }
    }
    
    // 运行形状推断
    try {
        onnx::shape_inference::InferShapes(&fixed_model);
    } catch (const std::exception& e) {
        LOG_WARN("Shape inference warning: %s", e.what());
    }
    
    // 保存模型
    std::ofstream ofs(output_path, std::ios::binary);
    if (!fixed_model.SerializeToOstream(&ofs)) {
        LOG_ERROR("Failed to save fixed model");
        return core::ErrorCode::FILE_WRITE_ERROR;
    }
    ofs.close();
    
    result.success = true;
    result.output_path = output_path;
    result.message = "Dynamic dimensions fixed successfully";
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.process_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    LOG_INFO("Dynamic dimensions fixed in %ld ms", result.process_time_ms);
    
    return core::ErrorCode::SUCCESS;
}

std::set<std::string> OnnxModelProcessor::getOperatorTypes() const {
    return model_info_.op_types;
}

bool OnnxModelProcessor::hasOperator(const std::string& op_type) const {
    return model_info_.op_types.find(op_type) != model_info_.op_types.end();
}

std::vector<TensorShape> OnnxModelProcessor::getInputShapes() const {
    return model_info_.inputs;
}

std::vector<TensorShape> OnnxModelProcessor::getOutputShapes() const {
    return model_info_.outputs;
}

core::ErrorCode OnnxModelProcessor::save(const std::string& output_path) {
    if (!model_loaded_) {
        LOG_ERROR("Model not loaded");
        return core::ErrorCode::MODEL_NOT_INITIALIZED;
    }
    
    std::ofstream ofs(output_path, std::ios::binary);
    if (!model_->SerializeToOstream(&ofs)) {
        LOG_ERROR("Failed to save model to: %s", output_path.c_str());
        return core::ErrorCode::FILE_WRITE_ERROR;
    }
    ofs.close();
    
    LOG_INFO("Model saved to: %s", output_path.c_str());
    return core::ErrorCode::SUCCESS;
}

const onnx::GraphProto* OnnxModelProcessor::getGraph() const {
    if (!model_loaded_) return nullptr;
    return &model_->graph();
}

void OnnxModelProcessor::printModelSummary() const {
    model_info_.print();
}

bool OnnxModelProcessor::checkCompatibility(int target_opset) const {
    if (!model_loaded_) return false;
    
    for (const auto& opset : model_->opset_import()) {
        if (opset.domain().empty() || opset.domain() == "ai.onnx") {
            if (static_cast<int>(opset.version()) > target_opset) {
                LOG_WARN("Model uses opset %d, target is %d", 
                         opset.version(), target_opset);
                return false;
            }
        }
    }
    return true;
}

core::ErrorCode OnnxModelProcessor::convertOpset(
    int target_opset,
    const std::string& output_path,
    ProcessResult& result) {
    
    if (!model_loaded_) {
        LOG_ERROR("Model not loaded");
        return core::ErrorCode::MODEL_NOT_INITIALIZED;
    }
    
    LOG_INFO("Converting opset to version %d", target_opset);
    
    // ONNX opset转换需要使用onnx库的版本转换功能
    // 这里提供基本框架
    
    result.success = false;
    result.message = "Opset conversion not fully implemented";
    
    return core::ErrorCode::SUCCESS;
}

core::ErrorCode OnnxModelProcessor::extractSubModel(
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& output_path,
    ProcessResult& result) {
    
    if (!model_loaded_) {
        LOG_ERROR("Model not loaded");
        return core::ErrorCode::MODEL_NOT_INITIALIZED;
    }
    
    LOG_INFO("Extracting sub-model with %zu inputs and %zu outputs",
             input_names.size(), output_names.size());
    
    // 子模型提取需要分析图的依赖关系
    // 这里提供基本框架
    
    result.success = false;
    result.message = "Sub-model extraction not fully implemented";
    
    return core::ErrorCode::SUCCESS;
}

core::ErrorCode OnnxModelProcessor::mergeModels(
    const std::string& other_model_path,
    const std::string& output_path,
    ProcessResult& result) {
    
    if (!model_loaded_) {
        LOG_ERROR("Model not loaded");
        return core::ErrorCode::MODEL_NOT_INITIALIZED;
    }
    
    LOG_INFO("Merging with model: %s", other_model_path.c_str());
    
    // 模型合并需要处理图的连接
    // 这里提供基本框架
    
    result.success = false;
    result.message = "Model merging not fully implemented";
    
    return core::ErrorCode::SUCCESS;
}

// ============================================================================
// ModelProcessorFactory 实现
// ============================================================================

std::unique_ptr<IModelProcessor> ModelProcessorFactory::create() {
    return std::make_unique<OnnxModelProcessor>();
}

std::unique_ptr<IModelProcessor> ModelProcessorFactory::create(
    const std::string& model_path) {
    
    auto processor = std::make_unique<OnnxModelProcessor>();
    if (processor->loadModel(model_path) != core::ErrorCode::SUCCESS) {
        return nullptr;
    }
    return processor;
}

// ============================================================================
// model_utils 实现
// ============================================================================

namespace model_utils {

bool isValidOnnxModel(const std::string& model_path) {
    std::ifstream ifs(model_path, std::ios::binary);
    if (!ifs.is_open()) {
        return false;
    }
    
    onnx::ModelProto model;
    bool success = model.ParseFromIstream(&ifs);
    ifs.close();
    
    return success;
}

std::string getIrVersionString(int64_t ir_version) {
    return std::to_string(ir_version);
}

std::string getDataTypeString(int32_t data_type) {
    switch (data_type) {
        case onnx::TensorProto::FLOAT: return "float32";
        case onnx::TensorProto::DOUBLE: return "float64";
        case onnx::TensorProto::INT32: return "int32";
        case onnx::TensorProto::INT64: return "int64";
        case onnx::TensorProto::UINT8: return "uint8";
        case onnx::TensorProto::INT8: return "int8";
        case onnx::TensorProto::UINT16: return "uint16";
        case onnx::TensorProto::INT16: return "int16";
        case onnx::TensorProto::BOOL: return "bool";
        case onnx::TensorProto::FLOAT16: return "float16";
        case onnx::TensorProto::BFLOAT16: return "bfloat16";
        default: return "unknown(" + std::to_string(data_type) + ")";
    }
}

int getDataTypeSize(int32_t data_type) {
    switch (data_type) {
        case onnx::TensorProto::FLOAT: return 4;
        case onnx::TensorProto::DOUBLE: return 8;
        case onnx::TensorProto::INT32: return 4;
        case onnx::TensorProto::INT64: return 8;
        case onnx::TensorProto::UINT8: return 1;
        case onnx::TensorProto::INT8: return 1;
        case onnx::TensorProto::UINT16: return 2;
        case onnx::TensorProto::INT16: return 2;
        case onnx::TensorProto::BOOL: return 1;
        case onnx::TensorProto::FLOAT16: return 2;
        case onnx::TensorProto::BFLOAT16: return 2;
        default: return 4;
    }
}

std::string formatFileSize(int64_t size) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double size_d = static_cast<double>(size);
    
    while (size_d >= 1024.0 && unit_index < 4) {
        size_d /= 1024.0;
        unit_index++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size_d << " " << units[unit_index];
    return oss.str();
}

std::string formatFlops(int64_t flops) {
    if (flops == 0) return "N/A";
    
    const char* units[] = {"", "K", "M", "G", "T", "P"};
    int unit_index = 0;
    double flops_d = static_cast<double>(flops);
    
    while (flops_d >= 1000.0 && unit_index < 5) {
        flops_d /= 1000.0;
        unit_index++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << flops_d << " " 
        << units[unit_index] << "FLOPs";
    return oss.str();
}

std::string formatParams(int64_t params) {
    const char* units[] = {"", "K", "M", "B", "T"};
    int unit_index = 0;
    double params_d = static_cast<double>(params);
    
    while (params_d >= 1000.0 && unit_index < 4) {
        params_d /= 1000.0;
        unit_index++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << params_d << " " 
        << units[unit_index] << " params";
    return oss.str();
}

} // namespace model_utils

} // namespace model_processor
} // namespace modules
} // namespace smart_video_analysis
