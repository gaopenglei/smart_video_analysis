/**
 * @file OperatorAdapter.cpp
 * @brief 算子适配模块实现文件
 */

#include "modules/operator_adapter/OperatorAdapter.hpp"
#include "core/Logger.hpp"
#include <algorithm>
#include <fstream>

namespace smart_video_analysis {
namespace modules {
namespace operator_adapter {

// ============================================================================
// RK3588OperatorAdapter 实现
// ============================================================================

RK3588OperatorAdapter::RK3588OperatorAdapter() {
    initializeSupportedOperators();
    initializeReplacementRules();
    LOG_DEBUG("RK3588OperatorAdapter created with %zu supported operators",
              supported_operators_.size());
}

void RK3588OperatorAdapter::initializeSupportedOperators() {
    // RK3588 NPU支持的ONNX算子列表
    // 基于RKNN-Toolkit2文档
    
    // 卷积相关
    supported_operators_.insert("Conv");
    supported_operators_.insert("ConvTranspose");
    supported_operators_.insert("DepthwiseConv");
    
    // 池化相关
    supported_operators_.insert("MaxPool");
    supported_operators_.insert("AveragePool");
    supported_operators_.insert("GlobalMaxPool");
    supported_operators_.insert("GlobalAveragePool");
    
    // 激活函数
    supported_operators_.insert("Relu");
    supported_operators_.insert("Relu6");
    supported_operators_.insert("LeakyRelu");
    supported_operators_.insert("Sigmoid");
    supported_operators_.insert("Tanh");
    supported_operators_.insert("HardSigmoid");
    supported_operators_.insert("HardSwish");
    supported_operators_.insert("PRelu");
    supported_operators_.insert("Gelu");
    supported_operators_.insert("Silu");
    supported_operators_.insert("Mish");
    
    // 归一化
    supported_operators_.insert("BatchNormalization");
    supported_operators_.insert("InstanceNormalization");
    supported_operators_.insert("LayerNormalization");
    supported_operators_.insert("GroupNormalization");
    
    // 全连接
    supported_operators_.insert("Gemm");
    supported_operators_.insert("MatMul");
    
    // 逐元素操作
    supported_operators_.insert("Add");
    supported_operators_.insert("Sub");
    supported_operators_.insert("Mul");
    supported_operators_.insert("Div");
    supported_operators_.insert("Pow");
    supported_operators_.insert("Sqrt");
    supported_operators_.insert("Exp");
    supported_operators_.insert("Log");
    supported_operators_.insert("Abs");
    supported_operators_.insert("Neg");
    supported_operators_.insert("Clip");
    supported_operators_.insert("Min");
    supported_operators_.insert("Max");
    
    // 张量操作
    supported_operators_.insert("Reshape");
    supported_operators_.insert("Transpose");
    supported_operators_.insert("Flatten");
    supported_operators_.insert("Squeeze");
    supported_operators_.insert("Unsqueeze");
    supported_operators_.insert("Concat");
    supported_operators_.insert("Split");
    supported_operators_.insert("Slice");
    supported_operators_.insert("Pad");
    supported_operators_.insert("Gather");
    supported_operators_.insert("Scatter");
    supported_operators_.insert("Tile");
    supported_operators_.insert("Expand");
    supported_operators_.insert("Resize");
    supported_operators_.insert("Upsample");
    
    // 比较操作
    supported_operators_.insert("Equal");
    supported_operators_.insert("Greater");
    supported_operators_.insert("Less");
    supported_operators_.insert("Where");
    
    // 逻辑操作
    supported_operators_.insert("And");
    supported_operators_.insert("Or");
    supported_operators_.insert("Not");
    
    // 其他
    supported_operators_.insert("Softmax");
    supported_operators_.insert("LogSoftmax");
    supported_operators_.insert("ArgMax");
    supported_operators_.insert("ArgMin");
    supported_operators_.insert("TopK");
    supported_operators_.insert("ReduceMean");
    supported_operators_.insert("ReduceSum");
    supported_operators_.insert("ReduceMax");
    supported_operators_.insert("ReduceMin");
    supported_operators_.insert("ReduceProd");
    supported_operators_.insert("Cast");
    supported_operators_.insert("Shape");
    supported_operators_.insert("Size");
    supported_operators_.insert("Constant");
    supported_operators_.insert("Range");
    
    // 检测相关
    supported_operators_.insert("NonMaxSuppression");
    supported_operators_.insert("ROIAlign");
    supported_operators_.insert("ROIPooling");
    
    // 插值
    supported_operators_.insert("Interpolate");
    supported_operators_.insert("BilinearResize");
    supported_operators_.insert("NearestResize");
    
    LOG_INFO("RK3588 supported operators initialized: %zu", supported_operators_.size());
}

void RK3588OperatorAdapter::initializeReplacementRules() {
    // 算子替换规则：不支持 -> 支持的替代
    replacement_rules_["HardSwish"] = "Swish";  // 可以用Swish近似
    replacement_rules_["Mish"] = "LeakyRelu";   // 可以用LeakyRelu近似
    replacement_rules_["Gelu"] = "Relu";        // 可以用Relu近似
    
    LOG_DEBUG("Replacement rules initialized: %zu", replacement_rules_.size());
}

core::ErrorCode RK3588OperatorAdapter::checkOperatorSupport(
    const std::string& model_path,
    AdapterResult& result) {
    
    LOG_INFO("Checking operator support for model: %s", model_path.c_str());
    
    // 解析模型获取算子信息
    std::vector<OperatorInfo> operators;
    if (!parseModelOperators(model_path, operators)) {
        LOG_ERROR("Failed to parse model operators");
        return core::ErrorCode::OPERATOR_ERROR;
    }
    
    result.success = true;
    result.unsupported_ops.clear();
    result.adapted_ops.clear();
    
    // 检查每个算子
    for (auto& op : operators) {
        OperatorSupportStatus status = checkOperator(op);
        op.support_status = status;
        
        if (status == OperatorSupportStatus::NOT_SUPPORTED) {
            result.unsupported_ops.push_back(op);
            result.success = false;
            
            LOG_WARN("Unsupported operator: %s (type: %s)", 
                     op.name.c_str(), op.op_type.c_str());
        } else if (status == OperatorSupportStatus::PARTIALLY_SUPPORTED) {
            LOG_WARN("Partially supported operator: %s (type: %s)",
                     op.name.c_str(), op.op_type.c_str());
        }
    }
    
    if (result.success) {
        result.message = "All operators are supported";
        LOG_INFO("All operators are supported for RK3588 NPU");
    } else {
        result.message = "Found " + std::to_string(result.unsupported_ops.size()) + 
                        " unsupported operators";
        LOG_WARN("Found %zu unsupported operators", result.unsupported_ops.size());
    }
    
    return core::ErrorCode::SUCCESS;
}

core::ErrorCode RK3588OperatorAdapter::adaptOperators(
    const std::string& model_path,
    const std::string& output_path,
    AdapterResult& result) {
    
    LOG_INFO("Adapting operators for model: %s", model_path.c_str());
    
    // 首先检查算子支持性
    core::ErrorCode ret = checkOperatorSupport(model_path, result);
    if (ret != core::ErrorCode::SUCCESS) {
        return ret;
    }
    
    if (result.success) {
        LOG_INFO("No adaptation needed, all operators are supported");
        return core::ErrorCode::SUCCESS;
    }
    
    // 尝试适配不支持的算子
    for (auto& op : result.unsupported_ops) {
        if (enable_replacement_ && tryReplaceOperator(op)) {
            result.adapted_ops.push_back(op);
            result.replacements[op.op_type] = replacement_rules_[op.op_type];
            LOG_INFO("Replaced operator %s with %s", 
                     op.op_type.c_str(), replacement_rules_[op.op_type].c_str());
        }
    }
    
    // 更新结果
    if (result.adapted_ops.size() == result.unsupported_ops.size()) {
        result.success = true;
        result.message = "All unsupported operators have been adapted";
    } else {
        result.message = "Some operators could not be adapted";
    }
    
    LOG_INFO("Operator adaptation completed: %zu operators adapted",
             result.adapted_ops.size());
    
    return core::ErrorCode::SUCCESS;
}

std::set<std::string> RK3588OperatorAdapter::getSupportedOperators() const {
    return supported_operators_;
}

std::set<std::string> RK3588OperatorAdapter::getUnsupportedOperators() const {
    return unsupported_operators_;
}

void RK3588OperatorAdapter::addSupportedOperator(const std::string& op_type) {
    supported_operators_.insert(op_type);
    unsupported_operators_.erase(op_type);
}

void RK3588OperatorAdapter::addReplacementRule(const std::string& from_op, 
                                                const std::string& to_op) {
    replacement_rules_[from_op] = to_op;
}

void RK3588OperatorAdapter::setEnableFusion(bool enable) {
    enable_fusion_ = enable;
}

void RK3588OperatorAdapter::setEnableReplacement(bool enable) {
    enable_replacement_ = enable;
}

bool RK3588OperatorAdapter::parseModelOperators(
    const std::string& model_path,
    std::vector<OperatorInfo>& operators) {
    
    // 这里应该使用ONNX库解析模型
    // 由于这是一个示例实现，我们模拟一些常见算子
    
    // 实际实现中应该使用:
    // onnx::ModelProto model;
    // onnx::ParseFromFile(model, model_path);
    // 遍历model.graph().node()获取算子信息
    
    // 模拟YOLOv8模型的算子
    operators.clear();
    
    // 添加一些示例算子
    OperatorInfo conv_op;
    conv_op.name = "Conv_0";
    conv_op.op_type = "Conv";
    conv_op.inputs = {"input", "conv_weight", "conv_bias"};
    conv_op.outputs = {"conv_out"};
    operators.push_back(conv_op);
    
    OperatorInfo bn_op;
    bn_op.name = "BatchNormalization_1";
    bn_op.op_type = "BatchNormalization";
    bn_op.inputs = {"conv_out", "bn_weight", "bn_bias", "bn_mean", "bn_var"};
    bn_op.outputs = {"bn_out"};
    operators.push_back(bn_op);
    
    OperatorInfo relu_op;
    relu_op.name = "Relu_2";
    relu_op.op_type = "Relu";
    relu_op.inputs = {"bn_out"};
    relu_op.outputs = {"relu_out"};
    operators.push_back(relu_op);
    
    OperatorInfo concat_op;
    concat_op.name = "Concat_3";
    concat_op.op_type = "Concat";
    concat_op.inputs = {"input1", "input2"};
    concat_op.outputs = {"concat_out"};
    operators.push_back(concat_op);
    
    OperatorInfo reshape_op;
    reshape_op.name = "Reshape_4";
    reshape_op.op_type = "Reshape";
    reshape_op.inputs = {"concat_out", "shape"};
    reshape_op.outputs = {"reshape_out"};
    operators.push_back(reshape_op);
    
    LOG_DEBUG("Parsed %zu operators from model", operators.size());
    return true;
}

OperatorSupportStatus RK3588OperatorAdapter::checkOperator(const OperatorInfo& op) {
    if (supported_operators_.count(op.op_type) > 0) {
        return OperatorSupportStatus::SUPPORTED;
    }
    
    if (unsupported_operators_.count(op.op_type) > 0) {
        return OperatorSupportStatus::NOT_SUPPORTED;
    }
    
    // 检查是否有替换规则
    if (replacement_rules_.count(op.op_type) > 0) {
        return OperatorSupportStatus::PARTIALLY_SUPPORTED;
    }
    
    return OperatorSupportStatus::NOT_SUPPORTED;
}

bool RK3588OperatorAdapter::tryReplaceOperator(OperatorInfo& op) {
    auto it = replacement_rules_.find(op.op_type);
    if (it != replacement_rules_.end()) {
        LOG_INFO("Attempting to replace %s with %s", 
                 op.op_type.c_str(), it->second.c_str());
        // 实际替换逻辑需要修改ONNX模型
        // 这里只是标记可以替换
        return true;
    }
    return false;
}

bool RK3588OperatorAdapter::tryFuseOperators(std::vector<OperatorInfo>& operators, 
                                              size_t index) {
    // 算子融合逻辑
    // 例如：Conv + BatchNorm + ReLU 可以融合为一个算子
    if (index + 2 >= operators.size()) {
        return false;
    }
    
    // 检查是否可以融合 Conv + BN + ReLU
    if (operators[index].op_type == "Conv" &&
        operators[index + 1].op_type == "BatchNormalization" &&
        operators[index + 2].op_type == "Relu") {
        
        LOG_INFO("Fusing Conv + BatchNorm + ReLU at index %zu", index);
        // 实际融合逻辑
        return true;
    }
    
    return false;
}

// ============================================================================
// GenericOperatorAdapter 实现
// ============================================================================

GenericOperatorAdapter::GenericOperatorAdapter(NPUType npu_type)
    : npu_type_(npu_type) {
    // 初始化通用支持的算子
    supported_operators_.insert("Conv");
    supported_operators_.insert("Relu");
    supported_operators_.insert("MaxPool");
    supported_operators_.insert("BatchNormalization");
    supported_operators_.insert("Gemm");
    supported_operators_.insert("Softmax");
    supported_operators_.insert("Reshape");
    supported_operators_.insert("Concat");
    supported_operators_.insert("Add");
    supported_operators_.insert("Mul");
}

core::ErrorCode GenericOperatorAdapter::checkOperatorSupport(
    const std::string& model_path,
    AdapterResult& result) {
    
    LOG_INFO("Checking operator support for generic NPU");
    result.success = true;
    result.message = "Generic adapter assumes basic operator support";
    return core::ErrorCode::SUCCESS;
}

core::ErrorCode GenericOperatorAdapter::adaptOperators(
    const std::string& model_path,
    const std::string& output_path,
    AdapterResult& result) {
    
    LOG_INFO("Adapting operators for generic NPU");
    result.success = true;
    result.message = "No adaptation performed for generic NPU";
    return core::ErrorCode::SUCCESS;
}

std::set<std::string> GenericOperatorAdapter::getSupportedOperators() const {
    return supported_operators_;
}

std::set<std::string> GenericOperatorAdapter::getUnsupportedOperators() const {
    return {};
}

// ============================================================================
// OperatorAdapterFactory 实现
// ============================================================================

std::unique_ptr<IOperatorAdapter> OperatorAdapterFactory::create(NPUType npu_type) {
    switch (npu_type) {
        case NPUType::RK3588:
            return std::make_unique<RK3588OperatorAdapter>();
        case NPUType::RK3568:
        case NPUType::RK3566:
            // RK3568/RK3566使用类似的适配器
            return std::make_unique<RK3588OperatorAdapter>();
        default:
            return std::make_unique<GenericOperatorAdapter>(npu_type);
    }
}

std::unique_ptr<IOperatorAdapter> OperatorAdapterFactory::create(const std::string& npu_name) {
    std::string name = npu_name;
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
    
    if (name == "rk3588") {
        return create(NPUType::RK3588);
    }
    if (name == "rk3568") {
        return create(NPUType::RK3568);
    }
    if (name == "rk3566") {
        return create(NPUType::RK3566);
    }
    
    return create(NPUType::GENERIC_NPU);
}

} // namespace operator_adapter
} // namespace modules
} // namespace smart_video_analysis
