/**
 * @file ModelProcessor.hpp
 * @brief ONNX模型处理模块头文件
 * 
 * 提供ONNX模型的加载、验证、优化和转换功能。
 * 支持模型信息查询、算子检查、模型简化等操作。
 */

#ifndef MODULES_MODEL_PROCESSOR_MODEL_PROCESSOR_HPP
#define MODULES_MODEL_PROCESSOR_MODEL_PROCESSOR_HPP

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>
#include "core/ErrorHandling.hpp"

// ONNX ProtoBuf 前向声明
namespace onnx {
    class ModelProto;
    class GraphProto;
    class NodeProto;
    class TensorProto;
}

namespace smart_video_analysis {
namespace modules {
namespace model_processor {

/**
 * @brief 张量形状信息结构体
 */
struct TensorShape {
    std::string name;                   ///< 张量名称
    std::vector<int64_t> dims;          ///< 维度信息
    std::string data_type;              ///< 数据类型
    std::vector<std::string> dim_params;///< 动态维度参数名
    
    /**
     * @brief 是否为动态形状
     */
    bool isDynamic() const;
    
    /**
     * @brief 获取元素数量（静态形状）
     */
    int64_t elementCount() const;
    
    /**
     * @brief 获取字节大小
     */
    int64_t byteSize() const;
    
    /**
     * @brief 转换为字符串
     */
    std::string toString() const;
};

/**
 * @brief 算子节点信息结构体
 */
struct OperatorNode {
    std::string name;                   ///< 节点名称
    std::string op_type;                ///< 算子类型
    std::vector<std::string> inputs;    ///< 输入名称
    std::vector<std::string> outputs;   ///< 输出名称
    std::map<std::string, std::string> attributes; ///< 属性
    int64_t flops;                      ///< 计算量估算
    
    std::string toString() const;
};

/**
 * @brief 模型信息结构体
 */
struct ModelInfo {
    std::string model_path;             ///< 模型路径
    std::string ir_version;             ///< IR版本
    std::string producer_name;          ///< 生产者名称
    std::string producer_version;       ///< 生产者版本
    std::string model_version;          ///< 模型版本
    std::string doc_string;             ///< 文档字符串
    
    std::vector<TensorShape> inputs;    ///< 输入张量信息
    std::vector<TensorShape> outputs;   ///< 输出张量信息
    
    std::vector<OperatorNode> nodes;    ///< 算子节点列表
    std::set<std::string> op_types;     ///< 算子类型集合
    
    int64_t total_params;               ///< 总参数量
    int64_t total_flops;                ///< 总计算量
    int64_t total_nodes;                ///< 总节点数
    int64_t file_size;                  ///< 文件大小
    
    std::map<std::string, int64_t> op_counts; ///< 各算子数量统计
    
    /**
     * @brief 打印模型信息
     */
    void print() const;
    
    /**
     * @brief 获取模型摘要字符串
     */
    std::string getSummary() const;
};

/**
 * @brief 模型优化选项结构体
 */
struct OptimizationOptions {
    bool eliminate_identity = true;     ///< 消除Identity节点
    bool eliminate_nop_transpose = true;///< 消除无用的Transpose
    bool eliminate_nop_pad = true;      ///< 消除无用的Pad
    bool eliminate_unused_nodes = true; ///< 消除未使用的节点
    bool fuse_consecutive_transposes = true; ///< 融合连续的Transpose
    bool fuse_consecutive_squeezes = true;   ///< 融合连续的Squeeze
    bool fuse_consecutive_reshapes = true;   ///< 融合连续的Reshape
    bool fuse_add_bias_into_conv = true;     ///< 将Bias融合到Conv中
    bool fuse_bn_into_conv = true;           ///< 将BN融合到Conv中
    bool skip_shape_inference = false;       ///< 跳过形状推断
    
    int optimization_level = 1;         ///< 优化级别 (0-2)
    bool skip_onnx_checker = false;     ///< 跳过ONNX检查器
    bool verbose = false;               ///< 详细输出
};

/**
 * @brief 模型转换选项结构体
 */
struct ConversionOptions {
    int opset_version = 12;             ///< ONNX opset版本
    std::vector<std::string> input_names;  ///< 输入名称
    std::vector<std::string> output_names; ///< 输出名称
    
    // 动态维度处理
    bool fix_dynamic_batch = true;      ///< 固定动态Batch
    int64_t fixed_batch_size = 1;       ///< 固定的Batch大小
    bool fix_dynamic_dims = true;       ///< 固定动态维度
    
    // 优化选项
    bool simplify = true;               ///< 是否简化模型
    bool optimize = true;               ///< 是否优化模型
    
    // 量化选项
    bool quantize = false;              ///< 是否量化
    std::string quantize_type = "fp16"; ///< 量化类型: fp16, int8
};

/**
 * @brief 模型处理结果结构体
 */
struct ProcessResult {
    bool success;                       ///< 是否成功
    std::string message;                ///< 结果消息
    std::string output_path;            ///< 输出路径
    
    ModelInfo original_info;            ///< 原始模型信息
    ModelInfo processed_info;           ///< 处理后模型信息
    
    int64_t original_size;              ///< 原始大小
    int64_t processed_size;             ///< 处理后大小
    double compression_ratio;           ///< 压缩比
    
    double process_time_ms;             ///< 处理耗时
    
    std::vector<std::string> warnings;  ///< 警告信息
    std::vector<std::string> errors;    ///< 错误信息
    
    ProcessResult() : success(false), original_size(0), processed_size(0),
                      compression_ratio(1.0), process_time_ms(0) {}
};

/**
 * @brief ONNX模型处理器抽象基类
 */
class IModelProcessor {
public:
    virtual ~IModelProcessor() = default;
    
    /**
     * @brief 加载ONNX模型
     * @param model_path 模型文件路径
     * @return 成功返回SUCCESS，失败返回错误码
     */
    virtual core::ErrorCode loadModel(const std::string& model_path) = 0;
    
    /**
     * @brief 获取模型信息
     * @param info 输出模型信息
     * @return 成功返回SUCCESS
     */
    virtual core::ErrorCode getModelInfo(ModelInfo& info) const = 0;
    
    /**
     * @brief 验证模型有效性
     * @return 有效返回true
     */
    virtual bool validate() const = 0;
    
    /**
     * @brief 优化模型
     * @param options 优化选项
     * @param output_path 输出路径
     * @param result 处理结果
     * @return 成功返回SUCCESS
     */
    virtual core::ErrorCode optimize(
        const OptimizationOptions& options,
        const std::string& output_path,
        ProcessResult& result) = 0;
    
    /**
     * @brief 简化模型
     * @param output_path 输出路径
     * @param result 处理结果
     * @return 成功返回SUCCESS
     */
    virtual core::ErrorCode simplify(
        const std::string& output_path,
        ProcessResult& result) = 0;
    
    /**
     * @brief 固定动态维度
     * @param batch_size 固定的batch大小
     * @param output_path 输出路径
     * @param result 处理结果
     * @return 成功返回SUCCESS
     */
    virtual core::ErrorCode fixDynamicDimensions(
        int64_t batch_size,
        const std::string& output_path,
        ProcessResult& result) = 0;
    
    /**
     * @brief 获取所有算子类型
     * @return 算子类型集合
     */
    virtual std::set<std::string> getOperatorTypes() const = 0;
    
    /**
     * @brief 检查是否包含指定算子
     * @param op_type 算子类型
     * @return 包含返回true
     */
    virtual bool hasOperator(const std::string& op_type) const = 0;
    
    /**
     * @brief 获取输入形状
     * @return 输入张量形状列表
     */
    virtual std::vector<TensorShape> getInputShapes() const = 0;
    
    /**
     * @brief 获取输出形状
     * @return 输出张量形状列表
     */
    virtual std::vector<TensorShape> getOutputShapes() const = 0;
    
    /**
     * @brief 保存模型
     * @param output_path 输出路径
     * @return 成功返回SUCCESS
     */
    virtual core::ErrorCode save(const std::string& output_path) = 0;
};

/**
 * @brief ONNX模型处理器实现类
 */
class OnnxModelProcessor : public IModelProcessor {
public:
    OnnxModelProcessor();
    ~OnnxModelProcessor() override;
    
    core::ErrorCode loadModel(const std::string& model_path) override;
    core::ErrorCode getModelInfo(ModelInfo& info) const override;
    bool validate() const override;
    
    core::ErrorCode optimize(
        const OptimizationOptions& options,
        const std::string& output_path,
        ProcessResult& result) override;
    
    core::ErrorCode simplify(
        const std::string& output_path,
        ProcessResult& result) override;
    
    core::ErrorCode fixDynamicDimensions(
        int64_t batch_size,
        const std::string& output_path,
        ProcessResult& result) override;
    
    std::set<std::string> getOperatorTypes() const override;
    bool hasOperator(const std::string& op_type) const override;
    std::vector<TensorShape> getInputShapes() const override;
    std::vector<TensorShape> getOutputShapes() const override;
    core::ErrorCode save(const std::string& output_path) override;
    
    /**
     * @brief 获取模型图
     */
    const onnx::GraphProto* getGraph() const;
    
    /**
     * @brief 打印模型摘要
     */
    void printModelSummary() const;
    
    /**
     * @brief 检查模型兼容性
     * @param target_opset 目标opset版本
     * @return 兼容返回true
     */
    bool checkCompatibility(int target_opset) const;
    
    /**
     * @brief 转换opset版本
     * @param target_opset 目标opset版本
     * @param output_path 输出路径
     * @param result 处理结果
     * @return 成功返回SUCCESS
     */
    core::ErrorCode convertOpset(
        int target_opset,
        const std::string& output_path,
        ProcessResult& result);
    
    /**
     * @brief 提取子模型
     * @param input_names 输入名称
     * @param output_names 输出名称
     * @param output_path 输出路径
     * @param result 处理结果
     * @return 成功返回SUCCESS
     */
    core::ErrorCode extractSubModel(
        const std::vector<std::string>& input_names,
        const std::vector<std::string>& output_names,
        const std::string& output_path,
        ProcessResult& result);
    
    /**
     * @brief 合并模型
     * @param other_model_path 其他模型路径
     * @param output_path 输出路径
     * @param result 处理结果
     * @return 成功返回SUCCESS
     */
    core::ErrorCode mergeModels(
        const std::string& other_model_path,
        const std::string& output_path,
        ProcessResult& result);

private:
    /**
     * @brief 解析模型信息
     */
    void parseModelInfo();
    
    /**
     * @brief 解析图节点
     */
    void parseNodes();
    
    /**
     * @brief 解析输入输出
     */
    void parseInputsOutputs();
    
    /**
     * @brief 计算模型统计信息
     */
    void calculateStatistics();
    
    /**
     * @brief 估算节点FLOPS
     */
    int64_t estimateNodeFlops(const OperatorNode& node) const;
    
    /**
     * @brief 运行ONNX形状推断
     */
    bool runShapeInference();
    
    /**
     * @brief 应用优化pass
     */
    bool applyOptimizationPass(const std::string& pass_name);

    // 成员变量
    std::unique_ptr<onnx::ModelProto> model_;
    ModelInfo model_info_;
    std::string model_path_;
    bool model_loaded_ = false;
};

/**
 * @brief 模型处理器工厂类
 */
class ModelProcessorFactory {
public:
    /**
     * @brief 创建模型处理器
     * @return 处理器实例
     */
    static std::unique_ptr<IModelProcessor> create();
    
    /**
     * @brief 创建并加载模型
     * @param model_path 模型路径
     * @return 处理器实例（加载失败返回nullptr）
     */
    static std::unique_ptr<IModelProcessor> create(const std::string& model_path);
};

/**
 * @brief 模型工具函数
 */
namespace model_utils {

/**
 * @brief 检查文件是否为有效的ONNX模型
 */
bool isValidOnnxModel(const std::string& model_path);

/**
 * @brief 获取ONNX IR版本字符串
 */
std::string getIrVersionString(int64_t ir_version);

/**
 * @brief 获取数据类型字符串
 */
std::string getDataTypeString(int32_t data_type);

/**
 * @brief 获取数据类型大小（字节）
 */
int getDataTypeSize(int32_t data_type);

/**
 * @brief 格式化文件大小
 */
std::string formatFileSize(int64_t size);

/**
 * @brief 格式化FLOPS
 */
std::string formatFlops(int64_t flops);

/**
 * @brief 格式化参数量
 */
std::string formatParams(int64_t params);

} // namespace model_utils

} // namespace model_processor
} // namespace modules
} // namespace smart_video_analysis

#endif // MODULES_MODEL_PROCESSOR_MODEL_PROCESSOR_HPP
