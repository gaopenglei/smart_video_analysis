/**
 * @file OperatorAdapter.hpp
 * @brief 算子适配模块头文件
 * 
 * 提供ONNX模型算子支持性检查和适配功能，
 * 针对RK3588 NPU等目标硬件进行算子兼容性处理。
 */

#ifndef MODULES_OPERATOR_ADAPTER_OPERATOR_ADAPTER_HPP
#define MODULES_OPERATOR_ADAPTER_OPERATOR_ADAPTER_HPP

#include <string>
#include <vector>
#include <set>
#include <map>
#include <memory>
#include "core/ErrorHandling.hpp"

namespace smart_video_analysis {
namespace modules {
namespace operator_adapter {

/**
 * @brief 算子支持状态枚举
 */
enum class OperatorSupportStatus {
    SUPPORTED,          ///< 完全支持
    PARTIALLY_SUPPORTED,///< 部分支持
    NOT_SUPPORTED,      ///< 不支持
    UNKNOWN             ///< 未知状态
};

/**
 * @brief 算子信息结构体
 */
struct OperatorInfo {
    std::string name;               ///< 算子名称
    std::string op_type;            ///< 算子类型
    std::vector<std::string> inputs;  ///< 输入名称
    std::vector<std::string> outputs; ///< 输出名称
    std::map<std::string, std::string> attributes; ///< 属性
    OperatorSupportStatus support_status; ///< 支持状态
    std::string support_message;    ///< 支持状态消息
    
    OperatorInfo() : support_status(OperatorSupportStatus::UNKNOWN) {}
};

/**
 * @brief 算子适配结果结构体
 */
struct AdapterResult {
    bool success;                   ///< 是否成功
    std::string message;            ///< 结果消息
    std::vector<OperatorInfo> unsupported_ops; ///< 不支持的算子列表
    std::vector<OperatorInfo> adapted_ops;     ///< 已适配的算子列表
    std::map<std::string, std::string> replacements; ///< 算子替换映射
    
    AdapterResult() : success(true) {}
};

/**
 * @brief NPU硬件类型枚举
 */
enum class NPUType {
    RK3588,         ///< 瑞芯微RK3588
    RK3568,         ///< 瑞芯微RK3568
    RK3566,         ///< 瑞芯微RK3566
    GENERIC_NPU,    ///< 通用NPU
    UNKNOWN         ///< 未知类型
};

/**
 * @brief 算子适配器基类
 */
class IOperatorAdapter {
public:
    virtual ~IOperatorAdapter() = default;
    
    /**
     * @brief 检查模型算子支持性
     * @param model_path ONNX模型路径
     * @param result 适配结果
     * @return 成功返回ErrorCode::SUCCESS
     */
    virtual core::ErrorCode checkOperatorSupport(
        const std::string& model_path,
        AdapterResult& result) = 0;
    
    /**
     * @brief 适配不支持的算子
     * @param model_path 输入模型路径
     * @param output_path 输出模型路径
     * @param result 适配结果
     * @return 成功返回ErrorCode::SUCCESS
     */
    virtual core::ErrorCode adaptOperators(
        const std::string& model_path,
        const std::string& output_path,
        AdapterResult& result) = 0;
    
    /**
     * @brief 获取支持的算子列表
     */
    virtual std::set<std::string> getSupportedOperators() const = 0;
    
    /**
     * @brief 获取不支持的算子列表
     */
    virtual std::set<std::string> getUnsupportedOperators() const = 0;
};

/**
 * @brief RK3588 NPU算子适配器
 * 
 * 针对瑞芯微RK3588 NPU的算子适配实现。
 */
class RK3588OperatorAdapter : public IOperatorAdapter {
public:
    RK3588OperatorAdapter();
    ~RK3588OperatorAdapter() override = default;
    
    core::ErrorCode checkOperatorSupport(
        const std::string& model_path,
        AdapterResult& result) override;
    
    core::ErrorCode adaptOperators(
        const std::string& model_path,
        const std::string& output_path,
        AdapterResult& result) override;
    
    std::set<std::string> getSupportedOperators() const override;
    std::set<std::string> getUnsupportedOperators() const override;
    
    /**
     * @brief 添加自定义支持的算子
     */
    void addSupportedOperator(const std::string& op_type);
    
    /**
     * @brief 添加算子替换规则
     */
    void addReplacementRule(const std::string& from_op, const std::string& to_op);
    
    /**
     * @brief 设置是否启用算子融合
     */
    void setEnableFusion(bool enable);
    
    /**
     * @brief 设置是否启用算子替换
     */
    void setEnableReplacement(bool enable);

private:
    /**
     * @brief 初始化RK3588支持的算子列表
     */
    void initializeSupportedOperators();
    
    /**
     * @brief 初始化算子替换规则
     */
    void initializeReplacementRules();
    
    /**
     * @brief 解析ONNX模型获取算子信息
     */
    bool parseModelOperators(const std::string& model_path,
                             std::vector<OperatorInfo>& operators);
    
    /**
     * @brief 检查单个算子是否支持
     */
    OperatorSupportStatus checkOperator(const OperatorInfo& op);
    
    /**
     * @brief 尝试替换算子
     */
    bool tryReplaceOperator(OperatorInfo& op);
    
    /**
     * @brief 尝试融合算子
     */
    bool tryFuseOperators(std::vector<OperatorInfo>& operators, size_t index);

    // 成员变量
    std::set<std::string> supported_operators_;
    std::set<std::string> unsupported_operators_;
    std::map<std::string, std::string> replacement_rules_;
    bool enable_fusion_ = true;
    bool enable_replacement_ = true;
};

/**
 * @brief 通用算子适配器
 */
class GenericOperatorAdapter : public IOperatorAdapter {
public:
    explicit GenericOperatorAdapter(NPUType npu_type = NPUType::GENERIC_NPU);
    ~GenericOperatorAdapter() override = default;
    
    core::ErrorCode checkOperatorSupport(
        const std::string& model_path,
        AdapterResult& result) override;
    
    core::ErrorCode adaptOperators(
        const std::string& model_path,
        const std::string& output_path,
        AdapterResult& result) override;
    
    std::set<std::string> getSupportedOperators() const override;
    std::set<std::string> getUnsupportedOperators() const override;

private:
    NPUType npu_type_;
    std::set<std::string> supported_operators_;
};

/**
 * @brief 算子适配器工厂类
 */
class OperatorAdapterFactory {
public:
    /**
     * @brief 创建算子适配器
     * @param npu_type NPU类型
     * @return 适配器实例
     */
    static std::unique_ptr<IOperatorAdapter> create(NPUType npu_type);
    
    /**
     * @brief 根据名称创建算子适配器
     * @param npu_name NPU名称 (如 "rk3588", "rk3568")
     * @return 适配器实例
     */
    static std::unique_ptr<IOperatorAdapter> create(const std::string& npu_name);
};

} // namespace operator_adapter
} // namespace modules
} // namespace smart_video_analysis

#endif // MODULES_OPERATOR_ADAPTER_OPERATOR_ADAPTER_HPP
