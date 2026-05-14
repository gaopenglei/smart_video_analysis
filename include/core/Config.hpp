/**
 * @file Config.hpp
 * @brief 配置管理模块头文件
 * 
 * 提供系统配置的加载、解析和管理功能，
 * 支持从YAML/JSON文件加载配置参数。
 */

#ifndef CORE_CONFIG_HPP
#define CORE_CONFIG_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace smart_video_analysis {
namespace core {

/**
 * @brief 配置值类型枚举
 */
enum class ConfigValueType {
    STRING,     ///< 字符串类型
    INTEGER,    ///< 整数类型
    FLOAT,      ///< 浮点数类型
    BOOLEAN,    ///< 布尔类型
    ARRAY,      ///< 数组类型
    OBJECT      ///< 对象类型
};

/**
 * @brief 配置值联合体
 */
struct ConfigValue {
    ConfigValueType type;
    std::string string_val;
    int int_val;
    float float_val;
    bool bool_val;
    std::vector<ConfigValue> array_val;
    std::map<std::string, ConfigValue> object_val;
    
    ConfigValue() : type(ConfigValueType::STRING), int_val(0), float_val(0.0f), bool_val(false) {}
};

/**
 * @brief 视频输入配置结构体
 */
struct VideoInputConfig {
    std::string source_type;        ///< 输入源类型: "camera", "file", "rtsp"
    std::string source_path;        ///< 输入源路径
    int camera_index = 0;           ///< 摄像头索引
    int frame_width = 1920;         ///< 帧宽度
    int frame_height = 1080;        ///< 帧高度
    int fps = 30;                   ///< 帧率
    bool use_hw_acceleration = false; ///< 是否使用硬件加速
    std::string hw_accel_device;    ///< 硬件加速设备
    int buffer_size = 10;           ///< 缓冲区大小
};

/**
 * @brief 预处理配置结构体
 */
struct PreprocessConfig {
    int target_width = 640;         ///< 目标宽度
    int target_height = 640;        ///< 目标高度
    bool normalize = true;          ///< 是否归一化
    float mean_r = 0.0f;            ///< R通道均值
    float mean_g = 0.0f;            ///< G通道均值
    float mean_b = 0.0f;            ///< B通道均值
    float std_r = 1.0f;             ///< R通道标准差
    float std_g = 1.0f;             ///< G通道标准差
    float std_b = 1.0f;             ///< B通道标准差
    bool swap_rb = true;            ///< 是否交换R和B通道
    std::string resize_mode = "letterbox"; ///< 缩放模式: "letterbox", "stretch", "crop"
    float scale = 1.0f / 255.0f;    ///< 缩放因子
};

/**
 * @brief 模型配置结构体
 */
struct ModelConfig {
    std::string model_path;         ///< 模型文件路径
    std::string model_type;         ///< 模型类型: "yolov8", "yolov11", "resnet", "mobilenet"
    std::string framework;          ///< 框架: "onnx", "pytorch"
    int input_batch = 1;            ///< 输入批次大小
    int input_channels = 3;         ///< 输入通道数
    int input_width = 640;          ///< 输入宽度
    int input_height = 640;         ///< 输入高度
    std::string input_name = "images";  ///< 输入节点名称
    std::string output_name = "output0"; ///< 输出节点名称
    std::vector<std::string> class_names; ///< 类别名称列表
    std::string class_names_file;   ///< 类别名称文件路径（每行一个类别名）
    int num_classes = 80;           ///< 类别数量
    float confidence_threshold = 0.25f; ///< 置信度阈值
    float nms_threshold = 0.45f;    ///< NMS阈值
};

/**
 * @brief 推理配置结构体
 */
struct InferenceConfig {
    int num_threads = 4;            ///< 推理线程数
    bool use_gpu = false;           ///< 是否使用GPU
    int gpu_device_id = 0;          ///< GPU设备ID
    std::string execution_provider = "CPU"; ///< 执行提供器: "CPU", "CUDA", "TensorRT", "OpenVINO", "NNAPI"
    std::string optimization_level = "all"; ///< 图优化级别: "all", "extended", "basic", "disable"
    bool enable_profiling = false;  ///< 是否启用性能分析
    std::string profiling_output_dir; ///< 性能分析输出目录
    int warmup_iterations = 3;      ///< 预热迭代次数
    bool enable_memory_pattern = true; ///< 是否启用内存模式优化
    bool enable_memory_allocator = true; ///< 是否启用内存分配器优化
};

/**
 * @brief 后处理配置结构体
 */
struct PostprocessConfig {
    float confidence_threshold = 0.25f; ///< 置信度阈值
    float nms_threshold = 0.45f;    ///< NMS阈值
    int max_detections = 100;       ///< 最大检测数量
    bool filter_small_objects = false; ///< 是否过滤小目标
    int min_object_size = 10;       ///< 最小目标尺寸
    std::vector<int> filter_classes; ///< 需要过滤的类别ID列表
};

/**
 * @brief 可视化配置结构体
 */
struct VisualizationConfig {
    bool enable = true;             ///< 是否启用可视化
    bool show_fps = true;           ///< 是否显示FPS
    bool show_confidence = true;    ///< 是否显示置信度
    bool show_class_name = true;    ///< 是否显示类别名称
    int box_line_width = 2;         ///< 边框线宽
    float font_scale = 0.6f;        ///< 字体缩放
    std::string output_type = "display"; ///< 输出类型: "display", "file", "rtsp"
    std::string output_path;        ///< 输出文件路径
    int output_fps = 30;            ///< 输出帧率
    std::string output_codec = "mp4v"; ///< 输出编码器
};

/**
 * @brief 算子适配配置结构体
 */
struct OperatorAdapterConfig {
    std::string target_npu = "RK3588"; ///< 目标NPU型号
    bool enable_fallback = true;    ///< 是否启用回退机制
    bool enable_fusion = true;      ///< 是否启用算子融合
    std::string unsupported_ops_file; ///< 不支持算子配置文件
};

/**
 * @brief 系统配置结构体
 */
struct SystemConfig {
    VideoInputConfig video_input;       ///< 视频输入配置
    PreprocessConfig preprocess;        ///< 预处理配置
    ModelConfig model;                  ///< 模型配置
    InferenceConfig inference;          ///< 推理配置
    PostprocessConfig postprocess;      ///< 后处理配置
    VisualizationConfig visualization;  ///< 可视化配置
    OperatorAdapterConfig operator_adapter; ///< 算子适配配置
    std::string log_level = "INFO";     ///< 日志级别
    std::string log_file = "./logs/smart_video_analysis.log"; ///< 日志文件路径
};

/**
 * @brief 配置管理器类
 * 
 * 提供配置文件的加载、解析、验证和访问功能。
 * 支持YAML格式配置文件，提供默认配置值。
 * 
 * @example
 * @code
 * auto& config = ConfigManager::getInstance();
 * config.loadFromFile("config/system.yaml");
 * SystemConfig sys_config = config.getSystemConfig();
 * @endcode
 */
class ConfigManager {
public:
    /**
     * @brief 获取单例实例
     * @return ConfigManager单例引用
     */
    static ConfigManager& getInstance();

    /**
     * @brief 从文件加载配置
     * @param config_file 配置文件路径
     * @return 成功返回true，失败返回false
     */
    bool loadFromFile(const std::string& config_file);

    /**
     * @brief 从字符串加载配置
     * @param config_str 配置字符串（YAML格式）
     * @return 成功返回true，失败返回false
     */
    bool loadFromString(const std::string& config_str);

    /**
     * @brief 保存配置到文件
     * @param config_file 配置文件路径
     * @return 成功返回true，失败返回false
     */
    bool saveToFile(const std::string& config_file);

    /**
     * @brief 获取系统配置
     * @return 系统配置结构体
     */
    SystemConfig getSystemConfig() const;

    /**
     * @brief 设置系统配置
     * @param config 系统配置结构体
     */
    void setSystemConfig(const SystemConfig& config);

    /**
     * @brief 获取视频输入配置
     * @return 视频输入配置结构体
     */
    VideoInputConfig getVideoInputConfig() const;

    /**
     * @brief 获取预处理配置
     * @return 预处理配置结构体
     */
    PreprocessConfig getPreprocessConfig() const;

    /**
     * @brief 获取模型配置
     * @return 模型配置结构体
     */
    ModelConfig getModelConfig() const;

    /**
     * @brief 获取推理配置
     * @return 推理配置结构体
     */
    InferenceConfig getInferenceConfig() const;

    /**
     * @brief 获取后处理配置
     * @return 后处理配置结构体
     */
    PostprocessConfig getPostprocessConfig() const;

    /**
     * @brief 获取可视化配置
     * @return 可视化配置结构体
     */
    VisualizationConfig getVisualizationConfig() const;

    /**
     * @brief 获取算子适配配置
     * @return 算子适配配置结构体
     */
    OperatorAdapterConfig getOperatorAdapterConfig() const;

    /**
     * @brief 获取配置值
     * @tparam T 值类型
     * @param key 配置键（支持点分隔符，如 "video_input.frame_width"）
     * @param default_val 默认值
     * @return 配置值
     */
    template<typename T>
    T get(const std::string& key, const T& default_val) const;

    /**
     * @brief 设置配置值
     * @tparam T 值类型
     * @param key 配置键
     * @param value 配置值
     */
    template<typename T>
    void set(const std::string& key, const T& value);

    /**
     * @brief 检查配置键是否存在
     * @param key 配置键
     * @return 存在返回true，否则返回false
     */
    bool has(const std::string& key) const;

    /**
     * @brief 验证配置有效性
     * @return 有效返回true，无效返回false
     */
    bool validate() const;

    /**
     * @brief 重置为默认配置
     */
    void reset();

    /**
     * @brief 打印当前配置
     */
    void print() const;

private:
    // 私有构造函数（单例模式）
    ConfigManager();
    ~ConfigManager() = default;

    // 禁止拷贝和赋值
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;

    /**
     * @brief 解析YAML配置
     * @param content YAML内容
     * @return 成功返回true，失败返回false
     */
    bool parseYAML(const std::string& content);

    /**
     * @brief 解析视频输入配置
     */
    void parseVideoInputConfig();

    /**
     * @brief 解析预处理配置
     */
    void parsePreprocessConfig();

    /**
     * @brief 解析模型配置
     */
    void parseModelConfig();

    /**
     * @brief 解析推理配置
     */
    void parseInferenceConfig();

    /**
     * @brief 解析后处理配置
     */
    void parsePostprocessConfig();

    /**
     * @brief 解析可视化配置
     */
    void parseVisualizationConfig();

    /**
     * @brief 解析算子适配配置
     */
    void parseOperatorAdapterConfig();

    /**
     * @brief 生成YAML字符串
     * @return YAML格式字符串
     */
    std::string generateYAML() const;

    // 成员变量
    SystemConfig system_config_;        ///< 系统配置
    std::map<std::string, ConfigValue> config_map_; ///< 配置映射表
    std::string config_file_path_;      ///< 配置文件路径
    mutable std::mutex mutex_;          ///< 互斥锁
    bool loaded_ = false;               ///< 是否已加载配置
};

} // namespace core
} // namespace smart_video_analysis

#endif // CORE_CONFIG_HPP
