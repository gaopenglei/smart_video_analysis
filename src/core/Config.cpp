/**
 * @file Config.cpp
 * @brief 配置管理模块实现文件
 */

#include "core/Config.hpp"
#include "core/Logger.hpp"
#include <algorithm>
#include <cctype>
#include <regex>

namespace smart_video_analysis {
namespace core {

// ============================================================================
// 简单的YAML解析器实现
// ============================================================================

namespace {

/**
 * @brief 去除字符串首尾空白
 */
std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

/**
 * @brief 分割字符串
 */
std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::istringstream iss(str);
    std::string token;
    while (std::getline(iss, token, delimiter)) {
        tokens.push_back(trim(token));
    }
    return tokens;
}

/**
 * @brief 解析YAML值
 */
ConfigValue parseYAMLValue(const std::string& value_str) {
    ConfigValue value;
    std::string str = trim(value_str);
    
    // 检查是否是布尔值
    if (str == "true" || str == "True" || str == "TRUE" || str == "yes" || str == "on") {
        value.type = ConfigValueType::BOOLEAN;
        value.bool_val = true;
        return value;
    }
    if (str == "false" || str == "False" || str == "FALSE" || str == "no" || str == "off") {
        value.type = ConfigValueType::BOOLEAN;
        value.bool_val = false;
        return value;
    }
    
    // 检查是否是数字
    bool is_number = true;
    bool has_dot = false;
    bool has_minus = false;
    
    for (size_t i = 0; i < str.size(); ++i) {
        char c = str[i];
        if (c == '-' && i == 0) {
            has_minus = true;
        } else if (c == '.' && !has_dot) {
            has_dot = true;
        } else if (!std::isdigit(c)) {
            is_number = false;
            break;
        }
    }
    
    if (is_number && !str.empty()) {
        if (has_dot) {
            value.type = ConfigValueType::FLOAT;
            value.float_val = std::stof(str);
        } else {
            value.type = ConfigValueType::INTEGER;
            value.int_val = std::stoi(str);
        }
        return value;
    }
    
    // 检查是否是数组
    if (str.front() == '[' && str.back() == ']') {
        value.type = ConfigValueType::ARRAY;
        std::string inner = str.substr(1, str.size() - 2);
        auto items = split(inner, ',');
        for (const auto& item : items) {
            if (!item.empty()) {
                value.array_val.push_back(parseYAMLValue(item));
            }
        }
        return value;
    }
    
    // 默认为字符串（去除引号）
    value.type = ConfigValueType::STRING;
    if ((str.front() == '"' && str.back() == '"') ||
        (str.front() == '\'' && str.back() == '\'')) {
        value.string_val = str.substr(1, str.size() - 2);
    } else {
        value.string_val = str;
    }
    
    return value;
}

} // anonymous namespace

// ============================================================================
// ConfigManager 实现
// ============================================================================

ConfigManager::ConfigManager() {
    reset();
}

ConfigManager& ConfigManager::getInstance() {
    static ConfigManager instance;
    return instance;
}

bool ConfigManager::loadFromFile(const std::string& config_file) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ifstream file(config_file);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open config file: %s", config_file.c_str());
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    file.close();
    
    if (!parseYAML(content)) {
        LOG_ERROR("Failed to parse config file: %s", config_file.c_str());
        return false;
    }
    
    config_file_path_ = config_file;
    loaded_ = true;
    
    // 解析各个配置模块
    parseVideoInputConfig();
    parsePreprocessConfig();
    parseModelConfig();
    parseInferenceConfig();
    parsePostprocessConfig();
    parseVisualizationConfig();
    parseOperatorAdapterConfig();
    
    LOG_INFO("Config loaded from: %s", config_file.c_str());
    return true;
}

bool ConfigManager::loadFromString(const std::string& config_str) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!parseYAML(config_str)) {
        LOG_ERROR("Failed to parse config string");
        return false;
    }
    
    loaded_ = true;
    
    parseVideoInputConfig();
    parsePreprocessConfig();
    parseModelConfig();
    parseInferenceConfig();
    parsePostprocessConfig();
    parseVisualizationConfig();
    parseOperatorAdapterConfig();
    
    return true;
}

bool ConfigManager::saveToFile(const std::string& config_file) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ofstream file(config_file);
    if (!file.is_open()) {
        LOG_ERROR("Failed to create config file: %s", config_file.c_str());
        return false;
    }
    
    file << generateYAML();
    file.close();
    
    LOG_INFO("Config saved to: %s", config_file.c_str());
    return true;
}

bool ConfigManager::parseYAML(const std::string& content) {
    std::istringstream stream(content);
    std::string line;
    std::vector<std::string> key_stack;
    
    while (std::getline(stream, line)) {
        // 跳过空行和注释
        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        
        // 计算缩进级别
        size_t indent = line.find_first_not_of(" \t");
        int level = static_cast<int>(indent / 2);
        
        // 调整key_stack
        while (static_cast<int>(key_stack.size()) > level) {
            key_stack.pop_back();
        }
        
        // 查找冒号
        size_t colon_pos = trimmed.find(':');
        if (colon_pos == std::string::npos) {
            continue;
        }
        
        std::string key = trim(trimmed.substr(0, colon_pos));
        std::string value_str = trim(trimmed.substr(colon_pos + 1));
        
        // 构建完整键名
        key_stack.push_back(key);
        std::string full_key;
        for (size_t i = 0; i < key_stack.size(); ++i) {
            if (i > 0) full_key += ".";
            full_key += key_stack[i];
        }
        
        // 如果有值，则解析
        if (!value_str.empty()) {
            config_map_[full_key] = parseYAMLValue(value_str);
        }
        
        // 如果没有值，则这是一个嵌套对象
        // 保持key_stack不变，等待下一行
    }
    
    return true;
}

void ConfigManager::parseVideoInputConfig() {
    auto& cfg = system_config_.video_input;
    
    if (has("video_input.source_type")) cfg.source_type = get<std::string>("video_input.source_type", cfg.source_type);
    if (has("video_input.source_path")) cfg.source_path = get<std::string>("video_input.source_path", cfg.source_path);
    if (has("video_input.camera_index")) cfg.camera_index = get<int>("video_input.camera_index", cfg.camera_index);
    if (has("video_input.frame_width")) cfg.frame_width = get<int>("video_input.frame_width", cfg.frame_width);
    if (has("video_input.frame_height")) cfg.frame_height = get<int>("video_input.frame_height", cfg.frame_height);
    if (has("video_input.fps")) cfg.fps = get<int>("video_input.fps", cfg.fps);
    if (has("video_input.use_hw_acceleration")) cfg.use_hw_acceleration = get<bool>("video_input.use_hw_acceleration", cfg.use_hw_acceleration);
    if (has("video_input.hw_accel_device")) cfg.hw_accel_device = get<std::string>("video_input.hw_accel_device", cfg.hw_accel_device);
    if (has("video_input.buffer_size")) cfg.buffer_size = get<int>("video_input.buffer_size", cfg.buffer_size);
}

void ConfigManager::parsePreprocessConfig() {
    auto& cfg = system_config_.preprocess;
    
    if (has("preprocess.target_width")) cfg.target_width = get<int>("preprocess.target_width", cfg.target_width);
    if (has("preprocess.target_height")) cfg.target_height = get<int>("preprocess.target_height", cfg.target_height);
    if (has("preprocess.normalize")) cfg.normalize = get<bool>("preprocess.normalize", cfg.normalize);
    if (has("preprocess.mean_r")) cfg.mean_r = get<float>("preprocess.mean_r", cfg.mean_r);
    if (has("preprocess.mean_g")) cfg.mean_g = get<float>("preprocess.mean_g", cfg.mean_g);
    if (has("preprocess.mean_b")) cfg.mean_b = get<float>("preprocess.mean_b", cfg.mean_b);
    if (has("preprocess.std_r")) cfg.std_r = get<float>("preprocess.std_r", cfg.std_r);
    if (has("preprocess.std_g")) cfg.std_g = get<float>("preprocess.std_g", cfg.std_g);
    if (has("preprocess.std_b")) cfg.std_b = get<float>("preprocess.std_b", cfg.std_b);
    if (has("preprocess.swap_rb")) cfg.swap_rb = get<bool>("preprocess.swap_rb", cfg.swap_rb);
    if (has("preprocess.resize_mode")) cfg.resize_mode = get<std::string>("preprocess.resize_mode", cfg.resize_mode);
    if (has("preprocess.scale")) cfg.scale = get<float>("preprocess.scale", cfg.scale);
}

void ConfigManager::parseModelConfig() {
    auto& cfg = system_config_.model;
    
    if (has("model.model_path")) cfg.model_path = get<std::string>("model.model_path", cfg.model_path);
    if (has("model.model_type")) cfg.model_type = get<std::string>("model.model_type", cfg.model_type);
    if (has("model.framework")) cfg.framework = get<std::string>("model.framework", cfg.framework);
    if (has("model.input_batch")) cfg.input_batch = get<int>("model.input_batch", cfg.input_batch);
    if (has("model.input_channels")) cfg.input_channels = get<int>("model.input_channels", cfg.input_channels);
    if (has("model.input_width")) cfg.input_width = get<int>("model.input_width", cfg.input_width);
    if (has("model.input_height")) cfg.input_height = get<int>("model.input_height", cfg.input_height);
    if (has("model.input_name")) cfg.input_name = get<std::string>("model.input_name", cfg.input_name);
    if (has("model.output_name")) cfg.output_name = get<std::string>("model.output_name", cfg.output_name);
    if (has("model.num_classes")) cfg.num_classes = get<int>("model.num_classes", cfg.num_classes);
    if (has("model.confidence_threshold")) cfg.confidence_threshold = get<float>("model.confidence_threshold", cfg.confidence_threshold);
    if (has("model.nms_threshold")) cfg.nms_threshold = get<float>("model.nms_threshold", cfg.nms_threshold);
}

void ConfigManager::parseInferenceConfig() {
    auto& cfg = system_config_.inference;
    
    if (has("inference.num_threads")) cfg.num_threads = get<int>("inference.num_threads", cfg.num_threads);
    if (has("inference.use_gpu")) cfg.use_gpu = get<bool>("inference.use_gpu", cfg.use_gpu);
    if (has("inference.gpu_device_id")) cfg.gpu_device_id = get<int>("inference.gpu_device_id", cfg.gpu_device_id);
    if (has("inference.execution_provider")) cfg.execution_provider = get<std::string>("inference.execution_provider", cfg.execution_provider);
    if (has("inference.enable_profiling")) cfg.enable_profiling = get<bool>("inference.enable_profiling", cfg.enable_profiling);
    if (has("inference.profiling_output_dir")) cfg.profiling_output_dir = get<std::string>("inference.profiling_output_dir", cfg.profiling_output_dir);
    if (has("inference.warmup_iterations")) cfg.warmup_iterations = get<int>("inference.warmup_iterations", cfg.warmup_iterations);
    if (has("inference.enable_memory_pattern")) cfg.enable_memory_pattern = get<bool>("inference.enable_memory_pattern", cfg.enable_memory_pattern);
    if (has("inference.enable_memory_allocator")) cfg.enable_memory_allocator = get<bool>("inference.enable_memory_allocator", cfg.enable_memory_allocator);
}

void ConfigManager::parsePostprocessConfig() {
    auto& cfg = system_config_.postprocess;
    
    if (has("postprocess.confidence_threshold")) cfg.confidence_threshold = get<float>("postprocess.confidence_threshold", cfg.confidence_threshold);
    if (has("postprocess.nms_threshold")) cfg.nms_threshold = get<float>("postprocess.nms_threshold", cfg.nms_threshold);
    if (has("postprocess.max_detections")) cfg.max_detections = get<int>("postprocess.max_detections", cfg.max_detections);
    if (has("postprocess.filter_small_objects")) cfg.filter_small_objects = get<bool>("postprocess.filter_small_objects", cfg.filter_small_objects);
    if (has("postprocess.min_object_size")) cfg.min_object_size = get<int>("postprocess.min_object_size", cfg.min_object_size);
}

void ConfigManager::parseVisualizationConfig() {
    auto& cfg = system_config_.visualization;
    
    if (has("visualization.enable")) cfg.enable = get<bool>("visualization.enable", cfg.enable);
    if (has("visualization.show_fps")) cfg.show_fps = get<bool>("visualization.show_fps", cfg.show_fps);
    if (has("visualization.show_confidence")) cfg.show_confidence = get<bool>("visualization.show_confidence", cfg.show_confidence);
    if (has("visualization.show_class_name")) cfg.show_class_name = get<bool>("visualization.show_class_name", cfg.show_class_name);
    if (has("visualization.box_line_width")) cfg.box_line_width = get<int>("visualization.box_line_width", cfg.box_line_width);
    if (has("visualization.font_scale")) cfg.font_scale = get<float>("visualization.font_scale", cfg.font_scale);
    if (has("visualization.output_type")) cfg.output_type = get<std::string>("visualization.output_type", cfg.output_type);
    if (has("visualization.output_path")) cfg.output_path = get<std::string>("visualization.output_path", cfg.output_path);
    if (has("visualization.output_fps")) cfg.output_fps = get<int>("visualization.output_fps", cfg.output_fps);
    if (has("visualization.output_codec")) cfg.output_codec = get<std::string>("visualization.output_codec", cfg.output_codec);
}

void ConfigManager::parseOperatorAdapterConfig() {
    auto& cfg = system_config_.operator_adapter;
    
    if (has("operator_adapter.target_npu")) cfg.target_npu = get<std::string>("operator_adapter.target_npu", cfg.target_npu);
    if (has("operator_adapter.enable_fallback")) cfg.enable_fallback = get<bool>("operator_adapter.enable_fallback", cfg.enable_fallback);
    if (has("operator_adapter.enable_fusion")) cfg.enable_fusion = get<bool>("operator_adapter.enable_fusion", cfg.enable_fusion);
    if (has("operator_adapter.unsupported_ops_file")) cfg.unsupported_ops_file = get<std::string>("operator_adapter.unsupported_ops_file", cfg.unsupported_ops_file);
}

std::string ConfigManager::generateYAML() const {
    std::ostringstream oss;
    
    oss << "# Smart Video Analysis System Configuration\n";
    oss << "# Generated automatically\n\n";
    
    oss << "# Video Input Configuration\n";
    oss << "video_input:\n";
    oss << "  source_type: " << system_config_.video_input.source_type << "\n";
    oss << "  source_path: " << system_config_.video_input.source_path << "\n";
    oss << "  camera_index: " << system_config_.video_input.camera_index << "\n";
    oss << "  frame_width: " << system_config_.video_input.frame_width << "\n";
    oss << "  frame_height: " << system_config_.video_input.frame_height << "\n";
    oss << "  fps: " << system_config_.video_input.fps << "\n";
    oss << "  use_hw_acceleration: " << (system_config_.video_input.use_hw_acceleration ? "true" : "false") << "\n";
    oss << "  buffer_size: " << system_config_.video_input.buffer_size << "\n\n";
    
    oss << "# Preprocessing Configuration\n";
    oss << "preprocess:\n";
    oss << "  target_width: " << system_config_.preprocess.target_width << "\n";
    oss << "  target_height: " << system_config_.preprocess.target_height << "\n";
    oss << "  normalize: " << (system_config_.preprocess.normalize ? "true" : "false") << "\n";
    oss << "  swap_rb: " << (system_config_.preprocess.swap_rb ? "true" : "false") << "\n";
    oss << "  resize_mode: " << system_config_.preprocess.resize_mode << "\n";
    oss << "  scale: " << system_config_.preprocess.scale << "\n\n";
    
    oss << "# Model Configuration\n";
    oss << "model:\n";
    oss << "  model_path: " << system_config_.model.model_path << "\n";
    oss << "  model_type: " << system_config_.model.model_type << "\n";
    oss << "  framework: " << system_config_.model.framework << "\n";
    oss << "  input_batch: " << system_config_.model.input_batch << "\n";
    oss << "  input_channels: " << system_config_.model.input_channels << "\n";
    oss << "  input_width: " << system_config_.model.input_width << "\n";
    oss << "  input_height: " << system_config_.model.input_height << "\n";
    oss << "  input_name: " << system_config_.model.input_name << "\n";
    oss << "  output_name: " << system_config_.model.output_name << "\n";
    oss << "  num_classes: " << system_config_.model.num_classes << "\n";
    oss << "  confidence_threshold: " << system_config_.model.confidence_threshold << "\n";
    oss << "  nms_threshold: " << system_config_.model.nms_threshold << "\n\n";
    
    oss << "# Inference Configuration\n";
    oss << "inference:\n";
    oss << "  num_threads: " << system_config_.inference.num_threads << "\n";
    oss << "  use_gpu: " << (system_config_.inference.use_gpu ? "true" : "false") << "\n";
    oss << "  execution_provider: " << system_config_.inference.execution_provider << "\n";
    oss << "  warmup_iterations: " << system_config_.inference.warmup_iterations << "\n\n";
    
    oss << "# Postprocessing Configuration\n";
    oss << "postprocess:\n";
    oss << "  confidence_threshold: " << system_config_.postprocess.confidence_threshold << "\n";
    oss << "  nms_threshold: " << system_config_.postprocess.nms_threshold << "\n";
    oss << "  max_detections: " << system_config_.postprocess.max_detections << "\n\n";
    
    oss << "# Visualization Configuration\n";
    oss << "visualization:\n";
    oss << "  enable: " << (system_config_.visualization.enable ? "true" : "false") << "\n";
    oss << "  show_fps: " << (system_config_.visualization.show_fps ? "true" : "false") << "\n";
    oss << "  output_type: " << system_config_.visualization.output_type << "\n\n";
    
    oss << "# Operator Adapter Configuration\n";
    oss << "operator_adapter:\n";
    oss << "  target_npu: " << system_config_.operator_adapter.target_npu << "\n";
    oss << "  enable_fallback: " << (system_config_.operator_adapter.enable_fallback ? "true" : "false") << "\n";
    oss << "  enable_fusion: " << (system_config_.operator_adapter.enable_fusion ? "true" : "false") << "\n";
    
    return oss.str();
}

SystemConfig ConfigManager::getSystemConfig() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return system_config_;
}

void ConfigManager::setSystemConfig(const SystemConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    system_config_ = config;
}

VideoInputConfig ConfigManager::getVideoInputConfig() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return system_config_.video_input;
}

PreprocessConfig ConfigManager::getPreprocessConfig() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return system_config_.preprocess;
}

ModelConfig ConfigManager::getModelConfig() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return system_config_.model;
}

InferenceConfig ConfigManager::getInferenceConfig() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return system_config_.inference;
}

PostprocessConfig ConfigManager::getPostprocessConfig() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return system_config_.postprocess;
}

VisualizationConfig ConfigManager::getVisualizationConfig() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return system_config_.visualization;
}

OperatorAdapterConfig ConfigManager::getOperatorAdapterConfig() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return system_config_.operator_adapter;
}

template<>
std::string ConfigManager::get<std::string>(const std::string& key, const std::string& default_val) const {
    auto it = config_map_.find(key);
    if (it != config_map_.end() && it->second.type == ConfigValueType::STRING) {
        return it->second.string_val;
    }
    return default_val;
}

template<>
int ConfigManager::get<int>(const std::string& key, const int& default_val) const {
    auto it = config_map_.find(key);
    if (it != config_map_.end() && it->second.type == ConfigValueType::INTEGER) {
        return it->second.int_val;
    }
    return default_val;
}

template<>
float ConfigManager::get<float>(const std::string& key, const float& default_val) const {
    auto it = config_map_.find(key);
    if (it != config_map_.end()) {
        if (it->second.type == ConfigValueType::FLOAT) {
            return it->second.float_val;
        } else if (it->second.type == ConfigValueType::INTEGER) {
            return static_cast<float>(it->second.int_val);
        }
    }
    return default_val;
}

template<>
bool ConfigManager::get<bool>(const std::string& key, const bool& default_val) const {
    auto it = config_map_.find(key);
    if (it != config_map_.end() && it->second.type == ConfigValueType::BOOLEAN) {
        return it->second.bool_val;
    }
    return default_val;
}

template<>
void ConfigManager::set<std::string>(const std::string& key, const std::string& value) {
    ConfigValue cv;
    cv.type = ConfigValueType::STRING;
    cv.string_val = value;
    config_map_[key] = cv;
}

template<>
void ConfigManager::set<int>(const std::string& key, const int& value) {
    ConfigValue cv;
    cv.type = ConfigValueType::INTEGER;
    cv.int_val = value;
    config_map_[key] = cv;
}

template<>
void ConfigManager::set<float>(const std::string& key, const float& value) {
    ConfigValue cv;
    cv.type = ConfigValueType::FLOAT;
    cv.float_val = value;
    config_map_[key] = cv;
}

template<>
void ConfigManager::set<bool>(const std::string& key, const bool& value) {
    ConfigValue cv;
    cv.type = ConfigValueType::BOOLEAN;
    cv.bool_val = value;
    config_map_[key] = cv;
}

bool ConfigManager::has(const std::string& key) const {
    return config_map_.find(key) != config_map_.end();
}

bool ConfigManager::validate() const {
    // 检查必要的配置项
    if (system_config_.model.model_path.empty()) {
        LOG_ERROR("Model path is not specified");
        return false;
    }
    
    if (system_config_.video_input.source_type.empty()) {
        LOG_ERROR("Video source type is not specified");
        return false;
    }
    
    if (system_config_.preprocess.target_width <= 0 || system_config_.preprocess.target_height <= 0) {
        LOG_ERROR("Invalid target dimensions");
        return false;
    }
    
    return true;
}

void ConfigManager::reset() {
    system_config_ = SystemConfig();
    config_map_.clear();
    loaded_ = false;
}

void ConfigManager::print() const {
    LOG_INFO("========== System Configuration ==========");
    LOG_INFO("Video Input:");
    LOG_INFO("  Source Type: %s", system_config_.video_input.source_type.c_str());
    LOG_INFO("  Source Path: %s", system_config_.video_input.source_path.c_str());
    LOG_INFO("  Frame Size: %dx%d", system_config_.video_input.frame_width, system_config_.video_input.frame_height);
    
    LOG_INFO("Preprocess:");
    LOG_INFO("  Target Size: %dx%d", system_config_.preprocess.target_width, system_config_.preprocess.target_height);
    LOG_INFO("  Normalize: %s", system_config_.preprocess.normalize ? "true" : "false");
    
    LOG_INFO("Model:");
    LOG_INFO("  Model Path: %s", system_config_.model.model_path.c_str());
    LOG_INFO("  Model Type: %s", system_config_.model.model_type.c_str());
    LOG_INFO("  Input Size: %dx%dx%d", system_config_.model.input_channels, 
             system_config_.model.input_width, system_config_.model.input_height);
    
    LOG_INFO("Inference:");
    LOG_INFO("  Num Threads: %d", system_config_.inference.num_threads);
    LOG_INFO("  Execution Provider: %s", system_config_.inference.execution_provider.c_str());
    
    LOG_INFO("==========================================");
}

} // namespace core
} // namespace smart_video_analysis
