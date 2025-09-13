# Extract Dialogue

>***本仓库为`huanhuan-chat`泛化版的一部分（文本对话抽取），欢迎大家给`huanhuan-chat`仓库star！本仓库的最大贡献就是为泛化的Character AI提供了从小说中建立数据集的功能。***
>
>`huanhuan-chat: https://github.com/KMnO4-zx/huanhuan-chat.git`

## Show

`repo`：https://github.com/KMnO4-zx/extract-dialogue.git

本项目利用多平台AI模型从小说中提取对话集，支持DeepSeek、OpenAI、SiliconFlow、Kimi等多个平台。提取的样本中包括`role`和`dialogue`，比如以下的形式：

```json
{
    "role": "艾伦",
    "dialogue": "不，不要提，这真是太倒霉了！我从楼梯上摔了下去，出现了较为严重的骨裂，只能打石膏做固定。"
}
{
    "role": "克莱恩",
    "dialogue": "真是不够走运啊。"
}
```

## QuickStart

- 克隆仓库并切换目录：`git clone https://github.com/KMnO4-zx/extract-dialogue.git `，`cd extract-dialogue`

- 安装依赖：`pip install -r requirements.txt`
- 复制环境变量配置文件：`cp env.example .env`
- 编辑`.env`文件，选择平台并填入对应API密钥（默认使用DeepSeek）
- 运行对话提取：`python dialogue_extractor.py data/test.txt --stats`

## 支持的AI平台

- [**DeepSeek**](https://platform.deepseek.com/usage)：DeepSeek AI平台，推荐使用
- **OpenAI**：OpenAI官方API，支持GPT系列模型
- [**SiliconFlow**](https://cloud.siliconflow.cn/i/ybUFvmqK)：SiliconFlow AI平台，提供多种开源模型
- [**Kimi**](https://platform.moonshot.cn/console/)：月之暗面Kimi平台，擅长长文本处理
- **自定义平台**：支持自定义OpenAI兼容API端点

## 基本用法

### 1. 列出所有支持的平台
```bash
python dialogue_extractor.py --list-platforms
```

### 2. 使用默认平台提取对话
```bash
python dialogue_extractor.py your_novel.txt --stats
```

### 3. 指定平台提取对话
```bash
# 使用OpenAI
python dialogue_extractor.py your_novel.txt -p openai --stats

# 使用Kimi
python dialogue_extractor.py your_novel.txt -p moonshot --stats

# 使用SiliconFlow
python dialogue_extractor.py your_novel.txt -p siliconflow --stats
```

### 4. 并发处理（推荐）
```bash
# 使用8个线程并发处理（默认）
python dialogue_extractor.py your_novel.txt --concurrent --stats

# 指定线程数
python dialogue_extractor.py your_novel.txt -t 16 --concurrent --stats
```

## 高级功能

### 1. Chunk管理
输出文件包含chunk-id信息，便于追踪对话来源：
```json
{
    "chunk_id": 5,
    "dialogue_index": 2,
    "role": "克莱恩",
    "dialogue": "这单免费，还有，叫我夏洛克就行了。"
}
```

### 2. 统计信息
程序会自动生成详细的统计信息：
- 总对话数和角色数量
- 角色对话分布
- 平均对话长度
- 文本块处理统计
- 错误处理汇总

### 3. 输出选项
```bash
# 不包含chunk-id（向后兼容）
python dialogue_extractor.py your_novel.txt --no-chunk-id

# 保存原始chunk文本
python dialogue_extractor.py your_novel.txt --save-chunk-text

# 完成后按chunk-id排序
python dialogue_extractor.py your_novel.txt --sort-output

# 生成旧格式文件
python dialogue_extractor.py your_novel.txt --legacy-format
```

## 环境配置

### 快速配置（DeepSeek）
```bash
# 复制配置文件
cp env.example .env

# 编辑.env文件，只需设置：
DEEPSEEK_API="your-deepseek-api-key"
LLM_PLATFORM="deepseek"
```

### 多平台配置示例
```bash
# 同时配置多个平台
DEEPSEEK_API="your-deepseek-api-key"
OPENAI_API_KEY="your-openai-api-key"
MOONSHOT_API_KEY="your-moonshot-api-key"
SILICONFLOW_API_KEY="your-siliconflow-api-key"

# 选择使用的平台
LLM_PLATFORM="openai"  # 切换到OpenAI
```

## 输出示例

### 标准输出（包含chunk-id）
```json
{"chunk_id": 0, "dialogue_index": 0, "role": "克莱恩", "dialogue": "在帮警察们调查那起连环杀人案，虽然不一定能有收获，但赏金足够诱人，而且，和警察部门建立良好的关系对我们私家侦探来说非常重要。"}
{"chunk_id": 0, "dialogue_index": 1, "role": "塔利姆", "dialogue": "这果然是大侦探忙碌的事情。"}
{"chunk_id": 1, "dialogue_index": 0, "role": "塔利姆", "dialogue": "莫里亚蒂先生，我能请教一个问题吗？"}
{"chunk_id": 1, "dialogue_index": 1, "role": "克莱恩", "dialogue": "这单免费，还有，叫我夏洛克就行了。"}
```

### 统计信息示例
```
=== 统计信息 ===
使用平台: deepseek
使用模型: deepseek-chat
处理方式: 多线程并发 (8 个线程)
输出格式: 包含chunk-id
总对话数: 1,247
角色数量: 15
平均对话长度: 45.2 字符
总块数: 42
平均每块对话数: 29.7

角色分布:
  克莱恩: 423 条
  塔利姆: 198 条
  梅丽莎: 156 条
  班森: 134 条
  ...
```


## 技术特性

### 🔧 核心功能
- **多平台支持**：DeepSeek、OpenAI、SiliconFlow、Kimi等
- **并发处理**：多线程并发提取，大幅提升处理速度
- **智能分块**：基于token的文本分块，保持上下文连贯性
- **去重优化**：自动识别和去除重复对话
- **错误恢复**：支持进度保存和恢复，防止意外中断

### 📊 高级特性
- **Chunk追踪**：输出包含chunk-id，便于定位对话来源
- **统计分析**：详细的处理统计和角色分布分析
- **灵活配置**：支持自定义提取模式和参数
- **向后兼容**：支持生成不含chunk-id的旧格式文件

### 🛠️ 架构设计

系统采用模块化架构：

1. **配置管理** (`config.py`)：统一管理多平台配置和环境变量
2. **对话提取器** (`dialogue_extractor.py`)：核心功能，整合所有处理逻辑
3. **文本处理**：智能分块算法，支持长文本处理
4. **API集成**：统一的OpenAI兼容接口，支持多平台切换
5. **错误处理**：完善的异常处理和重试机制

## 性能优化

### 并发处理
- 默认使用8个线程并发处理
- 支持自定义线程数量
- 线程安全的去重和结果写入
- 智能任务调度和负载均衡

### 内存优化
- 流式处理大型文件
- 智能文本分块，避免内存溢出
- 结果缓冲写入，减少I/O操作

### 准确性优化
- 基于token的精确分块
- 块间重叠保持上下文
- 结构化提示工程
- 自动去重和验证

## 常见问题

### Q: 如何切换不同的AI平台？
A: 编辑`.env`文件，设置`LLM_PLATFORM`为对应平台名称，并配置相应的API密钥。

### Q: 处理大文件时内存不足怎么办？
A: 系统已经实现了流式处理和智能分块，可以处理任意大小的文件。如仍遇到问题，可以减少`MAX_TOKEN_LEN`配置。

### Q: 如何提高提取准确性？
A:
1. 根据小说类型修改提取模式
2. 调整`TEMPERATURE`参数
3. 使用支持长文本的平台（如Kimi）
4. 优化`cover_content`参数以保持更好的上下文

### Q: 输出文件格式说明？
A: 标准格式包含`chunk_id`和`dialogue_index`字段，便于追踪对话来源。旧格式仅包含`role`和`dialogue`字段，用于向后兼容。

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request！请确保：
1. 代码符合现有风格
2. 添加必要的测试
3. 更新相关文档

## 联系方式

- 项目主页：https://github.com/KMnO4-zx/extract-dialogue
- Issues：https://github.com/KMnO4-zx/extract-dialogue/issues