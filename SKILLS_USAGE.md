# Skills系统使用指南

## 概述

本项目已经配置了完整的Claude Skills系统，包含多个专业的数据分析技能。

## 目录结构

```
.claude/
├── skills/                       # 技能目录
│   ├── ab-testing-analyzer/      # AB测试分析技能
│   ├── attribution-analysis-modeling/  # 归因分析建模技能
│   ├── content-analysis/         # 内容分析技能
│   ├── data-exploration-visualization/ # 数据探索可视化技能
│   ├── funnel-analysis/          # 漏斗分析技能
│   ├── growth-model-analyzer/    # 增长模型分析技能
│   ├── ltv-predictor/           # LTV预测技能
│   ├── recommender-system/      # 推荐系统技能
│   ├── regression-analysis-modeling/  # 回归分析建模技能
│   ├── retention-analysis/      # 留存分析技能
│   ├── rfm-customer-segmentation/  # RFM客户分群技能
│   └── user-profiling-analysis/  # 用户画像分析技能
├── agents/                       # 子代理配置
├── commands/                     # 斜杠命令定义
├── hooks/                       # 自动化脚本
└── settings.json                # Claude Code设置
```

## 可用技能列表

### 1. ab-testing-analyzer
全面的AB测试分析工具，支持实验设计、统计检验、用户分群分析和可视化报告生成。

### 2. attribution-analysis-modeling
多触点归因分析工具，使用马尔可夫链、Shapley值和自定义归因模型。

### 3. content-analysis
使用传统NLP和LLM增强方法进行文本内容分析，提取情感、主题、关键词和洞察。

### 4. data-exploration-visualization
自动化数据探索和可视化工具，提供从数据加载到专业报告生成的完整EDA解决方案。

### 5. funnel-analysis
分析用户转化漏斗，计算逐步转化率，创建交互式可视化，识别优化机会。

### 6. growth-model-analyzer
增长模型分析技能，提供全面的增长黑客分析工具，包括裂变策略评估、用户细分等。

### 7. ltv-predictor
对新客户进行快速LTV预测分析。

### 8. recommender-system
智能推荐系统分析工具，提供多种推荐算法实现、评估框架和可视化分析。

### 9. regression-analysis-modeling
使用线性回归、决策树和随机森林进行全面的回归分析和预测建模。

### 10. retention-analysis
使用生存分析、队列分析和机器学习分析用户留存和流失。

### 11. rfm-customer-segmentation
对电商数据进行RFM（最近、频率、金额）客户细分分析。

### 12. user-profiling-analysis
电商用户画像分析与客户分群工具，基于消费数据进行RFM分析、用户价值评估等。

## 使用方法

### 通过斜杠命令使用

最简单的方式是通过斜杠命令直接调用技能：

```bash
# 分析数据
/analyze [dataset] [analysis_type]

# 创建可视化
/visualize [dataset] [chart_type]

# 生成分析代码
/generate [language] [analysis_type]

# 生成分析报告
/report [dataset] [format]

# 执行数据质量检查
/quality [dataset] [action]

# 生成研究假设
/hypothesis [dataset] [domain]
```

### 通过Skill工具使用

也可以通过Skill工具直接调用：

```bash
# 调用特定技能
/skill [skill_name] [args]

# 例如：
/skill ab-testing-analyzer data/ab_test.csv
/skill data-exploration-visualization data/sales_data.csv
/skill rfm-customer-segmentation data/customer_data.csv
```

## 技能开发规范

每个技能都包含以下标准结构：

1. **SKILL.md**: 技能详细文档，包含功能描述、使用方法和示例
2. **README.md**: 使用指南和快速开始
3. **scripts/**: 核心功能模块
4. **examples/**: 示例和样本数据
5. **quick_test.py**: 快速功能测试

## 集成说明

Skills系统与现有的agents和commands完全集成：

- **Agents**: 提供基础的分析能力
- **Commands**: 提供用户友好的斜杠命令接口
- **Skills**: 提供专业化的分析功能

三者协同工作，提供完整的数据分析体验。

## 注意事项

1. 确保数据文件放在正确的位置（通常是`data_storage/`目录）
2. 使用斜杠命令时，遵循正确的参数格式
3. 查看每个技能的文档了解具体的使用方法和参数
4. 技能会自动加载，无需手动配置

## 获取帮助

要获取关于特定技能的帮助，可以：

1. 查看技能目录下的文档文件
2. 使用斜杠命令查看帮助信息
3. 运行技能的测试脚本了解功能

---

现在您可以使用任何已安装的技能进行数据分析了！