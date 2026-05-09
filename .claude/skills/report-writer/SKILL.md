---
name: "report-writer"
description: "Generates comprehensive analysis reports, executive summaries, and technical documentation. Invoke when user wants to create written reports from data analysis."
---

# Report Writer

Expert report writer specializing in comprehensive data analysis documentation, executive summaries, and technical documentation.

## When to Invoke This Skill

Invoke this skill when user:
- Wants to generate analysis reports
- Needs executive summaries for stakeholders
- Wants to document analysis findings
- Needs technical documentation
- Asks for business intelligence reports
- Needs research-style documentation
- Specifies a report type (summary, complete, executive, technical, custom)

## Report Types (Advanced Mode)

用户可以指定报告类型：

### 1. summary (简要报告)
**适用场景**: 快速摘要

包含:
- 关键发现摘要
- 核心指标
- 简要建议

### 2. complete (完整报告)
**适用场景**: 全面分析

包含:
- 执行摘要
- 数据概述
- 详细分析结果
- 可视化图表
- 业务建议

### 3. executive (高管报告)
**适用场景**: 决策层汇报

包含:
- 高层摘要
- 关键业务指标
- 战略建议
- 风险提示
- 行动项

### 4. technical (技术报告)
**适用场景**: 技术团队

包含:
- 方法论详述
- 统计检验细节
- 数据处理流程
- 技术局限说明
- 代码和算法

### 5. custom (自定义)
根据用户需求定制报告结构

## Core Capabilities

### Report Types
- **Executive Summaries**: High-level insights for decision-makers
- **Technical Reports**: Detailed analysis for technical stakeholders
- **Business Intelligence Reports**: Data-driven business insights
- **Research Papers**: Academic-style documentation
- **Dashboard Documentation**: Interactive report explanations

### Writing Styles
- **Executive**: Concise, business-focused, action-oriented
- **Technical**: Detailed, methodology-focused
- **Academic**: Rigorous, cited style
- **Journalistic**: Engaging, narrative-driven

## Report Structure

### Executive Summary Template
```markdown
# 执行摘要

## 关键发现
- 发现 1: [简要描述和关键指标]
- 发现 2: [简要描述和关键指标]
- 发现 3: [简要描述和关键指标]

## 业务影响
- 财务影响: [量化影响声明]
- 运营影响: [运营改进领域]
- 战略影响: [战略意义]

## 建议
1. [行动项目及时间线和负责人]
2. [行动项目及时间线和负责人]
3. [行动项目及时间线和负责人]

## 下一步
- [立即下一步]
- [后续分析需求]
- [实施时间表]
```

### Technical Report Template
```markdown
# 技术分析报告: [数据集名称]

## 摘要
[分析目标、方法和关键发现的简要总结]

## 1. 引言
### 1.1 背景
[分析背景和目的]

### 1.2 目标
[具体研究问题和分析目标]

### 1.3 方法论概述
[分析方法的高级描述]

## 2. 数据描述
### 2.1 数据来源
[数据的来源和收集方法]

### 2.2 数据结构
[数据结构和变量的详细描述]

### 2.3 数据质量
[数据质量评估和局限性]

## 3. 分析方法
### 3.1 数据准备
[数据清理和转换过程]

### 3.2 统计方法
[使用的统计技术的详细描述]

## 4. 分析结果
### 4.1 主要发现
[最重要的分析结果]

### 4.2 详细结果
[所有分析结果的详细说明]

## 5. 结论和建议
### 5.1 结论
[基于分析的主要结论]

### 5.2 建议
[基于发现的可操作建议]

## 6. 局限性和未来工作
### 6.1 分析局限性
[数据和方法的局限性]

### 6.2 未来工作建议
[进一步分析的建议]
```

## Output Standards

### File Formats
- **Markdown**: `.md` (source format)
- **HTML**: `.html` (web format)
- **PDF**: `.pdf` (print format)

### Output Directory
- `./analysis_reports/`

### Quality Requirements
- Professional formatting
- Clear structure and flow
- Data-driven insights
- Actionable recommendations
- Chinese language content

## Collaboration

Work with other skills:
- **data-explorer**: Get detailed analysis findings
- **visualization-specialist**: Include charts in reports
- **code-generator**: Document analysis code

## Language

All reports must be written in **Chinese**:
- Report titles and headings
- Body content
- Table and figure captions
- Executive summaries
