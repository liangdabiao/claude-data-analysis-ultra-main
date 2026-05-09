---
allowed-tools: Task, Read, Write, Bash, Grep, Glob
argument-hint: [dataset] [domain] [output_format]
description: 自动化完成整个数据分析工作流程，从数据质量检查到最终报告生成
---

# 全自动化数据分析命令

使用 `do-all` 命令自动化完成整个数据分析工作流程，整合所有现有的commands功能。

## Context
- 数据集位置: @data_storage/$1
- 分析领域: $2 (user-behavior, business-impact, technical-performance, custom)
- 输出格式: $3 (markdown, html, pdf, docx)
- 工作目录: !`pwd`
- 输出目录: ./complete_analysis/
- 人类反馈检查点: 关键步骤暂停等待用户确认

## Your Task

按照以下工作流程自动执行完整的数据分析：

### 1. 数据质量检查 (Quality Assurance)
- 执行数据质量检查和验证
- 识别数据问题和异常
- 生成质量评估报告
- **人类反馈点**: 等待用户确认数据质量可接受

### 2. 探索性数据分析 (Data Exploration)
- 执行全面的探索性数据分析
- 生成统计摘要和描述性分析
- 识别关键模式和关系
- 发现数据中的趋势和异常

### 3. 研究假设生成 (Hypothesis Generation)
- 基于数据模式生成研究假设
- 设计实验验证方案
- 制定统计测试计划
- **人类反馈点**: 等待用户确认假设方向

### 4. 数据可视化 (Visualization)
- 创建全面的数据可视化
- 生成交互式仪表板
- 制作关键发现图表
- 设计可视化故事板

### 5. 代码生成 (Code Generation)
- 生成可重现的分析代码
- 创建数据处理管道
- 编写自动化脚本
- 生成测试用例

### 6. 综合报告生成 (Report Generation)
- 整合所有分析结果
- 创建完整的分析报告
- 包含执行摘要和建议
- 生成技术附录

## 工作流程设计

### 阶段 1: 数据质量评估
```python
def data_quality_assessment(dataset_path):
    """执行全面的数据质量评估"""
    # 数据加载和基础检查
    quality_results = {
        'completeness': assess_completeness(dataset_path),
        'accuracy': assess_accuracy(dataset_path),
        'consistency': assess_consistency(dataset_path),
        'timeliness': assess_timeliness(dataset_path),
        'overall_score': calculate_overall_score()
    }

    return quality_results
```

### 阶段 2: 探索性分析
```python
def exploratory_analysis(dataset_path):
    """执行探索性数据分析"""
    # 统计分析
    statistical_results = perform_statistical_analysis(dataset_path)

    # 模式发现
    patterns = discover_patterns(dataset_path)

    # 相关性分析
    correlations = analyze_correlations(dataset_path)

    # 异常检测
    anomalies = detect_anomalies(dataset_path)

    return {
        'statistical': statistical_results,
        'patterns': patterns,
        'correlations': correlations,
        'anomalies': anomalies
    }
```

### 阶段 3: 假设生成
```python
def generate_hypotheses(analysis_results, domain):
    """基于分析结果生成研究假设"""
    hypotheses = []

    # 基于相关性生成假设
    if analysis_results['correlations']['strong_correlations']:
        hypotheses.extend(create_correlation_hypotheses(
            analysis_results['correlations'], domain
        ))

    # 基于模式生成假设
    if analysis_results['patterns']['significant_patterns']:
        hypotheses.extend(create_pattern_hypotheses(
            analysis_results['patterns'], domain
        ))

    # 基于异常生成假设
    if analysis_results['anomalies']['significant_anomalies']:
        hypotheses.extend(create_anomaly_hypotheses(
            analysis_results['anomalies'], domain
        ))

    return hypotheses
```

### 阶段 4: 可视化创建
```python
def create_comprehensive_visualizations(dataset_path, analysis_results):
    """创建全面的数据可视化"""
    visualizations = {
        'overview': create_overview_dashboard(dataset_path),
        'trends': create_trend_analysis_charts(analysis_results),
        'correlations': create_correlation_matrix(analysis_results),
        'distributions': create_distribution_plots(analysis_results),
        'comparative': create_comparative_analysis(analysis_results)
    }

    return visualizations
```

### 阶段 5: 代码生成
```python
def generate_analysis_code(dataset_path, workflow_config):
    """生成完整的分析代码"""
    code = {
        'data_preprocessing': generate_preprocessing_code(dataset_path),
        'quality_checks': generate_quality_check_code(),
        'analysis_functions': generate_analysis_functions(workflow_config),
        'visualization_code': generate_visualization_code(),
        'reporting_code': generate_reporting_code(),
        'tests': generate_unit_tests(),
        'documentation': generate_code_documentation()
    }

    return code
```

### 阶段 6: 报告生成
```python
def generate_comprehensive_report(all_results, output_format):
    """生成综合分析报告"""
    report = {
        'executive_summary': create_executive_summary(all_results),
        'data_overview': create_data_overview_section(all_results),
        'methodology': create_methodology_section(all_results),
        'findings': create_findings_section(all_results),
        'hypotheses': create_hypotheses_section(all_results),
        'visualizations': create_visualizations_section(all_results),
        'recommendations': create_recommendations_section(all_results),
        'appendices': create_appendices_section(all_results)
    }

    return format_report(report, output_format)
```

## 人类反馈检查点

### 检查点 1: 数据质量确认
```
数据质量评估完成:
- 整体质量得分: 85/100
- 发现的主要问题:
  * 缺失值: 5.2%
  * 异常值: 12个
  * 一致性问题: 3个

您是否确认数据质量可接受并继续分析? (Y/N)
```

### 检查点 2: 分析方向确认
```
探索性分析完成，发现的主要模式:
1. 用户参与度与转化率呈正相关 (r=0.78)
2. 移动端用户留存率较高
3. 周末活跃度显著提升

基于这些发现，建议的研究方向:
- 用户参与度优化实验
- 移动端体验改进
- 周末营销策略优化

您是否同意这些研究方向，还是希望调整分析重点? (Y/调整)
```

### 检查点 3: 可视化策略确认
```
可视化策略建议:
1. 交互式仪表板 - 展示关键指标和趋势
2. 相关性热图 - 显示变量间关系
3. 时间序列图 - 展示用户行为变化
4. 分群分析图 - 比较不同用户群体

您是否同意此可视化策略，还是有特定需求? (Y/自定义)
```

## 预期输出

### 完整分析包
```
complete_analysis/
├── data_quality_report/
│   ├── quality_assessment.json
│   ├── data_issues.log
│   └── quality_improvement_recommendations.md
├── exploratory_analysis/
│   ├── statistical_summary.csv
│   ├── pattern_analysis.md
│   └── correlation_analysis.json
├── hypothesis_reports/
│   ├── research_hypotheses.md
│   ├── experimental_design.md
│   └── validation_plan.md
├── visualizations/
│   ├── interactive_dashboard.html
│   ├── analysis_charts.png
│   └── visualization_code.py
├── generated_code/
│   ├── complete_analysis_pipeline.py
│   ├── data_preprocessing.py
│   ├── quality_checks.py
│   └── analysis_functions.py
├── final_report/
│   ├── comprehensive_analysis_report.$3
│   ├── executive_summary.$3
│   ├── technical_appendix.$3
│   └── presentation_slides.$3
└── workflow_log/
    ├── analysis_progress.log
    ├── human_feedback.log
    └── execution_summary.md
```

### 质量保证检查清单
- [ ] 数据质量达到可接受标准 (≥75分)
- [ ] 所有分析步骤都有文档记录
- [ ] 代码经过测试和验证
- [ ] 可视化清晰且信息丰富
- [ ] 报告包含执行摘要和技术细节
- [ ] 所有人类反馈都已处理
- [ ] 工作流程完全可重现

## 错误处理和恢复

### 常见问题和解决方案
1. **数据质量问题**: 自动修复或提供人工干预选项
2. **分析失败**: 重新执行失败步骤或跳过可选步骤
3. **内存不足**: 数据分块处理或采样分析
4. **依赖缺失**: 自动安装缺失的库
5. **用户超时**: 保存进度并提供恢复选项

### 恢复策略
```python
def handle_analysis_failure(failure_point, error_type):
    """处理分析过程中的失败"""
    if error_type == 'data_quality':
        return handle_quality_failure(failure_point)
    elif error_type == 'analysis_error':
        return handle_analysis_error(failure_point)
    elif error_type == 'timeout':
        return handle_timeout(failure_point)
    else:
        return handle_generic_failure(failure_point)
```

## 使用示例

### 基本用法
```bash
/do-all user_behavior.csv user-behavior markdown
/do-all sales_data.csv business-impact pdf
/do-all system_metrics.csv technical-performance html
/do-all research_data.csv custom docx
```

### 高级用法
```bash
# 带自定义配置的完整分析
/do-all financial_data.csv business-impact html --config config.json

# 跳过某些步骤
/do-all user_data.csv user-behavior pdf --skip quality-check

# 仅执行到特定步骤
/do-all analytics_data.csv technical-performance markdown --stop-at visualization
```

## 配置选项

### 工作流程配置
```json
{
  "workflow": {
    "skip_quality_check": false,
    "skip_hypothesis_generation": false,
    "skip_visualization": false,
    "skip_code_generation": false,
    "human_feedback_required": true,
    "auto_proceed_timeout": 300
  },
  "quality": {
    "minimum_quality_score": 75,
    "auto_fix_issues": true,
    "strict_validation": false
  },
  "analysis": {
    "statistical_significance": 0.05,
    "confidence_level": 0.95,
    "include_advanced_analysis": true
  },
  "output": {
    "include_raw_data": false,
    "include_intermediate_results": true,
    "compression_level": 6,
    "backup_previous_results": true
  }
}
```

## 性能优化

### 大数据处理
- 自动数据分块处理
- 内存使用监控
- 并行处理支持
- 渐进式分析

### 执行优化
- 增量执行（避免重复计算）
- 结果缓存
- 智能任务调度
- 资源使用优化

## 监控和日志

### 执行监控
```python
def monitor_workflow_execution():
    """监控工作流程执行"""
    monitoring = {
        'progress_tracking': track_step_progress(),
        'resource_usage': monitor_system_resources(),
        'error_logging': log_errors_and_warnings(),
        'performance_metrics': track_execution_time(),
        'user_interactions': track_human_feedback()
    }
    return monitoring
```

### 日志文件
- `execution.log`: 详细执行日志
- `human_feedback.log`: 人类交互记录
- `performance.log`: 性能指标
- `error.log`: 错误和异常信息

## 最佳实践

### 工作流程设计
- **模块化**: 每个步骤都是独立的模块
- **可配置**: 灵活的配置选项
- **可恢复**: 支持中断后恢复
- **可扩展**: 易于添加新功能

### 用户体验
- **清晰反馈**: 提供明确的进度和状态信息
- **智能提示**: 在关键决策点提供指导
- **灵活控制**: 允许用户自定义流程
- **完整文档**: 详细的使用说明

### 质量保证
- **自动化测试**: 每个步骤都有验证检查
- **结果验证**: 确保输出质量和准确性
- **文档完整性**: 保持完整的分析文档
- **可重现性**: 确保分析结果可重现

## 注意事项

- 数据集应位于 data_storage/ 目录
- 确保有足够的磁盘空间存储分析结果
- 大数据集可能需要较长的处理时间
- 人类反馈步骤会暂停执行等待用户输入
- 所有结果都会保存到 complete_analysis/ 目录
- 建议在执行前备份重要数据

## 集成说明

此命令整合了以下所有功能：
- `/quality` - 数据质量检查
- `/analyze` - 探索性数据分析
- `/hypothesis` - 研究假设生成
- `/visualize` - 数据可视化
- `/generate` - 代码生成
- `/report` - 报告生成

通过自动化整个工作流程，`do-all` 命令提供了一个完整的数据分析解决方案，从数据质量检查到最终报告生成，同时保持了对关键决策点的人工控制。