---
allowed-tools: Task, Read, Write, Bash, Grep, Glob, Skill, Edit, DeleteFile, RunCommand, StopCommand, CheckCommandStatus, TodoWrite, SearchCodebase, WebSearch, LS
description: 自动观察data_storage目录中的数据，智能匹配并执行合适的skills进行数据分析
skills:
---

# 智能数据分析命令 (do-more)

使用 `do-more` 命令自动观察 `data_storage` 目录中的数据，智能匹配并执行合适的skills进行数据分析。

## Context
- 数据目录: @data_storage/
- 技能目录: @skills/
- 工作目录: !`pwd`
- 输出目录: ./do_more_analysis/
- 执行模式: 顺序执行每个匹配的技能

## Your Task

按照以下工作流程自动执行智能数据分析：

### 1. 数据观察与识别 (Data Observation)
- 扫描 `data_storage` 目录中的所有数据文件
- 识别数据类型和结构
- 分析数据字段和特征
- 确定数据领域（电商、用户行为、营销等）

### 2. 技能匹配 (Skill Matching)
- 基于数据特征自动匹配合适的skills
- 按优先级排序技能执行顺序
- 确保每个技能的数据依赖满足
- 生成技能执行计划

### 3. 顺序执行技能 (Sequential Execution)
- 按照匹配结果逐个执行技能
- 每个技能执行前进行数据准备
- 执行过程中记录日志和结果
- 处理技能执行错误

### 4. 结果整合 (Result Integration)
- 收集所有技能的输出结果
- 整合分析结果和洞察
- 生成综合分析报告
- 创建可视化仪表板
- 创建一个图文并茂的HTML综合报告，包含所有分析结果和可视化图表。

## 技能匹配逻辑

### 数据特征识别

#### 电商订单数据
**识别特征**:
- 包含订单相关文件 (Orders.csv, Order Items.csv, Order Payments.csv)
- 包含客户数据 (Customers.csv)
- 包含产品数据 (Products.csv, Categories.csv)
- 包含评价数据 (Reviews.csv)

**匹配技能**:
1. **rfm-customer-segmentation** - 客户分群分析
2. **ltv-predictor** - 客户生命周期价值预测
3. **retention-analysis** - 用户留存分析
4. **funnel-analysis** - 转化漏斗分析
5. **growth-model-analyzer** - 增长模型分析
6. **recommender-system** - 推荐系统分析

#### 用户行为数据
**识别特征**:
- 包含用户活动日志
- 包含页面访问记录
- 包含点击流数据
- 包含用户交互数据

**匹配技能**:
1. **user-profiling-analysis** - 用户画像分析
2. **retention-analysis** - 用户留存分析
3. **funnel-analysis** - 转化漏斗分析
4. **data-exploration-visualization** - 探索性数据分析

#### 营销数据
**识别特征**:
- 包含营销活动数据
- 包含渠道数据
- 包含广告投放数据
- 包含转化追踪数据

**匹配技能**:
1. **attribution-analysis-modeling** - 营销归因分析
2. **growth-model-analyzer** - 增长模型分析
3. **ab-testing-analyzer** - A/B测试分析

#### 内容数据
**识别特征**:
- 包含文本数据
- 包含评论数据
- 包含社交媒体数据
- 包含内容标签数据

**匹配技能**:
1. **content-analysis** - 内容分析
2. **recommender-system** - 推荐系统分析

#### 通用数值数据
**识别特征**:
- 包含数值型数据
- 包含时间序列数据
- 包含分类变量
- 包含连续变量

**匹配技能**:
1. **data-exploration-visualization** - 探索性数据分析
2. **regression-analysis-modeling** - 回归分析建模

### 技能执行优先级

#### 电商数据优先级
1. **data-exploration-visualization** (基础分析)
2. **rfm-customer-segmentation** (客户分群)
3. **ltv-predictor** (价值预测)
4. **retention-analysis** (留存分析)
5. **funnel-analysis** (转化分析)
6. **growth-model-analyzer** (增长分析)
7. **recommender-system** (推荐分析)

#### 用户行为数据优先级
1. **data-exploration-visualization** (基础分析)
2. **user-profiling-analysis** (用户画像)
3. **retention-analysis** (留存分析)
4. **funnel-analysis** (转化分析)

#### 营销数据优先级
1. **data-exploration-visualization** (基础分析)
2. **attribution-analysis-modeling** (归因分析)
3. **growth-model-analyzer** (增长分析)
4. **ab-testing-analyzer** (A/B测试)

## 工作流程设计

### 阶段 1: 数据观察
```python
def observe_data(data_dir):
    """观察并识别数据特征"""
    data_files = scan_directory(data_dir)
    
    data_profile = {
        'files': data_files,
        'types': identify_data_types(data_files),
        'fields': extract_fields(data_files),
        'domain': classify_domain(data_files),
        'quality': assess_data_quality(data_files)
    }
    
    return data_profile
```

### 阶段 2: 技能匹配
```python
def match_skills(data_profile):
    """基于数据特征匹配技能"""
    matched_skills = []
    
    for skill in available_skills:
        if is_skill_compatible(skill, data_profile):
            priority = calculate_priority(skill, data_profile)
            matched_skills.append({
                'name': skill['name'],
                'path': skill['path'],
                'priority': priority,
                'requirements': skill['requirements']
            })
    
    matched_skills.sort(key=lambda x: x['priority'])
    return matched_skills
```

### 阶段 3: 顺序执行
```python
def execute_skills_sequentially(matched_skills, data_dir, output_dir):
    """顺序执行匹配的技能"""
    execution_results = []
    
    for skill in matched_skills:
        print(f"执行技能: {skill['name']}")
        
        try:
            result = execute_skill(
                skill['path'],
                data_dir,
                output_dir
            )
            
            execution_results.append({
                'skill': skill['name'],
                'status': 'success',
                'result': result
            })
            
        except Exception as e:
            execution_results.append({
                'skill': skill['name'],
                'status': 'failed',
                'error': str(e)
            })
            
            print(f"技能 {skill['name']} 执行失败: {e}")
    
    return execution_results
```

### 阶段 4: 结果整合
```python
def integrate_results(execution_results, output_dir):
    """整合所有技能的执行结果"""
    integrated_report = {
        'summary': create_execution_summary(execution_results),
        'skills': collect_skill_results(execution_results),
        'insights': extract_cross_skill_insights(execution_results),
        'recommendations': generate_recommendations(execution_results),
        'visualizations': create_integrated_dashboard(execution_results)
    }
    
    save_integrated_report(integrated_report, output_dir)
    return integrated_report
```

## 技能执行详情

### 1. data-exploration-visualization
**执行条件**: 总是首先执行
**数据要求**: 任意数据文件
**输出内容**:
- 数据质量报告
- 统计摘要
- 分布可视化
- 相关性分析
- 异常值检测

### 2. rfm-customer-segmentation
**执行条件**: 存在订单和客户数据
**数据要求**: Orders.csv, Customers.csv
**输出内容**:
- RFM分析结果
- 客户分群
- VIP客户列表
- 分群可视化

### 3. ltv-predictor
**执行条件**: 存在订单数据且客户数>100
**数据要求**: Orders.csv, Order Items.csv
**输出内容**:
- RFM特征
- LTV预测
- 模型评估
- 客户价值排名

### 4. retention-analysis
**执行条件**: 存在时间序列订单数据
**数据要求**: Orders.csv, Customers.csv
**输出内容**:
- 留存率分析
- 生存曲线
- 流失预测
- 留存策略

### 5. funnel-analysis
**执行条件**: 存在订单流程数据
**数据要求**: Orders.csv, Order Items.csv
**输出内容**:
- 转化漏斗
- 转化率分析
- 流失分析
- 分段对比

### 6. growth-model-analyzer
**执行条件**: 存在时间序列数据
**数据要求**: Orders.csv
**输出内容**:
- 增长指标
- 增长模型
- ROI分析
- 增长策略

### 7. recommender-system
**执行条件**: 存在用户-商品交互数据
**数据要求**: Orders.csv, Order Items.csv, Products.csv
**输出内容**:
- 推荐结果
- 协同过滤
- 推荐评估
- 个性化推荐

### 8. attribution-analysis-modeling
**执行条件**: 存在营销渠道数据
**数据要求**: 营销相关数据
**输出内容**:
- 归因分析
- 渠道贡献
- 触点分析
- 营销优化

### 9. ab-testing-analyzer
**执行条件**: 存在A/B测试数据
**数据要求**: 测试分组数据
**输出内容**:
- A/B测试结果
- 统计显著性
- 效果评估
- 决策建议

### 10. content-analysis
**执行条件**: 存在文本数据
**数据要求**: Reviews.csv, 文本字段
**输出内容**:
- 情感分析
- 主题分析
- 文本可视化
- 内容洞察

### 11. regression-analysis-modeling
**执行条件**: 存在数值型目标变量
**数据要求**: 数值数据
**输出内容**:
- 回归模型
- 特征重要性
- 预测结果
- 模型评估

## 预期输出

### 输出目录结构
```
do_more_analysis/
├── data_observation/
│   ├── data_profile.json
│   ├── data_types.json
│   └── data_quality_report.md
├── skill_matching/
│   ├── matched_skills.json
│   ├── execution_plan.json
│   └── skill_priorities.json
├── skill_execution/
│   ├── data-exploration-visualization/
│   │   ├── exploration_report.html
│   │   ├── visualizations/
│   │   └── analysis_results.json
│   ├── rfm-customer-segmentation/
│   │   ├── customer_segments.csv
│   │   ├── vip_customers.csv
│   │   └── segmentation_dashboard.png
│   ├── ltv-predictor/
│   │   ├── ltv_predictions.csv
│   │   ├── model_evaluation.json
│   │   └── ltv_report.html
│   ├── retention-analysis/
│   │   ├── retention_rates.csv
│   │   ├── survival_curves.png
│   │   └── retention_report.html
│   ├── funnel-analysis/
│   │   ├── funnel_analysis.html
│   │   ├── conversion_rates.csv
│   │   └── funnel_dashboard.html
│   ├── growth-model-analyzer/
│   │   ├── growth_metrics.csv
│   │   ├── growth_report.html
│   │   └── roi_analysis.json
│   ├── recommender-system/
│   │   ├── recommendations.csv
│   │   ├── model_evaluation.json
│   │   └── recommendation_report.html
│   ├── attribution-analysis-modeling/
│   │   ├── attribution_results.json
│   │   ├── channel_analysis.csv
│   │   └── attribution_report.html
│   ├── ab-testing-analyzer/
│   │   ├── ab_test_results.json
│   │   ├── statistical_tests.csv
│   │   └── ab_test_report.html
│   ├── content-analysis/
│   │   ├── sentiment_analysis.csv
│   │   ├── topic_analysis.json
│   │   └── content_report.html
│   └── regression-analysis-modeling/
│       ├── regression_model.joblib
│       ├── predictions.csv
│       └── regression_report.html
├── integrated_results/
│   ├── comprehensive_report.html
│   ├── executive_summary.md
│   ├── key_insights.md
│   ├── recommendations.md
│   └── integrated_dashboard.html
└── execution_logs/
    ├── execution_log.txt
    ├── skill_execution_status.json
    └── error_log.txt
```

## 执行日志

### 日志格式
```json
{
  "timestamp": "2025-12-24T10:00:00Z",
  "skill": "rfm-customer-segmentation",
  "status": "success",
  "duration": 45.2,
  "output_files": [
    "customer_segments.csv",
    "vip_customers.csv"
  ],
  "key_metrics": {
    "customers_analyzed": 1000,
    "segments_created": 5,
    "vip_customers": 50
  }
}
```

### 执行状态跟踪
- **pending**: 等待执行
- **in_progress**: 正在执行
- **completed**: 执行完成
- **failed**: 执行失败
- **skipped**: 跳过执行

## 错误处理

### 常见错误处理
1. **数据格式错误**: 尝试数据转换，失败则跳过该技能
2. **依赖缺失**: 自动安装依赖，失败则跳过该技能
3. **内存不足**: 数据分块处理，失败则跳过该技能
4. **执行超时**: 增加超时时间，失败则跳过该技能
5. **权限错误**: 记录错误并跳过该技能

### 恢复策略
```python
def handle_skill_failure(skill, error):
    """处理技能执行失败"""
    log_error(skill, error)
    
    if is_recoverable(error):
        retry_skill(skill)
    else:
        skip_skill(skill)
        continue_with_next_skill()
```

## 使用示例

### 基本用法
```bash
/do-more
```

### 指定输出目录
```bash
/do-more --output custom_output_dir
```

### 仅匹配特定技能
```bash
/do-more --skills rfm,ltv,retention
```

### 跳过某些技能
```bash
/do-more --skip content-analysis,ab-testing-analyzer
```

### 显示执行计划
```bash
/do-more --plan-only
```

## 配置选项

### 执行配置
```json
{
  "execution": {
    "parallel_execution": false,
    "max_concurrent_skills": 1,
    "timeout_per_skill": 300,
    "continue_on_error": true,
    "retry_failed_skills": false,
    "max_retries": 1
  },
  "matching": {
    "auto_match_skills": true,
    "min_data_quality": 70,
    "require_all_dependencies": true,
    "allow_partial_execution": true
  },
  "output": {
    "generate_integrated_report": true,
    "create_dashboard": true,
    "include_intermediate_results": true,
    "compress_output": false
  }
}
```

## 性能优化

### 执行优化
- 智能缓存避免重复计算
- 增量执行支持
- 资源使用监控
- 内存优化处理

### 数据优化
- 自动数据采样
- 智能数据分块
- 并行数据处理
- 增量数据更新

## 最佳实践

### 技能匹配
- 基于数据特征智能匹配
- 考虑技能间的依赖关系
- 优化技能执行顺序
- 避免重复分析

### 执行管理
- 记录详细执行日志
- 提供清晰的进度反馈
- 支持中断后恢复
- 灵活的错误处理

### 结果整合
- 跨技能洞察提取
- 统一的可视化风格
- 一致的报告格式
- 可操作的建议

## 注意事项

- 确保data_storage目录包含有效数据
- 某些技能可能需要特定的数据格式
- 执行时间取决于数据量和技能数量
- 大数据集可能需要较长的处理时间
- 建议在执行前备份重要数据
- 确保有足够的磁盘空间存储结果

## 与do-all的区别

| 特性 | do-all | do-more |
|------|--------|---------|
| 数据来源 | 指定数据集 | 自动观察data_storage |
| 技能选择 | 手动指定领域 | 自动匹配技能 |
| 执行方式 | 完整工作流程 | 顺序执行技能 |
| 人类反馈 | 多个检查点 | 无需人工干预 |
| 输出格式 | 指定格式 | HTML综合报告 |
| 灵活性 | 预定义流程 | 动态匹配 |

## 未来扩展

### 技能增强
- 支持更多技能类型
- 智能技能组合推荐
- 技能执行效果评估

### 分析增强
- 跨技能关联分析
- 自动化洞察生成
- 预测性分析集成

### 用户体验
- 交互式执行控制
- 实时进度可视化
- 自定义技能选择
