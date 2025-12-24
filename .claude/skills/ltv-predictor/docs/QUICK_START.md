# LTV预测技能快速入门指南

本指南将帮助您在10分钟内快速上手LTV预测技能，完成第一个客户生命周期价值分析。

## 🚀 第一步：环境准备

### 1. 检查Python环境
确保您的系统已安装Python 3.7或更高版本：
```bash
python --version
```

### 2. 安装依赖包
```bash
# 进入技能目录
cd .claude/skills/ltv-predictor

# 安装必需的Python包
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
```

### 3. 验证安装
```bash
# 运行快速验证
python -c "import pandas, sklearn, matplotlib, seaborn; print('✅ 所有依赖包安装成功')"
```

## 📊 第二步：准备数据

### 1. 使用示例数据（推荐新手）
技能已提供示例数据，位于：
```
data/sample_orders.csv
```

### 2. 使用自己的数据
如果您有自己的订单数据，请确保CSV文件包含以下列：

| 列名 | 说明 | 示例 |
|------|------|------|
| 订单号 | 唯一订单编号 | 1001 |
| 产品码 | 产品标识 | PROD001 |
| 消费日期 | 购买时间 | 2022-06-01 09:15 |
| 产品说明 | 产品描述 | 绿联usb分线器 |
| 数量 | 购买数量 | 2 |
| 单价 | 产品单价 | 25.50 |
| 用户码 | 客户唯一标识 | CUST001 |
| 城市 | 客户所在城市 | 北京 |

## 🎯 第三步：运行第一个分析

### 方法1：使用Python脚本（推荐）

创建一个新文件 `my_first_analysis.py`：

```python
from scripts.quick_analysis import quick_ltv_analysis

# 使用示例数据进行分析
results = quick_ltv_analysis(
    file_path='data/sample_orders.csv',
    feature_period_months=3,      # 使用前3个月数据
    prediction_period_months=12,  # 预测后12个月LTV
    output_dir='./my_results',    # 结果保存目录
    generate_charts=True,         # 生成图表
    generate_reports=True         # 生成报告
)

# 查看结果
print("🎉 分析完成！")
print(f"最佳模型: {results['summary']['model_summary']['best_model']}")
print(f"模型R²分数: {results['summary']['model_summary']['best_r2_score']:.4f}")
print(f"分析客户数: {results['summary']['data_summary']['total_customers']}")
print(f"平均LTV: {results['summary']['data_summary']['avg_ltv']:,.0f}")
```

运行脚本：
```bash
python my_first_analysis.py
```

### 方法2：使用命令行工具

```bash
# 基础分析
python scripts/quick_analysis.py analyze data/sample_orders.csv --output-dir ./cli_results

# 查看帮助
python scripts/quick_analysis.py --help
```

## 📈 第四步：查看结果

分析完成后，您将在输出目录中看到以下文件：

### 1. 数据文件
- `rfm_features.csv` - RFM特征和LTV数据
- `models/` - 训练好的模型文件

### 2. 可视化图表
- `charts/01_rfm_distribution.png` - RFM特征分布
- `charts/02_customer_segments.png` - 客户分层分布
- `charts/03_model_performance.png` - 模型性能比较
- `charts/04_feature_importance.png` - 特征重要性
- `charts/06_prediction_analysis.png` - 预测结果分析

### 3. 分析报告
- `reports/ltv_analysis_report.html` - 完整HTML报告
- `reports/ltv_analysis_report.md` - Markdown报告
- `reports/ltv_analysis_report.xlsx` - Excel报告

### 4. 分析摘要
- `analysis_summary.json` - JSON格式结果摘要

## 🔍 第五步：理解结果

### RFM特征说明
- **R值**：客户最近购买距离现在的天数（越小越好）
- **F值**：客户在特征期内的购买次数（越大越好）
- **M值**：客户在特征期内的总消费金额（越大越好）

### 客户价值分层
- **钻石客户**：R、F、M值都很高
- **白金客户**：F、M值较高
- **黄金客户**：中等水平的客户
- **白银客户**：F、M值较低
- **青铜客户**：低价值客户

### 模型性能指标
- **R²分数**：模型解释能力（0-1，越接近1越好）
- **MAE**：平均绝对误差
- **RMSE**：均方根误差
- **MAPE**：平均绝对百分比误差

## 🎯 下一步建议

### 1. 深入学习
- 运行完整示例：`python examples/ecommerce_ltv_analysis.py`
- 查看技术文档：`SKILL.md`

### 2. 实际应用
- 使用自己的订单数据进行分析
- 调整时间窗口参数
- 尝试不同的模型配置

### 3. 高级功能
- 预测新客户LTV：`python examples/quick_ltv_prediction.py`
- 批量预测功能
- 自定义特征工程

## ❓ 常见问题

### Q1: 分析报错怎么办？
**A**: 首先检查：
1. 数据文件路径是否正确
2. 数据格式是否符合要求
3. 所有依赖包是否安装成功

查看详细错误信息，常见问题：
- 中文编码问题：确保文件使用UTF-8编码
- 日期格式问题：检查消费日期列格式
- 内存不足：减少数据量或调整参数

### Q2: 模型准确性不高？
**A**: 尝试以下优化：
1. 增加历史数据量
2. 调整特征计算期和预测期
3. 启用超参数调优
4. 检查数据质量

### Q3: 如何处理新客户？
**A**: 使用 `predict_new_customers` 功能：
```python
from scripts.quick_analysis import predict_new_customers
predict_new_customers('./models', 'new_customers.csv')
```

### Q4: 大数据集处理？
**A**: 对于大数据集：
1. 分批处理数据
2. 关闭可视化生成
3. 使用更简单的模型
4. 增加系统内存

## 🎉 恭喜！

您已成功完成LTV预测技能的入门学习。现在可以：
- 使用示例数据进行各种实验
- 应用到自己的业务数据
- 探索高级功能和配置

更多资源：
- 📖 [完整文档](../SKILL.md)
- 🛠️ [API参考](../README.md)
- 💡 [示例代码](../examples/)

开始您的客户价值分析之旅吧！