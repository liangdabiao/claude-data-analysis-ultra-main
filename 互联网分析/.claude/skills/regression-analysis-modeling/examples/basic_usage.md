# 回归分析与预测建模基本使用指南
# Regression Analysis & Predictive Modeling Basic Usage Guide

## 快速开始 | Quick Start

### 1. 环境准备 | Environment Setup

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 基本回归分析 | Basic Regression Analysis

```python
# 导入核心模块
from core_regression import RegressionAnalyzer

# 创建分析器
analyzer = RegressionAnalyzer()

# 运行完整分析
results = analyzer.run_complete_analysis(
    'data.csv',           # 数据文件路径
    'target_column'       # 目标变量列名
)

# 查看结果
print(f"最佳模型: {analyzer.best_model_name}")
print(f"R² 分数: {results['summary']['best_r2']:.4f}")
```

### 3. 房价预测 | Housing Price Prediction

```python
# 房价预测示例
analyzer = RegressionAnalyzer()

# 启用交互特征
housing_results = analyzer.run_complete_analysis(
    'housing_data.csv',
    '房价',
    create_interactions=True
)

# 获取特征重要性
feature_importance = analyzer.get_feature_importance()
print(feature_importance.head(10))
```

## 完整分析流程 | Complete Analysis Pipeline

### 步骤1：数据准备 | Data Preparation

```python
# 加载和验证数据
X, y = analyzer.load_and_validate_data('data.csv', 'target')

# 数据预处理
X_clean, y_clean = analyzer.preprocess_data(X, y)

# 编码分类特征
X_encoded = analyzer.encode_categorical_features(X_clean)
```

### 步骤2：特征工程 | Feature Engineering

```python
# 创建交互特征
X_interactions = analyzer.create_interaction_features(X_encoded)

# 或者使用独立特征工程模块
from feature_engineering import FeatureEngineering

fe = FeatureEngineering()

# 时间特征提取
X_temporal = fe.extract_temporal_features(df, ['date_column'])

# 多项式特征
X_polynomial = fe.create_polynomial_features(X, degree=2)

# 特征选择
X_selected, scores = fe.select_features(X, y, k=15)
```

### 步骤3：模型训练 | Model Training

```python
# 训练多个模型
results = analyzer.train_models(X_final, y)

# 自动选择最佳模型
best_model = analyzer.best_model
best_model_name = analyzer.best_model_name
```

### 步骤4：模型评估 | Model Evaluation

```python
from model_evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# 计算综合指标
metrics = evaluator.calculate_comprehensive_metrics(y_test, y_pred, "Model Name")

# 残差分析
residual_results = evaluator.perform_residual_analysis(y_test, y_pred, "Model Name")

# 模型比较
comparison = evaluator.compare_models(results)

# 学习曲线分析
learning_results = evaluator.analyze_learning_curves(model, X, y)
```

### 步骤5：结果可视化 | Result Visualization

```python
from prediction_visualizer import PredictionVisualizer

visualizer = PredictionVisualizer()

# 综合仪表板
visualizer.create_comprehensive_dashboard(
    model_results,
    feature_importance
)

# 详细分析图表
visualizer.create_individual_analysis_plots(model_results)
```

## 数据格式要求 | Data Format Requirements

### 通用格式 | General Format

```csv
feature1,feature2,feature3,target_variable
value1,value2,value3,target_value
value1,value2,value3,target_value
...
```

### 房价预测数据 | Housing Price Data

```csv
房屋ID,面积,房间数,卫生间数,楼层,建造年份,地铁距离,装修等级,朝向,房价
1,120,3,2,15,2010,500,精装修,南,850000
2,85,2,1,8,2015,300,简装修,东,520000
...
```

## 高级功能 | Advanced Features

### 自定义特征工程 | Custom Feature Engineering

```python
# 创建比例特征
ratio_pairs = [
    ('area', 'rooms', 'area_per_room'),
    ('income', 'family_size', 'income_per_person')
]

X_ratios = fe.create_ratio_features(X, ratio_pairs)

# 创建分箱特征
binning_config = {
    'age': {'type': 'uniform', 'bins': 5},
    'income': {'type': 'quantile', 'bins': 4}
}

X_binned = fe.create_binning_features(X, binning_config)
```

### 模型微调 | Model Fine-tuning

```python
# 特征缩放
X_scaled = fe.scale_features(X, method='standard')

# 重新训练
results = analyzer.train_models(X_scaled, y)
```

### 业务洞察提取 | Business Insights Extraction

```python
# 特征重要性解释
feature_importance = analyzer.get_feature_importance(top_n=10)

for idx, row in feature_importance.iterrows():
    feature_name = row['feature']
    importance = row['importance']

    # 业务解释
    business_insight = translate_feature_to_business(feature_name)
    print(f"{business_insight}: 重要性 {importance:.4f}")

def translate_feature_to_business(feature_name):
    """将技术特征名转换为业务语言"""
    translations = {
        'R_最近购买天数': '客户活跃度',
        'F_购买频次': '购买频率',
        'M_总消费金额': '消费能力',
        'area': '房屋面积',
        'subway_distance': '交通便利性',
        'decoration_level': '装修质量'
    }
    return translations.get(feature_name, feature_name)
```

## 输出文件说明 | Output Files

### 数据文件 | Data Files
- `model_results.csv`: 完整预测结果
- `feature_importance.csv`: 特征重要性排名
- `model_comparison.csv`: 模型性能对比

### 可视化文件 | Visualization Files
- `regression_dashboard.png`: 综合分析仪表板
- `model_comparison.png`: 模型性能对比
- `learning_curves.png`: 学习曲线分析
- `*_residual_analysis.png`: 残差诊断图

### 报告文件 | Report Files
- `model_evaluation_report.md`: 详细评估报告
- `regression_analysis_report.md`: 综合分析报告

## 性能优化 | Performance Optimization

### 大数据处理 | Big Data Processing

```python
# 分块读取
chunk_size = 10000
for chunk in pd.read_csv('large_data.csv', chunksize=chunk_size):
    # 处理每个数据块
    processed_chunk = preprocess_chunk(chunk)
    results.append(processed_chunk)

# 使用高效数据类型
dtypes = {
    'category_col': 'category',
    'numeric_col': 'float32'
}
df = pd.read_csv('data.csv', dtype=dtypes)
```

### 并行计算 | Parallel Computing

```python
# 使用多核处理
from sklearn.model_selection import cross_val_score

# 并行交叉验证
scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)
```

## 故障排除 | Troubleshooting

### 常见问题 | Common Issues

1. **内存不足**
   ```python
   # 减少特征数量
   X_selected = X[['most_important_features']]

   # 或使用增量学习
   from sklearn.linear_model import SGDRegressor
   model = SGDRegressor()
   ```

2. **模型过拟合**
   ```python
   # 增加正则化
   from sklearn.linear_model import Ridge
   model = Ridge(alpha=10.0)

   # 减少特征
   from sklearn.feature_selection import SelectKBest
   selector = SelectKBest(k=10)
   ```

3. **中文显示问题**
   ```python
   import matplotlib.pyplot as plt
   plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
   plt.rcParams['axes.unicode_minus'] = False
   ```

4. **数据编码问题**
   ```python
   # 尝试不同编码
   try:
       df = pd.read_csv('data.csv', encoding='utf-8')
   except:
       df = pd.read_csv('data.csv', encoding='gbk')
   ```

## 最佳实践 | Best Practices

### 数据质量 | Data Quality
- 检查并处理缺失值
- 识别和处理异常值
- 验证数据的业务合理性

### 特征工程 | Feature Engineering
- 理解业务背景，创造有意义特征
- 避免数据泄露
- 保持特征的可解释性

### 模型选择 | Model Selection
- 使用交叉验证评估模型
- 比较多个算法
- 考虑模型的可解释性需求

### 结果验证 | Result Validation
- 检查残差分布
- 分析特征重要性的合理性
- 与业务专家验证结果

---

*更多高级用法请参考各模块的详细文档*