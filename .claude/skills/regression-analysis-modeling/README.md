# Regression Analysis & Predictive Modeling Skill

一个用于回归分析和预测建模的综合性Claude Code技能，支持多种算法和自动化机器学习流程。

## 功能特性 | Features

- 🤖 **多算法支持**: 线性回归、决策树、随机森林、梯度提升等多种回归算法
- 🔧 **自动化特征工程**: 时间特征、交互特征自动生成
- 📊 **智能模型评估**: R²、MAE、RMSE等多维度评估指标和诊断分析
- 🎯 **可视化仪表板**: 综合分析图表和模型性能可视化
- 🌏 **中文支持**: 完整支持中文数据和可视化显示
- 📈 **业务洞察**: 特征重要性分析和业务价值解读

## 安装依赖 | Installation

```bash
pip install -r requirements.txt
```

## 快速开始 | Quick Start

### 基本使用

```python
from core_regression import RegressionAnalyzer
from model_evaluation import ModelEvaluator
from prediction_visualizer import PredictionVisualizer

# 1. 数据分析和模型训练
analyzer = RegressionAnalyzer()
analysis_results = analyzer.run_complete_analysis(
    'data.csv',
    'target_column',
    create_interactions=True
)

# 2. 模型评估
evaluator = ModelEvaluator()
comparison_results = evaluator.compare_models(analysis_results['results'])
evaluation_report = evaluator.generate_evaluation_report(analysis_results['results'])

# 3. 可视化分析
visualizer = PredictionVisualizer()
visualizer.create_comprehensive_dashboard(
    analysis_results['results'],
    analysis_results['feature_importance']
)
```

### 销售预测

```python
# 销售预测示例
analyzer = RegressionAnalyzer()
results = analyzer.run_complete_analysis(
    'sales_data.csv',
    'monthly_sales',
    create_time_features=True,
    time_config={
        'date_col': '销售日期',
        'frequency': 'M'
    }
)
```

### 房价预测

```python
# 房价预测示例
results = analyzer.run_complete_analysis(
    'housing_data.csv',
    '房价',
    create_interactions=True
)
```

## 数据格式要求 | Data Format Requirements

### 通用回归分析
```csv
feature1,feature2,feature3,target_variable
value1,value2,value3,target_value
...
```

### 销售预测
```csv
日期,产品类别,销售额,促销活动,节假日
2024-01-01,电子产品,15000,False,False
2024-01-02,电子产品,12000,False,False
...
```

### 房价预测
```csv
房屋ID,面积,房间数,卫生间数,楼层,总楼层,建造年份,地铁距离,房价
1,120,3,2,15,30,2010,500,850000
...
```

## 文件结构 | File Structure

```
regression-analysis-modeling/
├── SKILL.md                    # 技能说明文档
├── core_regression.py          # 核心回归分析引擎
├── feature_engineering.py      # 特征工程工具
├── model_evaluation.py         # 模型评估与比较
├── prediction_visualizer.py    # 预测结果可视化
├── requirements.txt            # Python依赖包
├── README.md                   # 使用说明
├── examples/
│   ├── housing_price_example.py       # 房价预测示例
│   └── sample_housing_data.csv       # 房价示例数据
└── templates/
    ├── regression_report_template.md # 回归分析报告模板
    └── model_comparison_template.md  # 模型比较模板
```

## 输出文件 | Output Files

### 数据文件
- `model_results.csv`: 完整模型预测结果
- `feature_importance.csv`: 特征重要性排名
- `model_comparison.csv`: 模型性能比较

### 可视化文件
- `regression_dashboard.png`: 综合分析仪表板
- `model_comparison.png`: 模型性能对比图
- `learning_curves.png`: 学习曲线分析图
- `{model_name}_residual_analysis.png`: 残差诊断图

### 报告文件
- `model_evaluation_report.md`: 详细评估报告
- `regression_analysis_report.md`: 综合分析报告

## 核心功能模块 | Core Modules

### 1. 回归算法引擎 (core_regression.py)
```python
class RegressionAnalyzer:
    def load_and_validate_data()      # 数据加载与验证
    def preprocess_data()             # 数据预处理
    def encode_categorical_features()  # 分类特征编码
    def create_interaction_features() # 交互特征生成
    def train_models()                # 多模型训练
    def get_feature_importance()      # 特征重要性分析
```

### 2. 特征工程工具 (feature_engineering.py)
```python
class FeatureEngineering:
    def extract_temporal_features()   # 时间特征提取
    def create_polynomial_features()  # 多项式特征
    def create_aggregation_features() # 聚合特征
    def create_ratio_features()       # 比例特征
    def select_features()             # 特征选择
```

### 3. 模型评估系统 (model_evaluation.py)
```python
class ModelEvaluator:
    def calculate_comprehensive_metrics()  # 综合指标计算
    def perform_residual_analysis()        # 残差分析
    def compare_models()                   # 模型比较
    def analyze_learning_curves()          # 学习曲线分析
    def generate_evaluation_report()       # 评估报告生成
```

### 4. 可视化工具 (prediction_visualizer.py)
```python
class PredictionVisualizer:
    def create_comprehensive_dashboard()  # 综合仪表板
    def create_individual_analysis_plots() # 详细分析图
    def _plot_model_performance_comparison() # 性能比较图
    def _plot_feature_importance()         # 特征重要性图
```

## 支持的算法 | Supported Algorithms

### 线性模型
- **线性回归**: 基础回归模型，支持系数解释
- **岭回归**: L2正则化，防止过拟合
- **Lasso回归**: L1正则化，特征选择

### 树模型
- **决策树回归**: 非线性关系建模
- **随机森林**: 集成学习，提高稳定性
- **梯度提升**: 逐步优化，高精度预测

## 评估指标 | Evaluation Metrics

### 准确性指标
- **R²分数**: 模型解释方差比例
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **MAPE**: 平均绝对百分比误差

### 诊断指标
- **残差分析**: 模型假设检验
- **学习曲线**: 过拟合/欠拟合诊断
- **特征重要性**: 预测因子排序
- **交叉验证**: 模型稳定性评估

## 使用场景 | Use Cases

### 商业分析
- **销售预测**: 基于历史数据预测未来销售趋势
- **风险评估**: 多维度风险评分模型

### 房地产分析
- **房价预测**: 基于房屋特征和市场因素预测价格
- **租金评估**: 租金定价策略优化
- **投资回报**: 房地产投资收益预测

### 运营优化
- **需求预测**: 库存管理和资源优化
- **价格策略**: 动态定价模型

## 高级用法 | Advanced Usage

### 自定义特征工程

```python
from feature_engineering import FeatureEngineering

fe = FeatureEngineering()

# 时间特征提取
X_temporal = fe.extract_temporal_features(df, ['date_column'])

# 特征选择
X_selected, scores = fe.select_features(X, y, method='f_regression', k=10)
```

### 详细模型诊断

```python
from model_evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# 残差分析
residual_results = evaluator.perform_residual_analysis(
    y_test, y_pred, "Random Forest"
)

# 学习曲线分析
learning_results = evaluator.analyze_learning_curves(
    model, X, y, cv=5
)
```

### 自定义可视化

```python
from prediction_visualizer import PredictionVisualizer

visualizer = PredictionVisualizer()

# 综合仪表板
visualizer.create_comprehensive_dashboard(
    model_results, feature_importance
)

# 详细分析图
visualizer.create_individual_analysis_plots(
    model_results, output_dir='analysis_plots'
)
```

## 故障排除 | Troubleshooting

### 常见问题

1. **内存不足**
   ```python
   # 分批处理大数据
   chunk_size = 10000
   for chunk in pd.read_csv('large_data.csv', chunksize=chunk_size):
       # 处理每个数据块
       pass
   ```

2. **中文显示问题**
   ```python
   plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
   ```

3. **数据编码问题**
   ```python
   df = pd.read_csv('data.csv', encoding='utf-8-sig')  # 或 'gbk'
   ```

4. **模型过拟合**
   ```python
   # 增加正则化
   model = Ridge(alpha=10.0)
   # 或使用交叉验证
   model = RandomForestRegressor(max_depth=5)
   ```

## 性能优化 | Performance Optimization

### 大数据优化
```python
# 使用更高效的数据类型
dtypes = {
    'user_id': 'category',
    'product_id': 'category',
    'amount': 'float32'
}
df = pd.read_csv('data.csv', dtype=dtypes)
```

### 并行处理
```python
# 使用多进程
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)
```

## 最佳实践 | Best Practices

### 数据质量
- 确保数据完整性，处理缺失值
- 检查异常值和离群点
- 验证特征的业务合理性

### 模型选择
- 始终使用交叉验证评估模型
- 比较多个算法，选择最适合的
- 考虑模型的可解释性需求

### 特征工程
- 理解业务背景，创造有意义的特征
- 避免数据泄露（target leakage）
- 保持特征的可解释性

## 版本历史 | Version History

- **v1.0** (2024-12): 初始版本发布
  - 多算法回归分析框架
  - 自动化特征工程
  - 综合评估和可视化系统
  - 完整的中文支持

---

*由回归分析与预测建模系统支持 | Powered by Regression Analysis Engine*