# 数据探索可视化技能

一个基于《数据分析咖哥十话》第2课理论的自动化数据探索和可视化工具，提供从数据加载到专业分析报告生成的完整EDA解决方案。

## ✨ 核心功能

### 🔍 智能数据探索
- **自动数据诊断**: 检测数据质量问题、异常值和缺失值模式
- **统计描述分析**: 生成全面的统计摘要和分布特征
- **相关性分析**: 识别特征间关系和依赖模式
- **数据质量报告**: 专业级数据质量评估和建议

### 📊 专业可视化生成
- **分布可视化**: 直方图、密度图、小提琴图、QQ图
- **统计可视化**: 箱线图、误差条图、置信区间图
- **关系可视化**: 散点图、热图、配对图、3D散点图
- **专门图表**: ROC曲线、混淆矩阵、特征重要性图
- **交互式图表**: Plotly驱动的动态可视化

### 🏥 医疗数据专精
- **医疗编码支持**: ICD-10、SNOMED CT等医疗标准
- **生物标记物分析**: 专门的医学指标处理
- **诊断模型构建**: 医疗预测模型和评估
- **医学可解释性**: 符合医学实践的解释框架

### 🤖 自动化建模评估
- **多算法支持**: 逻辑回归、随机森林、XGBoost、神经网络
- **自动特征工程**: 特征选择、转换和优化
- **超参数调优**: 网格搜索和贝叶斯优化
- **模型可解释性**: SHAP值、特征重要性、部分依赖图

### 📋 专业报告生成
- **HTML报告**: 可发表级交互式分析报告
- **PDF导出**: 高质量文档格式输出
- **Markdown支持**: 轻量级报告格式
- **自定义模板**: 可配置的报告模板系统

## 🚀 快速开始

### 1. 环境安装

```bash
# 安装基础依赖
pip install pandas numpy matplotlib seaborn scipy scikit-learn plotly jinja2

# 安装可选依赖（用于高级功能）
pip install xgboost lightgbm shap imbalanced-learn weasyprint
```

### 2. 快速测试

```bash
# 运行快速测试验证功能
python quick_test.py
```

### 3. 基础使用

```python
from scripts.eda_analyzer import EDAAnalyzer
from scripts.visualizer import DataVisualizer
from scripts.modeling_evaluator import ModelingEvaluator
from scripts.report_generator import ReportGenerator

# 1. 加载和分析数据
analyzer = EDAAnalyzer()
data = analyzer.load_data('your_data.csv')
eda_results = analyzer.auto_eda(data)

# 2. 生成可视化
visualizer = DataVisualizer()
charts = visualizer.auto_visualize(data, target_col='your_target')

# 3. 建模分析
modeler = ModelingEvaluator()
model_results = modeler.auto_modeling(data, target_col='your_target')

# 4. 生成报告
generator = ReportGenerator()
report_path = generator.generate_comprehensive_report(
    data=data,
    eda_results=eda_results,
    model_results=model_results,
    output_path='analysis_report.html'
)
```

## 📁 项目结构

```
data-exploration-visualization/
├── scripts/                    # 核心功能模块
│   ├── eda_analyzer.py        # EDA分析器
│   ├── visualizer.py          # 可视化生成器
│   ├── data_preprocessor.py   # 数据预处理器
│   ├── modeling_evaluator.py  # 建模评估器
│   └── report_generator.py    # 报告生成器
├── examples/                   # 示例脚本
│   ├── quick_start_example.py # 快速开始示例
│   ├── medical_data_analysis.py # 医疗数据分析
│   └── financial_data_analysis.py # 金融数据分析
├── SKILL.md                   # 技能说明文档
├── quick_test.py              # 快速测试脚本
└── README.md                  # 项目说明
```

## 🎯 使用场景

### 🏥 医疗健康领域
- **疾病预测**: 基于临床数据的疾病风险预测
- **诊断辅助**: 医学影像和检验结果分析
- **流行病学研究**: 疫情数据分析和趋势预测
- **临床试验**: 试验数据统计分析和可视化

### 💰 金融风控领域
- **信用评估**: 个人和企业信用风险建模
- **欺诈检测**: 异常交易模式识别
- **投资分析**: 市场趋势和风险评估
- **合规报告**: 监管要求的分析报告

### 🛒 电商零售领域
- **用户分析**: 客户行为和偏好分析
- **销售预测**: 销量预测和库存优化
- **推荐系统**: 个性化推荐算法评估
- **市场细分**: 客户群体分析和画像

### 🎓 科研教育领域
- **学术研究**: 数据驱动的学术研究支持
- **教学案例**: 数据分析教学和实践
- **论文写作**: 研究数据分析和图表制作
- **技能培训**: 数据科学技能培训工具

## 🔧 配置选项

### EDA分析器配置
```python
config = {
    'max_categories': 20,        # 分类变量最大显示类别数
    'correlation_threshold': 0.5, # 相关性阈值
    'outlier_detection': True,   # 是否检测异常值
    'statistical_tests': True    # 是否进行统计检验
}
```

### 可视化器配置
```python
config = {
    'figure_size': (12, 8),     # 默认图表尺寸
    'style': 'seaborn-v0_8',    # 图表样式
    'color_palette': 'husl',     # 颜色调色板
    'interactive_charts': True,  # 是否生成交互式图表
    'save_format': 'png'         # 保存格式
}
```

### 建模评估器配置
```python
config = {
    'cv_folds': 5,                     # 交叉验证折数
    'scoring_metric': 'accuracy',      # 评估指标
    'enable_hyperparameter_tuning': True, # 是否调参
    'n_iter_search': 50,               # 搜索迭代次数
    'ensemble_models': True            # 是否使用集成方法
}
```

### 报告生成器配置
```python
config = {
    'report_title': '数据分析报告',     # 报告标题
    'author': '数据分析助手',         # 作者
    'theme': 'modern',                # 主题样式
    'include_toc': True,              # 是否包含目录
    'medical_specialization': False   # 是否医疗专化
}
```

## 📊 示例数据

### 医疗数据示例
```python
# 模拟医疗数据
medical_data = {
    'patient_id': ['P001', 'P002', ...],
    'diagnosis': ['Malignant', 'Benign', ...],
    'radius_mean': [17.99, 20.57, ...],
    'texture_mean': [10.38, 17.77, ...],
    'perimeter_mean': [122.8, 132.9, ...]
}
```

### 金融数据示例
```python
# 模拟金融数据
financial_data = {
    'customer_id': ['C001', 'C002', ...],
    'credit_score': [720, 680, ...],
    'income': [85000, 62000, ...],
    'debt_ratio': [0.15, 0.32, ...],
    'default': [0, 1, ...]
}
```

## 🧪 运行示例

### 快速开始示例
```bash
python examples/quick_start_example.py
```

### 医疗数据分析示例
```bash
python examples/medical_data_analysis.py
```

### 金融数据分析示例
```bash
python examples/financial_data_analysis.py
```

## 📋 常见问题

### Q: 如何处理中文数据？
A: 技能自动检测和处理中文编码，支持UTF-8、GBK等多种编码格式。

### Q: 支持哪些数据格式？
A: 支持CSV、Excel、JSON、Parquet等常见格式，也支持数据库连接。

### Q: 如何自定义可视化样式？
A: 可以通过配置文件自定义颜色、字体、图表布局等样式参数。

### Q: 模型准确性如何保证？
A: 技能采用交叉验证、多种评估指标和集成方法来确保模型的可靠性和泛化能力。

### Q: 如何处理大数据集？
A: 技能自动采样大数据集，并提供内存优化建议。对于超大数据集，建议使用分布式处理框架。

## ⚡ 性能优化

### 内存优化
- 自动数据采样减少内存占用
- 智能分块处理大文件
- 垃圾回收优化

### 计算优化
- 并行处理提升计算速度
- 缓存机制避免重复计算
- 增量更新支持

### 可视化优化
- 大数据集采样显示
- 图表渲染优化
- 交互式图表懒加载

## 🔄 更新日志

### v1.0.0 (2025-01-19)
- 初始版本发布
- 完整的EDA功能
- 基础可视化支持
- 逻辑回归建模
- HTML报告生成

### 未来计划
- 支持更多机器学习算法
- 增加深度学习模型支持
- 扩展医疗数据分析功能
- 云端部署支持
- 实时数据分析能力

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出改进建议：

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 🙏 致谢

- 《数据分析咖哥十话》提供的理论指导
- Scikit-learn、Pandas、Plotly 等优秀开源库
- 数据科学社区的支持和反馈

---

通过这个技能，您可以大幅提升数据分析效率，从重复性工作中解放出来，专注于洞察发现和决策支持。