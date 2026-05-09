# 推荐系统分析技能 (Recommender System Skill)

一个功能完整的智能推荐系统分析工具，基于"数据分析咖哥十话"的推荐系统模块开发。

## 🎯 技能概述

本技能提供从数据处理到推荐生成的完整推荐系统解决方案，支持多种推荐算法、评估方法和可视化分析。

### ✨ 核心特性

- **🤖 多种推荐算法**
  - 基于用户的协同过滤 (User-Based CF)
  - 基于物品的协同过滤 (Item-Based CF)
  - SVD矩阵分解 (Singular Value Decomposition)
  - 混合推荐策略 (Hybrid Recommendation)

- **📊 全面的评估框架**
  - 离线评估指标 (Precision@K, Recall@K, MAE, RMSE)
  - 多种评估方法 (留一法, K折交叉验证, 时间序列验证)
  - 算法性能对比分析

- **📈 丰富的可视化功能**
  - 推荐结果可视化
  - 评估指标图表
  - 用户行为分析图
  - 算法比较图

- **🔧 智能数据处理**
  - 数据质量检查
  - 冷启动问题检测
  - 用户画像分析
  - 数据稀疏性分析

## 🚀 快速开始

### 1. 环境要求

```bash
# 依赖包
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. 基础使用

```python
from scripts.recommendation_engine import RecommendationEngine
from scripts.recommender_evaluator import RecommenderEvaluator
from scripts.recommender_visualizer import RecommenderVisualizer

# 初始化组件
engine = RecommendationEngine()
evaluator = RecommenderEvaluator()
visualizer = RecommenderVisualizer()

# 加载数据
user_data, item_data = engine.load_data('user_behavior.csv', 'item_info.csv')

# 训练模型
engine.train_user_based_cf()
engine.train_item_based_cf()
engine.train_svd(n_components=50)

# 生成推荐
recommendations = engine.recommend_hybrid('U001', top_k=10)
print(f"推荐结果: {recommendations}")
```

### 3. 运行示例

```bash
# 快速测试
python quick_test.py

# 简化示例 (推荐)
python examples/simple_recommendation_example.py

# 完整功能示例
python examples/basic_recommendation_example.py

# 高级功能演示
python examples/advanced_recommendation_example.py
```

## 📁 项目结构

```
recommender-system/
├── SKILL.md                     # 技能详细文档
├── README.md                    # 使用指南 (本文件)
├── quick_test.py               # 快速功能测试
├── test_skill.py               # 完整测试套件
│
├── scripts/                    # 核心功能模块
│   ├── __init__.py
│   ├── recommendation_engine.py    # 推荐算法引擎
│   ├── recommender_evaluator.py    # 评估框架
│   ├── data_analyzer.py            # 数据分析器
│   └── recommender_visualizer.py   # 可视化器
│
└── examples/                   # 示例和数据
    ├── sample_data/            # 样本数据
    │   ├── sample_user_behavior.csv
    │   └── sample_item_info.csv
    ├── simple_recommendation_example.py  # 简化示例
    ├── basic_recommendation_example.py    # 基础示例
    └── advanced_recommendation_example.py # 高级示例
```

## 💡 主要功能

### 1. 推荐算法

#### 基于用户的协同过滤
```python
# 训练模型
engine.train_user_based_cf(similarity_metric='cosine', normalize=True)

# 生成推荐
recommendations = engine.recommend_user_based_cf('U001', top_k=10)
```

#### 基于物品的协同过滤
```python
# 训练模型
engine.train_item_based_cf(similarity_metric='cosine')

# 生成推荐
recommendations = engine.recommend_item_based_cf('U001', top_k=10)
```

#### SVD矩阵分解
```python
# 训练模型
engine.train_svd(n_components=50, random_state=42)

# 生成推荐
recommendations = engine.recommend_svd('U001', top_k=10)
```

#### 混合推荐
```python
# 配置权重
weights = {'user_cf': 0.3, 'item_cf': 0.3, 'svd': 0.4}

# 生成推荐
recommendations = engine.recommend_hybrid('U001', top_k=10, weights=weights)
```

### 2. 系统评估

#### 基本评估指标
```python
# 计算推荐质量
precision = evaluator.precision_at_k(recommendations, ground_truth, k=5)
recall = evaluator.recall_at_k(recommendations, ground_truth, k=5)
f1 = evaluator.f1_score_at_k(recommendations, ground_truth, k=5)

# 计算评分准确性
mae = evaluator.mean_absolute_error(predictions, actual_ratings)
rmse = evaluator.root_mean_square_error(predictions, actual_ratings)
```

#### 留一法评估
```python
loo_results = evaluator.leave_one_out_evaluation(
    engine, user_item_matrix,
    k_values=[5, 10, 20],
    num_users=50
)
```

#### 算法比较
```python
algorithm_results = {
    'User-CF': user_cf_metrics,
    'Item-CF': item_cf_metrics,
    'SVD': svd_metrics,
    'Hybrid': hybrid_metrics
}

comparison_df = evaluator.compare_algorithms(algorithm_results)
```

### 3. 数据分析

#### 用户行为分析
```python
from scripts.data_analyzer import DataAnalyzer

analyzer = DataAnalyzer()
user_data, item_data = analyzer.load_data('user_behavior.csv', 'item_info.csv')

# 用户行为分析
user_analysis = analyzer.analyze_user_behavior(user_data)

# 商品热度分析
item_analysis = analyzer.analyze_item_popularity(user_data, item_data)

# 冷启动问题检测
cold_start = analyzer.detect_cold_start(user_data)
```

### 4. 可视化分析

#### 推荐结果可视化
```python
# 推荐结果柱状图
fig = visualizer.plot_recommendation_results(
    recommendations, user_id='U001',
    title='用户U001的推荐结果'
)

# 算法比较图
fig = visualizer.plot_algorithm_comparison(comparison_df)

# 用户行为分析图
fig = visualizer.plot_user_behavior_analysis(user_data)
```

## 📊 数据格式

### 用户行为数据 (user_behavior.csv)

```csv
用户ID,商品ID,评分,时间戳,商品类别,商品价格,用户年龄,用户性别,用户城市,行为类型
U001,P001,5,2024-01-01 10:30:00,电子产品,2999,28,男,北京,购买
U002,P002,4,2024-01-02 14:20:00,服装,399,32,女,上海,购买
```

### 商品信息数据 (item_info.csv)

```csv
商品ID,商品名称,商品类别,价格,品牌,上架时间,销量,库存,商品描述,平均评分
P001,iPhone 15 Pro,电子产品,3999,苹果,2023-09-15,1250,89,最新款苹果手机,4.8
P002,优衣库羊毛大衣,服装,599,优衣库,2023-10-01,890,156,秋冬保暖羊毛大衣,4.5
```

## 🎯 应用场景

### 电商推荐
- 商品个性化推荐
- 相似商品推荐
- 跨品类推荐
- 购物车补充推荐

### 内容推荐
- 文章推荐
- 视频推荐
- 音乐推荐
- 课程推荐

### 游戏推荐
- 游戏推荐
- 游戏内物品推荐
- 社交推荐

## ⚙️ 高级配置

### 算法参数调优

```python
# 协同过滤参数
engine.train_user_based_cf(
    similarity_metric='cosine',  # 'cosine' 或 'pearson'
    normalize=True              # 是否标准化
)

# SVD参数
engine.train_svd(
    n_components=50,           # 降维组件数
    random_state=42            # 随机种子
)

# 混合推荐权重
weights = {
    'user_cf': 0.3,    # 用户协同过滤权重
    'item_cf': 0.3,    # 物品协同过滤权重
    'svd': 0.4         # SVD权重
}
```

### 评估参数配置

```python
# 留一法评估
loo_results = evaluator.leave_one_out_evaluation(
    engine, user_item_matrix,
    k_values=[5, 10, 20],    # 评估的K值
    num_users=100,           # 评估用户数
    n_neighbors=50           # 相似邻居数
)

# 交叉验证评估
cv_results = evaluator.cross_validation_evaluation(
    engine, user_item_matrix,
    cv_folds=5,              # 交叉验证折数
    k_values=[5, 10]         # 评估的K值
)
```

## 🐛 常见问题

### Q: 如何处理新用户冷启动问题？
A: 系统自动使用热门商品推荐策略为新用户生成推荐。

### Q: 如何选择最优的算法参数？
A: 建议使用交叉验证评估不同参数组合的效果，选择最优配置。

### Q: 如何处理大规模数据？
A: 可以通过采样减少计算量，或使用增量学习方法。

### Q: 如何提高推荐的多样性？
A: 可以在混合推荐中加入多样性权重，或使用基于内容的推荐。

## 📈 性能优化

- 使用矩阵运算库 (如 numpy, scipy)
- 实现增量学习机制
- 使用缓存机制存储计算结果
- 采用并行计算加速训练过程

## 📚 技术原理

### 协同过滤
基于用户或物品的相似度进行推荐，核心是"相似的人喜欢相似的物品"。

### 矩阵分解
通过降维技术发现用户和商品的隐含特征，解决数据稀疏性问题。

### 混合推荐
结合多种推荐算法的优势，提高推荐的准确性和覆盖率。

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进这个技能。

## 📄 许可证

MIT License

---

## 🎉 开始使用

现在你已经了解了推荐系统技能的所有功能，可以开始使用了：

```bash
# 快速验证功能
python quick_test.py

# 运行示例
python examples/simple_recommendation_example.py
```

享受你的推荐系统之旅！🚀