# RFM客户分群分析基本使用指南
# RFM Customer Segmentation Basic Usage Guide

## 快速开始 | Quick Start

### 1. 准备数据 | Data Preparation

准备包含以下字段的电商订单数据CSV文件：

```csv
订单号,用户码,消费日期,产品码,产品说明,数量,单价
ORD001,USER001,2024-01-15,PROD001,智能手机保护壳,2,29.90
ORD002,USER002,2024-01-16,PROD002,无线耳机,1,199.00
ORD003,USER001,2024-01-20,PROD003,充电宝,1,89.00
```

**必需字段说明：**
- `用户码`: 客户唯一标识符
- `消费日期`: 购买日期 (格式: YYYY-MM-DD)
- `数量`: 购买数量 (必须为正数)
- `单价`: 商品单价 (必须为正数)

### 2. 运行分析 | Run Analysis

```python
# 导入分析引擎
from core_analysis import RFMAnalyzer

# 创建分析器实例
analyzer = RFMAnalyzer()

# 运行完整分析
results, summary = analyzer.run_complete_analysis('your_data.csv')

# 保存结果
results.to_csv('customer_segments.csv', index=False, encoding='utf-8-sig')
```

### 3. 生成可视化报告 | Generate Visualizations

```python
# 导入可视化工具
from visualization import RFMVisualizer

# 创建可视化器
visualizer = RFMVisualizer()

# 生成综合仪表板
visualizer.create_comprehensive_dashboard(results, 'dashboard.png')

# 生成详细图表
visualizer.create_individual_charts(results, 'charts/')
```

### 4. 导出营销报告 | Export Marketing Reports

```python
# 导入报告生成器
from report_generator import RFMReportGenerator

# 创建报告生成器
generator = RFMReportGenerator()

# 生成所有报告
generator.save_all_reports(results)
```

## 命令行使用 | Command Line Usage

### 一键完整分析

```bash
# 创建并运行分析脚本
python -c "
from core_analysis import RFMAnalyzer
from visualization import RFMVisualizer
from report_generator import RFMReportGenerator

# 加载和分析数据
analyzer = RFMAnalyzer()
results, summary = analyzer.run_complete_analysis('电商历史订单.csv')

# 生成可视化
visualizer = RFMVisualizer()
visualizer.create_comprehensive_dashboard(results)

# 生成报告
generator = RFMReportGenerator()
generator.save_all_reports(results)
print('RFM分析完成！')
"
```

### 使用示例数据测试

```bash
# 运行示例
python core_analysis.py
python visualization.py
python report_generator.py
```

## 输出文件说明 | Output Files

分析完成后会生成以下文件：

### 数据文件 | Data Files
- `customer_segments.csv`: 完整的客户分群结果
- `vip_customers_marketing_list.csv`: VIP客户营销清单

### 可视化文件 | Visualization Files
- `rfm_dashboard.png`: 综合分析仪表板
- `charts/`: 详细分析图表文件夹
  - `segment_distribution_detailed.png`: 分群分布详细图
  - `rfm_score_distribution.png`: RFM评分分布图
  - `customer_lifetime_value.png`: 客户生命周期价值分析

### 报告文件 | Report Files
- `reports/executive_summary.md`: 执行摘要报告
- `reports/detailed_analysis.md`: 详细技术分析报告
- `reports/marketing_insights.md`: 营销洞察报告

## 高级用法 | Advanced Usage

### 自定义聚类数

```python
# 指定特定的聚类数量
results, summary = analyzer.run_complete_analysis(
    'data.csv',
    n_clusters=5  # 自定义聚类数
)
```

### 仅运行RFM计算

```python
# 仅计算RFM指标
analyzer = RFMAnalyzer()
original_df, cleaned_df = analyzer.load_and_clean_data('data.csv')
rfm_metrics = analyzer.calculate_rfm_metrics(cleaned_df)
clustered_df = analyzer.apply_clustering(rfm_metrics)
final_df = analyzer.calculate_rfm_scores(clustered_df)
```

### 自定义可视化

```python
# 仅生成特定图表
visualizer = RFMVisualizer()
visualizer._create_segment_distribution_detailed(results)
plt.savefig('custom_segment_chart.png')
```

## 故障排除 | Troubleshooting

### 常见问题 | Common Issues

1. **中文显示问题**
   ```python
   # 设置中文字体
   plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
   ```

2. **数据编码问题**
   ```python
   # 尝试不同编码
   try:
       df = pd.read_csv('data.csv', encoding='utf-8')
   except:
       df = pd.read_csv('data.csv', encoding='gbk')
   ```

3. **内存不足**
   ```python
   # 分批处理大数据
   chunk_size = 10000
   for chunk in pd.read_csv('large_data.csv', chunksize=chunk_size):
       # 处理每个数据块
       pass
   ```

### 数据质量检查

```python
# 检查数据质量
def check_data_quality(df):
    print(f"总订单数: {len(df):,}")
    print(f"唯一客户数: {df['用户码'].nunique():,}")
    print(f"日期范围: {df['消费日期'].min()} 到 {df['消费日期'].max()}")

    # 检查异常值
    negative_qty = len(df[df['数量'] <= 0])
    negative_price = len(df[df['单价'] <= 0])

    print(f"负数量订单: {negative_qty:,}")
    print(f"负价格订单: {negative_price:,}")

    return negative_qty == 0 and negative_price == 0
```

## 性能优化 | Performance Optimization

### 大数据集优化

```python
# 使用数据类型优化
dtypes = {
    '用户码': 'category',
    '产品码': 'category',
    '数量': 'int32',
    '单价': 'float32'
}

df = pd.read_csv('large_data.csv', dtype=dtypes)
```

### 并行处理

```python
# 使用多进程加速聚类计算
from sklearn.cluster import MiniBatchKMeans

# 对于大数据集使用MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=3, batch_size=1000, random_state=42)
```

## API参考 | API Reference

### RFMAnalyzer类

```python
class RFMAnalyzer:
    def __init__(self, chinese_font='SimHei')
    def load_and_clean_data(self, file_path) -> tuple
    def calculate_rfm_metrics(self, df) -> pd.DataFrame
    def apply_clustering(self, rfm_data, n_clusters=None) -> pd.DataFrame
    def calculate_rfm_scores(self, rfm_df) -> pd.DataFrame
    def run_complete_analysis(self, file_path, n_clusters=None) -> tuple
```

### RFMVisualizer类

```python
class RFMVisualizer:
    def __init__(self, chinese_font='SimHei')
    def create_comprehensive_dashboard(self, rfm_df, save_path='rfm_dashboard.png')
    def create_individual_charts(self, rfm_df, output_dir='charts')
```

### RFMReportGenerator类

```python
class RFMReportGenerator:
    def __init__(self)
    def generate_executive_summary(self, rfm_df, summary_stats) -> str
    def generate_detailed_analysis_report(self, rfm_df, original_data=None) -> str
    def generate_marketing_insights(self, rfm_df) -> str
    def export_vip_customer_list(self, rfm_df, top_n=100) -> pd.DataFrame
    def save_all_reports(self, rfm_df, original_data=None, output_dir='reports')
```

---

*更多高级用法请参考各模块的详细文档*