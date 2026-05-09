# RFM Customer Segmentation Analysis Skill

一个用于电商客户分群和RFM分析的综合性Claude Code技能，支持中文数据处理和智能聚类。

## 功能特性 | Features

- 🎯 **智能RFM分析**: 自动计算Recency、Frequency、Monetary指标
- 📊 **K-means聚类**: 自动确定最优聚类数，无需手动调参
- 🌏 **中文支持**: 完整支持中文数据字段和可视化显示
- 📈 **可视化仪表板**: 自动生成综合分析图表和报告
- 🎯 **营销洞察**: 提供可操作的营销建议和VIP客户清单
- 🔧 **即用型设计**: 一键式数据处理流水线

## 安装依赖 | Installation

```bash
pip install -r requirements.txt
```

## 快速开始 | Quick Start

### 基本使用

```python
from core_analysis import RFMAnalyzer
from visualization import RFMVisualizer
from report_generator import RFMReportGenerator

# 1. 数据分析
analyzer = RFMAnalyzer()
results, summary = analyzer.run_complete_analysis('your_data.csv')

# 2. 生成可视化
visualizer = RFMVisualizer()
visualizer.create_comprehensive_dashboard(results)

# 3. 生成报告
generator = RFMReportGenerator()
generator.save_all_reports(results)
```

### 数据格式要求

CSV文件需要包含以下字段：

```csv
订单号,用户码,消费日期,产品码,产品说明,数量,单价,城市
ORD001,USER001,2024-01-15,PROD001,智能手机保护壳,2,29.90,北京
```

**必需字段：**
- `用户码`: 客户唯一标识
- `消费日期`: 购买日期 (YYYY-MM-DD)
- `数量`: 购买数量 (必须为正数)
- `单价`: 商品单价 (必须为正数)

## 文件结构 | File Structure

```
rfm-customer-segmentation/
├── SKILL.md                    # 技能说明文档
├── core_analysis.py            # 核心RFM分析引擎
├── visualization.py            # 可视化工具
├── report_generator.py         # 报告生成器
├── requirements.txt            # Python依赖包
├── README.md                   # 使用说明
├── examples/
│   ├── sample_data.csv         # 示例数据
│   └── basic_usage.md          # 使用指南
└── templates/
    ├── analysis_template.md    # 分析报告模板
    └── vip_list_template.csv   # VIP客户模板
```

## 输出文件 | Output Files

### 数据文件
- `customer_segments.csv`: 完整客户分群结果
- `vip_customers_marketing_list.csv`: VIP客户营销清单

### 可视化文件
- `rfm_dashboard.png`: 综合分析仪表板
- `charts/`: 详细分析图表

### 报告文件
- `reports/executive_summary.md`: 执行摘要
- `reports/detailed_analysis.md`: 详细技术分析
- `reports/marketing_insights.md`: 营销洞察报告

## 核心功能 | Core Features

### 1. RFM指标计算
- **Recency (R)**: 客户最近购买时间
- **Frequency (F)**: 客户购买频次
- **Monetary (M)**: 客户消费金额

### 2. 智能聚类
- 自动使用肘部法则确定最优聚类数
- K-means算法进行客户分群
- 标准化处理确保各指标权重一致

### 3. 客户价值评分
- 1-3分制评分系统
- 基于33%和66%分位数划分
- 综合评分确定客户价值等级

### 4. 可视化分析
- 客户分群分布饼图
- 收入份额分析图
- RFM散点图和箱线图
- 客户价值相关性热力图

### 5. 营销支持
- VIP客户识别和排名
- 针对性营销策略建议
- 客户激活和挽留方案
- ROI预测和KPI监控

## 使用场景 | Use Cases

### 电商客户分析
- 识别高价值VIP客户
- 制定个性化营销策略
- 客户流失预警和挽留
- 提升客户终身价值

### 营销活动优化
- 精准营销目标客户筛选
- 营销预算优化分配
- 活动效果预测和评估
- A/B测试客户分组

### 客户关系管理
- 客户生命周期管理
- 客户价值分层服务
- 客户满意度提升策略
- 客户推荐和奖励机制

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
    def generate_executive_summary(self, rfm_df, summary_stats) -> str
    def generate_detailed_analysis_report(self, rfm_df, original_data=None) -> str
    def generate_marketing_insights(self, rfm_df) -> str
    def export_vip_customer_list(self, rfm_df, top_n=100) -> pd.DataFrame
```

## 故障排除 | Troubleshooting

### 常见问题

1. **中文字体显示问题**
   ```python
   plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
   ```

2. **数据编码问题**
   ```python
   df = pd.read_csv('data.csv', encoding='utf-8-sig')  # 或 'gbk'
   ```

3. **依赖包安装问题**
   ```bash
   pip install --upgrade pandas scikit-learn matplotlib seaborn
   ```

## 最佳实践 | Best Practices

### 数据准备
- 确保数据质量，去除异常订单
- 统一日期格式
- 客户ID保持唯一性

### 分析优化
- 至少1000个订单用于可靠聚类
- 定期更新客户分群（建议月度）
- 结合业务背景解释结果

### 营销应用
- 建立客户分群效果监控机制
- 持续优化营销策略
- 测试不同触达方式的效果

## 版本历史 | Version History

- **v1.0** (2024-12): 初始版本发布
  - 基础RFM分析功能
  - K-means聚类算法
  - 可视化仪表板
  - 营销报告生成

## 技术支持 | Support

如有问题或建议，请：
1. 检查数据格式是否符合要求
2. 确认依赖包已正确安装
3. 查看错误日志信息
4. 参考示例数据和代码

---

*由RFM客户分群分析系统支持 | Powered by RFM Customer Segmentation Engine*