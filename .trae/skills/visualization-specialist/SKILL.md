---
name: "visualization-specialist"
description: "Creates data visualizations, charts, and interactive dashboards. Invoke when user wants to create plots, graphs, or visual representations of data."
---

# Visualization Specialist

Expert data visualization specialist for creating interactive, insightful, and publication-quality visualizations.

## When to Invoke This Skill

Invoke this skill when user:
- Wants to create data visualizations or charts
- Needs to visualize patterns or trends
- Wants interactive dashboards
- Needs publication-quality plots
- Asks for specific chart types (bar, line, scatter, etc.)
- Needs data story telling through visuals
- Specifies a chart type (all, trends, distribution, correlation, comparison)

## Chart Types (Advanced Mode)

用户可以指定图表类型：

### 1. all (完整仪表板)
创建包含多种图表类型的综合仪表板：
- 数据概览
- 关键变量可视化
- 交互式探索仪表板

### 2. trends (趋势分析)
时间序列相关图表：
- 折线图
- 移动平均图
- 趋势分解图
- 季节性分析图

### 3. distribution (分布分析)
分布相关图表：
- 直方图
- 密度图
- 箱线图
- 小提琴图

### 4. correlation (相关性分析)
相关性可视化：
- 散点图
- 相关性热力图
- 配对图

### 5. comparison (对比分析)
对比类图表：
- 分组条形图
- 堆叠条形图
- 对比折线图

### 6. custom (自定义)
根据用户需求创建特定图表

## Core Capabilities

### Visualization Types
- **Statistical Charts**: Histograms, box plots, scatter plots, correlation matrices
- **Time Series**: Line charts, area charts, candlestick charts
- **Categorical Data**: Bar charts, pie charts, heatmaps, treemaps
- **Distribution Analysis**: Density plots, violin plots, Q-Q plots
- **Multivariate Data**: Parallel coordinates, radar charts, bubble charts
- **Geographic Data**: Choropleth maps, point maps
- **Comparative Analysis**: Side-by-side charts, small multiples

### Design Principles
- **Data-Ink Ratio**: Maximize data-ink, minimize chart junk
- **Color Theory**: Use appropriate, accessible color schemes
- **Accessibility**: Ensure colorblind-friendly designs
- **Labeling**: Clear, concise labels and titles
- **Scale**: Appropriate scaling for data

### Technical Skills
- **Matplotlib/Seaborn**: Static visualizations
- **Plotly**: Interactive web visualizations
- **Pandas**: Built-in plotting

## Chart Selection Guide

### For Numerical Data
- **Distribution**: Histogram, box plot, violin plot, density plot
- **Comparison**: Bar chart, line chart, scatter plot
- **Relationship**: Scatter plot, correlation heatmap
- **Trend**: Line chart, area chart

### For Categorical Data
- **Frequency**: Bar chart, pie chart
- **Comparison**: Grouped bar chart, stacked bar chart
- **Relationship**: Heatmap, mosaic plot

### For Time Series
- **Trend**: Line chart, area chart
- **Seasonality**: Seasonal decomposition
- **Comparison**: Multiple line charts

## Chinese Font Support

**IMPORTANT**: When creating visualizations with Chinese text, always configure proper fonts:

```python
import matplotlib.pyplot as plt
import matplotlib

# Windows
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
# Mac
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
# Linux
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei']

# Must have this to show minus signs correctly
matplotlib.rcParams['axes.unicode_minus'] = False
```

## Usage Examples

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# Load data
df = pd.read_csv('./data_storage/your_data.csv')

# Create visualization
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data=df, x='column_name', kde=True, ax=ax)
ax.set_title('数据分布图', fontsize=14)
ax.set_xlabel('列名', fontsize=12)
ax.set_ylabel('频数', fontsize=12)
plt.tight_layout()
plt.savefig('./visualizations/distribution.png', dpi=300, bbox_inches='tight')
```

## Output Standards

### File Formats
- **Static Images**: PNG (300 dpi), SVG, PDF
- **Interactive**: HTML (Plotly)
- **Output Directory**: `./visualizations/`

### Quality Requirements
- High resolution (300 dpi for static)
- Proper Chinese labels and titles
- Clear legends and annotations
- Consistent color schemes
- Responsive layout

## Collaboration

Work with other skills:
- **data-explorer**: Get statistical insights to visualize
- **report-writer**: Supply visualizations for reports
- **code-generator**: Generate reusable plotting code

## Language

All visualization labels, titles, and annotations must be in **Chinese**:
- Chart titles
- Axis labels
- Legend text
- Annotations and tooltips
