# Claude Data Analysis Assistant

A modern, intelligent data analysis platform built with Claude Code's sub-agents, slash-commands, skills, and hooks. Transform your data analysis workflow with AI-powered assistance and specialized analysis tools.

简单的一句话： 2个命令， /do-all 常规数据分析 ； /do-more 互联网数据分析 。 而分析数据是放在 /data_storage 。就这么简单，用起来吧！

注意： 下载项目下来，分析数据是放在 /data_storage [删去原来的demo数据] ，你需要先删除 complete_analysis 和 do_more_analysis 这两个文件夹。我这里放着是给你参考最终的分析结果，作为例子。

## 🚀 Quick Start

### 1. Set Up Your Data
Place your dataset in the `data_storage/` directory:
```bash
cp your_data.csv ./data_storage/
```

### 2. Start Analysis
Use intuitive slash commands to analyze your data:

```bash
# Complete interactive workflow with human feedback checkpoints
/do-all

# ⭐ NEW: Automatic multi-skill analysis
/do-more

# Basic exploratory analysis
/analyze user_behavior_sample.csv exploratory

# Create visualizations
/visualize user_behavior_sample.csv all

# Generate analysis code
/generate python data-cleaning

# Create comprehensive report
/report user_behavior_sample.csv complete markdown


```

## 🎯 Key Features

### ⭐ /do-more vs /do-all: Which Should You Use?

#### `/do-more`: Automatic Multi-Skill Analysis
**Best for:** Quick, automated analysis without configuration

```bash
/do-more  # No parameters needed!
```

**What it does:**
- ✅ Automatically scans `data_storage/` directory
- ✅ Identifies data types (e-commerce, user behavior, etc.)
- ✅ Intelligently matches 7+ relevant skills
- ✅ Executes skills in optimal order
- ✅ Generates comprehensive HTML report
- ✅ No human intervention required
- ✅ Fast execution (2-5 minutes)

**Output:** `do_more_analysis/integrated_results/Comprehensive_Analysis_Report.html`

---

#### `/do-all`: Complete Interactive Analysis Workflow
**Best for:** Thorough analysis with human oversight and feedback

```bash
/do-all
```

**What it does:**
- ✅ Reads data from `data_storage/` (no parameters needed!)
- ✅ 6-stage workflow with quality checks
- ✅ **3 Human feedback checkpoints** at critical stages
- ✅ Interactive hypothesis generation
- ✅ Custom code generation
- ✅ Comprehensive documentation
- ✅ Multiple output formats (HTML, PDF, Markdown, DOCX)

**Workflow Stages:**
1. Data Quality Assessment → **⚠️ [human checkpoint #1]** - Confirm data quality
2. Exploratory Analysis - Statistical summaries, patterns, trends
3. Hypothesis Generation → **⚠️ [human checkpoint #2]** - Review research directions
4. Visualization → **⚠️ [human checkpoint #3]** - Approve visualization strategy
5. Code Generation - Reproducible analysis pipeline
6. Report Generation - Comprehensive final report

**Output Directory:**
```
complete_analysis/
├── data_quality_report/          # Stage 1 output
├── exploratory_analysis/         # Stage 2 output
├── hypothesis_reports/           # Stage 3 output
├── visualizations/               # Stage 4 output
├── generated_code/               # Stage 5 output
├── final_report/                 # Stage 6 output
└── workflow_log/                 # Execution logs
```

**Execution Time:** 10-30 minutes (depends on data size)

---

### Comparison Summary

| Feature | `/do-more` | `/do-all` |
|---------|-----------|-----------|
| **Data Source** | Auto-scans data_storage/ | Reads from data_storage/ |
| **Parameters** | None required | None |
| **Human Feedback** | No | Yes (3 checkpoints) |
| **Execution Time** | 2-5 minutes | 10-30 minutes |
| **Skills Used** | 7+ auto-selected | Complete workflow (no skills) |
| **Output Format** | HTML report | Multi-format (HTML/PDF/MD/DOCX) |
| **Code Generation** | No | Yes (complete pipeline) |
| **Analysis Stages** | Integrated execution | 6 separate stages |
| **Interactive** | No | Yes (at checkpoints) |
| **Report Detail** | Comprehensive | Extensive + technical |
| **Best For** | Quick insights | Thorough analysis |
| **Customization** | Automatic | Interactive |

### Specialized Analysis Skills
12 domain-specific skills for expert-level analysis:

**Customer Analysis:**
- `rfm-customer-segmentation` - Customer value segmentation
- `ltv-predictor` - Lifetime value prediction
- `retention-analysis` - Customer retention and churn
- `user-profiling-analysis` - User behavior profiling

**Marketing Analysis:**
- `attribution-analysis-modeling` - Marketing attribution
- `growth-model-analyzer` - Growth hacking analysis
- `ab-testing-analyzer` - A/B test validation
- `funnel-analysis` - Conversion funnels

**Data Analysis:**
- `data-exploration-visualization` - Automated EDA
- `regression-analysis-modeling` - Predictive modeling
- `content-analysis` - Text and NLP analysis
- `recommender-system` - Recommendation engines

### Intelligent Sub-Agents
- **data-explorer**: Expert statistical analysis and pattern discovery
- **visualization-specialist**: Beautiful, insightful charts and graphs
- **code-generator**: Production-ready analysis code
- **report-writer**: Comprehensive analysis reports
- **quality-assurance**: Data validation and quality control
- **hypothesis-generator**: Research hypothesis and insights

### Intuitive Slash Commands
- `/do-more` - **⭐ RECOMMENDED** Automatic multi-skill analysis (no parameters)
- `/do-all` - Complete interactive workflow with human feedback (no parameters)
- `/analyze [dataset] [type]` - Perform data analysis
- `/visualize [dataset] [type]` - Create visualizations
- `/generate [language] [type]` - Generate analysis code
- `/report [dataset] [format]` - Generate reports
- `/quality [dataset] [action]` - Quality assurance
- `/hypothesis [dataset] [domain]` - Generate hypotheses

### Automated Workflows
- **Data Validation**: Automatic quality checks on data upload
- **Smart Context**: Project-aware analysis suggestions
- **Reproducible Analysis**: Complete documentation and code generation
- **Beautiful Reports**: HTML, Markdown, and PDF output formats

## 📊 Usage Examples

### ⭐ Automatic Multi-Skill Analysis
```bash
# Easiest way - no parameters needed!
/do-more

# Output (2-5 minutes):
# do_more_analysis/integrated_results/
# └── Comprehensive_Analysis_Report.html
```

### Interactive Complete Analysis
```bash
# For thorough analysis with human feedback checkpoints
/do-all

# Includes:
# ✓ Data Quality Assessment → [your confirmation]
# ✓ Exploratory Analysis
# ✓ Hypothesis Generation → [your approval]
# ✓ Visualizations → [your review]
# ✓ Code Generation
# ✓ Comprehensive Report
```

### E-commerce Data Analysis
```bash
# Quick automated analysis
/do-more

# Or specific customer analysis
/rfm-customer-segmentation olist_orders.csv
/ltv-predictor order_items.csv
/retention-analysis orders.csv customers.csv
```

### User Behavior Analysis
```bash
# Complete analysis workflow
/analyze user_behavior.csv exploratory
/visualize user_behavior.csv trends
/quality user_behavior.csv clean
/report user_behavior.csv complete html
/generate python user-segmentation
```

### Sales Data Analysis
```bash
# Sales performance analysis
/analyze sales_data.csv statistical
/visualize sales_data.csv trends
/generate sql revenue-analysis
/report sales_data.csv executive pdf
```

### Customer Analytics
```bash
# Customer segmentation
/analyze customer_data.csv predictive
/visualize customer_data.csv distribution
/generate r clustering-analysis
/hypothesis customer_data churn-prediction
```

## 🛠️ Project Structure

```
claude-data-analysis/
├── .claude/
│   ├── agents/          # Sub-agent configurations
│   ├── commands/        # Slash command definitions
│   │   ├── do-more.md   # ⭐ NEW! Automatic multi-skill analysis
│   ├── hooks/          # Automation scripts
│   ├── settings.json   # Claude Code settings
│   └── skills/         # ⭐ 12 Specialized analysis skills
│       ├── rfm-customer-segmentation/
│       ├── ltv-predictor/
│       ├── retention-analysis/
│       ├── funnel-analysis/
│       ├── growth-model-analyzer/
│       ├── content-analysis/
│       └── ... (9 more skills)
├── data_storage/       # Your data files
│   ├── Orders.csv
│   ├── Customers.csv
│   └── ... (Olist datasets included)
├── do_more_analysis/   # ⭐ NEW! /do-more output directory
│   ├── skill_execution/  # Individual skill results
│   │   ├── data-exploration-visualization/
│   │   ├── rfm-customer-segmentation/
│   │   ├── ltv-predictor/
│   │   ├── retention-analysis/
│   │   ├── funnel-analysis/
│   │   ├── growth-model-analyzer/
│   │   └── content-analysis/
│   └── integrated_results/
│       └── Comprehensive_Analysis_Report.html  # ⭐ Interactive report
├── analysis_reports/   # Generated analysis reports
├── visualizations/     # Generated charts
├── generated_code/     # Analysis code
└── examples/          # Example datasets
```

## 🎨 Sample Data

The project includes Olist Brazilian E-commerce datasets in `data_storage/`:

- **Orders.csv** (99,441 records): Order information, status, timestamps
- **Customers.csv** (99,441 records): Customer demographics, location
- **Order Items.csv**: Order details, products, pricing
- **Order Payments.csv**: Payment methods, installments
- **Products.csv**: Product catalog, categories
- **Reviews.csv** (99,224 records): Customer reviews, ratings, comments
- **Categories.csv**: Product categories
- **Sellers.csv**: Seller information
- **Geolocation.csv**: Geographic data

**Sample Workflow:**
```bash
# 1. Data already in data_storage/
# 2. Run automatic analysis
/do-more

# 3. View results
# Open: do_more_analysis/integrated_results/Comprehensive_Analysis_Report.html
```

## 🔧 Configuration

### Environment Setup
The project uses Claude Code's configuration system. Key settings:

1. **Hooks**: Automated validation and context loading
2. **Sub-agents**: Specialized AI assistants for different tasks
3. **Commands**: Custom slash commands for common operations

### Requirements
- Python 3.8+ for data analysis
- Claude Code with sub-agents enabled
- Data files in CSV, JSON, or Excel format

## 📚 Getting Started Guide

### For New Users
1. **Place your data** in `data_storage/`
2. **Run exploratory analysis**: `/analyze your_data.csv exploratory`
3. **Create visualizations**: `/visualize your_data.csv all`
4. **Generate report**: `/report your_data.csv complete markdown`

### For Advanced Users
1. **Customize agents**: Modify `.claude/agents/` configurations
2. **Create custom commands**: Add new commands in `.claude/commands/`
3. **Set up automation**: Configure hooks in `.claude/settings.json`
4. **Extend functionality**: Add custom analysis scripts

## 🎯 Analysis Types

### Exploratory Analysis
- Data quality assessment
- Summary statistics
- Pattern discovery
- Initial insights

### Statistical Analysis
- Hypothesis testing
- Correlation analysis
- Regression analysis
- Confidence intervals

### Predictive Analysis
- Feature importance
- Predictive modeling
- Variable relationships
- Model recommendations

### Complete Analysis
- All analysis types
- Comprehensive reports
- Visualizations
- Actionable insights

## 📈 Visualization Types

### All Visualizations
- Comprehensive dashboard
- Multiple chart types
- Interactive exploration
- Executive summary

### Specific Charts
- **Trends**: Time series, moving averages
- **Distribution**: Histograms, box plots, density plots
- **Correlation**: Heatmaps, scatter plots, correlation matrices
- **Comparison**: Bar charts, grouped charts, small multiples

## 🔍 Code Generation

### Supported Languages
- **Python**: Pandas, NumPy, Scikit-learn, Matplotlib
- **R**: Tidyverse, ggplot2, caret
- **SQL**: All major dialects
- **JavaScript**: D3.js, Plotly.js, TensorFlow.js

### Analysis Types
- Data cleaning and preprocessing
- Statistical analysis
- Machine learning
- Visualization code
- Custom analysis

## 📋 Project Status

**Current Phase**: Active Development with 12 Specialized Skills ✅

### Completed Features
- [x] Project structure and configuration
- [x] Data Explorer sub-agent
- [x] Visualization Specialist sub-agent
- [x] Code Generator sub-agent
- [x] Report Writer sub-agent
- [x] Quality Assurance sub-agent
- [x] Hypothesis Generator sub-agent
- [x] **12 Specialized Analysis Skills**
  - [x] data-exploration-visualization
  - [x] rfm-customer-segmentation
  - [x] ltv-predictor
  - [x] retention-analysis
  - [x] funnel-analysis
  - [x] growth-model-analyzer
  - [x] attribution-analysis-modeling
  - [x] ab-testing-analyzer
  - [x] content-analysis
  - [x] regression-analysis-modeling
  - [x] recommender-system
  - [x] user-profiling-analysis
- [x] Core slash commands (/analyze, /visualize, /generate, /report, /quality, /hypothesis)
- [x] **⭐ NEW: /do-more command** - Automatic multi-skill analysis
- [x] Automation hooks (context-loader, validate-analysis)
- [x] Olist e-commerce sample datasets
- [x] Interactive HTML report generation
- [x] Comprehensive documentation

### Recent Enhancements
- ⭐ **/do-more Command**: One-command automatic multi-skill analysis
- ⭐ **Interactive HTML Reports**: Beautiful, embedded charts with navigation
- ⭐ **Smart Skill Matching**: Automatic skill selection based on data characteristics
- ⭐ **Integrated Workflows**: Sequential execution of multiple skills
- ⭐ **Enhanced Visualizations**: 20+ chart types across all skills

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
3. **Add your improvements**
4. **Test your changes**
5. **Submit a pull request**

### Development Guidelines
- Follow the established code style
- Add comprehensive documentation
- Include unit tests for new features
- Update the README as needed

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Claude Code](https://claude.ai/code)
- Inspired by the [DATAGEN](https://github.com/starpig1129/DATAGEN) project
- Powered by modern data science tools and frameworks

## 📞 Support

For support and questions:
- Check the documentation in the `docs/` directory
- Review the examples in `examples/`
- Use the `/help` command for usage assistance

---

**Start analyzing your data smarter, not harder!** 🚀