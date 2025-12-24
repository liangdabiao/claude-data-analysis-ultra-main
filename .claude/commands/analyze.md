---
allowed-tools: Task, Read, Write, Bash, Grep, Glob
argument-hint: [dataset] [analysis_type]
description: Perform comprehensive data analysis on specified dataset
---

# Data Analysis Command

Execute data analysis on dataset `$1` with analysis type `$2` using the data-explorer subagent.

## Context
- Dataset location: @data_storage/$1
- Analysis type: $2 (exploratory, statistical, predictive, complete)
- Current working directory: !`pwd`
- Available visualization libraries: matplotlib, seaborn, plotly
- Python data science stack: pandas, numpy, scipy

## Your Task

Use the data-explorer subagent to perform comprehensive data analysis:

### 1. Data Assessment
- Load and inspect the dataset structure
- Check data types, missing values, and duplicates
- Generate initial summary statistics
- Identify data quality issues

### 2. Statistical Analysis
- Perform descriptive statistics analysis
- Calculate correlations between variables
- Identify outliers and anomalies
- Conduct appropriate statistical tests

### 3. Pattern Discovery
- Identify trends and patterns in the data
- Discover relationships between variables
- Detect seasonal patterns or cycles
- Find clusters or segments in the data

### 4. Generate Insights
- Extract key findings from the analysis
- Identify actionable insights
- Suggest areas for deeper investigation
- Recommend visualization approaches

## Analysis Types

### Exploratory Analysis
- Basic data understanding
- Summary statistics
- Data quality assessment
- Initial pattern identification

### Statistical Analysis
- Advanced statistical testing
- Correlation and regression analysis
- Hypothesis testing
- Confidence intervals

### Predictive Analysis
- Feature importance analysis
- Predictive modeling preparation
- Variable relationships
- Model recommendation

### Complete Analysis
- All of the above plus
- Comprehensive report generation
- Visualization recommendations
- Next steps planning

## Expected Output

### Analysis Report
Create a comprehensive analysis report with:
- **Executive Summary**: Key findings in plain language
- **Data Overview**: Dataset characteristics and quality
- **Statistical Findings**: Detailed statistical analysis
- **Key Insights**: Actionable discoveries
- **Recommendations**: Next steps for deeper analysis
- **Limitations**: Data and method constraints

### File Outputs
- `analysis_reports/analysis_summary_$1.md` - Detailed analysis report
- `analysis_reports/statistical_summary_$1.csv` - Statistical summary table
- `analysis_reports/data_quality_$1.json` - Data quality assessment

## Quality Assurance
- Validate all statistical calculations
- Cross-check important findings
- Document all assumptions and limitations
- Ensure reproducible analysis

## Example Usage
```bash
/analyze user_behavior.csv exploratory
/analyze sales_data.csv statistical
/analyze customer_data.csv predictive
/analyze financial_data.csv complete
```

## Notes
- Dataset should be located in the data_storage/ directory
- Analysis results will be saved to analysis_reports/ directory
- Use Task tool to delegate to data-explorer subagent
- Consider following up with /visualize command for charts