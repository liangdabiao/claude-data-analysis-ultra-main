---
allowed-tools: Task, Read, Write, Bash, Grep, Glob
argument-hint: [dataset] [skill_type] [analysis_objective]
description: Execute specialized data analysis skills including AB testing, attribution modeling, funnel analysis, LTV prediction, and more for deep, targeted insights.
---

# Specialized Skills Analysis Command

Execute specialized data analysis on dataset `$1` using skill type `$2` with analysis objective `$3` using the skills-specialist subagent.

## Context
- Dataset location: @data_storage/$1
- Skill type: $2 (ab-testing, attribution, funnel, ltv, rfm, regression, retention, growth, recommender)
- Analysis objective: $3 (describe your specific analysis goal)
- Current working directory: !`pwd`
- Available specialized skills: AB Testing, Attribution Modeling, Funnel Analysis, LTV Prediction, RFM Segmentation, Regression Analysis, Retention Analysis, Growth Modeling, Recommender Systems

## Your Task

Use the skills-specialist subagent to perform specialized data analysis:

### 1. Skill Assessment
- Evaluate dataset suitability for selected skill
- Confirm data requirements are met
- Identify any necessary data preprocessing

### 2. Analysis Execution
- Apply selected specialized analytical model
- Perform calculations and statistical analysis
- Validate model assumptions and results

### 3. Insight Generation
- Extract key findings from specialized analysis
- Translate technical results into business insights
- Identify actionable recommendations

### 4. Reporting
- Document analysis methodology and assumptions
- Visualize specialized analysis results
- Provide implementation guidance

## Skill Types

### AB Testing
- Statistical significance testing for experiments
- User segmentation and cohort analysis
- Test result visualization
- Sample size calculation and power analysis

### Attribution Modeling
- Multi-channel marketing attribution
- First-touch, last-touch, linear attribution
- Markov chain and Shapley value modeling
- Channel performance optimization

### Funnel Analysis
- Customer journey conversion analysis
- Drop-off point identification
- Funnel optimization recommendations
- Cohort-based funnel comparison

### LTV Prediction
- Customer lifetime value modeling
- CLV segmentation and targeting
- Predictive LTV forecasting
- ROI analysis for customer acquisition

### RFM Segmentation
- Recency, Frequency, Monetary analysis
- Customer tier classification
- Targeted marketing strategies
- Churn risk prediction

### Regression Analysis
- Linear and multiple regression modeling
- Logistic regression for classification
- Time series forecasting
- Feature selection and validation

### Retention Analysis
- Customer retention rate calculation
- Cohort analysis for retention trends
- Churn prediction modeling
- Retention optimization strategies

### Growth Modeling
- User growth forecasting
- Viral coefficient calculation
- Growth rate analysis
- Scenario planning for growth strategies

### Recommender Systems
- Collaborative filtering algorithms
- Content-based recommendation models
- Hybrid recommendation systems
- Performance metrics evaluation

## Expected Output

### Specialized Analysis Report
Create a detailed analysis report with:
- **Executive Summary**: Key findings and business impact
- **Methodology**: Analytical approach and model details
- **Results**: Detailed analysis outputs
- **Insights**: Actionable discoveries
- **Recommendations**: Implementation strategies
- **Limitations**: Data and model constraints

### File Outputs
- `analysis_reports/skills_analysis_$1_$2.md` - Detailed specialized analysis report
- `analysis_reports/skills_results_$1_$2.csv` - Raw analysis results
- `analysis_reports/skills_visualizations_$1_$2.json` - Visualization specifications

## Quality Assurance
- Validate all model assumptions
- Cross-check calculations with alternative methods
- Document all data transformations
- Ensure reproducibility of analysis
- Provide clear interpretation of results

## Example Usage
```bash
/skills user_test_data.csv ab-testing "Analyze conversion rate difference between control and variant groups"
/skills marketing_data.csv attribution "Evaluate channel contribution using multiple attribution models"
/skills customer_journey.csv funnel "Identify drop-off points in the registration funnel"
/skills customer_data.csv ltv "Predict customer lifetime value for segmentation"
/skills sales_data.csv rfm "Segment customers based on purchase behavior"
/skills financial_data.csv regression "Model revenue drivers and forecast future sales"
/skills user_retention.csv retention "Analyze monthly retention trends and predict churn"
/skills growth_data.csv growth "Forecast user acquisition and growth rate"
/skills product_data.csv recommender "Build product recommendation system"
```

## Integration with Other Commands

### Pre-analysis
- Use `/analyze` for initial data exploration before applying specialized skills
- Use `/hypothesis` to generate testable hypotheses for skill application

### Post-analysis
- Use `/visualize` to create advanced visualizations of skill results
- Use `/report` to generate comprehensive business reports
- Use `/quality` to validate specialized analysis results