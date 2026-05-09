---
allowed-tools: Task, Read, Write, Bash, Grep, Glob
argument-hint: [dataset] [chart_type]
description: Create data visualizations for the specified dataset
---

# Data Visualization Command

Create comprehensive data visualizations for dataset `$1` with chart type `$2` using the visualization-specialist subagent.

## Context
- Dataset location: @data_storage/$1
- Chart type: $2 (all, trends, distribution, correlation, comparison, custom)
- Current working directory: !`pwd`
- Visualization output directory: ./visualizations/
- Available libraries: matplotlib, seaborn, plotly, bokeh

## Your Task

Use the visualization-specialist subagent to create informative visualizations:

### 1. Data Preparation
- Load and prepare the dataset
- Handle missing values and outliers
- Select appropriate variables for visualization
- Prepare data for different chart types

### 2. Visualization Planning
- Determine the best chart types for the data
- Plan color schemes and styling
- Consider the target audience and purpose
- Plan layout and composition

### 3. Chart Creation
- Create multiple complementary visualizations
- Ensure proper labeling and annotations
- Use appropriate scales and ranges
- Apply consistent styling and colors

### 4. Quality Assurance
- Test visualizations with different data scenarios
- Verify data accuracy in visualizations
- Check accessibility and readability
- Optimize for different screen sizes

## Chart Types

### All Visualizations
- Comprehensive dashboard with multiple chart types
- Overview of all key variables and relationships
- Executive summary visualizations
- Interactive exploration dashboard

### Trends
- Time series line charts
- Moving average plots
- Trend decomposition
- Seasonal analysis charts

### Distribution
- Histograms and density plots
- Box plots and violin plots
- Q-Q plots for normality
- Statistical distribution charts

### Correlation
- Correlation heatmaps
- Scatter plot matrices
- Pair plots
- Regression analysis plots

### Comparison
- Bar charts and column charts
- Grouped and stacked charts
- Small multiples
- Comparative analysis charts

### Custom
- User-specified custom visualizations
- Domain-specific charts
- Interactive dashboards
- Animated visualizations

## Expected Output

### Visualization Files
- `visualizations/dashboard_$1.html` - Interactive dashboard
- `visualizations/summary_$1.png` - Summary charts
- `visualizations/detailed_$1.pdf` - Detailed analysis charts
- `visualizations/charts_$1.py` - Reproducible code

### Documentation
- **Chart Descriptions**: Explanation of each visualization
- **Data Sources**: Documentation of data transformations
- **Interpretation Guide**: How to read and understand the charts
- **Customization Options**: How to modify and extend visualizations

## Technical Requirements

### File Formats
- **Static Images**: PNG (high-resolution), SVG (vector)
- **Interactive**: HTML with JavaScript (Plotly, D3.js)
- **Print**: PDF with high resolution
- **Code**: Python/R scripts for reproducibility

### Design Standards
- **Color Schemes**: Colorblind-friendly palettes
- **Typography**: Clear, readable fonts
- **Layout**: Responsive and well-organized
- **Accessibility**: WCAG compliant where possible

## Quality Assurance

### Validation Checks
- Verify data accuracy in all visualizations
- Test with different screen sizes and devices
- Check color accessibility
- Ensure proper labeling and annotations

### Performance
- Optimize file sizes for web display
- Ensure fast loading times
- Test interactivity and responsiveness
- Validate cross-browser compatibility

## Example Usage
```bash
/visualize user_behavior.csv all
/visualize sales_data.csv trends
/visualize customer_data.csv distribution
/visualize financial_data.csv correlation
/visualize performance_data.csv comparison
/visualize custom_data.csv custom
```

## Best Practices

### Design Principles
- **Data-Ink Ratio**: Maximize the ratio of data-ink to total ink
- **Chart Junk**: Eliminate non-data ink and decorative elements
- **Clarity**: Ensure the message is immediately understandable
- **Consistency**: Use consistent styling across all visualizations

### Data Integrity
- Validate data before visualization
- Handle missing values appropriately
- Use appropriate scales and ranges
- Document all data transformations

### User Experience
- Consider the target audience
- Provide clear labels and legends
- Include interactive features where helpful
- Offer multiple views of the same data

## Notes
- Dataset should be located in the data_storage/ directory
- Visualizations will be saved to visualizations/ directory
- Use Task tool to delegate to visualization-specialist subagent
- Consider using /analyze command first for data insights
- Interactive visualizations require web browser for viewing

## Integration with Other Commands
- Use after `/analyze` for data-driven visualizations
- Combine with `/report` for comprehensive analysis reports
- Follow with `/generate` for visualization code snippets