---
allowed-tools: Task, Read, Write, Bash, Grep, Glob
argument-hint: [dataset] [report_type] [format]
description: Generate comprehensive analysis reports for specified dataset
---

# Report Generation Command

Generate comprehensive analysis report for dataset `$1` with report type `$2` in `$3` format using the report-writer subagent.

## Context
- Dataset location: @data_storage/$1
- Report type: $2 (summary, complete, executive, technical, custom)
- Output format: $3 (markdown, html, pdf, json, docx)
- Current working directory: !`pwd`
- Output directory: ./analysis_reports/
- Available templates and formatting options

## Your Task

Use the report-writer subagent to create professional, comprehensive analysis reports:

### 1. Report Planning
- Determine appropriate report structure based on type and audience
- Gather all relevant analysis results and visualizations
- Plan narrative flow and storytelling approach
- Select appropriate level of technical detail

### 2. Content Generation
- Write compelling executive summary (for executive reports)
- Create detailed methodology sections (for technical reports)
- Present key findings with supporting evidence
- Include data visualizations with explanations
- Provide actionable recommendations

### 3. Quality Assurance
- Verify accuracy of all statistics and claims
- Ensure logical consistency throughout the report
- Check for clarity and readability
- Validate proper citation of sources and methods

### 4. Format and Deliver
- Apply appropriate formatting for the selected output format
- Create professional document structure
- Add necessary metadata and references
- Generate final deliverable files

## Report Types

### Summary Report
- Brief overview of key findings
- Essential statistics and insights
- High-level recommendations
- 1-2 pages in length

### Complete Report
- Comprehensive analysis documentation
- Detailed methodology and results
- Full statistical analysis
- Extensive visualizations and explanations
- Technical appendices and references

### Executive Report
- Business-focused summary
- Key performance indicators
- Strategic recommendations
- Action items and implementation roadmap
- Financial impact assessment

### Technical Report
- Detailed methodology documentation
- Statistical analysis results
- Technical appendices
- Code and algorithm documentation
- Peer-review ready format

### Custom Report
- User-defined structure and content
- Tailored to specific requirements
- Flexible formatting options
- Custom sections and emphasis

## Output Formats

### Markdown (md)
- Web-friendly format
- Version control friendly
- Easy to convert to other formats
- Good for documentation and collaboration

### HTML (html)
- Interactive web format
- Embedded visualizations
- Responsive design
- Web browser accessible

### PDF (pdf)
- Professional print format
- Fixed layout
- High-quality output
- Distribution ready

### JSON (json)
- Structured data format
- Machine readable
- API friendly
- Integration ready

### DOCX (docx)
- Microsoft Word format
- Business standard
- Editable format
- Corporate ready

## Expected Output

### Report Files
- `analysis_reports/$1_$2_report.$3` - Main report file
- `analysis_reports/$1_appendix.$3` - Technical appendix
- `analysis_reports/$1_visualizations.$3` - Visualization compilation
- `analysis_reports/$1_metadata.json` - Report metadata

### Content Structure
- **Title Page**: Report title, author, date, version
- **Executive Summary**: Key findings and recommendations
- **Introduction**: Background and objectives
- **Methodology**: Analysis approach and methods
- **Results**: Detailed findings and statistics
- **Discussion**: Interpretation and implications
- **Conclusions**: Summary and next steps
- **References**: Sources and citations
- **Appendices**: Technical details and supplementary material

## Quality Standards

### Content Quality
- **Accuracy**: All statistics and claims must be verified
- **Clarity**: Clear, concise, and understandable language
- **Completeness**: Cover all important aspects of the analysis
- **Relevance**: Focus on audience-relevant information
- **Objectivity**: Balanced and unbiased presentation

### Technical Quality
- **Statistical Validity**: Proper statistical methods and interpretation
- **Methodological Rigor**: Sound analysis methodology
- **Data Integrity**: Accurate representation of data
- **Reproducibility**: Sufficient detail for reproduction
- **Documentation**: Comprehensive documentation of methods

### Presentation Quality
- **Professional Formatting**: Industry-standard formatting
- **Visual Integration**: Seamless integration of visualizations
- **Readability**: Clear typography and layout
- **Accessibility**: Accessible design and content
- **Consistency**: Consistent styling throughout

## Report Templates

### Executive Summary Template
```markdown
# Executive Summary: $1 Analysis

## Key Findings
- **Primary Insight**: [Key statistical finding]
- **Business Impact**: [Business-relevant implication]
- **Performance Metrics**: [KPIs and measurements]

## Recommendations
1. **Immediate Action**: [Specific action item with timeline]
2. **Strategic Initiative**: [Long-term recommendation]
3. **Investment Priority**: [Resource allocation recommendation]

## Next Steps
- **Phase 1 (0-30 days)**: [Immediate actions]
- **Phase 2 (30-90 days)**: [Medium-term initiatives]
- **Phase 3 (90+ days)**: [Long-term strategic actions]
```

### Technical Report Template
```markdown
# Technical Analysis Report: $1

## Abstract
Brief summary of analysis objectives, methods, and key findings.

## 1. Introduction
### 1.1 Background and Objectives
Context and purpose of the analysis.

### 1.2 Data Description
Source, structure, and characteristics of the dataset.

## 2. Methodology
### 2.1 Data Preparation
Data cleaning, transformation, and preprocessing steps.

### 2.2 Analytical Methods
Statistical methods and algorithms used in the analysis.

## 3. Results
### 3.1 Descriptive Statistics
Summary statistics and data characteristics.

### 3.2 Inferential Statistics
Hypothesis testing results and confidence intervals.

### 3.3 Key Findings
Detailed presentation of analysis results.

## 4. Discussion
### 4.1 Interpretation
Explanation of what the findings mean.

### 4.2 Limitations
Constraints and limitations of the analysis.

## 5. Conclusions
### 5.1 Summary
Recap of key insights and discoveries.

### 5.2 Recommendations
Actionable recommendations based on findings.

## 6. References
Citations to relevant literature and methods.

## 7. Appendices
### 7.1 Technical Details
Additional technical information.
### 7.2 Data Dictionary
Detailed variable descriptions.
### 7.3 Code Listings
Relevant code snippets.
```

## Best Practices

### Report Writing
- **Know Your Audience**: Tailor content to audience expertise level
- **Tell a Story**: Create a narrative flow from data to insights
- **Be Concise**: Include only relevant information
- **Use Visuals**: Support text with appropriate visualizations
- **Provide Context**: Explain the significance of findings

### Data Presentation
- **Use Appropriate Charts**: Select chart types based on data type
- **Label Clearly**: Ensure all charts and tables are clearly labeled
- **Include Units**: Always include units of measurement
- **Highlight Key Points**: Emphasize important findings
- **Ensure Accessibility**: Make content accessible to all users

### Recommendations
- **Be Specific**: Provide concrete, actionable recommendations
- **Prioritize**: Rank recommendations by importance and impact
- **Include Timeline**: Specify when actions should be taken
- **Assign Ownership**: Identify who should take action
- **Measure Success**: Define how success will be measured

## Example Usage
```bash
/report user_behavior.csv complete markdown
/report sales_data.csv executive pdf
/report customer_data.csv technical html
/report financial_data.json summary json
/report custom_data custom docx
```

## Integration with Other Commands
- Use after `/analyze` to have analysis results for reporting
- Combine with `/visualize` to include visualizations in reports
- Follow with `/quality` to validate report data quality
- Precede with `/hypothesis` to include research findings

## Notes
- Dataset should be located in the data_storage/ directory
- Reports will be saved to analysis_reports/ directory
- Use Task tool to delegate to report-writer subagent
- Consider the target audience when selecting report type
- Ensure all data analysis is complete before generating reports
- Review and customize reports before final distribution