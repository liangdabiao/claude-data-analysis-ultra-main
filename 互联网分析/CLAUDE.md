# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Claude Data Analysis Assistant is an intelligent data analysis platform that leverages Claude Code's sub-agents, slash-commands, skills, and hooks to provide a comprehensive data analysis workflow. This project combines custom skills for specialized analysis (RFM segmentation, attribution modeling, funnel analysis, etc.) with a flexible agent-based architecture.

## Quick Start

### ⭐ NEW: Automatic Multi-Skill Analysis
The easiest way to analyze your data:

```bash
# 1. Place data files in data_storage/
# 2. Run this single command:
/do-more

# That's it! It will:
# - Auto-discover all data files
# - Match relevant analysis skills
# - Execute skills in optimal order
# - Generate comprehensive HTML report
# - Takes 2-5 minutes, no human intervention needed
```

### Traditional Usage
1. Place your data files in the `data_storage/` directory
2. Use specialized skills directly (e.g., `/rfm-customer-segmentation`) or universal commands
3. Use `/analyze [filename]` for general data analysis
4. Use `/visualize [filename]` to create visualizations

### ⚡ Quick Decision Guide: Which Command Should I Use?

**Choose `/do-more` when you want:**
- ✅ Fast, automated analysis (2-5 minutes)
- ✅ No configuration or parameters
- ✅ Quick insights and overview
- ✅ Interactive HTML report
- ✅ Set-it-and-forget experience

**Choose `/do-all` when you want:**
- ✅ Thorough, deep analysis
- ✅ Human oversight and feedback at key stages
- ✅ Interactive hypothesis generation
- ✅ Custom code generation
- ✅ Multiple output formats (HTML, PDF, DOCX)
- ✅ Complete documentation pipeline
- ✅ Time for interactive session (10-30 minutes)

**Example comparison:**
```bash
# Quick insights (2-5 min, automatic)
/do-more

# OR thorough analysis (10-30 min, interactive)
/do-all
# → [checkpoint 1] Review data quality
# → [checkpoint 2] Approve hypotheses
# → [checkpoint 3] Review visualizations
# → Receive complete report with code
```

## Core Architecture

### Skills System (Primary Analysis Tools)
The project uses 12 specialized skills for different analysis types:

**Customer Analysis Skills:**
- `rfm-customer-segmentation`: RFM analysis for e-commerce customer segmentation
- `user-profiling-analysis`: Comprehensive user behavior analysis and profiling
- `ltv-predictor`: Customer lifetime value prediction
- `retention-analysis`: User retention and churn analysis

**Marketing Analysis Skills:**
- `attribution-analysis-modeling`: Multi-touch attribution using Markov chains and Shapley values
- `growth-model-analyzer`: Growth hacking and user acquisition analysis
- `ab-testing-analyzer`: A/B test analysis and statistical validation
- `funnel-analysis`: Conversion funnel analysis with segmentation

**Data Analysis Skills:**
- `data-exploration-visualization`: Automated EDA and visualization
- `regression-analysis-modeling`: Predictive modeling and regression analysis
- `content-analysis`: Text and content analysis using NLP
- `recommender-system`: Recommendation engine implementation

### Universal Commands
- **`/do-more`**: ⭐ **RECOMMENDED** - Automatic multi-skill analysis (no parameters, observes data_storage/)
- **`/do-all`**: Complete interactive workflow with human feedback checkpoints (no parameters, uses data from data_storage/)
- `/analyze [dataset] [analysis_type]`: General data analysis (exploratory, statistical, predictive, complete)
- `/visualize [dataset] [chart_type]`: Create visualizations
- `/generate [language] [analysis_type]`: Generate analysis code
- `/report [dataset] [format]`: Generate analysis reports
- `/quality [dataset] [action]`: Data quality validation
- `/hypothesis [dataset] [domain]`: Generate research hypotheses

### /do-more Command Details

**Purpose**: Automatically observe data in `data_storage/`, intelligently match skills, execute analyses, and generate integrated reports

**Workflow**:
1. **Data Observation**: Scans all files in `data_storage/`
2. **Type Detection**: Identifies data types (e-commerce, user behavior, marketing, etc.)
3. **Skill Matching**: Selects relevant skills based on data characteristics
4. **Sequential Execution**: Runs skills in optimal order
5. **Result Integration**: Combines all outputs into comprehensive reports

**Output Structure**:
```
do_more_analysis/
├── skill_execution/
│   ├── data-exploration-visualization/
│   ├── rfm-customer-segmentation/
│   ├── ltv-predictor/
│   ├── retention-analysis/
│   ├── funnel-analysis/
│   ├── growth-model-analyzer/
│   └── content-analysis/
└── integrated_results/
    └── Comprehensive_Analysis_Report.html  # Interactive HTML report
```

**Supported Data Types**:
- **E-commerce**: Orders, Customers, Products, Reviews
- **User Behavior**: Clickstream, sessions, events
- **Marketing**: Campaigns, attributions, funnels
- **General**: Any CSV/JSON/Excel dataset

### Sub-Agents System
- **data-explorer**: Statistical analysis and pattern discovery
- **visualization-specialist**: Chart and graph creation
- **code-generator**: Analysis code generation
- **report-writer**: Comprehensive report generation
- **quality-assurance**: Data validation and quality control
- **hypothesis-generator**: Research hypothesis generation

## Directory Structure
```
claude-data-analysis/
├── .claude/
│   ├── agents/          # Sub-agent configurations
│   ├── commands/        # Universal slash command definitions
│   │   └── do-more.md   # ⭐ NEW: /do-more command definition
│   ├── hooks/          # Automation scripts (context-loader, validate-analysis)
│   ├── settings.json   # Claude Code configuration with skill discovery
│   ├── settings.local.json # Permissions configuration
│   └── skills/         # Specialized analysis skills (12 modules)
│       ├── rfm-customer-segmentation/
│       ├── ltv-predictor/
│       ├── retention-analysis/
│       ├── funnel-analysis/
│       ├── growth-model-analyzer/
│       ├── content-analysis/
│       ├── attribution-analysis-modeling/
│       ├── regression-analysis-modeling/
│       └── ... (4 more skills)
├── data_storage/       # Data files directory (Olist e-commerce datasets)
│   ├── Orders.csv
│   ├── Customers.csv
│   ├── Order Items.csv
│   └── ... (9 datasets total)
├── do_more_analysis/   # ⭐ NEW: /do-more command output directory
│   ├── skill_execution/  # Individual skill results with visualizations
│   │   ├── data-exploration-visualization/
│   │   ├── rfm-customer-segmentation/
│   │   ├── ltv-predictor/
│   │   ├── retention-analysis/
│   │   ├── funnel-analysis/
│   │   ├── growth-model-analyzer/
│   │   └── content-analysis/
│   └── integrated_results/
│       ├── Comprehensive_Analysis_Report.html  # ⭐ Interactive HTML report
│       ├── Comprehensive_Analysis_Report.md
│       └── integrated_summary.json
├── analysis_reports/   # Generated analysis reports
├── visualizations/     # Generated charts and plots
└── examples/          # Example datasets and workflows
```

## Key Configuration Features

### Skill Auto-Discovery
The system automatically discovers and applies skills based on:
- `auto-discovery: true` - Skills are automatically detected
- `auto-apply: true` - Skills are automatically applied when relevant
- `matching-strategy: semantic` - Skills matched by semantic analysis

### Automation Hooks
- **PostToolUse Hook**: Runs `validate-analysis.py` after any file write/edit for quality control
- **UserPromptSubmit Hook**: Runs `context-loader.py` to provide project context for analysis

## Development Workflow and Commands

### Testing Individual Skills
Each skill includes a `quick_test.py` for functionality verification:
```bash
# Test any skill from its directory
cd .claude/skills/[skill_name]/
python quick_test.py

# Example for RFM segmentation
cd .claude/skills/rfm-customer-segmentation/
python quick_test.py
```

### Installing Skill Dependencies
Skills have independent Python requirements. Install per skill:
```bash
# Navigate to skill directory
cd .claude/skills/[skill_name]/

# Install dependencies
pip install -r requirements.txt

# Common core dependencies across skills
# pandas>=1.3.0, numpy>=1.21.0, scikit-learn>=1.0.0, matplotlib>=3.5.0, seaborn>=0.11.0
```

### Running Analysis
```bash
# ⭐ RECOMMENDED: Automatic multi-skill analysis (no parameters!)
/do-more

# OR: Interactive complete workflow (with human feedback)
/do-all

# Direct skill invocation
/skill [skill_name] [dataset_path]

# Other universal commands
/analyze [dataset] [analysis_type]
/visualize [dataset] [chart_type]
```

### `/do-more` vs `/do-all` - Detailed Comparison

**`/do-more`** - Automatic Analysis (Recommended):
```bash
/do-more
```
- ✅ No parameters required
- ✅ Auto-scans data_storage/
- ✅ Executes 7+ relevant skills
- ✅ Generates HTML report
- ✅ Takes 2-5 minutes
- ✅ Zero human intervention
- 📁 Output: `do_more_analysis/integrated_results/Comprehensive_Analysis_Report.html`

**`/do-all`** - Interactive Complete Workflow:
```bash
/do-all
```
- ✅ No parameters required
- ✅ Uses data from data_storage/
- ✅ Complete analysis workflow with 3 human feedback checkpoints
- ✅ Generates code + reports
- ✅ Takes 10-30 minutes
- 📁 Output: `complete_analysis/` directory

**Key Differences:**
| Aspect | `/do-more` | `/do-all` |
|--------|-----------|-----------|
| Parameters | None | None |
| Skills | Auto-selected (7+) | Complete workflow (no skills) |
| Checkpoints | None | 3 interactive |
| Duration | 2-5 min | 10-30 min |
| Code gen | No | Yes |
| Best for | Quick insights | Complete analysis |

### Validation and Quality Control
The validation hook runs automatically, but can be triggered manually:
```bash
# Run validation on any file
python .claude/hooks/validate-analysis.py < file_path
```

## Common Workflows

### ⭐ Recommended: Automatic Multi-Skill Analysis
```bash
# E-commerce complete analysis (one command!)
/do-more

# Output (2-5 minutes):
# ✓ Data Exploration & Visualization (5 charts)
# ✓ RFM Customer Segmentation (5 segments + VIP list)
# ✓ LTV Prediction (ML models with predictions)
# ✓ Retention Analysis (cohort heatmaps + retention curves)
# ✓ Funnel Analysis (conversion metrics + dashboard)
# ✓ Growth Model Analysis (CAGR + trend analysis)
# ✓ Content Analysis (sentiment + keyword extraction)
# ✓ Interactive HTML report with all findings
```

### Interactive Complete Analysis (When You Need Depth)
```bash
# Complete analysis with human oversight
/do-all

# Workflow (10-30 minutes with checkpoints):
# 1. Data Quality Assessment → [you confirm]
# 2. Exploratory Analysis
# 3. Hypothesis Generation → [you approve]
# 4. Visualizations → [you review]
# 5. Code Generation
# 6. Comprehensive Report
```

### Customer Segmentation Workflow
```bash
# Quick automated analysis
/do-more  # Includes RFM + 6 more skills!

# OR use specific skill
/rfm-customer-segmentation
```

### Marketing Analysis Workflow
```bash
# Quick automated analysis
/do-more  # Auto-detects marketing data

# OR specific analyses
/attribution-analysis-modeling
/funnel-analysis
/growth-model-analyzer
/ab-testing-analyzer
```

### General Data Analysis Workflow
```bash
# Option 1: Quick and automatic (recommended)
/do-more

# Option 2: Interactive and thorough
/do-all

# Option 3: Step-by-step manual
/quality [dataset.csv] validate
/analyze [dataset.csv] exploratory
/visualize [dataset.csv] all
/report [dataset.csv] complete markdown
```

## Sample Data

The project includes Olist e-commerce datasets in `data_storage/`:
- `Orders.csv`: Order information (99,441 records)
- `Customers.csv`: Customer data (99,441 records)
- `Order Items.csv`: Order item details
- `Order Payments.csv`: Payment information
- `Products.csv`: Product catalog
- `Reviews.csv`: Customer reviews (99,224 records)
- `Categories.csv`: Product categories
- `Sellers.csv`: Seller information
- `Geolocation.csv`: Geographic data

**Try it now:**
```bash
# Just run /do-more to see the full analysis in action!
/do-more

# Then open: do_more_analysis/integrated_results/Comprehensive_Analysis_Report.html
```

## /do-all Command Deep Dive

### What Makes /do-all Special?

**1. Complete Interactive Workflow**
- 6-stage comprehensive analysis pipeline
- Human feedback checkpoints at 3 critical stages
- Interactive hypothesis generation
- Custom code generation
- Multiple output formats

**2. Comprehensive Coverage**
- Complete analysis workflow (no skills required)
- Data quality assessment
- Exploratory + statistical + predictive analysis
- Visualizations + code generation
- Professional documentation

**3. Human-in-the-Loop**
- Quality checkpoint: Review data quality before proceeding
- Hypothesis checkpoint: Approve research directions
- Visualization checkpoint: Review and adjust visual strategy
- Maintain control while leveraging automation

### Workflow Stages in Detail

**Stage 1: Data Quality Assessment**
- Completeness, accuracy, consistency checks
- Missing value analysis
- Outlier detection
- Quality score calculation
- **[Human Checkpoint]**: Confirm data quality acceptable

**Stage 2: Exploratory Data Analysis**
- Statistical summaries
- Distribution analysis
- Correlation analysis
- Pattern discovery
- Anomaly detection

**Stage 3: Hypothesis Generation**
- Data-driven hypothesis creation
- Experimental design
- Validation planning
- **[Human Checkpoint]**: Approve or adjust research direction

**Stage 4: Visualization**
- Interactive dashboards
- Chart generation
- Storyboards
- **[Human Checkpoint]**: Review visual strategy

**Stage 5: Code Generation**
- Reproducible analysis code
- Data processing pipelines
- Automation scripts
- Test cases

**Stage 6: Report Generation**
- Executive summary
- Technical findings
- Recommendations
- Format: HTML, PDF, Markdown, or DOCX

### When to Use Each Command

| Scenario | Use `/do-more` | Use `/do-all` |
|----------|----------------|--------------|
| Quick business review | ✅ Yes | ❌ |
| Executive presentation prep | ✅ Yes | ❌ |
| Initial data exploration | ✅ Yes | ❌ |
| Deep research project | ⚠️ Maybe | ✅ Yes |
| Code generation needed | ❌ No | ✅ Yes |
| Custom analysis direction | ❌ No | ✅ Yes |
| Publish-quality report | ⚠️ Maybe | ✅ Yes |
| Interactive session needed | ❌ No | ✅ Yes |
| Time-constrained (under 10 min) | ✅ Yes | ❌ |
| Thorough documentation needed | ⚠️ Maybe | ✅ Yes |

### Output Structure

**`/do-more` Output**:
```
do_more_analysis/
├── skill_execution/
│   ├── data-exploration-visualization/
│   ├── rfm-customer-segmentation/
│   ├── ltv-predictor/
│   ├── retention-analysis/
│   ├── funnel-analysis/
│   ├── growth-model-analyzer/
│   └── content-analysis/
└── integrated_results/
    └── Comprehensive_Analysis_Report.html
```

**`/do-all` Output**:
```
complete_analysis/
├── data_quality_report/
│   ├── quality_assessment.json
│   ├── data_issues.log
│   └── quality_improvement_recommendations.md
├── exploratory_analysis/
│   ├── statistical_summary.csv
│   ├── pattern_analysis.md
│   └── correlation_analysis.json
├── hypothesis_reports/
│   ├── research_hypotheses.md
│   ├── experimental_design.md
│   └── validation_plan.md
├── visualizations/
│   ├── interactive_dashboard.html
│   ├── analysis_charts.png
│   └── visualization_code.py
├── generated_code/
│   ├── complete_analysis_pipeline.py
│   ├── data_preprocessing.py
│   ├── quality_checks.py
│   └── analysis_functions.py
├── final_report/
│   ├── comprehensive_analysis_report.html/pdf/docx
│   ├── executive_summary.html/pdf/docx
│   └── technical_appendix.html/pdf/docx
└── workflow_log/
    ├── analysis_progress.log
    ├── human_feedback.log
    └── execution_summary.md
```

### Usage Examples

**Quick Business Insights** (2-5 minutes):
```bash
/do-more
# → Open: do_more_analysis/integrated_results/Comprehensive_Analysis_Report.html
# → Review KPIs, charts, and recommendations
# → Make data-driven decisions
```

**Complete Research Analysis** (10-30 minutes):
```bash
/do-all
# → [Checkpoint 1] Review data quality: 85/100, proceed?
# → [Checkpoint 2] Hypotheses: "Mobile users have higher retention", approve?
# → [Checkpoint 3] Visualizations: Interactive dashboard ready, good?
# → Receive comprehensive report with code, dashboards, and documentation
# → complete_analysis/ directory with all artifacts
```

### Best Practices

**For `/do-more`:**
1. Ensure data is clean and in `data_storage/`
2. Run when you need quick insights
3. Use HTML report for interactive exploration
4. Export VIP lists for immediate action

**For `/do-all`:**
1. Schedule dedicated time for interactive session
2. Prepare to provide feedback at checkpoints
3. Choose domain based on your goals
4. Select format based on your audience
5. Review generated code before production use

## /do-more Command Deep Dive

### What Makes /do-more Special?

**1. Zero Configuration**
- No need to specify which skills to run
- No need to manually select data files
- No complex parameters required

**2. Intelligent Analysis**
- Automatically detects data types (e-commerce, user behavior, marketing, etc.)
- Matches optimal skills based on data characteristics
- Executes skills in scientifically determined order

**3. Comprehensive Output**
- 7+ specialized analyses in one run
- 20+ visualizations automatically generated
- Interactive HTML report with embedded charts
- CSV exports and detailed markdown reports

**4. Production Ready**
- Handles missing data gracefully
- Robust error handling and logging
- Professional-grade visualizations
- Executive-ready reporting

### Skill Matching Logic

**For E-commerce Data** (Orders + Customers + Products):
- ✓ Data Exploration & Visualization
- ✓ RFM Customer Segmentation
- ✓ LTV Prediction
- ✓ Retention Analysis
- ✓ Funnel Analysis
- ✓ Growth Model Analysis
- ✓ Content Analysis (if reviews available)

**For User Behavior Data** (Events + Sessions):
- ✓ Data Exploration & Visualization
- ✓ User Profiling Analysis
- ✓ Retention Analysis
- ✓ Funnel Analysis
- ✓ Recommender System

**For Marketing Data** (Campaigns + Attributions):
- ✓ Data Exploration & Visualization
- ✓ Attribution Analysis
- ✓ Growth Model Analysis
- ✓ A/B Testing Analyzer
- ✓ Funnel Analysis

### Output Formats

**Interactive HTML Report** (Comprehensive_Analysis_Report.html):
- Modern gradient design with smooth animations
- Sticky navigation for quick section jumping
- KPI cards with key metrics
- Embedded charts (base64 encoded)
- Responsive layout for all devices
- Strategic insights and recommendations

**Individual Skill Outputs**:
- Each skill generates its own directory with:
  - Analysis scripts (Python)
  - CSV exports (data, predictions, lists)
  - PNG visualizations (3-5 charts per skill)
  - Markdown reports
  - JSON summaries

### Best Practices

1. **Data Preparation**
   - Place all related data files in `data_storage/`
   - Use consistent naming conventions
   - Ensure proper date formats

2. **Running Analysis**
   - Simply run `/do-more` - no arguments needed
   - Wait for all skills to complete (2-5 minutes)
   - Check the generated HTML report

3. **Reviewing Results**
   - Start with the HTML report for overview
   - Dive into individual skill directories for details
   - Export VIP customer lists for marketing
   - Use insights for strategic planning

4. **Iterating**
   - Add new data to `data_storage/`
   - Run `/do-more` again for updated analysis
   - Compare results over time

## Important Implementation Notes

### Chinese Language Support
- All skills support Chinese field names and data formats
- Reports and documentation should be generated in Chinese when possible
- Visualization skills handle Chinese font rendering automatically

### Analysis Code Persistence
- All detailed analysis code and validation results must be saved to the project
- Skills automatically generate comprehensive documentation and validation reports
- Use `analysis_reports/` directory for all analytical outputs

### Quality Assurance
- The validate-analysis hook runs automatically after any code modification
- Skills include built-in data quality validation and error handling
- All analysis results include statistical validation and confidence intervals

### Security Considerations
- Python execution is explicitly allowed in settings.local.json
- Validation hook checks for dangerous code patterns (os.system, eval, exec, subprocess)
- All file operations are logged and validated

## Usage Examples by Data Type

### E-commerce Transaction Data
```bash
/rfm-customer-segmentation olist_order_items_dataset.csv
/user-profiling-analysis [customer_behavior_data.csv]
/ltv-predictor [customer_ltv_data.csv]
```

### Marketing Campaign Data
```bash
/attribution-analysis-modeling [marketing_touchpoints.csv]
/growth-model-analyzer [user_acquisition_data.csv]
/funnel-analysis [conversion_funnel_data.csv]
```

### General Dataset Analysis
```bash
/do-all  # Complete automated workflow
/data-exploration-visualization [dataset.csv]  # EDA focused
/regression-analysis-modeling [dataset.csv]  # Predictive modeling
```

## Development Notes

### Adding New Skills
1. Create skill directory in `.claude/skills/`
2. Add `SKILL.md` with skill definition and usage instructions
3. Implement analysis scripts following existing patterns
4. Include Chinese language support and proper documentation
5. Add quality validation and statistical testing
6. Create `quick_test.py` for functionality verification
7. Add `requirements.txt` with dependencies

### Skill Structure Standards
Each skill must contain:
- `SKILL.md`: Detailed skill documentation
- `README.md`: Usage guides and examples
- `scripts/`: Core functionality modules
- `examples/`: Sample datasets and usage examples
- `requirements.txt`: Python dependencies
- `quick_test.py`: Quick functionality tests

### Custom Analysis Workflows
- Skills can be combined for complex multi-stage analysis
- The `/do-all` command provides complete analysis workflow (no skills required)
- Hooks ensure consistent validation and context loading across all operations

### Environment Requirements
- Python 3.8+ (tested with Python 3.11.9)
- Claude Code with sub-agents and skills enabled
- Core data science stack: pandas, numpy, scikit-learn, matplotlib, seaborn
- Additional dependencies per skill (see individual requirements.txt files)