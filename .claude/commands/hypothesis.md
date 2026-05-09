---
allowed-tools: Task, Read, Write, Bash, Grep, Glob
argument-hint: [dataset] [domain]
description: Generate research hypotheses and experimental designs based on data patterns
---

# Hypothesis Generation Command

Generate research hypotheses and experimental designs for dataset `$1` in domain `$2` using the hypothesis-generator subagent.

## Context
- Dataset location: @data_storage/$1
- Analysis domain: $2 (user-behavior, business-impact, technical-performance, custom)
- Current working directory: !`pwd`
- Output directory: ./hypothesis_reports/
- Available research methodologies and experimental designs
- Statistical analysis capabilities

## Your Task

Use the hypothesis-generator subagent to create rigorous, testable hypotheses:

### 1. Pattern Analysis
- Identify significant correlations and relationships
- Detect temporal patterns and trends
- Discover clusters and segments in the data
- Recognize anomalies and unusual patterns

### 2. Hypothesis Formulation
- Create clear, testable hypotheses
- Define null and alternative hypotheses
- Specify variables and their relationships
- Establish measurable outcomes and success criteria

### 3. Experimental Design
- Select appropriate research methodologies
- Design experimental approaches to test hypotheses
- Determine sample size and power requirements
- Plan data collection and measurement procedures

### 4. Validation Strategy
- Define statistical testing approaches
- Establish success criteria and metrics
- Plan for replication and verification
- Consider alternative explanations and approaches

## Analysis Domains

### User Behavior
- **Engagement Patterns**: User interaction and engagement hypotheses
- **Conversion Optimization**: Conversion rate and funnel analysis hypotheses
- **Retention and Churn**: User retention and churn prediction hypotheses
- **Segmentation**: User behavior segmentation hypotheses
- **Journey Analysis**: User journey and path analysis hypotheses

### Business Impact
- **Revenue Optimization**: Revenue generation and growth hypotheses
- **Cost Reduction**: Cost efficiency and optimization hypotheses
- **Market Expansion**: Market growth and expansion hypotheses
- **Customer Satisfaction**: Customer experience and satisfaction hypotheses
- **Operational Efficiency**: Process improvement and efficiency hypotheses

### Technical Performance
- **System Optimization**: Performance and scalability hypotheses
- **Reliability**: System stability and reliability hypotheses
- **Security**: Security vulnerability and protection hypotheses
- **User Experience**: Technical UX and performance hypotheses
- **Integration**: System integration and compatibility hypotheses

### Custom
- **Domain-Specific**: Custom domain-specific hypotheses
- **Research-Oriented**: Academic and research hypotheses
- **Experimental**: Novel experimental hypotheses
- **Predictive**: Predictive modeling hypotheses

## Hypothesis Types

### Descriptive Hypotheses
Describe patterns and relationships in the data without inferring causation.

**Example**: "There is a positive correlation between user engagement time and conversion rates."

### Explanatory Hypotheses
Explain underlying mechanisms and causal relationships.

**Example**: "Increased user engagement time leads to higher conversion rates due to improved product understanding."

### Predictive Hypotheses
Forecast future outcomes based on current patterns.

**Example**: "Users with engagement time > 5 minutes are 3x more likely to convert within 30 days."

### Prescriptive Hypotheses
Recommend optimal actions and interventions.

**Example**: "Implementing personalized recommendations will increase user engagement by 25%."

## Research Methodologies

### Experimental Designs
- **A/B Testing**: Randomized controlled experiments
- **Multivariate Testing**: Multiple variable experiments
- **Longitudinal Studies**: Time-series analysis
- **Cross-sectional Studies**: Point-in-time analysis
- **Quasi-experiments**: Non-randomized designs
- **Case Studies**: In-depth analysis of specific cases

### Statistical Approaches
- **Hypothesis Testing**: Statistical significance testing
- **Confidence Intervals**: Estimation of effect sizes
- **Bayesian Methods**: Bayesian hypothesis testing
- **Power Analysis**: Statistical power calculation
- **Effect Size Measurement**: Quantifying relationship strength

### Validation Methods
- **Cross-validation**: Model validation techniques
- **Bootstrapping**: Resampling validation
- **Sensitivity Analysis**: Testing robustness
- **Replication Studies**: Independent verification
- **Meta-analysis**: Synthesis of multiple studies

## Expected Output

### Hypothesis Documentation
- `hypothesis_reports/$1_$2_hypotheses.md` - Hypothesis documentation
- `hypothesis_reports/$1_$2_experimental_design.md` - Experimental design
- `hypothesis_reports/$1_$2_validation_plan.md` - Validation strategy
- `hypothesis_reports/$1_$2_research_proposal.md` - Research proposal

### Hypothesis Structure
Each hypothesis includes:
- **Clear Statement**: Testable hypothesis statement
- **Rationale**: Justification and theoretical foundation
- **Variables**: Independent and dependent variables
- **Methodology**: Proposed testing approach
- **Success Criteria**: Metrics for validation
- **Expected Outcomes**: Predicted results
- **Alternative Explanations**: Other possible explanations

### Experimental Design
Comprehensive experimental design including:
- **Research Questions**: Clear research questions
- **Methodology**: Detailed experimental approach
- **Sample Size**: Power analysis and justification
- **Variables**: Measurement and operationalization
- **Procedure**: Step-by-step experimental process
- **Analysis Plan**: Statistical analysis approach
- **Timeline**: Project schedule and milestones
- **Resources**: Required resources and budget

## Working Process

### 1. Data Pattern Discovery
```python
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def discover_research_patterns(data):
    """Discover patterns that suggest research hypotheses"""
    patterns = {}

    # Correlation analysis
    correlation_matrix = data.corr()
    strong_correlations = correlation_matrix[(correlation_matrix.abs() > 0.5) & (correlation_matrix.abs() < 1.0)]
    patterns['strong_correlations'] = strong_correlations

    # Trend analysis
    patterns['trends'] = analyze_temporal_trends(data)

    # Cluster analysis
    patterns['clusters'] = perform_cluster_analysis(data)

    # Statistical anomalies
    patterns['anomalies'] = detect_statistical_anomalies(data)

    return patterns
```

### 2. Hypothesis Generation
```python
def generate_domain_hypotheses(patterns, domain):
    """Generate hypotheses based on domain-specific patterns"""
    hypotheses = []

    if domain == 'user-behavior':
        hypotheses.extend(generate_user_behavior_hypotheses(patterns))
    elif domain == 'business-impact':
        hypotheses.extend(generate_business_hypotheses(patterns))
    elif domain == 'technical-performance':
        hypotheses.extend(generate_technical_hypotheses(patterns))
    else:
        hypotheses.extend(generate_custom_hypotheses(patterns, domain))

    return hypotheses

def generate_user_behavior_hypotheses(patterns):
    """Generate user behavior specific hypotheses"""
    hypotheses = []

    # Engagement hypotheses
    if 'engagement_time' in patterns['trends']:
        hypothesis = {
            'type': 'predictive',
            'statement': 'Users with higher engagement time show increased conversion rates',
            'null_hypothesis': 'H₀: Engagement time has no effect on conversion rates',
            'alternative_hypothesis': 'H₁: Higher engagement time leads to increased conversion rates',
            'variables': {
                'independent': 'engagement_time',
                'dependent': 'conversion_rate'
            },
            'methodology': 'A/B testing with engagement time manipulation'
        }
        hypotheses.append(hypothesis)

    return hypotheses
```

### 3. Experimental Design
```python
def design_experiment(hypothesis, data_characteristics):
    """Design experimental approach to test hypothesis"""
    experiment = {
        'hypothesis': hypothesis,
        'research_design': select_research_design(hypothesis),
        'sample_size': calculate_sample_size(hypothesis, data_characteristics),
        'variables': operationalize_variables(hypothesis),
        'measurement_protocol': define_measurements(hypothesis),
        'data_collection': plan_data_collection(hypothesis),
        'analysis_plan': define_analysis_plan(hypothesis),
        'quality_controls': define_quality_controls(hypothesis),
        'timeline': estimate_timeline(hypothesis),
        'resources': estimate_resources(hypothesis),
        'risks': identify_risks(hypothesis),
        'mitigation': develop_mitigation_strategies(hypothesis)
    }

    return experiment
```

### 4. Validation Strategy
```python
def create_validation_strategy(hypothesis, experiment):
    """Create comprehensive validation strategy"""
    validation = {
        'statistical_tests': select_statistical_tests(hypothesis),
        'success_criteria': define_success_criteria(hypothesis),
        'confidence_level': 0.95,  # 95% confidence level
        'power_analysis': {
            'target_power': 0.80,
            'effect_size': estimate_effect_size(hypothesis),
            'sample_size': experiment['sample_size']
        },
        'validation_timeline': plan_validation_timeline(experiment),
        'replication_plan': design_replication_strategy(hypothesis),
        'alternative_approaches': generate_alternative_methods(hypothesis),
        'quality_assurance': define_quality_measures(hypothesis)
    }

    return validation
```

## Quality Standards

### Hypothesis Quality
- **Testability**: Must be empirically testable
- **Falsifiability**: Must be possible to prove wrong
- **Specificity**: Clear predictions about relationships
- **Measurability**: Variables must be quantifiable
- **Relevance**: Addresses meaningful questions
- **Theoretical Foundation**: Based on sound reasoning

### Experimental Rigor
- **Control**: Appropriate control conditions
- **Randomization**: Proper random assignment
- **Blinding**: Single or double blinding as appropriate
- **Replication**: Designed for reproducibility
- **Statistical Power**: Adequate sample size
- **Internal Validity**: Minimizes confounding variables
- **External Validity**: Generalizable to target population

### Methodological Soundness
- **Appropriate Methods**: Suitable research methodology
- **Valid Measurements**: Reliable and valid measurement instruments
- **Statistical Validity**: Appropriate statistical tests
- **Ethical Considerations**: Ethical research practices
- **Feasibility**: Practical implementation considerations

## Best Practices

### Hypothesis Development
- **Data-Driven**: Base hypotheses on observed patterns
- **Theoretical Grounding**: Support with relevant theory
- **Clear Articulation**: Unambiguous hypothesis statements
- **Variable Definition**: Clear operational definitions
- **Scope Management**: Manageable scope and complexity

### Experimental Design
- **Research Question Alignment**: Design directly addresses research questions
- **Methodology Selection**: Choose appropriate research methods
- **Sample Size Justification**: Power analysis for sample size
- **Control Group**: Appropriate control conditions
- **Confounding Variables**: Identify and control for confounders

### Validation Planning
- **Success Criteria**: Clear criteria for hypothesis validation
- **Statistical Planning**: Appropriate statistical tests and methods
- **Multiple Approaches**: Consider multiple validation approaches
- **Replication Planning**: Plan for independent verification
- **Risk Assessment**: Identify and mitigate risks

## Example Usage
```bash
/hypothesis user_behavior.csv user-behavior
/hypothesis sales_data.csv business-impact
/hypothesis system_metrics.csv technical-performance
/hypothesis research_data.csv custom
```

## Integration with Other Commands
- Use after `/analyze` to base hypotheses on data insights
- Combine with `/visualize` to visualize patterns supporting hypotheses
- Follow with `/generate` to create experimental code
- Precede with `/quality` to ensure hypothesis data quality

## Notes
- Dataset should be located in the data_storage/ directory
- Hypothesis reports will be saved to hypothesis_reports/ directory
- Use Task tool to delegate to hypothesis-generator subagent
- Consider both statistical and practical significance
- Plan for replication and verification of findings
- Document assumptions and limitations of hypotheses