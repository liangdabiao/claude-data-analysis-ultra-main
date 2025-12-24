---
allowed-tools: Task, Read, Write, Bash, Grep, Glob
argument-hint: [dataset] [action]
description: Perform data quality validation, checks, and monitoring for specified dataset
---

# Data Quality Command

Execute data quality operations on dataset `$1` with action `$2` using the quality-assurance subagent.

## Context
- Dataset location: @data_storage/$1
- Quality action: $2 (check, clean, validate, monitor, profile)
- Current working directory: !`pwd`
- Output directory: ./quality_reports/
- Quality rules and validation thresholds
- Available quality metrics and KPIs

## Your Task

Use the quality-assurance subagent to perform comprehensive data quality operations:

### 1. Quality Assessment
- Analyze data completeness and accuracy
- Check data consistency and validity
- Assess data uniqueness and timeliness
- Evaluate overall data integrity

### 2. Issue Identification
- Detect missing values and data gaps
- Identify outliers and anomalies
- Find duplicate records and inconsistencies
- Discover format violations and data type issues

### 3. Quality Improvement
- Implement data cleaning procedures
- Apply data validation rules
- Execute data transformation operations
- Perform data standardization

### 4. Monitoring and Reporting
- Generate quality metrics and KPIs
- Create quality assessment reports
- Set up ongoing quality monitoring
- Provide quality improvement recommendations

## Quality Actions

### Check
Perform basic data quality assessment:
- Completeness analysis
- Basic accuracy validation
- Simple consistency checks
- Summary quality metrics

### Clean
Execute data cleaning operations:
- Remove duplicate records
- Handle missing values
- Correct format violations
- Standardize data formats

### Validate
Comprehensive data validation:
- Statistical validation
- Business rule validation
- Cross-field validation
- Referential integrity checks

### Monitor
Set up quality monitoring:
- Continuous quality tracking
- Alert threshold configuration
- Quality trend analysis
- Performance metrics monitoring

### Profile
Generate comprehensive data profile:
- Detailed data statistics
- Distribution analysis
- Relationship analysis
- Data lineage documentation

## Quality Dimensions

### Completeness
- **Missing Value Analysis**: Identify and quantify missing data
- **Required Field Validation**: Check presence of mandatory fields
- **Record Completeness**: Assess completeness of individual records
- **Data Coverage**: Evaluate coverage of expected data range

### Accuracy
- **Statistical Validation**: Verify statistical properties
- **Business Rule Validation**: Check against business constraints
- **Range Validation**: Ensure values within expected ranges
- **Format Validation**: Verify correct data formats

### Consistency
- **Cross-Field Validation**: Check logical consistency between fields
- **Temporal Consistency**: Validate time-based consistency
- **Referential Integrity**: Check relationship consistency
- **Format Consistency**: Ensure consistent formatting

### Timeliness
- **Data Currency**: Assess how current the data is
- **Update Frequency**: Evaluate data refresh rates
- **Latency Analysis**: Measure data processing delays
- **Freshness Metrics**: Track data age and relevance

### Uniqueness
- **Duplicate Detection**: Identify and eliminate duplicate records
- **Primary Key Validation**: Verify unique identifiers
- **Record Uniqueness**: Assess overall uniqueness
- **Relationship Uniqueness**: Check unique relationships

### Validity
- **Data Type Validation**: Verify correct data types
- **Domain Validation**: Check against allowed value domains
- **Pattern Validation**: Validate against expected patterns
- **Constraint Validation**: Check database and business constraints

## Expected Output

### Quality Reports
- `quality_reports/$1_quality_check.json` - Quality assessment results
- `quality_reports/$1_data_profile.json` - Comprehensive data profile
- `quality_reports/$1_validation_report.md` - Detailed validation report
- `quality_reports/$1_monitoring_config.json` - Monitoring configuration

### Quality Metrics
- **Overall Quality Score**: Composite quality metric (0-100)
- **Dimension Scores**: Individual quality dimension scores
- **Issue Counts**: Number and severity of quality issues
- **Improvement Metrics**: Quality improvement tracking

### Data Outputs
- **Cleaned Data**: Quality-improved dataset versions
- **Validation Logs**: Detailed validation results
- **Error Reports**: Specific error descriptions and locations
- **Recommendations**: Actionable improvement suggestions

## Working Process

### 1. Data Loading and Profiling
```python
import pandas as pd
import numpy as np
from scipy import stats

def load_and_profile_data(dataset_path):
    """Load dataset and create initial profile"""
    data = pd.read_csv(dataset_path)

    profile = {
        'basic_info': {
            'shape': data.shape,
            'columns': list(data.columns),
            'data_types': data.dtypes.to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum()
        },
        'quality_metrics': {
            'completeness': calculate_completeness(data),
            'uniqueness': calculate_uniqueness(data),
            'consistency': calculate_consistency(data)
        }
    }

    return data, profile
```

### 2. Quality Assessment
```python
def comprehensive_quality_assessment(data):
    """Perform comprehensive data quality assessment"""
    assessment = {
        'completeness': assess_completeness(data),
        'accuracy': assess_accuracy(data),
        'consistency': assess_consistency(data),
        'timeliness': assess_timeliness(data),
        'uniqueness': assess_uniqueness(data),
        'validity': assess_validity(data)
    }

    # Calculate overall quality score
    dimension_scores = [assessment[dim]['score'] for dim in assessment]
    overall_score = np.mean(dimension_scores)

    assessment['overall_score'] = overall_score
    assessment['quality_grade'] = assign_quality_grade(overall_score)

    return assessment
```

### 3. Data Cleaning
```python
def clean_data(data, quality_issues):
    """Clean data based on identified quality issues"""
    cleaned_data = data.copy()

    # Handle missing values
    if quality_issues['completeness']['missing_values']:
        cleaned_data = handle_missing_values(cleaned_data, quality_issues['completeness'])

    # Remove duplicates
    if quality_issues['uniqueness']['duplicates'] > 0:
        cleaned_data = remove_duplicates(cleaned_data)

    # Fix format issues
    if quality_issues['validity']['format_issues']:
        cleaned_data = fix_format_issues(cleaned_data, quality_issues['validity'])

    # Apply business rules
    cleaned_data = apply_business_rules(cleaned_data)

    return cleaned_data
```

### 4. Monitoring Setup
```python
def setup_quality_monitoring(data, monitoring_config):
    """Set up ongoing quality monitoring"""
    monitoring_plan = {
        'data_source': data,
        'frequency': monitoring_config['frequency'],
        'metrics': define_quality_metrics(),
        'thresholds': define_quality_thresholds(),
        'alerts': define_alert_rules(),
        'reporting': define_reporting_schedule()
    }

    return monitoring_plan
```

## Quality Thresholds and Standards

### Quality Score Interpretation
- **90-100**: Excellent - High quality data suitable for critical analysis
- **80-89**: Good - Reliable data with minor issues
- **70-79**: Fair - Usable data with some limitations
- **60-69**: Poor - Data requires significant cleaning
- **Below 60**: Unacceptable - Data not suitable for analysis

### Dimension-Specific Thresholds
- **Completeness**: ≥ 95% for critical fields, ≥ 85% for others
- **Accuracy**: ≥ 98% validation success rate
- **Consistency**: ≥ 95% consistency across related fields
- **Timeliness**: Data age ≤ 24 hours for real-time needs
- **Uniqueness**: ≥ 99% unique records in key fields
- **Validity**: ≥ 97% valid format compliance

## Best Practices

### Quality Management
- **Prevention Over Detection**: Focus on preventing quality issues
- **Continuous Monitoring**: Implement ongoing quality tracking
- **Root Cause Analysis**: Address underlying causes of quality issues
- **Standardization**: Use consistent quality standards and processes
- **Documentation**: Thoroughly document quality rules and procedures

### Validation Design
- **Comprehensive Coverage**: Validate all critical data elements
- **Risk-Based Approach**: Prioritize high-impact data elements
- **Statistical Rigor**: Use appropriate statistical methods
- **Business Alignment**: Align validation with business requirements
- **Performance Consideration**: Optimize validation efficiency

### Issue Management
- **Prioritization**: Address issues based on business impact
- **Systematic Resolution**: Use structured problem-solving approaches
- **Preventive Action**: Implement measures to prevent recurrence
- **Stakeholder Communication**: Keep stakeholders informed
- **Continuous Improvement**: Refine quality processes over time

## Error Handling and Recovery

### Common Quality Issues
1. **Missing Data**: Identify patterns and implement appropriate handling
2. **Outliers**: Detect and handle anomalous values appropriately
3. **Format Issues**: Standardize formats and validate compliance
4. **Consistency Violations**: Resolve logical inconsistencies
5. **Referential Integrity**: Fix relationship violations

### Recovery Strategies
```python
class QualityRecoveryManager:
    def __init__(self):
        self.recovery_strategies = {
            'missing_data': self.recover_missing_data,
            'outliers': self.handle_outliers,
            'format_issues': self.fix_format_issues,
            'consistency_violations': self.resolve_inconsistencies
        }

    def recover_missing_data(self, data, missing_pattern):
        """Recover missing data based on identified patterns"""
        if missing_pattern['mechanism'] == 'MCAR':
            # Use mean/median imputation for MCAR
            return self.impute_mcar(data, missing_pattern)
        elif missing_pattern['mechanism'] == 'MAR':
            # Use model-based imputation for MAR
            return self.impute_mar(data, missing_pattern)
        else:
            # Use specialized techniques for MNAR
            return self.impute_mnar(data, missing_pattern)
```

## Integration with Other Systems

### Data Pipeline Integration
- **Pre-processing Validation**: Validate data before analysis
- **Real-time Monitoring**: Monitor streaming data quality
- **Batch Processing**: Validate batch processing results
- **Data Lake Governance**: Ensure data lake quality standards

### Business Intelligence Integration
- **Quality Metrics in Dashboards**: Include quality indicators
- **Data Confidence Levels**: Show reliability of data insights
- **Decision Support**: Provide quality context for decisions
- **Performance Tracking**: Monitor quality trends over time

## Example Usage
```bash
/quality user_behavior.csv check
/quality sales_data.csv clean
/quality customer_data.csv validate
/quality financial_data.csv monitor
/quality inventory_data.csv profile
```

## Integration with Other Commands
- Use before `/analyze` to ensure data quality
- Combine with `/visualize` for quality visualization
- Follow with `/report` for quality reporting
- Precede with `/generate` for quality-aware code generation

## Notes
- Dataset should be located in the data_storage/ directory
- Quality reports will be saved to quality_reports/ directory
- Use Task tool to delegate to quality-assurance subagent
- Consider business context when setting quality thresholds
- Review and understand quality issues before analysis
- Implement continuous quality monitoring for critical data