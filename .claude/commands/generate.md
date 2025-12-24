---
allowed-tools: Task, Read, Write, Bash, Grep, Glob
argument-hint: [language] [analysis_type]
description: Generate analysis code in specified language and analysis type
---

# Code Generation Command

Generate data analysis code in `$1` language for `$2` analysis type using the code-generator subagent.

## Context
- Programming language: $1 (python, r, sql, javascript)
- Analysis type: $2 (data-cleaning, statistical, visualization, machine-learning, custom)
- Current working directory: !`pwd`
- Output directory: ./generated_code/
- Available libraries and frameworks based on language

## Your Task

Use the code-generator subagent to create high-quality, production-ready analysis code:

### 1. Requirements Analysis
- Understand the specific analysis requirements
- Identify appropriate libraries and frameworks
- Consider data types and volumes
- Plan for scalability and performance

### 2. Code Architecture
- Design modular, reusable code structure
- Implement proper error handling
- Include comprehensive documentation
- Add unit tests where appropriate

### 3. Implementation
- Write clean, efficient, and maintainable code
- Include proper data validation
- Implement best practices for the language
- Add logging and debugging capabilities

### 4. Documentation
- Create comprehensive code documentation
- Include usage examples and tutorials
- Provide troubleshooting guidance
- Document dependencies and requirements

## Language Support

### Python
- **Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn, plotly
- **Use Cases**: Data cleaning, statistical analysis, machine learning, visualization
- **Output**: Jupyter notebooks, Python scripts, modules

### R
- **Libraries**: tidyverse, ggplot2, dplyr, caret, shiny
- **Use Cases**: Statistical analysis, data visualization, bioinformatics
- **Output**: R scripts, R Markdown documents, Shiny apps

### SQL
- **Dialects**: PostgreSQL, MySQL, SQLite, BigQuery, Redshift
- **Use Cases**: Data extraction, aggregation, reporting, ETL
- **Output**: SQL queries, stored procedures, views

### JavaScript
- **Libraries**: D3.js, Plotly.js, Chart.js, TensorFlow.js
- **Use Cases**: Web visualizations, interactive dashboards, client-side ML
- **Output**: HTML/JS files, Node.js scripts, web applications

## Analysis Types

### Data Cleaning
- Missing value handling
- Outlier detection and treatment
- Data type conversion
- Normalization and standardization
- Feature engineering

### Statistical Analysis
- Descriptive statistics
- Hypothesis testing
- Correlation and regression
- Time series analysis
- ANOVA and t-tests

### Visualization
- Chart creation code
- Dashboard implementation
- Interactive visualizations
- Custom plot types
- Animation and transitions

### Machine Learning
- Data preprocessing
- Model training and evaluation
- Feature selection
- Hyperparameter tuning
- Model deployment

### Custom
- User-specific requirements
- Domain-specific analysis
- Integration with existing systems
- Performance optimization
- Custom algorithms

## Expected Output

### Code Files
- `generated_code/$1_$2_analysis.py` - Main analysis script
- `generated_code/$1_$2_utils.py` - Utility functions
- `generated_code/$1_$2_config.py` - Configuration settings
- `generated_code/$1_$2_test.py` - Unit tests
- `generated_code/requirements_$1.txt` - Dependencies

### Documentation
- **README.md**: Usage instructions and examples
- **API Documentation**: Function and class documentation
- **Tutorials**: Step-by-step guides
- **Troubleshooting**: Common issues and solutions

## Code Quality Standards

### Python Code Standards
```python
"""
High-quality Python code template for data analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path

class DataAnalyzer:
    """
    Data analysis class with comprehensive functionality

    Args:
        data_path (str): Path to input data file
        config (Dict): Configuration parameters

    Attributes:
        data (pd.DataFrame): Loaded dataset
        config (Dict): Configuration settings
        logger (logging.Logger): Logger instance
    """

    def __init__(self, data_path: str, config: Dict = None):
        self.data_path = Path(data_path)
        self.config = config or {}
        self.data = None
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

    def load_data(self) -> pd.DataFrame:
        """
        Load data from file with error handling

        Returns:
            pd.DataFrame: Loaded dataset

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data format is invalid
        """
        try:
            # Implementation with proper error handling
            pass
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
```

### SQL Code Standards
```sql
-- High-quality SQL template for data analysis
-- Include proper comments and documentation

-- Analysis: Customer Segmentation
-- Purpose: Identify customer segments based on purchase behavior
-- Dependencies: customers, orders, order_items tables

WITH customer_summary AS (
    -- Calculate customer-level metrics
    SELECT
        c.customer_id,
        c.customer_name,
        c.signup_date,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(oi.quantity * oi.unit_price) AS total_revenue,
        AVG(oi.quantity * oi.unit_price) AS avg_order_value,
        MAX(o.order_date) AS last_order_date
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    GROUP BY c.customer_id, c.customer_name, c.signup_date
),

segment_calculation AS (
    -- Calculate RFM metrics and segments
    SELECT
        customer_id,
        customer_name,
        total_orders,
        total_revenue,
        avg_order_value,
        -- Recency: days since last order
        DATEDIFF(CURRENT_DATE, last_order_date) AS recency_days,
        -- Frequency: total orders
        total_orders AS frequency,
        -- Monetary: total revenue
        total_revenue AS monetary,
        -- Segment assignment
        CASE
            WHEN total_revenue > 10000 THEN 'High Value'
            WHEN total_revenue > 5000 THEN 'Medium Value'
            WHEN total_revenue > 1000 THEN 'Low Value'
            ELSE 'New Customer'
        END AS customer_segment
    FROM customer_summary
    WHERE last_order_date IS NOT NULL
)

-- Final result with segment insights
SELECT
    customer_segment,
    COUNT(customer_id) AS customer_count,
    AVG(total_orders) AS avg_orders,
    AVG(total_revenue) AS avg_revenue,
    AVG(recency_days) AS avg_recency,
    ROUND(COUNT(customer_id) * 100.0 / SUM(COUNT(customer_id)) OVER (), 2) AS percentage
FROM segment_calculation
GROUP BY customer_segment
ORDER BY total_revenue DESC;
```

## Quality Assurance

### Code Review Checklist
- [ ] Code follows language-specific style guidelines
- [ ] Proper error handling and validation
- [ ] Comprehensive documentation and comments
- [ ] Unit tests included where appropriate
- [ ] Performance considerations addressed
- [ ] Security best practices followed
- [ ] Dependencies clearly specified

### Testing Strategy
- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test data flow and dependencies
- **Performance Tests**: Validate with large datasets
- **Security Tests**: Check for vulnerabilities

## Example Usage
```bash
/generate python data-cleaning
/generate r statistical
/generate sql reporting
/generate javascript visualization
/generate python machine-learning
/generate custom custom-analysis
```

## Best Practices

### General Principles
- **DRY**: Don't Repeat Yourself - write reusable code
- **KISS**: Keep It Simple, Stupid - avoid unnecessary complexity
- **YAGNI**: You Ain't Gonna Need It - avoid over-engineering
- **SOLID**: Follow SOLID principles for object-oriented design

### Documentation Standards
- **Docstrings**: Comprehensive function documentation
- **Comments**: Explain complex logic and algorithms
- **README**: Project overview and setup instructions
- **Examples**: Provide usage examples and tutorials

### Version Control
- Include .gitignore file
- Document versioning strategy
- Include changelog
- Tag releases appropriately

## Notes
- Generated code will be saved to generated_code/ directory
- Code includes comprehensive documentation and examples
- Use Task tool to delegate to code-generator subagent
- Review and test generated code before production use
- Consider using /analyze command first to understand data requirements

## Integration with Other Commands
- Use after `/analyze` to understand data requirements
- Combine with `/visualize` for visualization code
- Follow with `/quality` for code quality validation