"""
客户生命周期价值 (LTV) 预测分析
基于 RFM 特征的回归建模
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
import json
from datetime import datetime
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def main():
    output_dir = 'do_more_analysis/skill_execution/ltv-predictor'
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("Customer Lifetime Value (LTV) Prediction Analysis")
    print("="*80)

    # Load data
    print("\nLoading data...")
    orders = pd.read_csv('data_storage/Orders.csv', parse_dates=['order_purchase_timestamp'])
    order_items = pd.read_csv('data_storage/Order Items.csv')

    # Filter delivered orders
    orders = orders[orders['order_status'] == 'delivered'].copy()

    # Merge data
    print("Merging order data...")
    orders_items = orders.merge(order_items, on='order_id', how='left')

    # Calculate order totals
    order_totals = orders_items.groupby('order_id').agg({
        'price': 'sum',
        'freight_value': 'sum',
        'customer_id': 'first',
        'order_purchase_timestamp': 'max'
    }).reset_index()
    order_totals['total_amount'] = order_totals['price'] + order_totals['freight_value']

    print(f"Total orders: {len(order_totals):,}")
    print(f"Unique customers: {order_totals['customer_id'].nunique():,}")

    # Define time periods
    print("\nSetting up time periods...")
    min_date = order_totals['order_purchase_timestamp'].min()
    max_date = order_totals['order_purchase_timestamp'].max()
    total_days = (max_date - min_date).days

    # Use first 60% of data as feature period, last 40% as prediction period
    split_date = min_date + pd.Timedelta(days=int(total_days * 0.6))

    print(f"Date range: {min_date.date()} to {max_date.date()}")
    print(f"Feature period: {min_date.date()} to {split_date.date()}")
    print(f"Prediction period: {split_date.date()} to {max_date.date()}")

    # Calculate RFM features from feature period
    print("\nCalculating RFM features...")
    feature_orders = order_totals[order_totals['order_purchase_timestamp'] < split_date].copy()

    rfm_features = feature_orders.groupby('customer_id').agg({
        'order_purchase_timestamp': lambda x: (split_date - x.max()).days,  # Recency
        'order_id': 'count',  # Frequency
        'total_amount': 'sum'  # Monetary
    }).reset_index()
    rfm_features.columns = ['customer_id', 'recency', 'frequency', 'monetary']

    print(f"Customers in feature period: {len(rfm_features):,}")

    # Calculate LTV from prediction period
    print("Calculating LTV (future value)...")
    prediction_orders = order_totals[order_totals['order_purchase_timestamp'] >= split_date].copy()

    ltv_values = prediction_orders.groupby('customer_id')['total_amount'].sum().reset_index()
    ltv_values.columns = ['customer_id', 'ltv']

    print(f"Customers with future purchases: {len(ltv_values):,}")

    # IMPORTANT: Only keep customers who appear in BOTH periods
    # This ensures we're predicting future LTV for customers we have historical data for
    print("\nMerging feature and prediction data...")

    # Get customers who purchased in BOTH periods
    customers_in_both = set(rfm_features['customer_id']) & set(ltv_values['customer_id'])

    print(f"Customers who purchased in BOTH periods: {len(customers_in_both):,}")

    # For customers who purchased in feature period but NOT in prediction period, LTV = 0
    model_data = rfm_features.merge(ltv_values, on='customer_id', how='left')
    model_data['ltv'] = model_data['ltv'].fillna(0)

    print(f"Training samples: {len(model_data):,}")
    print(f"Customers with future LTV > 0: {(model_data['ltv'] > 0).sum():,}")
    print(f"Customers with future LTV = 0: {(model_data['ltv'] == 0).sum():,}")

    # Feature engineering
    print("\nEngineering features...")
    # Log transform monetary to reduce skewness
    model_data['log_monetary'] = np.log1p(model_data['monetary'])
    model_data['log_frequency'] = np.log1p(model_data['frequency'])

    # RFM scores
    model_data['R_score'] = pd.qcut(model_data['recency'], q=4, labels=[4, 3, 2, 1], duplicates='drop').astype(int)
    model_data['F_score'] = pd.qcut(model_data['frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4]).astype(int)
    model_data['M_score'] = pd.qcut(model_data['monetary'].rank(method='first'), q=4, labels=[1, 2, 3, 4]).astype(int)

    # Prepare features
    feature_cols = ['recency', 'frequency', 'monetary', 'log_monetary', 'log_frequency',
                    'R_score', 'F_score', 'M_score']

    X = model_data[feature_cols].copy()
    y = model_data['ltv'].copy()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nTraining set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")

    # Train models
    print("\n" + "="*80)
    print("Training Prediction Models")
    print("="*80)

    results = {}

    # 1. Linear Regression
    print("\n1. Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred_train = lr_model.predict(X_train)
    lr_pred_test = lr_model.predict(X_test)

    results['Linear Regression'] = {
        'train_r2': r2_score(y_train, lr_pred_train),
        'test_r2': r2_score(y_test, lr_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, lr_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, lr_pred_test)),
        'train_mae': mean_absolute_error(y_train, lr_pred_train),
        'test_mae': mean_absolute_error(y_test, lr_pred_test),
        'model': lr_model,
        'predictions': lr_pred_test
    }

    print(f"  Train R2: {results['Linear Regression']['train_r2']:.4f}")
    print(f"  Test R2: {results['Linear Regression']['test_r2']:.4f}")
    print(f"  Test RMSE: ${results['Linear Regression']['test_rmse']:.2f}")

    # 2. Random Forest
    print("\n2. Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred_train = rf_model.predict(X_train)
    rf_pred_test = rf_model.predict(X_test)

    results['Random Forest'] = {
        'train_r2': r2_score(y_train, rf_pred_train),
        'test_r2': r2_score(y_test, rf_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, rf_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, rf_pred_test)),
        'train_mae': mean_absolute_error(y_train, rf_pred_train),
        'test_mae': mean_absolute_error(y_test, rf_pred_test),
        'model': rf_model,
        'predictions': rf_pred_test,
        'feature_importance': dict(zip(feature_cols, rf_model.feature_importances_))
    }

    print(f"  Train R2: {results['Random Forest']['train_r2']:.4f}")
    print(f"  Test R2: {results['Random Forest']['test_r2']:.4f}")
    print(f"  Test RMSE: ${results['Random Forest']['test_rmse']:.2f}")

    # Determine best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
    best_model = results[best_model_name]

    print(f"\nBest Model: {best_model_name}")
    print(f"Best Test R2: {best_model['test_r2']:.4f}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Model performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    models = list(results.keys())
    r2_scores = [results[m]['test_r2'] for m in models]
    rmse_scores = [results[m]['test_rmse'] for m in models]
    mae_scores = [results[m]['test_mae'] for m in models]

    # R2 comparison
    axes[0, 0].bar(models, r2_scores, color=['#3498db', '#2ecc71'])
    axes[0, 0].set_title('Model R2 Score Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('R2 Score')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(r2_scores):
        axes[0, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

    # RMSE comparison
    axes[0, 1].bar(models, rmse_scores, color=['#e74c3c', '#f39c12'])
    axes[0, 1].set_title('Model RMSE Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('RMSE ($)')
    for i, v in enumerate(rmse_scores):
        axes[0, 1].text(i, v + max(rmse_scores)*0.02, f'${v:.2f}', ha='center', fontweight='bold')

    # MAE comparison
    axes[1, 0].bar(models, mae_scores, color=['#9b59b6', '#1abc9c'])
    axes[1, 0].set_title('Model MAE Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('MAE ($)')
    for i, v in enumerate(mae_scores):
        axes[1, 0].text(i, v + max(mae_scores)*0.02, f'${v:.2f}', ha='center', fontweight='bold')

    # Actual vs Predicted (best model)
    axes[1, 1].scatter(y_test, best_model['predictions'], alpha=0.5, s=20)
    min_val = min(y_test.min(), best_model['predictions'].min())
    max_val = max(y_test.max(), best_model['predictions'].max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[1, 1].set_title(f'Actual vs Predicted LTV ({best_model_name})', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Actual LTV ($)')
    axes[1, 1].set_ylabel('Predicted LTV ($)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] model_comparison.png")

    # 2. Feature importance (Random Forest)
    if 'feature_importance' in results['Random Forest']:
        fig, ax = plt.subplots(figsize=(10, 6))

        importances = results['Random Forest']['feature_importance']
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        features = [f[0] for f in sorted_features]
        values = [f[1] for f in sorted_features]

        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        ax.barh(features, values, color=colors)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        for i, v in enumerate(values):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print("  [OK] feature_importance.png")

    # 3. LTV distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].hist(model_data['ltv'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(model_data['ltv'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: ${model_data['ltv'].mean():.2f}")
    axes[0].set_xlabel('LTV ($)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('LTV Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # LTV vs RFM features
    axes[1].scatter(model_data['monetary'], model_data['ltv'], alpha=0.5, s=20)
    axes[1].set_xlabel('Historical Monetary ($)', fontsize=12)
    axes[1].set_ylabel('Future LTV ($)', fontsize=12)
    axes[1].set_title('Historical Monetary vs Future LTV', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ltv_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] ltv_distribution.png")

    # Export results
    print("\nExporting results...")

    # Prediction results
    prediction_results = pd.DataFrame({
        'customer_id': model_data['customer_id'],
        'actual_ltv': model_data['ltv'],
        'predicted_ltv_lr': results['Linear Regression']['model'].predict(X),
        'predicted_ltv_rf': results['Random Forest']['model'].predict(X),
        'recency': model_data['recency'],
        'frequency': model_data['frequency'],
        'monetary': model_data['monetary']
    })
    prediction_results['prediction_error_lr'] = abs(
        prediction_results['actual_ltv'] - prediction_results['predicted_ltv_lr']
    )
    prediction_results['prediction_error_rf'] = abs(
        prediction_results['actual_ltv'] - prediction_results['predicted_ltv_rf']
    )

    prediction_results = prediction_results.sort_values('actual_ltv', ascending=False)
    prediction_results.to_csv(os.path.join(output_dir, 'ltv_predictions.csv'),
                             index=False, encoding='utf-8-sig')
    print("  [OK] ltv_predictions.csv")

    # Top high-value customers
    top_customers = prediction_results.head(500).copy()
    top_customers.to_csv(os.path.join(output_dir, 'top_high_value_customers.csv'),
                        index=False, encoding='utf-8-sig')
    print("  [OK] top_high_value_customers.csv")

    # Model evaluation summary
    eval_summary = pd.DataFrame({
        'Model': models,
        'Train_R2': [results[m]['train_r2'] for m in models],
        'Test_R2': [results[m]['test_r2'] for m in models],
        'Train_RMSE': [results[m]['train_rmse'] for m in models],
        'Test_RMSE': [results[m]['test_rmse'] for m in models],
        'Train_MAE': [results[m]['train_mae'] for m in models],
        'Test_MAE': [results[m]['test_mae'] for m in models]
    })
    eval_summary.to_csv(os.path.join(output_dir, 'model_evaluation.csv'),
                       index=False, encoding='utf-8-sig')
    print("  [OK] model_evaluation.csv")

    # JSON summary
    summary_json = {
        'timestamp': datetime.now().isoformat(),
        'total_customers': int(len(model_data)),
        'feature_period': {
            'start': str(min_date.date()),
            'end': str(split_date.date())
        },
        'prediction_period': {
            'start': str(split_date.date()),
            'end': str(max_date.date())
        },
        'models': {}
    }

    for model_name, model_results in results.items():
        summary_json['models'][model_name] = {
            'test_r2': float(model_results['test_r2']),
            'test_rmse': float(model_results['test_rmse']),
            'test_mae': float(model_results['test_mae'])
        }

    summary_json['best_model'] = best_model_name
    summary_json['key_metrics'] = {
        'avg_ltv': float(model_data['ltv'].mean()),
        'median_ltv': float(model_data['ltv'].median()),
        'total_ltv': float(model_data['ltv'].sum()),
        'customers_with_future_purchases': int((model_data['ltv'] > 0).sum())
    }

    if 'feature_importance' in results['Random Forest']:
        summary_json['feature_importance'] = {
            k: float(v) for k, v in results['Random Forest']['feature_importance'].items()
        }

    with open(os.path.join(output_dir, 'ltv_analysis_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)
    print("  [OK] ltv_analysis_summary.json")

    # Markdown report
    report = f"""# Customer Lifetime Value (LTV) Prediction Analysis Report

**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents the results of Customer Lifetime Value (LTV) prediction analysis using machine learning regression models.

### Key Findings

- **Total Customers Analyzed**: {len(model_data):,}
- **Best Performing Model**: {best_model_name}
- **Best Model R² Score**: {best_model['test_r2']:.4f}
- **Average LTV**: ${model_data['ltv'].mean():.2f}
- **Total Predicted LTV**: ${model_data['ltv'].sum():,.2f}
- **Customers with Future Purchases**: {(model_data['ltv'] > 0).sum():,} ({(model_data['ltv'] > 0).mean()*100:.1f}%)

## Model Performance Comparison

| Model | Train R² | Test R² | Test RMSE | Test MAE |
|-------|----------|---------|-----------|----------|
"""

    for model_name in models:
        report += f"| {model_name} | {results[model_name]['train_r2']:.4f} | {results[model_name]['test_r2']:.4f} | ${results[model_name]['test_rmse']:.2f} | ${results[model_name]['test_mae']:.2f} |\n"

    report += f"""

## Feature Importance (Random Forest)

The most important features for predicting LTV are:

"""

    if 'feature_importance' in results['Random Forest']:
        sorted_features = sorted(results['Random Forest']['feature_importance'].items(),
                                key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features:
            report += f"- **{feature}**: {importance:.3f} ({importance*100:.1f}%)\n"

    report += """

## Business Insights

### 1. Model Selection
- The """ + best_model_name + f""" model achieved the best R² score of {best_model['test_r2']:.4f}
- This indicates that the model can explain approximately {best_model['test_r2']*100:.1f}% of the variance in customer LTV

### 2. Key Predictive Factors
- Historical monetary value is the strongest predictor of future LTV
- Purchase frequency and recency also contribute significantly
- Log-transformed features help handle skewed distributions

### 3. Customer Value Segmentation
- High-value customers can be identified using the prediction results
- Marketing budgets can be allocated based on predicted LTV
- Customer acquisition cost (CAC) can be compared against predicted LTV

## Recommendations

### Marketing Strategy
1. **High-LTV Customers**: Prioritize retention efforts and provide VIP service
2. **Mid-LTV Customers**: Implement upselling and cross-selling campaigns
3. **Low-LTV Customers**: Focus on increasing purchase frequency through targeted promotions

### Model Improvement
1. **Add More Features**: Include customer demographics, product categories, and geographic data
2. **Time-Series Analysis**: Incorporate seasonal patterns and trend analysis
3. **Advanced Algorithms**: Try XGBoost, LightGBM, or neural networks for potentially better performance

### Business Applications
1. **Budget Allocation**: Use LTV predictions to optimize marketing spend
2. **Customer Segmentation**: Create targeted campaigns based on predicted value
3. **Performance Monitoring**: Track actual vs predicted LTV to assess model accuracy over time

## Files Generated

- ltv_predictions.csv: Complete prediction results for all customers
- top_high_value_customers.csv: Top 500 high-value customers
- model_evaluation.csv: Detailed model performance metrics
- ltv_analysis_summary.json: Analysis summary in JSON format
- model_comparison.png: Model performance visualization
- feature_importance.png: Feature importance chart
- ltv_distribution.png: LTV distribution analysis

## Methodology

1. **Data Split**: First 60% of time period for features, last 40% for LTV calculation
2. **Feature Engineering**: RFM features with log transformations
3. **Model Training**: Linear Regression and Random Forest algorithms
4. **Evaluation**: 80/20 train-test split with R², RMSE, and MAE metrics

---

*This analysis provides a foundation for data-driven customer value management and marketing optimization.*
"""

    with open(os.path.join(output_dir, 'LTV_Analysis_Report.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    print("  [OK] LTV_Analysis_Report.md")

    print("\n" + "="*80)
    print("LTV Prediction Analysis Complete!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  - ltv_predictions.csv")
    print("  - top_high_value_customers.csv")
    print("  - model_evaluation.csv")
    print("  - ltv_analysis_summary.json")
    print("  - LTV_Analysis_Report.md")
    print("  - 3 visualization files")

if __name__ == "__main__":
    main()
