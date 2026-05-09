# -*- coding: utf-8 -*-
"""
数据质量检查模块
完整性、一致性、有效性检查
"""
import pandas as pd
import numpy as np

def check_completeness(df):
    """检查数据完整性"""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    result = pd.DataFrame({
        'column': df.columns,
        'missing_count': missing,
        'missing_pct': missing_pct
    }).sort_values('missing_pct', ascending=False)
    
    return result

def check_duplicates(df):
    """检查重复"""
    duplicates = df.duplicated().sum()
    return {
        'duplicate_count': duplicates,
        'duplicate_pct': (duplicates / len(df)) * 100
    }

def check_data_types(df):
    """检查数据类型"""
    return df.dtypes.astype(str).to_dict()

def check_outliers_series(series, method='iqr'):
    """检查异常值"""
    if method == 'iqr':
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = (series < lower) | (series > upper)
        return {
            'count': outliers.sum(),
            'pct': outliers.sum() / len(series) * 100,
            'lower_bound': lower,
            'upper_bound': upper
        }
    return {}

def run_full_quality_check(df, name='dataset'):
    """运行完整质量检查"""
    print(f"=== 质量检查: {name} ===")
    print(f"形状: {df.shape}")
    
    print("\n完整性检查:")
    completeness = check_completeness(df)
    print(completeness[completeness['missing_pct'] > 0].head(10))
    
    print("\n重复检查:")
    duplicates = check_duplicates(df)
    print(f"重复行数: {duplicates['duplicate_count']} ({duplicates['duplicate_pct']:.2f}%)")
    
    return {
        'completeness': completeness,
        'duplicates': duplicates
    }

def main():
    print("数据质量检查模块")

if __name__ == "__main__":
    main()
