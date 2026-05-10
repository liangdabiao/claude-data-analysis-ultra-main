import pandas as pd
import os

files = {
    'Amazon': 'eCommerce-dataset-samples/amazon-products.csv',
    'Lazada': 'eCommerce-dataset-samples/lazada-products.csv',
    'SHEIN': 'eCommerce-dataset-samples/shein-products.csv',
    'Shopee': 'eCommerce-dataset-samples/shopee-products.csv',
    'Walmart': 'eCommerce-dataset-samples/walmart-products.csv',
}

for name, path in files.items():
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    try:
        df = pd.read_csv(path, engine='python', on_bad_lines='skip')
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nNulls (>0):")
        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0]
        print(nulls.to_dict() if len(nulls) > 0 else "  None")
        for col in df.columns:
            if any(k in col.lower() for k in ['price','cost']):
                if pd.api.types.is_numeric_dtype(df[col]):
                    print(f"\n{col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}, median={df[col].median():.2f}")
        for col in df.columns:
            if 'category' in col.lower() and df[col].dtype == 'object':
                print(f"\n{col} unique: {df[col].nunique()}")
                top3 = df[col].value_counts().head(3)
                for v, c in top3.items():
                    print(f"  {str(v)[:50]}: {c}")
        for col in df.columns:
            if 'brand' in col.lower() and df[col].dtype == 'object':
                print(f"\n{col} unique: {df[col].nunique()}")
                top3 = df[col].value_counts().head(3)
                for v, c in top3.items():
                    print(f"  {v}: {c}")
        if 'currency' in df.columns:
            print(f"\ncurrency: {df['currency'].unique()}")
        if 'country_code' in df.columns:
            print(f"country_code: {df['country_code'].unique()}")
        if 'rating' in df.columns:
            print(f"rating: mean={df['rating'].mean():.2f}, median={df['rating'].median():.2f}")
        if 'reviews_count' in df.columns:
            print(f"reviews_count: mean={df['reviews_count'].mean():.1f}, max={df['reviews_count'].max()}")
        if 'in_stock' in df.columns:
            print(f"in_stock: {df['in_stock'].value_counts().to_dict()}")
        if 'color' in df.columns:
            print(f"color unique: {df['color'].nunique()}")
    except Exception as e:
        print(f"Error: {e}")
    print()
