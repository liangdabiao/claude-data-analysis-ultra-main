import pandas as pd
files = ['olist_customers_dataset','olist_orders_dataset','olist_order_items_dataset',
         'olist_order_payments_dataset','olist_order_reviews_dataset','olist_products_dataset',
         'olist_sellers_dataset','product_category_name_translation']
for f in files:
    df = pd.read_csv(f'data_storage/{f}.csv', nrows=3)
    print(f'{f}: cols={list(df.columns)}')
print('All files loaded OK')
