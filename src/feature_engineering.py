import os
import yaml
import dask.dataframe as dd
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_feature_engineering(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Paths
    processed_dir = config['directories']['processed_data']
    final_dir = config['directories']['final_data']
    output_filename = config['files']['final_output']
    
    os.makedirs(final_dir, exist_ok=True)

    logging.info("⏳ Loading Parquet data using Dask...")
    
    # Read Parquet files
    train_dd = dd.read_parquet(os.path.join(processed_dir, 'train.parquet'))
    members_dd = dd.read_parquet(os.path.join(processed_dir, 'members.parquet'))
    transactions_dd = dd.read_parquet(os.path.join(processed_dir, 'transactions.parquet'))
    user_logs_dd = dd.read_parquet(os.path.join(processed_dir, 'user_logs.parquet'))

    # --- 1. Aggregating Transactions ---
    logging.info("⚙️  Aggregating Transactions...")
    
    # Flag promos (0 paid amount)
    transactions_dd['is_promo'] = (transactions_dd['actual_amount_paid'] == 0).astype(int)
    
    # Determine snapshot date for recency calculation
    max_trans_date = transactions_dd['transaction_date'].max().compute()
    snapshot_date = pd.to_datetime(max_trans_date, format='%Y%m%d')

    trans_agg = transactions_dd.groupby('msno').agg({
        'payment_method_id': 'count',           
        'actual_amount_paid': 'sum',            
        'payment_plan_days': 'mean',            
        'is_cancel': 'sum',                     
        'is_promo': 'sum',
        'transaction_date': 'max'
    })
    
    trans_agg.columns = ['total_transactions', 'total_payment', 'avg_plan_days', 
                         'total_cancel_count', 'promo_transaction_count', 'last_transaction_date']

    # --- 2. Aggregating User Logs (With Trends) ---
    logging.info("⚙️  Aggregating User Logs & Calculating Trends...")
    
    user_logs_dd['total_songs_daily'] = (
        user_logs_dd['num_25'] + user_logs_dd['num_50'] + 
        user_logs_dd['num_75'] + user_logs_dd['num_985'] + 
        user_logs_dd['num_100']
    )
    
    user_logs_dd['date_parsed'] = dd.to_datetime(user_logs_dd['date'], format='%Y%m%d')
    max_log_date = user_logs_dd['date_parsed'].max().compute()
    mid_month = max_log_date - pd.Timedelta(days=15)
    
    # Split into recent vs older logs for trend analysis
    logs_first_half = user_logs_dd[user_logs_dd['date_parsed'] < mid_month]
    logs_second_half = user_logs_dd[user_logs_dd['date_parsed'] >= mid_month]
    
    # Overall aggregates
    logs_overall = user_logs_dd.groupby('msno').agg({
        'total_secs': 'sum',
        'num_unq': 'sum',
        'total_songs_daily': 'sum',
        'num_100': 'sum',
        'date': 'count'
    })
    logs_overall.columns = ['total_secs_played', 'total_unique_songs', 
                            'total_songs_played', 'total_songs_100_percent', 'active_days']

    # Trend aggregates
    agg_first = logs_first_half.groupby('msno').agg({'date': 'count', 'total_secs': 'sum'})
    agg_first.columns = ['active_days_first_half', 'total_secs_first_half']

    agg_second = logs_second_half.groupby('msno').agg({'date': 'count', 'total_secs': 'sum'})
    agg_second.columns = ['active_days_second_half', 'total_secs_second_half']

    # --- 3. Compute & Merge ---
    logging.info("💾 Computing and Merging (Dask -> Pandas)...")
    
    # Trigger Dask computation
    df_train = train_dd.compute()
    df_members = members_dd.compute()
    df_trans = trans_agg.compute()
    df_logs = logs_overall.compute()
    df_first = agg_first.compute()
    df_second = agg_second.compute()
    
    # Post-Compute Transaction Feature: Days Since Last
    df_trans['last_transaction_date'] = pd.to_datetime(df_trans['last_transaction_date'], format='%Y%m%d')
    df_trans['days_since_last_transaction'] = (snapshot_date - df_trans['last_transaction_date']).dt.days
    df_trans = df_trans.drop(columns=['last_transaction_date'])

    # Reset indices to make merging easy
    df_trans = df_trans.reset_index()
    df_logs = df_logs.reset_index()
    df_first = df_first.reset_index()
    df_second = df_second.reset_index()
    
    logging.info("   -> Merging tables...")
    master_df = df_train.merge(df_members, on='msno', how='left')
    master_df = master_df.merge(df_trans, on='msno', how='left')
    master_df = master_df.merge(df_logs, on='msno', how='left')
    master_df = master_df.merge(df_first, on='msno', how='left')
    master_df = master_df.merge(df_second, on='msno', how='left')

    # --- 4. Final Cleanup & Type Casting ---
    logging.info("🧹 Performing Final Type Cleanup...")

    # A. Fill NaNs with 0 for Count/Sum columns (missing means no activity)
    fill_0_cols = [
        'total_transactions', 'total_payment', 'total_cancel_count', 
        'promo_transaction_count', 'avg_plan_days', 
        'total_secs_played', 'total_unique_songs', 'total_songs_played', 
        'total_songs_100_percent', 'active_days',
        'active_days_first_half', 'total_secs_first_half',
        'active_days_second_half', 'total_secs_second_half'
    ]
    # Only fill columns that exist
    cols_to_fill = [c for c in fill_0_cols if c in master_df.columns]
    master_df[cols_to_fill] = master_df[cols_to_fill].fillna(0)
    
    # B. Fill missing recency with -1 (Standard practice for Tree models)
    if 'days_since_last_transaction' in master_df.columns:
        master_df['days_since_last_transaction'] = master_df['days_since_last_transaction'].fillna(-1)

    # C. Force Integer Types
    int_cols = [
        'total_transactions', 'total_cancel_count', 'promo_transaction_count',
        'total_unique_songs', 'total_songs_played', 'total_songs_100_percent', 'active_days',
        'active_days_first_half', 'active_days_second_half', 
        'days_since_last_transaction', 'is_churn'
    ]
    
    for col in int_cols:
        if col in master_df.columns:
            master_df[col] = master_df[col].fillna(-1).astype(int)

    # --- 5. Save Gold Data ---
    output_file = os.path.join(final_dir, output_filename)
    logging.info(f"✅ Saving Master Table to {output_file}...")
    
    master_df.to_parquet(output_file, index=False)
    logging.info(f"🎉 Done! Final Shape: {master_df.shape}")

if __name__ == "__main__":
    run_feature_engineering()