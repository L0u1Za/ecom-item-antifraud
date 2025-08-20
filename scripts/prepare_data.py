#!/usr/bin/env python3
"""
Data preparation script for e-commerce anti-fraud project.
This script should be run once to prepare the training and test data.

Extracts data preparation logic from notebooks and text processing logic from TextProcessor.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import argparse
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

from preprocessing.text.cleaner import TextCleaner
from preprocessing.text.normalizer import normalize_text
from preprocessing.text.business_rules import BusinessRulesChecker


def clean_and_process_text(df, config, text_columns=['description', 'name_rus']):
    """
    Clean and process text columns, add fraud indicators based on config.
    Extracted from TextProcessor class.
    """
    text_config = config.preprocessing.text
    
    # Add fraud indicators based on brand_name and description
    if text_config.add_fraud_indicators:
        print("Adding fraud indicators...")
        if 'brand_name' in df.columns and 'description' in df.columns:
            checker = BusinessRulesChecker()
            
            # Collect all fraud indicators for all rows
            all_indicators = []
            for idx, row in df.iterrows():
                brand_name = str(row['brand_name']) if pd.notna(row['brand_name']) else ""
                description = str(row['description']) if pd.notna(row['description']) else ""
                title = str(row['name_rus']) if pd.notna(row['description']) else ""
                indicators_desc = checker(brand_name, description)
                indicators_title = checker(brand_name, title)
                indicators_desc = {f"desc_{indicator[0]}": indicator[1] for indicator in indicators_desc.items()}
                indicators_title = {f"title_{indicator[0]}": indicator[1] for indicator in indicators_title.items()}
                all_indicators.append(indicators_desc)
                all_indicators[-1].update(indicators_title)
            
            # Get all possible fraud indicator keys
            all_keys = set()
            for indicators in all_indicators:
                all_keys.update(indicators.keys())
            
            # Create separate columns for each fraud indicator
            print(f"Creating {len(all_keys)} fraud indicator columns...")
            for key in sorted(all_keys):
                column_name = f"fraud_{key}"
                df[column_name] = [indicators.get(key, False) for indicators in all_indicators]
                print(f"  - {column_name}")
    else:
        print("Skipping fraud indicators (disabled in config)")
    
    if text_config.apply_cleaning:
        print("Cleaning and processing text data...")
        
        # Initialize text processing components
        cleaner = TextCleaner(text_config.nltk_data_dir)
        
        for col in text_columns:
            if col in df.columns:
                print(f"Processing {col}...")
                # Clean text
                df[col] = df[col].astype(str)
                df[col] = df[col].apply(lambda x: cleaner.clean_text(x))
                df[col] = df[col].apply(lambda x: cleaner.clean_repeating_chars(x))
                df[col] = df[col].apply(lambda x: cleaner.truncate_text(x, text_config.max_length))
                
                # Apply normalization if enabled
                if text_config.apply_lemmatization:
                    df[col] = df[col].apply(lambda x: normalize_text(x))
    else:
        print("Skipping text cleaning (disabled in config)")
    
    return df


def add_engineered_features(df_train, df_test):
    """
    Add engineered features to both train and test datasets.
    Extracted from data preparation notebook.
    """
    print("Adding engineered features...")
    
    for df in [df_train, df_test]:
        # --- Ratios & rates ---
        df['return_rate_30'] = df['item_count_returns30'] / (df['item_count_sales30'] + 1e-6)
        df['fake_return_rate_30'] = df['item_count_fake_returns30'] / (df['item_count_sales30'] + 1e-6)
        df['refund_value_ratio_30'] = df['ExemplarReturnedValueTotal30'] / (df['GmvTotal30'] + 1e-6)

        # --- Growth / trend features ---
        df['sales_growth_7_30'] = (df['item_count_sales7']+1) / (df['item_count_sales30']+1)
        df['sales_growth_30_90'] = (df['item_count_sales30']+1) / (df['item_count_sales90']+1)

        # --- Activity rates ---
        df['sales_velocity'] = df['item_count_sales30'] / (df['item_time_alive'] + 1e-6)
        df['seller_velocity'] = df['item_count_sales30'] / (df['seller_time_alive'] + 1e-6)

        # --- Text features ---
        df['desc_len'] = df['description'].astype(str).str.len()
        df['desc_word_count'] = df['description'].astype(str).str.split().str.len()
        df['name_len'] = df['name_rus'].astype(str).str.len()

        # --- Interaction features ---
        df['price_return_interaction'] = df['PriceDiscounted'] * df['return_rate_30']
        df['gmv_per_day'] = df['GmvTotal30'] / (df['item_time_alive'] + 1)

    # --- Seller-level aggregations ---
    print("Computing seller-level aggregations...")
    seller_stats = df_train.groupby('SellerID').agg(
        seller_total_items=('ItemID','count'),
        seller_total_sales=('item_count_sales30','sum'),
        seller_avg_return_rate=('return_rate_30','mean')
    ).reset_index()

    # Get numeric columns for anomaly detection
    numeric_columns = df_train.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target column if present
    if 'resolution' in numeric_columns:
        numeric_columns.remove('resolution')
    
    # --- Anomaly score (optional, unsupervised) ---
    print("Computing anomaly scores...")
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(df_train[numeric_columns].fillna(0))

    for df in [df_train, df_test]:
        df = df.merge(seller_stats, on='SellerID', how='left')
        df['anomaly_score'] = iso.predict(df[numeric_columns].fillna(0))

    return df_train, df_test


def prepare_data(config, train_path=None, test_path=None, output_dir=None):
    """
    Main data preparation function using Hydra config.
    """
    print("Starting data preparation...")
    
    # Use config defaults if not provided
    if train_path is None:
        train_path = config.data_preparation.get('train_path', 'data/ml_ozon_train.csv')
    if test_path is None:
        test_path = config.data_preparation.get('test_path', 'data/ml_ozon_test.csv')
    if output_dir is None:
        output_dir = config.data_preparation.get('output_dir', 'data')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load raw data
    print("Loading raw data...")
    print(f"Training data: {train_path}")
    print(f"Test data: {test_path}")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")
    
    df_train = pd.read_csv(train_path, index_col=0)
    df_test = pd.read_csv(test_path, index_col=0)
    
    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")
    print(f"Target distribution in train:")
    print(df_train['resolution'].value_counts())
    
    
    # Fill nulls
    cat_cols = ['brand_name', 'CommercialTypeName4']
    numeric_columns = df_train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col != 'resolution']

    df_train[numeric_columns] = df_train[numeric_columns].fillna(0)
    df_test[numeric_columns] = df_test[numeric_columns].fillna(0)

    df_train[cat_cols] = df_train[cat_cols].fillna("unknown")
    df_test[cat_cols] = df_test[cat_cols].fillna("unknown")
    
    # Clean and process text data using config
    df_train = clean_and_process_text(df_train, config)
    df_test = clean_and_process_text(df_test, config)
    
    # Add engineered features
    df_train, df_test = add_engineered_features(df_train, df_test)
    
    # Save prepared datasets
    print("Saving prepared datasets...")
    
    # Save full prepared datasets
    train_prepared_path = os.path.join(output_dir, 'train_prepared.csv')
    test_prepared_path = os.path.join(output_dir, 'test_prepared.csv')
    
    df_train.to_csv(train_prepared_path)
    df_test.to_csv(test_prepared_path)
    
    print(f"Saved prepared training data to: {train_prepared_path}")
    print(f"Saved prepared test data to: {test_prepared_path}")
    
    # Create train/validation split
    print("Creating train/validation split...")
    df_train_clean = df_train.drop(columns=['SellerID'], errors='ignore')
    df_test_clean = df_test.drop(columns=['SellerID'], errors='ignore')
    
    df_train_train, df_train_val = train_test_split(
        df_train_clean, test_size=0.2, random_state=42, 
        stratify=df_train_clean['resolution']
    )
    
    # Save train/validation splits
    train_split_path = os.path.join(output_dir, 'train.csv')
    val_split_path = os.path.join(output_dir, 'val.csv')
    test_final_path = os.path.join(output_dir, 'test.csv')
    
    df_train_train.to_csv(train_split_path)
    df_train_val.to_csv(val_split_path)
    df_test_clean.to_csv(test_final_path)
    
    print(f"Saved training split to: {train_split_path}")
    print(f"Saved validation split to: {val_split_path}")
    print(f"Saved final test data to: {test_final_path}")
    
    print("Data preparation completed successfully!")
    
    return df_train, df_test


@hydra.main(version_base=None, config_path="../src/config", config_name="config")
def main(config: DictConfig) -> None:
    """
    Main function using Hydra configuration.
    """
    print("Configuration:")
    print(OmegaConf.to_yaml(config))
    
    try:
        prepare_data(config)
    except Exception as e:
        print(f"Error during data preparation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
