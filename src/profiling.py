# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 21:54:34 2024

@author: piece
"""

import pandas as pd
import joblib


def profiling_campaign(df,features):
    
    df_NotContacted = df[df['Contacted']=='No'] # we have a restriction, we cant call those already called in 2022
    
    # Load the saved models
    loaded_lr = joblib.load('models/logistic_regression_model.pkl')
    loaded_rf = joblib.load('models/random_forest_model.pkl')
    loaded_xgb = joblib.load('models/xgboost_model.pkl')
    
    categorical_cols = df_NotContacted[features].select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numerical_cols = df_NotContacted[features].select_dtypes(include=['number']).columns.tolist()
    
    def update(df):
        for c in categorical_cols: 
            df[c] = df[c].fillna('Missing')
            df[c] = df[c].astype('category')
        return df

    df_NotContacted = update(df_NotContacted)
    
    def create_mapping(column, dataset):
        unique_values = dataset[column].unique()
        return {value: idx for idx, value in enumerate(unique_values)}
    
    for col in categorical_cols:
        mapping_df = create_mapping(col, df_NotContacted)
        df_NotContacted[col] = df_NotContacted[col].replace(mapping_df).astype(int)
    
    X_new = df_NotContacted[features]
    
    df_NotContacted['Affinity'] = (loaded_xgb.predict_proba(X_new)[:, 1])
    df_NotContacted = df_NotContacted.sort_values(by='Affinity', ascending=False)
    
    # Optionally, reset the index after sorting
    df_NotContacted.reset_index(drop=True, inplace=True)
    
    Next_Campaign = df_NotContacted.head(100000)[['ID', 'FirstName','LastName','email', 'Affinity','Years_since_joined', 'Amount_last_5_years',
       'Times_donated_last_5_years', 'Total_times_donated']]
    
    print(Next_Campaign.head(10))
    print(Next_Campaign.tail(10))
    
    return Next_Campaign
        
    
    
    