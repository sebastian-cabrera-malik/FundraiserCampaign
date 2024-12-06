# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 22:03:34 2024

@author: piece
"""

from src.data_preprocessing import merge_datasets,create_master_table
from src.model_training import train_models
from src.profiling import profiling_campaign

import pandas as pd


# 1. Data Merging
merge_datasets('data/DonationHistory.csv', 
               'data/MembersList.csv',
               'data/ListContacted.csv')

# 2. Creating Master Table for 2022 use in training
create_master_table('data/Donations_Merge.csv',current_year = 2022)

# 3. Model Training
df = pd.read_csv('data/Master_2022.csv')

# Available features for training
features = ['Woman', 'Age', 'Salary',
       'Education', 'City', 'Joined',
        'Amount_last_2_years', 'Amount_last_3_years',
       'Amount_last_4_years', 'Amount_last_5_years', 'Avg_last_2_years',
       'Avg_last_3_years', 'Avg_last_4_years', 'Avg_last_5_years',
       'Max_last_2_years', 'Max_last_3_years', 'Max_last_4_years',
       'Max_last_5_years', 'Years_since_joined', 'Times_donated_last_2_years',
       'Times_donated_last_3_years', 'Times_donated_last_4_years',
       'Times_donated_last_5_years', 'Total_times_donated']

train_models(df,features,test_size=0.2,trials=2)

# 4. Creating Master Table for 2023 use in prioritizing
create_master_table('data/Donations_Merge.csv',current_year = 2023)

# 5. Profiling Next Campaign
df = pd.read_csv('data/Master_2023.csv')
next_campaign = profiling_campaign(df, features)
next_campaign.to_csv('data/Next_Campaign.csv', index=False)