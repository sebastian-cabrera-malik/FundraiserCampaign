# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 21:41:37 2024

@author: piece
"""

import pandas as pd
import numpy as np

def generate_donations_summary(df):
    # Ensure the 'Yr' column is treated as a string (if it's not already)
    df['Yr'] = df['Yr'].astype(str)
    
    # Pivot the table so each year becomes a separate column for Amount
    donations_pivot = df.pivot_table(
        index='ID',  # Group by ID
        columns='Yr',  # Create columns for each year
        values='Amount',  # Values to aggregate
        aggfunc='sum',  # Sum the Amount per year
        fill_value=0  # Fill NaN values with 0
    )
    
    # Calculate total amount for each ID across all years
    donations_pivot['Amount'] = donations_pivot.sum(axis=1)
    
    # Optionally, if you want to add summary statistics for each year (like mean, min, max, etc.)
    summary_stats = donations_pivot.describe(percentiles=[.05, .95]).transpose()
    
    return donations_pivot, summary_stats

def merge_datasets(file1, file2, file3):
    # Merge logic here
    donationHistory = pd.read_csv(file1)
    MemberList = pd.read_csv(file2)
    ListContacted = pd.read_csv(file3,header=None)
    ListContacted.rename(columns={0: 'ID'}, inplace=True)
    ListContacted['Contacted']='Yes'
    
    donations_pivot, summary_stats = generate_donations_summary(donationHistory)

    # Merging logic
    df = MemberList.merge(donations_pivot,on='ID',how="left")
    df = df.merge(ListContacted,on='ID',how="left")
    df['Contacted'].fillna('No', inplace=True)
    df.fillna(0, inplace=True)
    df['Woman']=np.where(df['Woman']==1,'Yes','No')
    
    df.to_csv("data/Donations_Merge.csv", index=False)
    
    return df

def calculate_donations_metrics(df, current_year=2022):
    
    # Get the donation columns dynamically based on the years in the dataframe
    donation_years = [str(year) for year in range(current_year-10, current_year)]  # Donations from current_year-10 to current_year

    # Create variables for donations in the last 3 and 5 years
    df['Amount_last_2_years'] = df[donation_years[-2:]].sum(axis=1)
    df['Amount_last_3_years'] = df[donation_years[-3:]].sum(axis=1)
    df['Amount_last_4_years'] = df[donation_years[-4:]].sum(axis=1)
    df['Amount_last_5_years'] = df[donation_years[-5:]].sum(axis=1)

    # Calculate the average donations in the last 3 and 5 years
    df['Avg_last_2_years'] = df[donation_years[-2:]].gt(0).sum(axis=1)  # Count non-zero donations
    df['Avg_last_2_years'] = df['Amount_last_2_years'] / df['Avg_last_2_years'].replace(0, 1)  # Prevent division by 0

    df['Avg_last_3_years'] = df[donation_years[-3:]].gt(0).sum(axis=1)  # Count non-zero donations
    df['Avg_last_3_years'] = df['Amount_last_3_years'] / df['Avg_last_3_years'].replace(0, 1)  # Prevent division by 0
    
    df['Avg_last_4_years'] = df[donation_years[-4:]].gt(0).sum(axis=1)  # Count non-zero donations
    df['Avg_last_4_years'] = df['Amount_last_4_years'] / df['Avg_last_4_years'].replace(0, 1)  # Prevent division by 0

    df['Avg_last_5_years'] = df[donation_years[-5:]].gt(0).sum(axis=1)  # Count non-zero donations
    df['Avg_last_5_years'] = df['Amount_last_5_years'] / df['Avg_last_5_years'].replace(0, 1)  # Prevent division by 0

    # Calculate the max donation in the last 3 and 5 years
    df['Max_last_2_years'] = df[donation_years[-2:]].max(axis=1)
    df['Max_last_3_years'] = df[donation_years[-3:]].max(axis=1)
    df['Max_last_4_years'] = df[donation_years[-4:]].max(axis=1)
    df['Max_last_5_years'] = df[donation_years[-5:]].max(axis=1)

    # Calculate the number of years since the member joined
    df['Years_since_joined'] = current_year - df['Joined']

    # Calculate the number of times donated in the last 3 and 5 years
    df['Times_donated_last_2_years'] = df[donation_years[-2:]].gt(0).sum(axis=1)
    df['Times_donated_last_3_years'] = df[donation_years[-3:]].gt(0).sum(axis=1)
    df['Times_donated_last_4_years'] = df[donation_years[-4:]].gt(0).sum(axis=1)
    df['Times_donated_last_5_years'] = df[donation_years[-5:]].gt(0).sum(axis=1)

    # Calculate the total number of years donated (non-zero donations across all years)
    df['Total_times_donated'] = df[donation_years].gt(0).sum(axis=1)

    return df

def create_master_table(file1,current_year=2022):
    # Merge logic here
    df = pd.read_csv(file1)
    df = calculate_donations_metrics(df, current_year=current_year)
    if current_year==2022:
        df['Donated'] = np.where(df[f'{current_year}'] > 0, 1, 0)
    df.to_csv(f'data/Master_{current_year}.csv', index=False)

    print("master table generated for year: ",current_year)
    
    


