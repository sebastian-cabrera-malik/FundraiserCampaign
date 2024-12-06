## Before running the main.py file, **unzip the folder located in data**. <br>

Running main.py starts the datacleaning-trainingwithcampaign2022-generatingtheleadsfor2023

### 1. Data Merging  - Merge the 3 data sources into a single file - creates file Donations_Merge.csv
merge_datasets('data/DonationHistory.csv', 
               'data/MembersList.csv',
               'data/ListContacted.csv')
               
### 2. Creating Master Table for 2022 use in training - adding variables at id level based on their donations behavior in past years, 2021 as first previous year
create_master_table('data/Donations_Merge.csv',current_year = 2022) 
    
### 3. Model Training - starts the training of a logistic reg, a random forest classifier and an xgboost classifier
train_models(df,features,test_size=0.2,trials=2)

### 4. Creating Master Table for 2023 use in prioritizing - Creates same features, but now the donations behavior consider 2022 as the first previous year
create_master_table('data/Donations_Merge.csv',current_year = 2023)

### 5. Profiling Next Campaign - Generates a csv file with ID, name, email of the members with highest probability to donate in 2023
df = pd.read_csv('data/Master_2023.csv')
next_campaign = profiling_campaign(df, features)
next_campaign.to_csv('data/Next_Campaign.csv', index=False)
