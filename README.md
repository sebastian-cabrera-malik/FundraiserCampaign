## The project context and objective are explained in the pdf. <br>

Donations-who-to-call-in-2023.pdf

## Before running the main.py file, **unzip the folder located in data**. <br>

Running main.py starts the datacleaning-trainingwithcampaign2022-generatingtheleadsfor2023

1. Data Merging  - Merge the 3 data sources into a single file - creates file Donations_Merge.csv 
               
2. Creating Master Table for 2022 use in training - adding variables at id level based on their donations behavior in past years, 2021 as first previous year
    
3. Model Training - starts the training of a logistic reg, a random forest classifier and an xgboost classifier

4. Creating Master Table for 2023 use in prioritizing - Creates same features, but now the donations behavior consider 2022 as the first previous year

5. Profiling Next Campaign - Generates a csv file with ID, name, email of the members with highest probability to donate in 2023

