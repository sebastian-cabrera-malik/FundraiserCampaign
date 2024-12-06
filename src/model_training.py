# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 21:46:52 2024

@author: piece
"""
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')


from src.utils import plot_roc_curve,performance_tables


# --- Model training ---
def train_models(df,features,test_size,trials=10):
    
    df = df[df['Contacted']=='Yes']
    
    categorical_cols = df[features].select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numerical_cols = df[features].select_dtypes(include=['number']).columns.tolist()
    
    def update(df):
        for c in categorical_cols: 
            df[c] = df[c].fillna('Missing')
            df[c] = df[c].astype('category')
        return df

    df = update(df)
    
    def create_mapping(column, dataset):
        unique_values = dataset[column].unique()
        return {value: idx for idx, value in enumerate(unique_values)}
    
    for col in categorical_cols:
        mapping_df = create_mapping(col, df)
        df[col] = df[col].replace(mapping_df).astype(int)

    df = df[['Donated']+features]
    
    X = df.drop(columns=['Donated'])
    y = df['Donated']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective_lr(trial):
        # Hyperparameter search space for Logistic Regression
        C = trial.suggest_loguniform('C', 1e-5, 100)
        solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs', 'saga'])
        max_iter = trial.suggest_int('max_iter', 100, 500)

        model = LogisticRegression(C=C, solver=solver, max_iter=max_iter, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate AUC and return it for optimization
        auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        return auc_score

    # --- Random Forest ---
    def objective_rf(trial):
        # Hyperparameter search space for Random Forest
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 2, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_split=min_samples_split, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate AUC and return it for optimization
        auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        return auc_score

    # --- XGBoost ---
    def objective_xgb(trial):
        # Hyperparameter search space for XGBoost
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 0.1)
        colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.3, 1.0)
        subsample = trial.suggest_uniform('subsample', 0.3, 1.0)
        
        model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                              learning_rate=learning_rate, colsample_bytree=colsample_bytree, 
                              subsample=subsample, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate AUC and return it for optimization
        auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        return auc_score
    
    
    study_lr = optuna.create_study(direction='maximize')
    study_lr.optimize(objective_lr, n_trials=trials)
    
    # Set up the study for Random Forest
    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(objective_rf, n_trials=trials)
    
    # Set up the study for XGBoost
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=trials)

    # --- Best Parameters & Results ---
    print("Best parameters for Logistic Regression: ", study_lr.best_params)
    print("Best parameters for Random Forest: ", study_rf.best_params)
    print("Best parameters for XGBoost: ", study_xgb.best_params)
    
    best_lr = LogisticRegression(**study_lr.best_params, random_state=42)
    best_lr.fit(X_train, y_train)
    best_rf = RandomForestClassifier(**study_rf.best_params, random_state=42)
    best_rf.fit(X_train, y_train)
    best_xgb = XGBClassifier(**study_xgb.best_params, random_state=42)
    best_xgb.fit(X_train, y_train)
    
    y_pred_proba_lr =  (best_lr.predict_proba(X_test)[:, 1])
    y_pred_proba_rf = (best_rf.predict_proba(X_test)[:, 1])
    y_pred_proba_xgb = (best_xgb.predict_proba(X_test)[:, 1])

    X_test['prob_lr']=y_pred_proba_lr
    X_test['prob_rf']=y_pred_proba_rf
    X_test['prob_xgb']=y_pred_proba_xgb
    
    X_test['prob_lr_bin'] = pd.qcut(X_test['prob_lr'], 10, labels=False)
    X_test['prob_rf_bin'] = pd.qcut(X_test['prob_rf'], 10, labels=False)
    X_test['prob_xgb_bin'] = pd.qcut(X_test['prob_xgb'], 10, labels=False)
    X_test['Actual']=y_test
    
    plot_roc_curve(y_test,y_pred_proba_lr,y_pred_proba_rf,y_pred_proba_xgb)
    
    Lift_performance_lr,Lift_performance_rf,Lift_performance_xgb = performance_tables(X_test)
    
    print(Lift_performance_xgb)
    
    joblib.dump(best_lr, 'models/logistic_regression_model.pkl')
    joblib.dump(best_rf, 'models/random_forest_model.pkl')
    joblib.dump(best_xgb, 'models/xgboost_model.pkl')
    
    print("Models saved successfully.")