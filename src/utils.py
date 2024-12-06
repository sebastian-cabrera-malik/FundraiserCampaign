# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 21:54:33 2024

@author: piece
"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix


def plot_roc_curve(y_test,y_pred_proba_lr,y_pred_proba_rf,y_pred_proba_xgb):
    
    # --- Calculate the ROC curve for each model ---
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
    
    # --- Calculate the AUC (Area Under the Curve) for each model ---
    auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
    auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
    auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
    
    # --- Plot the ROC curves ---
    plt.figure(figsize=(6, 6))
    
    # Plot for Logistic Regression
    plt.plot(fpr_lr, tpr_lr, color='blue', label=f'Logistic Regression (AUC = {auc_lr:.2f})')
    
    # Plot for Random Forest
    plt.plot(fpr_rf, tpr_rf, color='green', label=f'Random Forest (AUC = {auc_rf:.2f})')
    
    # Plot for XGBoost
    plt.plot(fpr_xgb, tpr_xgb, color='red', label=f'XGBoost (AUC = {auc_xgb:.2f})')
    
    # --- Plot the diagonal (random classifier) ---
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
    
    # --- Customize the plot ---
    plt.title('ROC Curve Comparison')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
def performance_tables(X_test):
    
    Lift_performance_lr = X_test.groupby('prob_lr_bin').agg(
        min_prob=('prob_lr', 'min'),
        max_prob=('prob_lr', 'max'),
        num_cases=('prob_lr_bin', 'size'),         # Count the number of cases in each bin
        num_actual_cases=('Actual', 'sum')         # Count the number of actual positive cases (Actual == 1)
    ).reset_index()
    
    # Calculate the percentage of actual cases in each bin
    Lift_performance_lr['percentage'] = (Lift_performance_lr['num_actual_cases'] / Lift_performance_lr['num_cases']) * 100
    Lift_performance_lr['lift'] = (Lift_performance_lr['percentage']/100)/(sum(X_test['Actual'])/len(X_test))
    
    Lift_performance_rf = X_test.groupby('prob_rf_bin').agg(
    min_prob=('prob_rf', 'min'),
    max_prob=('prob_rf', 'max'),
    num_cases=('prob_rf_bin', 'size'),         # Count the number of cases in each bin
    num_actual_cases=('Actual', 'sum')         # Count the number of actual positive cases (Actual == 1)
    ).reset_index()
    
    # Calculate the percentage of actual cases in each bin
    Lift_performance_rf['percentage'] = (Lift_performance_rf['num_actual_cases'] / Lift_performance_rf['num_cases']) * 100
    Lift_performance_rf['lift'] = (Lift_performance_rf['percentage']/100)/(sum(X_test['Actual'])/len(X_test))
    
    Lift_performance_xgb = X_test.groupby('prob_xgb_bin').agg(
    min_prob=('prob_xgb', 'min'),
    max_prob=('prob_xgb', 'max'),
    num_cases=('prob_xgb_bin', 'size'),         # Count the number of cases in each bin
    num_actual_cases=('Actual', 'sum')         # Count the number of actual positive cases (Actual == 1)
    ).reset_index()
    
    # Calculate the percentage of actual cases in each bin
    Lift_performance_xgb['percentage'] = (Lift_performance_xgb['num_actual_cases'] / Lift_performance_xgb['num_cases']) * 100
    Lift_performance_xgb['lift'] = (Lift_performance_xgb['percentage']/100)/(sum(X_test['Actual'])/len(X_test))
    

    
    return Lift_performance_lr,Lift_performance_rf,Lift_performance_xgb