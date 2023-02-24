import logging
import numpy
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from math import *
from datetime import datetime
from operator import itemgetter
from zipfile import ZipFile
from io import BytesIO
import pickle
from urllib.request import urlopen

from IPython.display import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from numpy import inf
from scipy.stats import kurtosis, skew

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

from fair_func import acfmetrics

class Generic:
    
    def __init__(self, data, protected_features, independent_features, target_variable, is_fair, is_sensitive=None):
        """function to initialize class"""
        self.data = data
        y=self.data[target_variable]*1000
        x=self.data.drop(columns=[target_variable])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.20, random_state=3000)
        self.protected_features = protected_features
        self.independent_features = independent_features
        self.target_variable = target_variable
        self.sens_test = self.X_test[self.protected_features] if self.protected_features is not "" else None
        self.counter_sens_test = None
        self.is_fair = is_fair 
        self.is_sensitive = is_sensitive
        self.model = None
        self.predictions = None
        self.pred_probs = None
    
    def get_residuals_train_data(self):
        """function to get residuals for train data
           residual = diff(predicted,actual)"""
        residuals_dict_train = {}
        sens_train=self.X_train[self.protected_features]
        for feature in self.independent_features:
            clf_feature_train = LinearRegression().fit(sens_train, self.X_train[feature])
            residual_train = self.X_train[feature] - clf_feature_train.predict(sens_train)
            residuals_dict_train[f"{feature}R"] = residual_train

        df_R_train = pd.DataFrame(residuals_dict_train)
        return df_R_train
    
    def fit(self, X_train=None):
        """function to fit model"""
        if X_train is None:
            self.model = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr').fit(self.X_train, self.y_train)
        else:            
            self.model = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr').fit(X_train, self.y_train)
        return self.model
    
    def get_residuals_test_data(self,is_cuf=False):
        """function to get residuals for test data
           residual = diff(predicted,actual)"""
        residuals_dict_test = {}
        for feature in self.independent_features:
            clf_feature_test = LinearRegression().fit(self.sens_test, self.X_test[feature])
            residual_test = self.X_test[feature] - clf_feature_test.predict(self.sens_test if is_cuf==False else self.counter_sens_test)
            residuals_dict_test[f"{feature}R"] = residual_test

        df_R_test = pd.DataFrame(residuals_dict_test)
        return df_R_test
        
    def predict(self, X_test=None):
        """function to predict using model"""
        if X_test is None:
            self.predictions = self.model.predict(self.X_test)
        else:            
            self.predictions = self.model.predict(X_test)
        return self.predictions
    
    def predict_proba(self, X_test=None):
        """function to predict probability using model"""
        if X_test is None:
            self.pred_probs = self.model.predict_proba(self.X_test)[:,0]
        else:            
            self.pred_probs = self.model.predict_proba(X_test)[:,0]
        return self.pred_probs
    
    def get_model_metrics(self):
        """function to get ROC AUC Score for model"""
        model_type = "ACF Model" if self.is_fair == True else "Full Model"
        sensitivity = " without sensitive features" if self.is_sensitive == False else " with sensitive features" if self.is_fair == False else "" 
        metrics_data = pd.DataFrame({f"ROC AUC Score ({model_type}{sensitivity})":[roc_auc_score(self.y_test, self.pred_probs)]})
        return metrics_data
    
    def get_fairness_metrics(self):
        """function to get Equal Odds, Demographic Parity and Predictive Parity for model"""
        model_type = "ACF Model" if self.is_fair == True else "Full Model"
        sensitivity = " without sensitive features" if self.is_sensitive == False else " with sensitive features" if self.is_fair == False else "" 
        fairness_metrics_dict = {"Column":[],f"Equal Odds ({model_type}{sensitivity})":[],f"Demographic Parity ({model_type}{sensitivity})":[],f"Predictive Parity ({model_type}{sensitivity})":[]}
        
        for feature in self.protected_features:
            fairness_metrics_dict["Column"].append(feature)
            tn_up, fp_up, fn_up, tp_up = confusion_matrix(self.y_test[self.X_test[feature]==1], self.predictions[self.X_test[feature]==1]).ravel()
            tn_p, fp_p, fn_p, tp_p = confusion_matrix(self.y_test[self.X_test[feature]==0], self.predictions[self.X_test[feature]==0]).ravel()
            fairness_metrics = acfmetrics(tn_up, fp_up, fn_up, tp_up, tn_p, fp_p, fn_p, tp_p)
            fairness_metrics_dict[f"Equal Odds ({model_type}{sensitivity})"].append(fairness_metrics[1])
            fairness_metrics_dict[f"Demographic Parity ({model_type}{sensitivity})"].append(fairness_metrics[3])
            fairness_metrics_dict[f"Predictive Parity ({model_type}{sensitivity})"].append(fairness_metrics[6])
            
        fairness_metrics_data = pd.DataFrame(fairness_metrics_dict)
        return fairness_metrics_data
        
    def get_model_metrics_difference(self):
        """function to get ROC AUC Score difference for protected features - privileged vs unprivileged"""
        model_type = "ACF Model" if self.is_fair == True else "Full Model"
        sensitivity = " without sensitive features" if self.is_sensitive == False else " with sensitive features" if self.is_fair == False else "" 
        model_metrics_difference_dict = {"Column":[],f"ROC AUC Score Difference ({model_type}{sensitivity})":[]}

        for feature in self.protected_features:
            model_metrics_difference_dict["Column"].append(feature)
            A_fair=roc_auc_score(self.y_test[self.sens_test[feature]==0], self.pred_probs[self.sens_test[feature]==0]) #pval = 0 is Privileged
            B_fair=roc_auc_score(self.y_test[self.sens_test[feature]==1], self.pred_probs[self.sens_test[feature]==1]) #pval = 1 is Unprivileged
            model_metrics_difference_dict[f"ROC AUC Score Difference ({model_type}{sensitivity})"].append(abs(B_fair-A_fair))

        model_metrics_difference_data = pd.DataFrame(model_metrics_difference_dict)
        return model_metrics_difference_data
    
    def get_fairness_metrics_plots(self,metrics,acf_fairness_metrics,fm_fairness_metrics):
        """function to get fairness metrics comparison plots for protected features - Full Model VS ACF Model"""
        index = np.arange(len(metrics))
        bar_width = 0.35
        plt.figure(figsize=(50,25))

        for i,feature in enumerate(self.protected_features):
            ax = plt.subplot(2,2,i+1)
            fairness_metrics_dict = {"Metrics":metrics,'Full Model':fm_fairness_metrics.iloc[[i],[1,2,3]].values.tolist()[0], 'ACF Fair Model':acf_fairness_metrics.iloc[[i],[1,2,3]].values.tolist()[0]}
            fairness_metrics_table = pd.DataFrame.from_dict(fairness_metrics_dict)
            a = ax.bar(index, fairness_metrics_table["Full Model"], bar_width,color="red",label="Full Model")
            b = ax.bar(index+bar_width, fairness_metrics_table["ACF Fair Model"], bar_width, color="black",label="ACF Fair Model")
            ax.set_title(f"Fairness Metrics Comparison for Full Model vs ACF model for {feature}",fontsize=30)
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels(metrics,fontsize=30)
            ax.tick_params(axis='y', which='major', labelsize=30)
            ax.legend(loc='upper right',fontsize=30)
        plt.show()
        
    def get_crosstab_total_amount(self,predictions):
        crosstab = pd.crosstab(self.y_test, predictions, rownames=['Actual'], colnames=['Predicted']).rename(columns={0.0:"Non-Defaulter",1000.0:"Defaulter"}).rename({0.0:"Non-Defaulter",1000.0:"Defaulter"},axis="rows")
        
        df = pd.DataFrame()
        df["Default_Actual"] = pd.DataFrame(self.y_test.reset_index(drop=True))["Default"]
        df["Default_Predicted"] = pd.DataFrame(predictions)[0]
        df["AppliedAmount"] = pd.DataFrame(self.X_test["AppliedAmount"].reset_index(drop=True))
        
        TP = df[(df["Default_Actual"]==0) & (df["Default_Predicted"]==0)]["AppliedAmount"]
        FN = df[(df["Default_Actual"]==0) & (df["Default_Predicted"]==1)]["AppliedAmount"]
        FP = df[(df["Default_Actual"]==1) & (df["Default_Predicted"]==0)]["AppliedAmount"]
        TN = df[(df["Default_Actual"]==1) & (df["Default_Predicted"]==1)]["AppliedAmount"]
        
        total = sum(TP)-sum(FN)-sum(FP)
        return crosstab, total
    
    def get_error(self, predictions):
        """function to get error = diff(predicted,actual)"""
        error = self.y_test - predictions 
        return error
        
    def get_counterfactual_data(self):
        """function to invert privileged and unprivileged classes in protected features"""
        self.counter_sens_test = self.sens_test.replace({0:1, 1:0})
        counter_X_test = pd.concat([self.X_test[self.independent_features], self.counter_sens_test], axis=1)
        return counter_X_test
    
    def get_plots_CUF(self, normal_predictions, cuf_predictions, normal_error, cuf_error):
        """function to get plots for model with and w/o CUF"""
        model_type = "ACF Fair Model" if self.is_fair == True else "Full Model"
        plt.figure(figsize=(8,5))
        p1=sns.kdeplot(normal_predictions, shade=True, color="r")
        p1=sns.kdeplot(cuf_predictions, shade=True, color="b")
        plt.title(f'Density plot of predictions for sensitive features VS counterfactual sensitive features for {model_type}', fontsize=10)
        plt.axvline(np.mean(normal_predictions), color="r")
        plt.axvline(np.mean(cuf_predictions), color="b")
        plt.legend(['Without CUF','With CUF'])
        