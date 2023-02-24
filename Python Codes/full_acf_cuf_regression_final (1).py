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

from fair_func import RMSE,mape

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
        self.sens_test = self.X_test[self.protected_features]
        self.counter_sens_test = None
        self.is_fair = is_fair
        self.is_sensitive = is_sensitive
        self.model = None
        self.predictions = None
    
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
            self.model = LinearRegression().fit(self.X_train, self.y_train)
        else:            
            self.model = LinearRegression().fit(X_train, self.y_train)
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
    
    def get_model_metrics(self):
        """function to get MSE, RMSE and MAPE for model"""
        model_type = "ACF Model" if self.is_fair == True else "Full Model"
        sensitivity = " without sensitive features" if self.is_sensitive == False else " with sensitive features" if self.is_fair == False else "" 
        metrics_data = pd.DataFrame({f"Mean Squared Error ({model_type}{sensitivity})":[mean_squared_error(self.y_test, self.predictions)], f"Root Mean Squared Error ({model_type}{sensitivity})":[RMSE(self.predictions, self.y_test)], f"Mean Absolute Percentage Error ({model_type}{sensitivity})":[mape(self.predictions, self.y_test)]})
        return metrics_data
    
    def get_model_performace_metrics_difference(self):
        """function to get MSE, RMSE and MAPE difference for protected features - privileged vs unprivileged"""
        model_type = "ACF Model" if self.is_fair == True else "Full Model"
        sensitivity = " without sensitive features" if self.is_sensitive == False else " with sensitive features" if self.is_fair == False else "" 
        performance_metrics_difference_dict = {"Column":[],f"Mean Squared Error Difference ({model_type}{sensitivity})":[],f"Root Mean Squared Error Difference ({model_type}{sensitivity})":[],f"Mean Absolute Percentage Error Difference ({model_type}{sensitivity})":[]}

        for feature in self.protected_features:
            performance_metrics_difference_dict["Column"].append(feature)

            A_fair=mean_squared_error(self.y_test[self.sens_test[feature]==0], self.predictions[self.sens_test[feature]==0]) #pval = 0 is Privileged
            B_fair=mean_squared_error(self.y_test[self.sens_test[feature]==1], self.predictions[self.sens_test[feature]==1]) #pval = 1 is Unprivileged
            performance_metrics_difference_dict[f"Mean Squared Error Difference ({model_type}{sensitivity})"].append(abs(B_fair-A_fair))

            A_fair=RMSE(self.y_test[self.sens_test[feature]==0], self.predictions[self.sens_test[feature]==0]) #pval = 0 is Privileged
            B_fair=RMSE(self.y_test[self.sens_test[feature]==1], self.predictions[self.sens_test[feature]==1]) #pval = 1 is Unprivileged
            performance_metrics_difference_dict[f"Root Mean Squared Error Difference ({model_type}{sensitivity})"].append(abs(B_fair-A_fair))

            A_fair=mape(self.y_test[self.sens_test[feature]==0], self.predictions[self.sens_test[feature]==0]) #pval = 0 is Privileged
            B_fair=mape(self.y_test[self.sens_test[feature]==1], self.predictions[self.sens_test[feature]==1]) #pval = 1 is Unprivileged
            performance_metrics_difference_dict[f"Mean Absolute Percentage Error Difference ({model_type}{sensitivity})"].append(abs(B_fair-A_fair))

        performance_metrics_difference_data = pd.DataFrame(performance_metrics_difference_dict)
        return performance_metrics_difference_data
    
    def get_density_plots(self):
        """function to get density plots of prediction distribution for protected features - privileged vs unprivileged"""
        model_type = "ACF Fair Model" if self.is_fair == True else "Full Model"
        sensitivity = " without sensitive features" if self.is_sensitive == False else " with sensitive features" if self.is_fair == False else "" 
        plt.figure(figsize=(20,10))

        for i,feature in enumerate(self.protected_features):

            plt.subplot(2,2,i+1)
            p1=sns.kdeplot(self.predictions[self.sens_test[feature]==0], shade=True, color="r")
            p1=sns.kdeplot(self.predictions[self.sens_test[feature]==1], shade=True, color="b")
            plt.title(f"Density Plot for prediction by {model_type}{sensitivity}: {feature}", fontsize=12)
            plt.axvline(np.mean(self.predictions[self.sens_test[feature]==0]), color="r" )
            plt.axvline(np.mean(self.predictions[self.sens_test[feature]==1]), color="b")
            plt.legend(['privileged','unprivileged'])

        plt.show()
        
    def get_prediction_distribution_metrics_difference(self):
        """function to get mean, skewness and kurtosis difference in prediction distribution for protected features - privileged vs unprivileged"""
        model_type = "ACF Model" if self.is_fair == True else "Full Model"
        sensitivity = " without sensitive features" if self.is_sensitive == False else " with sensitive features" if self.is_fair == False else "" 
        distribution_metrics_dict = {"Column":[],f"Mean difference of predicted target value ({model_type}{sensitivity})":[],
                                     f"Skewness difference of predicted target value ({model_type}{sensitivity})":[],
                                     f"Kurtosis difference of predicted target value ({model_type}{sensitivity})":[],
                                     f"Mean ratio of predicted target value ({model_type}{sensitivity})":[],
                                     f"Skewness ratio of predicted target value ({model_type}{sensitivity})":[],
                                     f"Kurtosis ratio of predicted target value ({model_type}{sensitivity})":[]}

        for feature in self.protected_features:
            distribution_metrics_dict["Column"].append(feature)
            distribution_metrics_dict[f"Mean difference of predicted target value ({model_type}{sensitivity})"].append(np.mean(self.predictions[self.sens_test[feature]==0]) - np.mean(self.predictions[self.sens_test[feature]==1]))
            distribution_metrics_dict[f"Skewness difference of predicted target value ({model_type}{sensitivity})"].append(skew(self.predictions[self.sens_test[feature]==0]) - skew(self.predictions[self.sens_test[feature]==1]))
            distribution_metrics_dict[f"Kurtosis difference of predicted target value ({model_type}{sensitivity})"].append(kurtosis(self.predictions[self.sens_test[feature]==0]) - kurtosis(self.predictions[self.sens_test[feature]==1]))
            distribution_metrics_dict[f"Mean ratio of predicted target value ({model_type}{sensitivity})"].append(np.mean(self.predictions[self.sens_test[feature]==0]) / np.mean(self.predictions[self.sens_test[feature]==1]))
            distribution_metrics_dict[f"Skewness ratio of predicted target value ({model_type}{sensitivity})"].append(skew(self.predictions[self.sens_test[feature]==0]) / skew(self.predictions[self.sens_test[feature]==1]))
            distribution_metrics_dict[f"Kurtosis ratio of predicted target value ({model_type}{sensitivity})"].append(kurtosis(self.predictions[self.sens_test[feature]==0]) / kurtosis(self.predictions[self.sens_test[feature]==1]))

        distribution_metrics_data = pd.DataFrame(distribution_metrics_dict)
        return distribution_metrics_data
    
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
        plt.figure(figsize=(10*3,6))
        
        plt.subplot(1,3,1)
        plt.scatter(self.y_test, normal_predictions, s=1, alpha = 0.3, color='r')
        plt.scatter(self.y_test, cuf_predictions, s=1, alpha=0.3, color='b')
        plt.title(f"Actual vs Fitted on {model_type}", fontsize=16)
        plt.legend(['Without CUF','With CUF'])
        
        plt.subplot(1,3,2)
        p1=sns.kdeplot(normal_predictions, shade=True, color="r")
        p1=sns.kdeplot(cuf_predictions, shade=True, color="b")
        plt.title(f'Density plot of predictions for sensitive features VS counterfactual sensitive features for {model_type}', fontsize=10)
        plt.axvline(np.mean(normal_predictions), color="r")
        plt.axvline(np.mean(cuf_predictions), color="b")
        plt.legend(['Without CUF','With CUF'])
        
        plt.subplot(1,3,3)
        p1=sns.kdeplot(normal_error, shade=True, color="r")
        p1=sns.kdeplot(cuf_error, shade=True, color="b")
        plt.title(f'Density plot of errors for sensitive features vs counterfactual sensitive features on target variable for {model_type}', fontsize=10)
        plt.axvline(np.mean(normal_error), color="r")
        plt.axvline(np.mean(cuf_error), color="b")
        plt.legend(['Without CUF','With CUF'])

        plt.show()
        
    def get_error_vs_amount_plot(self,amount_col,acf_predictions):
        residuals = self.y_test - acf_predictions
        plt.figure(figsize=(10,6))
        plt.scatter(amount_col,residuals,s=5)
        plt.title("Error vs AppliedAmount", fontsize=16)
        plt.xlabel("Applied Amount")
        plt.ylabel("Error")
        m, b = np.polyfit(amount_col, residuals, 1)
        plt.plot(amount_col, m*amount_col+b,color='r')