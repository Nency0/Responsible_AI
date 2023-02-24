## Importing the desired Packages/ Libraries.
## Data Analysis packages
import pandas as pd
import numpy as np

# Visualization packages
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="darkgrid")

# ## Machine learning packages
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# ## Model Interpretation package
#!pip install pdpbox
import pdpbox as pdb
from pdpbox import pdp, get_dataset, info_plots
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
from time import time
#!pip install pygam
import pygam
from pygam import LinearGAM, s, f
from pygam import LogisticGAM
# For generating Counter factuals
#pip install dice-ml
import dice_ml
from dice_ml.utils import helpers

import os

import warnings
warnings.filterwarnings('ignore')



## Declaring class for IV and PDP functions:
class IV_PDP:
    
    def __init__(self, data, independent_features, target_variable):
        """function to initialize class"""
        self.data = data
        y=self.data[target_variable]
        x=self.data.drop(columns=[target_variable])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.20, random_state=555)
        self.independent_features = independent_features
        self.target_variable = target_variable
        self.model = None
        self.predictions = None
        self.PDP = None
        self.IV = None
        
    def fit(self):
        """function to fit model"""
        self.model = RandomForestClassifier().fit(self.X_train, self.y_train)
        return self.model
    
        
    def predict(self):
        """function to predict using model"""
        self.predictions = self.model.predict(self.X_test)
        return self.predictions
    
    def PDP_PLOT(self):
        """function to show pdp plots"""
        self.PDP = plot_partial_dependence(self.model, self.X_train, self.independent_features,
                        n_jobs=3, grid_resolution=20)
        fig = plt.gcf()
        fig.set_size_inches(14, 9)
        fig.suptitle('Partial Dependence Plots')
        fig.subplots_adjust(wspace=0.5, hspace=0.4)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        
    def IV_PLOT(self):
        """function to show iv plots"""
        for i in range(0,4):
            self.IV = info_plots.actual_plot(self.model, self.X_train,
                                       feature=self.independent_features[i],feature_name=self.independent_features[i],predict_kwds={})
    def EDA_PLOT(self):
        """function to show EDA plots"""
        for i in range(0,2):
            self.EDA = info_plots.target_plot(df=self.data, feature=self.independent_features[i], feature_name=self.independent_features[i], 
                                          target=self.target_variable, show_percentile=True)
            
            
## Declaring class for Split and Compare Quantiles functions:
class SCQ:
    
    def __init__(self, data, independent_features, target_variable):
        """function to initialize class"""
        self.data = data
        y=self.data[target_variable]
        x=self.data.drop(columns=[target_variable])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.20, random_state=555)
        self.independent_features = independent_features
        self.target_variable = target_variable
        self.model = None
        self.predictions = None
        self.view = None
    
    def fit(self):
        """function to fit model"""
        self.model = RandomForestClassifier(random_state=66).fit(self.X_train.loc[:,self.independent_features], self.y_train)
        return self.model
    
        
    def predict(self):
        """function to predict using model"""
        self.score = pd.DataFrame(self.model.predict_proba(self.X_test.loc[:,self.independent_features])[:,0])
        return self.score
    
    def create_view(self,amount_col,score_col,col_list):
        """function to create a dataframe"""
        ytest = pd.DataFrame(self.y_test)
        ytest = ytest.reset_index(drop=True)
        amount=self.X_test[amount_col]
        amount.reset_index(drop=True,inplace=True)
        self.view = pd.concat([ytest,self.score,amount], axis=1)
        self.view.columns = col_list
        self.view = self.view.reset_index(drop=True)
        self.view['Score2']=self.view[score_col]*100
        return self.view 
        

## Declaring function for Split and Compare Quantiles Plots:
def Show_quantile_label(df1,df2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    margin = 0.05
    width = (1.-2.*margin)/2
    if df2 is None:
        pal = sns.color_palette("RdYlBu", len(df1['Range']))
        chart = df1.set_index('Range').drop(columns=['Label 1'])
        chart2 = df1.set_index('Range').drop(columns=['Label 0'])

    elif df1 is None:
        pal = sns.color_palette("RdYlBu", len(df2['Range']))
        chart = df2.set_index('Range').drop(columns=['Amount:1'])
        chart2 = df2.set_index('Range').drop(columns=['Amount:0'])

    chart.T.plot.bar(stacked=True, color = pal, grid=True, ax=ax1, width=width, legend=False)
    for rect in ax1.patches:
        height = rect.get_height()
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()   
        label_text = f'{height:.2f}'      
        label_x = x + width/2
        label_y = (y + height/2)
        ax1.text(label_x, label_y, label_text, ha='center', va='center', fontsize=12)
       

    chart2.T.plot.bar(stacked=True, color = pal, grid=True, ax=ax2, width=width, legend=False)
    for rect in ax2.patches:
        height = rect.get_height()
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()    
        label_text = f'{height:.2f}'      
        label_x = x + width/2
        label_y = (y + height/2)
        ax2.text(label_x, label_y, label_text, ha='center', va='center', fontsize=12)
        

    plt.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend()
    plt.show()
    

## Declaring class for GAM:
class GAM:
    
    def __init__(self, data, independent_features, target_variable):
        """function to initialize class"""
        self.data = data
        y=self.data[target_variable]
        x=self.data.drop(columns=[target_variable])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.20, random_state=555)
        self.independent_features = independent_features
        self.target_variable = target_variable
        self.model = None
        self.train_acc = None
        self.test_acc = None
        self.summary = None
    
    def fit(self):
        """function to fit model"""
        self.model = LogisticGAM().fit(self.X_train.loc[:,self.independent_features], self.y_train)
        return self.model
    
        
    def predict(self):
        """function to evaulate the accuarcy and show summary of model"""
        self.train_acc = self.model.accuracy(self.X_train.loc[:,self.independent_features], self.y_train)
        self.test_acc = self.model.accuracy(self.X_test.loc[:,self.independent_features], self.y_test)
        self.summary = self.model.summary
        return self.train_acc, self.test_acc, self.summary
    
    def GAM_PLOT(self):
        titles=self.X_train.columns.to_list()

        for i, term in enumerate(self.model.terms):
            if term.isintercept:
                continue
            XX = self.model.generate_X_grid(term=i)
            pdep, confi = self.model.partial_dependence(term=i, X=XX, width=0.95)
            plt.figure()
            plt.rcParams['figure.figsize'] = (10, 6)

            plt.plot(XX[:, term.feature], pdep)
            plt.plot(XX[:, term.feature], confi, c='r', ls='--')
            plt.title(repr(titles[i]))
            plt.show()
            
            
## Declaring class for Counterfactuals:            
class CF:
    
    def __init__(self, data, independent_features, target_variable):
        """function to initialize class"""
        self.data = data
        y=self.data[target_variable]
        #x=self.data.drop(columns=[target_variable])
        self.train_dataset, self.test_dataset, self.y_train, self.y_test = train_test_split(self.data,y,test_size=0.2,random_state=0,stratify=y)
        self.X_train = self.train_dataset.drop(columns = [target_variable])
        self.X_test = self.test_dataset.drop(columns = [target_variable])
        self.independent_features = independent_features
        self.target_variable = target_variable
        self.model = None
        self.exp = None
    
    def fit(self):
        """function to fit model"""
        self.model = RandomForestClassifier().fit(self.X_train, self.y_train)
        return self.model
    
        
    def dice_ml(self):
        """function to evaulate the accuarcy and show summary of model"""
        d = dice_ml.Data(dataframe=self.train_dataset, continuous_features=self.independent_features, outcome_name=self.target_variable)
        m = dice_ml.Model(model=self.model, backend="sklearn")
        self.exp = dice_ml.Dice(d, m, method="random")
        return self.exp
    
    def CF_EXP(self):
        self.e1 = self.exp.generate_counterfactuals(self.X_test[3:4], total_CFs=4, 
                                                    desired_class="opposite",features_to_vary=self.independent_features[1:4])
        self.e1.visualize_as_dataframe(show_only_changes=True)
        
        
##The End!
            

           
