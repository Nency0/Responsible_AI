import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


class ACF:

    def __init__(self, X_train, X_test, y_train, y_test, sens_train, sens_test, independent_vars):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.independent_vars = independent_vars
        self.sens_train = sens_train
        self.sens_test = sens_test

    def regression_model(self, sens, independent_variable):
        reg_model = LinearRegression().fit(sens, self.X_train[independent_variable])

        return reg_model

    def calc_residual(self, X, independent_variable, reg_model, sens):
        residual = X[independent_variable] - reg_model.predict(sens)

        return residual

    def create_residual_df(self, df_columns, residuals):
        residuals_dict = {df_columns[i]: residuals[i] for i in range(len(df_columns))}
        df_R = pd.DataFrame(residuals_dict)

        return df_R

    def fit(self, main_model):
        self.proxy_models = []
        residuals = []
        for var in range(0, len(self.independent_vars)):
            acf_model = self.regression_model(self.sens_train, self.independent_vars[var])
            residual = self.calc_residual(self.X_train, self.independent_vars[var], acf_model, self.sens_train)
            self.proxy_models.append(acf_model)
            residuals.append(residual)

        self.df_R_train = self.create_residual_df(self.independent_vars, residuals)
        self.fair = main_model.fit(self.df_R_train, self.y_train)

        return self.fair

    def transform_test_data(self):
        residuals = []
        for var in range(0, len(self.independent_vars)):
            residual = self.calc_residual(self.X_test, self.independent_vars[var], self.proxy_models[var], self.sens_test)
            residuals.append(residual)

        df_R_test = self.create_residual_df(self.independent_vars, residuals)
        return df_R_test

    def predict(self):
        df_R_test = self.transform_test_data()
        self.y_pred_fair = self.fair.predict(df_R_test)

        return self.y_pred_fair

    def predict_proba(self):
        df_R_test = self.transform_test_data()
        self.y_pred_prob_fair = self.fair.predict_proba(df_R_test)[:,0]

        return self.y_pred_prob_fair

    def score(self):
        df_R_test = self.transform_test_data()
        acc_score = self.fair.score(df_R_test, self.y_test)
        print("Accuracy of additive counterfactual fair model:", acc_score)

        return acc_score

    def confusion_mat(self, protected_group, binary_selector):
        tn, fp, fn, tp = confusion_matrix(self.y_test[self.sens_test[protected_group] == binary_selector],
                                          self.y_pred_fair[self.sens_test[protected_group] == binary_selector]).ravel()

        return tn, fp, fn, tp




