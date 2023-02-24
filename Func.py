#!/usr/bin/env python
# coding: utf-8

# In[1]:
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy import spatial
from sklearn.model_selection import train_test_split


def Accuracy_difference(X_test,y_test,model_name,model):
    y_pred_wow=model.predict(X_test)
    y_pred_prob_wow=model.predict_proba(X_test)[:,0]
    print("Accuracy of the model {} without weights:{}".format(model_name,model.score(X_test, y_test)))

#     for i in cols:
#         A_wow=model.score(X_test[X_test[i]==X_test[i].unique()[0]], y_test[X_test[i]==X_test[i].unique()[0]])
#         B_wow=model.score(X_test[X_test[i]==X_test[i].unique()[1]], y_test[X_test[i]==X_test[i].unique()[1]])
#         print("Accuracy difference between two groups {} :{}{} ".format(i,abs(B_wow-A_wow)*100 ,"%"))
    return y_pred_wow,y_pred_prob_wow

def model_metrics(y_true, y_pred_prob_ww, y_pred_prob_wow, Y_pred_binary_ww, Y_pred_binary_wow, X_test):
    """
    Model accuracy metrics for models with sample weights and without sample weights

    Parameters
    ----------

    :param y_true: Actual binary outcome
    :param y_pred_prob_ww: predicted probabilities with weights
    :param y_pred_prob_wow: predicted probabilities without weights
    :param Y_pred_binary_ww: predicted binary outcome with weights
    :param Y_pred_binary_wow: predicted binary outcome without weights
    :param X_test: Xtest data [not being used here]
    :return: roc, gini, avg precision, precision, sensitivity, tnr, fnr, f1, cost


    Examples
    --------
    model_perf=[model_metrics(y_test, y_pred_prob_ww, y_pred_prob_wow,
                          y_pred_ww, y_pred_wow, X_test1)]

    """
    tn_ww, fp_ww, fn_ww, tp_ww = confusion_matrix(y_true, Y_pred_binary_ww).ravel() #y_true, y_pred
    tn_wow, fp_wow, fn_wow, tp_wow = confusion_matrix(y_true, Y_pred_binary_wow).ravel()

    roc_ww = roc_auc_score(y_true, y_pred_prob_ww)
    roc_wow = roc_auc_score(y_true, y_pred_prob_wow)

    gini_ww = gini_normalized(y_true, y_pred_prob_ww)
    gini_wow = gini_normalized(y_true, y_pred_prob_wow)


    ps_ww = average_precision_score(y_true, Y_pred_binary_ww)
    ps_wow = average_precision_score(y_true, Y_pred_binary_wow)


    prec_ww = tp_ww / (tp_ww + fp_ww)
    prec_wow = tp_wow / (tp_wow + fp_wow)


    sensitivity_ww = tp_ww/(tp_ww+fn_ww)
    sensitivity_wow = tp_wow/(tp_wow+fn_wow)

    tnr_ww = tn_ww/(tn_ww + fp_ww)
    tnr_wow = tn_wow/(tn_wow + fp_wow)


    fnr_ww = fn_ww/(fn_ww+tp_ww)
    fnr_wow = fn_wow/(fn_wow+tp_wow)

    f1_ww = (2*tp_ww)/((2*tp_ww)+fp_ww+fn_ww)
    f1_wow = (2*tp_wow)/((2*tp_wow)+fp_wow+fn_wow)



    cost_ww = (fp_ww*700) + (fn_ww*300)
    cost_wow = (fp_wow*700) + (fn_wow*300)

    return roc_ww, gini_ww, ps_ww, prec_ww, sensitivity_ww, fnr_ww, f1_ww, cost_ww, roc_wow, gini_wow, ps_wow,  prec_wow, sensitivity_wow, fnr_wow, f1_wow, cost_wow


def fair_metrics(y_actual, y_pred_prob, y_pred_binary, X_test, protected_group_name,
                 adv_val, disadv_val):
    """
    Fairness performance metrics for a model to compare advantageous and disadvantageous groups of a protected variable

    Parameters
    ----------

    :param y_actual: Actual binary outcome
    :param y_pred_prob: predicted probabilities
    :param y_pred_binary: predicted binary outcome
    :param X_test: Xtest data
    :param protected_group_name: Sensitive feature
    :param adv_val: Priviliged value of protected label
    :param disadv_val: Unpriviliged value of protected label
    :return: roc, avg precision, Eq of Opportunity, Equalised Odds, Precision/Predictive Parity, Demographic Parity, Avg Odds Diff,
            Predictive Equality, Treatment Equality

    Examples
    --------
    fairness_metrics=[fair_metrics(y_test, y_pred_prob, y_pred,
                     X_test, choice, adv_val, disadv_val)]


    """
    tn_adv, fp_adv, fn_adv, tp_adv = confusion_matrix(
        y_actual[X_test[protected_group_name] == adv_val],
        y_pred_binary[X_test[protected_group_name] == adv_val]).ravel()

    tn_disadv, fp_disadv, fn_disadv, tp_disadv = confusion_matrix(
        y_actual[X_test[protected_group_name] == disadv_val],
        y_pred_binary[X_test[protected_group_name] == disadv_val]).ravel()

    # Receiver operating characteristic
    roc_adv = roc_auc_score(y_actual[X_test[protected_group_name] == adv_val],
                            y_pred_prob[X_test[protected_group_name] == adv_val])
    roc_disadv = roc_auc_score(y_actual[X_test[protected_group_name] == disadv_val],
                               y_pred_prob[X_test[protected_group_name] == disadv_val])

    roc_diff = abs(roc_disadv - roc_adv)

    # Average precision score
    ps_adv = average_precision_score(
        y_actual[X_test[protected_group_name] == adv_val],
        y_pred_prob[X_test[protected_group_name] == adv_val])
    ps_disadv = average_precision_score(
        y_actual[X_test[protected_group_name] == disadv_val],
        y_pred_prob[X_test[protected_group_name] == disadv_val])

    ps_diff = abs(ps_disadv - ps_adv)

    # Equal Opportunity - advantageous and disadvantageous groups have equal FNR
    FNR_adv = fn_adv / (fn_adv + tp_adv)
    FNR_disadv = fn_disadv / (fn_disadv + tp_disadv)
    EOpp_diff = abs(FNR_disadv - FNR_adv)

    # Predictive equality  - advantageous and disadvantageous groups have equal FPR
    FPR_adv = fp_adv / (fp_adv + tn_adv)
    FPR_disadv = fp_disadv / (fp_disadv + tn_disadv)
    pred_eq_diff = abs(FPR_disadv - FPR_adv)

    # Equalised Odds - advantageous and disadvantageous groups have equal TPR + FPR
    TPR_adv = tp_adv / (tp_adv + fn_adv)
    TPR_disadv = tp_disadv / (tp_disadv + fn_disadv)
    EOdds_diff = abs((TPR_disadv + FPR_disadv) - (TPR_adv + FPR_adv))

    # Predictive Parity - advantageous and disadvantageous groups have equal PPV/Precision (TP/TP+FP)
    prec_adv = (tp_adv)/(tp_adv + fp_adv)

    prec_disadv = (tp_disadv)/(tp_disadv + fp_disadv)

    prec_diff = abs(prec_disadv - prec_adv)


    # Demographic Parity - ratio of (instances with favourable prediction) / (total instances)
    demo_parity_adv = (tp_adv + fp_adv) / (tn_adv + fp_adv + fn_adv + tp_adv)
    demo_parity_disadv = (tp_disadv + fp_disadv) /             (tn_disadv + fp_disadv + fn_disadv + tp_disadv)
    demo_parity_diff = abs(demo_parity_disadv - demo_parity_adv)

    # Average of Difference in FPR and TPR for advantageous and disadvantageous groups
    AOD = 0.5*((FPR_disadv - FPR_adv) + (TPR_disadv - TPR_adv))

    # Treatment Equality  - advantageous and disadvantageous groups have equal ratio of FN/FP


    return [('Equal Opps', EOpp_diff),
            ('PredEq', pred_eq_diff), ('Equal Odds',
                                       EOdds_diff), 
            ('DemoParity', demo_parity_diff), ('AOD', abs(AOD))]


def acf_fair_metrics(tn_disadv, fp_disadv, fn_disadv, tp_disadv, tn_adv, fp_adv, fn_adv, tp_adv):
    """
    Fairness performance metrics for a additive counterfactually fair model to compare advantageous and
    disadvantageous groups of a protected variable

    :param tn_disadv: disadvantaged group's true negative
    :param fp_disadv: disadvantaged group's false positive
    :param fn_disadv: disadvantaged group's false negative
    :param tp_disadv: disadvantaged group's true positive
    :param tn_adv: advantaged group's true negative
    :param fp_adv: advantaged group's false positive
    :param fn_adv: advantaged group's false negative
    :param tp_adv: advantaged group's true positive
    :return: Equal Opportunity, Predictive Equality, Equalised Odds, Precision/Predictive Parity, Demographic Parity,
        Avg Odds Diff, Treatment Equality

    Examples
    --------
    acf_metrics=acf_fair_metrics(tn_disadv, fp_disadv, fn_disadv, tp_disadv, tn_adv, fp_adv, fn_adv, tp_adv)
    """

    # Equal Opportunity - advantageous and disadvantageous groups have equal FNR
    FNR_adv = fn_adv / (fn_adv + tp_adv)
    FNR_disadv = fn_disadv / (fn_disadv + tp_disadv)
    EOpp_diff = abs(FNR_disadv - FNR_adv)

    # Predictive equality  - advantageous and disadvantageous groups have equal FPR
    FPR_adv = fp_adv / (fp_adv + tn_adv)
    FPR_disadv = fp_disadv / (fp_disadv + tn_disadv)
    pred_eq_diff = abs(FPR_disadv - FPR_adv)

    # Equalised Odds - advantageous and disadvantageous groups have equal TPR + FPR
    TPR_adv = tp_adv / (tp_adv + fn_adv)
    TPR_disadv = tp_disadv / (tp_disadv + fn_disadv)
    EOdds_diff = abs((TPR_disadv + FPR_disadv) - (TPR_adv + FPR_adv))

    # Predictive Parity - advantageous and disadvantageous groups have equal PPV/Precision (TP/TP+FP)
    prec_adv = (tp_adv)/(tp_adv + fp_adv)
    prec_disadv = (tp_disadv)/(tp_disadv + fp_disadv)
    prec_diff = abs(prec_disadv - prec_adv)

    # Demographic Parity - ratio of (instances with favourable prediction) / (total instances)
    demo_parity_adv = (tp_adv + fp_adv) / (tn_adv + fp_adv + fn_adv + tp_adv)
    demo_parity_disadv = (tp_disadv + fp_disadv) /             (tn_disadv + fp_disadv + fn_disadv + tp_disadv)
    demo_parity_diff = abs(demo_parity_disadv - demo_parity_adv)

    # Average of Difference in FPR and TPR for advantageous and disadvantageous groups
    AOD = 0.5*((FPR_disadv - FPR_adv) + (TPR_disadv - TPR_adv))

    # Treatment Equality  - advantageous and disadvantageous groups have equal ratio of FN/FP
    TE_adv = fn_adv/fp_adv
    TE_disadv = fn_disadv/fp_disadv
    TE_diff = abs(TE_disadv - TE_adv)

    return [('Equal Opps', EOpp_diff),
            ('PredEq', pred_eq_diff), ('Equal Odds',
                                       EOdds_diff), 
            ('DemoParity', demo_parity_diff), ('AOD', abs(AOD))]



def circle(result):
        r = 1
        d = 10 * r * (1 - result)
        circle1=plt.Circle((0, 0), r, alpha=.2)
        circle2=plt.Circle((d, 0), r, alpha=.2)
        plt.ylim([-1.1, 1.1])
        plt.xlim([-1.1, 1.1 + d])
        fig = plt.gcf()
        fig.gca().add_artist(circle1)
        fig.gca().add_artist(circle2)

def cosine_similarity(df,n,lower_limit,upper_limit):
    for i in np.arange(lower_limit,upper_limit):
        result = 1 - spatial.distance.cosine(df[n], df.iloc[:,i])
        print ('cosine distance between {}  and {}'.format(n,df.columns[i]), result)
        circle(result)
def oneHotEnc(df,col_name):
    one_hot = pd.get_dummies(df[col_name], prefix=col_name)
    df = df.drop(col_name, axis=1)
    df = df.join(one_hot)
    return df

def heatmap_protected(df,i):

    pivot=df.reset_index().pivot_table(index=i, columns='default payment next month', aggfunc='count')
#     print(pivot)
    ax = sns.heatmap(pivot, annot=True,fmt='g')
    plt.show()
    return pivot

def statistical_parity_test(df,protected_group,Sa_label,Sd_label,y,fav_label,statistical_parity,disparate_impact):
    Sa=df[df[protected_group] == Sa_label]
    fav_Sa=Sa[Sa[y] == fav_label]
    fav_Sa_count = len(fav_Sa)
    Sd=df[df[protected_group] == Sd_label]
    fav_Sd=Sd[Sd[y] == fav_label]
    fav_Sd_count = len(fav_Sd)
    adv=len(Sa)
    disadv=len(Sd)
    statistical_parity.append((fav_Sd_count/disadv)-(fav_Sa_count/adv))
    disparate_impact.append((fav_Sd_count/disadv)/(fav_Sa_count/adv))


def plot_SPD_DI(cols,statistical_parity,disparate_impact):
    d = pd.DataFrame({'Protected_feature':cols,'Statistical_Parity':statistical_parity,'Disparate_Impact':disparate_impact})
    d['DI_normal']=d["Disparate_Impact"].apply(lambda x: 1/x if x < 1 else x)
    d['SP_normal']=d["Statistical_Parity"].apply(lambda x: abs(x) if x < 0 else x)

    fig = plt.figure() 
    ax = fig.add_subplot(111) 
    ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

    fig.suptitle('Normalised Statistical Parity Difference and Disparate Impact for comparison', fontsize=40, y=1)

    width = 0.3
    ax.set_ylim(0, 0.25) 
    ax2.set_ylim(0, 1.5) 

    d.plot(x ='Protected_feature', y='SP_normal', kind = 'bar', ax=ax, width=width, 
           position=1, color='green', legend=False, figsize=(30,10), fontsize=20)
    d.plot(x ='Protected_feature', y='DI_normal', kind = 'bar', ax=ax2, width=width, 
           position=0, color='black', legend=False, figsize=(30,10), fontsize=20)

    ax.axhline(y=0.10, linestyle='dashed', linewidth=2, alpha=0.7, color='green')
    ax2.axhline(y=0.80, linestyle='dashed', linewidth=2, alpha=0.7, color='black')

    patches, labels = ax.get_legend_handles_labels()
    ax.legend(patches, ['Stat Parity Diff'], loc='upper left', fontsize=25)

    patches, labels = ax2.get_legend_handles_labels()
    ax2.legend(patches, ['Disparate Impact'], loc='upper right', fontsize=25)

    labels = [item.get_text() for item in ax.get_xticklabels()]

    ax.set_xticklabels(labels)
    ax.set_xlabel('Protected Groups', fontsize=25)
    ax.set_ylabel('Statistical Parity Difference', fontsize=25)
    ax2.set_ylabel('Disparate Impact', fontsize=25)

    plt.show()

def Reweighing1 (data, choice, target_feature, pval, upval, fav=0, unfav=1):


    dummy = np.repeat(1, len(data)) 
    data['dummy'] = dummy

    n = np.sum(data['dummy']) #Total number of instances
    sa = np.sum(data['dummy'][data[choice]==pval]) #Total number of privileged
    sd = np.sum(data['dummy'][data[choice]==upval]) #Total number of unprivileged
    ypos = np.sum(data['dummy'][data[target_feature]==fav]) #Total number of favourable
    yneg = np.sum(data['dummy'][data[target_feature]==unfav]) #Total number of unfavourable

    data_sa_ypos = data[(data[choice]==pval) & (data[target_feature]==fav)] # priviliged and favourable
    data_sa_yneg = data[(data[choice]==pval) & (data[target_feature]==unfav)] # priviliged and unfavourable
    data_sd_ypos = data[(data[choice]==upval) & (data[target_feature]==fav)] # unpriviliged and favourable
    data_sd_yneg = data[(data[choice]==upval) & (data[target_feature]==unfav)] # unpriviliged and unfavourable

    sa_ypos = np.sum(data_sa_ypos['dummy']) #Total number of privileged and favourable
    sa_yneg = np.sum(data_sa_yneg['dummy']) #Total number of privileged and unfavourable
    sd_ypos = np.sum(data_sd_ypos['dummy']) #Total number of unprivileged and favourable
    sd_yneg = np.sum(data_sd_yneg['dummy']) #Total number of unprivileged and unfavourable

    w_sa_ypos= (ypos*sa) / (n*sa_ypos) #weight for privileged and favourable
    w_sa_yneg = (yneg*sa) / (n*sa_yneg) #weight for privileged and unfavourable
    w_sd_ypos = (ypos*sd) / (n*sd_ypos) #weight for unprivileged and favourable
    w_sd_yneg = (yneg*sd) / (n*sd_yneg) #weight for unprivileged and unfavourable

    datatest=data #.copy()

#     print (w_sa_ypos, w_sa_yneg, w_sd_ypos, w_sd_yneg)

    DiscriminationBefore=(sa_ypos/sa)-(sd_ypos/sd)
    DiscriminationAfter=(sa_ypos/sa * w_sa_ypos)-(sd_ypos/sd * w_sd_ypos)


    print ("DiscriminationBefore: {} \nDiscriminationAfter: {}  ".format(DiscriminationBefore, DiscriminationAfter))

    datatest['NewWeights']= np.repeat(999, len(datatest)) 
    datatest.loc[(datatest[choice]==pval) & (datatest[target_feature]==fav), 'NewWeights'] = w_sa_ypos
    datatest.loc[(datatest[choice]==pval) & (datatest[target_feature]==unfav), 'NewWeights'] = w_sa_yneg
    datatest.loc[(datatest[choice]==upval) & (datatest[target_feature]==fav), 'NewWeights'] = w_sd_ypos
    datatest.loc[(datatest[choice]==upval) & (datatest[target_feature]==unfav), 'NewWeights'] = w_sd_yneg

    return datatest['NewWeights']

def split_data(df):
    x=df.drop("default payment next month", axis=1)

    y=df[['default payment next month']]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=6666)

    return X_train, X_test, y_train, y_test


def gini(actual, pred):
    """

    :param actual: actual values
    :param pred: predicted probablities
    :return: gini scores
    """
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_normalized(actual, pred):
    """

    :param actual: actual values
    :param pred: predicted probablities
    :return: normalized gini scores
    """
    return gini(actual, pred) / gini(actual, actual)

def model_perff(y_test, y_pred_prob_ww, y_pred_prob_wow, y_pred_ww, y_pred_wow, X_test1):

    model_perf=[model_metrics(y_test, y_pred_prob_ww, y_pred_prob_wow, 
                              y_pred_ww, y_pred_wow, X_test1)]

    headers=["AUC", "Gini", "Avg Precision Score", "Precision", "Sensitivity", "False Negative Rate", 
             "F1 Score", "Total Cost"]


    #full_metric={'With Weights':B, 
    #             'Without_Weights':list(ww[0]), 'Without_Weights':list(wow[0])}

    #compare_table=pd.DataFrame.from_dict(ww_wow)

    B = list(model_perf[0])[:len(list(model_perf[0]))//2]
    C = list(model_perf[0])[len(list(model_perf[0]))//2:]


    model_table={'Metrics':headers, 
                 'With_Weights':B, 'Without_Weights':C}

    model_table_df=pd.DataFrame.from_dict(model_table)
    model_table_df.loc[8] = ['Total Cost (in Mn)', model_table_df.iloc[7,1]/10000000, model_table_df.iloc[7,2]/10000000]
    return model_table_df

def data_acf(df)  :
    dataacf = df[['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'PAY_0', 'PAY_2', 'PAY_3',
       'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
       'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
       'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
       'default payment next month', 'MARRIAGE_0', 'MARRIAGE_1', 'MARRIAGE_2',
       'MARRIAGE_3', 'age_bins_(21, 41]', 'age_bins_(41, 61]',
       'age_bins_(61, 81]']]
    return dataacf

def metrics(choices,X_test,y_test):
    for choice in choices:
        
        A_full=log_reg.score(X_test[X_test[choice]==0], y_test[X_test[choice]==0]) 
        B_full=log_reg.score(X_test[X_test[choice]==1], y_test[X_test[choice]==1]) 
        print("Accuracy difference between two groups:", abs(B_full-A_full)*100, "%")


# In[ ]:




