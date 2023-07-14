# -*- coding: utf-8 -*-
"""
Author: Saima Gaffar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedKFold
from collections import Counter

#AAC
#Import training file
data=pd.read_csv('AAC/Train_AAC.csv', header=None)

#Import test file
data2=pd.read_csv('AAC/Ind_1301_AAC.csv', header=None)

#Import test file
data3=pd.read_csv('AAC/Ind_425_AAC.csv', header=None)

X_train1 =  data.iloc[:,1:].values
X_test1=  data2.iloc[:1049,1:].values
X_test11=  data3.iloc[:,1:].values

#DPC
#Import training file
data=pd.read_csv('DPC/Train_DPC.csv', header=None)

#Import test file
data2=pd.read_csv('DPC/Ind_1301_DPC.csv', header=None)

#Import test file
data3=pd.read_csv('DPC/Ind_425_DPC.csv', header=None)

X_train2 =  data.iloc[:,1:].values
X_test2=  data2.iloc[:1049,1:].values
X_test22=  data3.iloc[:,1:].values

# #PAAC
# #Import training file
# data=pd.read_csv('PAAC/Train_PAAC.csv', header=None)

# #Import test file
# data2=pd.read_csv('PAAC/Ind_1301_PAAC.csv', header=None)

# #Import test file
# data3=pd.read_csv('PAAC/Ind_425_PAAC.csv', header=None)

# X_train3 =  data.iloc[:,1:].values
# X_test3=  data2.iloc[:1049,1:].values
# X_test33=  data3.iloc[:,1:].values

# #SOCN
# #Import training file
# data=pd.read_csv('SCON/Train_SCON.csv')

# #Import test file
# data2=pd.read_csv('SCON/Ind_1301_SCON.csv')

# #Import test file
# data3=pd.read_csv('SCON/Ind_425_SCON.csv')

# X_train4=  data.iloc[:,1:-1].values
# X_test4 = data2.iloc[:1049,1:-1].values
# X_test44= data3.iloc[:,1:-1].values

#QSO
#Import training file
data=pd.read_csv('QSO/Train_QSO.csv')

#Import test file
data2=pd.read_csv('QSO/Ind_1301_QSO.csv')

#Import test file
data3=pd.read_csv('QSO/Ind_425_QSO.csv')

X_train5 =  data.iloc[:,1:-1].values
X_test5=  data2.iloc[:1049,1:-1].values
X_test55=  data3.iloc[:,1:-1].values

#CKSAAGP
#Import training file
data=pd.read_csv('CKSAAGP/Train_CKSAAGP_k_3.csv',header=None)

#Import test file
data2=pd.read_csv('CKSAAGP/Ind_1301_k_3.csv', header=None)

#Import test file
data3=pd.read_csv('CKSAAGP/Ind_425_k_3.csv', header=None)

X_train6 =  data.iloc[:,1:].values
X_test6=  data2.iloc[:1049,1:].values
X_test66=  data3.iloc[:,1:].values

#GTPC
#Import training file
data=pd.read_csv('GTPC/Train_GTPC.csv', header=None)

#Import test file
data2=pd.read_csv('GTPC/Ind_1301_GTPC.csv', header=None)

#Import test file
data3=pd.read_csv('GTPC/Ind_425_GTPC.csv', header=None)

X_train7 =  data.iloc[:,1:].values
X_test7=  data2.iloc[:1049,1:].values
X_test77=  data3.iloc[:,1:].values

# #PAAC
# #Import training file
# data=pd.read_csv('APAAC/Train_APAAC.csv', header=None)

# #Import test file
# data2=pd.read_csv('APAAC/Ind_1301_APAAC.csv', header=None)

# #Import test file
# data3=pd.read_csv('APAAC/Ind_425_APAAC.csv', header=None)

# X_train8 =  data.iloc[:,1:].values
# X_test8=  data2.iloc[:1049,1:].values
# X_test88=  data3.iloc[:,1:].values

X_train=np.concatenate((X_train1,X_train2,X_train5,X_train6,X_train7), axis=1, out=None)

X_test01=np.concatenate((X_test1,X_test2,X_test5,X_test6,X_test7), axis=1, out=None)

X_test02=np.concatenate((X_test11,X_test22,X_test55,X_test66,X_test77), axis=1, out=None)

X_train.shape, X_test01.shape, X_test02.shape

def evaluate_model_test(model, X_test, y_test):
    from sklearn import metrics

    # Predict Test Data 
    y_pred = model.predict(X_test)
    
    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    # Calculate area under curve (AUC)
    y_pred_proba = model.predict_proba(X_test)[::,1]
    #fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    
    #MCC
    mcc=matthews_corrcoef(y_test, y_pred)
    
    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    total=sum(sum(cm))
    
    #accuracy=(cm[0,0]+cm[1,1])/total
    spec = cm[0,0]/(cm[0,1]+cm[0,0])
    sen= cm[1,1]/(cm[1,0]+cm[1,1])
    
    
    # Print result
    print('\t Accuracy:', acc)
    print('\t Precision:', prec)
    print('\t Recall:', rec)
    print('\t F1 Score:', f1)
    print('\t Area Under Curve:', auc)
    print('\t Sensitivity : ',sen)
    print('\t Specificity : ', spec)
    print('\t MCC Score : ', mcc)
    print('\t Confusion Matrix:\n', cm)
    print('\n')
    print('\n')


    return

def evaluate_model_train(model, X_train, y_train):
    from sklearn import metrics
    conf_matrix_list_of_arrays = []
    mcc_array=[]
    #cv = KFold(n_splits=5)
    #cv = StratifiedKFold(n_splits=5)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)
    lst_accu = []
    AUC_list=[]
    
    
    prec_train=np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring='precision'))
    recall_train=np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring='recall'))
    f1_train=np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring='f1'))
    Acc=np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy'))
    
    
    
    for train_index, test_index in cv.split(X_train, y_train): 
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index] 
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        
        
        model.fit(X_train_fold, y_train_fold)
        y_pred=model.predict(X_test_fold)
        
        lst_accu.append(model.score(X_test_fold, y_test_fold))
        acc=np.mean(lst_accu)
        
        conf_matrix = confusion_matrix(y_test_fold, y_pred)
        conf_matrix_list_of_arrays.append(conf_matrix)
        cm = np.mean(conf_matrix_list_of_arrays, axis=0)
        
        mcc_array.append(matthews_corrcoef(y_test_fold, model.predict(X_test_fold)))
        mcc=np.mean(mcc_array, axis=0)
        
        AUC=metrics.roc_auc_score( y_test_fold, model.predict_proba(X_test_fold)[:,1])
        AUC_list.append(AUC)
        auc=np.mean(AUC_list)
        
        
    total=sum(sum(cm))
    acc=(cm[0,0]+cm[1,1])/total
    specificity = cm[0,0]/(cm[0,1]+cm[0,0])
    sensitivity= cm[1,1]/(cm[1,0]+cm[1,1])
    
    
    #print("\t Confusion Matrix is: \n", cm)
    print ('\t Accuracy : ', Acc)
    print('\t Sensitivity : ', sensitivity)
    print('\t Specificity : ', specificity)
    print("\t Mean of Matthews Correlation Coefficient is: ", mcc)
    print("\t The Acc value from CM is: ", acc)
    print("\t The Recall value is: ", recall_train)
    print("\t The F1 score is: ", f1_train)
    print('\t The area under curve is:',auc)
    print('\n')

# label the dataset
pos1 = np.ones(1451)
neg1 = np.zeros(2339)
y_train = np.concatenate((pos1,neg1),axis=0)


pos_test = np.ones(420)
neg_test = np.zeros(629)
y_test1 = np.concatenate((pos_test,neg_test),axis=0)


pos_test2 = np.ones(173)
neg_test2 = np.zeros(252)
y_test2 = np.concatenate((pos_test2,neg_test2),axis=0)



# transform the dataset
import imblearn
from imblearn.over_sampling import SMOTE,  ADASYN
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
oversample = SMOTETomek()
X_train, y_train = oversample.fit_resample(X_train, y_train)
print(sorted(Counter(y_train).items()))

cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=1)

# Random Forest


import optuna
from sklearn.ensemble import RandomForestClassifier
def RF_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    max_depth = trial.suggest_int('max_depth', 1, 60)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 1000)
    min_samples_split= trial.suggest_int("min_samples_split", 2, 20)
    
    ## Create Model
    model = RandomForestClassifier(max_depth = max_depth, min_samples_split=min_samples_split,
                                   n_estimators = n_estimators,n_jobs=2,
                                     random_state=25)
    
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    accuracy_mean = score.mean()
    return accuracy_mean

#Execute optuna and set hyperparameters
RF_study = optuna.create_study(direction='maximize')
RF_study.optimize(RF_objective, n_trials=2)

optimized_RF=RandomForestClassifier(**RF_study.best_params)

# tsne1=rfc.predict_proba(X_train)[:,1]
# tsne1=pd.DataFrame(tsne1)

# ExtraTreeClassifier


from sklearn.ensemble import ExtraTreesClassifier
import optuna
def objective(trial):
    """Define the objective function"""
    params = {
            'n_estimators' : trial.suggest_int('n_estimators', 100, 2000),
            'max_depth' : trial.suggest_int('max_depth', 10, 60),
            'max_leaf_nodes' : trial.suggest_int('max_leaf_nodes', 15, 100),
            'criterion' : trial.suggest_categorical('criterion', ['gini', 'entropy'])

    }


    # Fit the model
    etc_model = ExtraTreesClassifier(**params)
    score = cross_val_score(etc_model, X_train, y_train, cv=cv, scoring="accuracy")
    accuracy_mean = score.mean()

    return accuracy_mean


#Execute optuna and set hyperparameters
etc_study = optuna.create_study(direction='maximize')
etc_study.optimize(objective, n_trials=2)

optimized_etc =ExtraTreesClassifier(**etc_study.best_params)

# XGB


from xgboost import XGBClassifier
#cv = RepeatedStratifiedKFold(n_splits=5)
import optuna
def objective(trial):
    """Define the objective function"""

    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 370),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 10.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'gamma': trial.suggest_float('gamma', 1e-8, 10.0),
        'subsample': trial.suggest_float('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0),
        #'eval_metric': 'mlogloss',
        #'use_label_encoder': False
    }

    # Fit the model
    xgb_model = XGBClassifier(**params,  eval_metric='mlogloss')
    score = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring="accuracy")
    accuracy_mean = score.mean()
    return accuracy_mean
#Execute optuna and set hyperparameters
XGB_study = optuna.create_study(direction='maximize')
XGB_study.optimize(objective, n_trials=2)
optimized_XGB =XGBClassifier(**XGB_study.best_params)

# tsne4=xgb.predict_proba(X_train)[:,1]
# tsne4=pd.DataFrame(tsne4)

# LGBM

import lightgbm as lgbm
import optuna
def objective(trial):
    """Define the objective function"""
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 2, 100), 
        'max_depth': trial.suggest_int('max_depth', 1, 100), 
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 1), 
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000), 
        #'objective': 'multiclass', 
        # 'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100), 
        'subsample': trial.suggest_float('subsample', 0.7, 1.0), 
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 0
    }


    # Fit the model
    lgbm_model = lgbm.LGBMClassifier(**params)
    score = cross_val_score(lgbm_model, X_train, y_train, cv=cv, scoring="accuracy")
    accuracy_mean = score.mean()

    return accuracy_mean


#Execute optuna and set hyperparameters
lgbm_study = optuna.create_study(direction='maximize')
lgbm_study.optimize(objective, n_trials=2)

optimized_lgbm =lgbm.LGBMClassifier(**lgbm_study.best_params)

# CatBoost


from catboost import CatBoostClassifier
def objective(trial):
    params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
            "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 1, 12),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
            "used_ram_limit": "3gb",
        }

#     if param["bootstrap_type"] == "Bayesian":
#         param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
#     elif param["bootstrap_type"] == "Bernoulli":
#         param["subsample"] = trial.suggest_float("subsample", 0.1, 1)


    # Fit the model
    cat_model = CatBoostClassifier(**params, silent=True)
    score = cross_val_score(cat_model, X_train, y_train, cv=cv, scoring="accuracy")
    accuracy_mean = score.mean()

    return accuracy_mean


#Execute optuna and set hyperparameters
cat_study = optuna.create_study(direction='maximize')
cat_study.optimize(objective, n_trials=2)

optimized_cat =CatBoostClassifier(**cat_study.best_params, silent=True)

# VotingClassifier

from sklearn.ensemble import VotingClassifier
v_clf = VotingClassifier(estimators=[('RF', optimized_RF), ('XGB', optimized_XGB), 
                                     ("Cat", optimized_cat), ('ETC', optimized_etc), 
                                     ('LGBM', optimized_lgbm)], voting='soft')

# Results 

model={'rfc': optimized_RF, 'etc':optimized_etc,
       'lgbm': optimized_lgbm, 'xgb':optimized_XGB, 
       'Cat':optimized_cat, 'voting':v_clf}

for key in model:
    
    print(model[key])

from termcolor import colored
for key in model:
    if key=='rfc':
        print('Cross validation results using ', key)
        train_eval = evaluate_model_train(model[key], X_train, y_train)
        print('Optimized results on 1st dataset using ' , key)
        dtc_eval = evaluate_model_test(model[key], X_test01, y_test1)
        print('Optimized results on 2nd dataset using ',key)
        dtc_eval = evaluate_model_test(model[key], X_test02, y_test2)
        print(colored('===================================================', 'red'))
    elif key=='etc':
        print('Cross validation results using ', key)
        train_eval = evaluate_model_train(model[key], X_train, y_train)
        print('Optimized Results on 1st dataset using ' , key)
        dtc_eval = evaluate_model_test(model[key], X_test01, y_test1)
        print('Optimized Results on 2nd dataset using ' , key)
        dtc_eval = evaluate_model_test(model[key], X_test02, y_test2)
        print(colored('===================================================', 'red'))
    elif key=='lgbm':
        print('Cross validation results using ', key)
        train_eval = evaluate_model_train(model[key], X_train, y_train)
        print('Optimized Results on 1st dataset using ' , key)
        dtc_eval = evaluate_model_test(model[key], X_test01, y_test1)
        print('Optimized Results on 2nd dataset using ' , key)
        dtc_eval = evaluate_model_test(model[key], X_test02, y_test2)
        print(colored('===================================================', 'red'))
    elif key=='xgb':
        print('Cross validation results using ', key)
        train_eval = evaluate_model_train(model[key], X_train, y_train)
        print('Optimized Results on 1st dataset using ' , key)
        dtc_eval = evaluate_model_test(model[key], X_test01, y_test1)
        print('Optimized Results on 2nd dataset using ' , key)
        dtc_eval = evaluate_model_test(model[key], X_test02, y_test2)
        print(colored('===================================================', 'red'))
    elif key=='voting':
        print('Cross validation results using ', key)
        train_eval = evaluate_model_train(model[key], X_train, y_train)
        print('Optimized Results on 1st dataset using ' , key)
        dtc_eval = evaluate_model_test(model[key], X_test01, y_test1)
        print('Optimized Results on 2nd dataset using ' , key)
        dtc_eval = evaluate_model_test(model[key], X_test02, y_test2)
        print(colored('===================================================', 'red'))     
    elif key=='Cat':
        print('Cross validation results using ', key)
        train_eval = evaluate_model_train(model[key], X_train, y_train)
        print('Optimized Results on 1st dataset using ' , key)
        dtc_eval = evaluate_model_test(model[key], X_test01, y_test1)
        print('Optimized Results on 2nd dataset using ' , key)
        dtc_eval = evaluate_model_test(model[key], X_test02, y_test2)
        print(colored('===================================================', 'red'))
    else:
        print('Cross validation results using ', key)
        train_eval = evaluate_model_train(model[key], X_train, y_train)
        print('Optimized Results on 1st dataset using ' , key)
        dtc_eval = evaluate_model_test(model[key], X_test01, y_test1)
        print('Optimized Results on 2nd dataset using ' , key)
        dtc_eval = evaluate_model_test(model[key], X_test02, y_test2)
        print(colored('===================================================', 'red')) 



