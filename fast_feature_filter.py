# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 09:12:12 2018

@author: june
"""

import pandas as pd
import numpy as np
import math
import os
import time
import dask.dataframe as dd
import numba

_DEBUG = True
global outputfilename
outputfilename=''
global outputfilepath
outputfilepath=''

def setEnvInfo(filepath, filename):
    """
    Configure data file path and file name. 
    This must be used before other method, as it generate log info storage path.
    Parameters
    ----------
    filepath : string
        log file path
    filename : string
        log file name
    Output
    -------
    Generate path filepath/filename/ to storage result.    
    """
    global outputfilename
    global outputfilepath
    outputfilename = filename
    outputfilepath = filepath
    if not os.path.exists(outputfilepath):
        os.mkdir(outputfilepath) 

def _log(*arg, mode):
    global outputfilename
    global outputfilepath
    if outputfilename == '' or outputfilepath == '':
        return  
    timeline = time.strftime("%Y_%m_%d", time.localtime()) 
    with open(outputfilepath+outputfilename+mode+timeline+'.filter', "a+") as text_file:
        print(*arg, file=text_file)

def trace(*arg):
    _log(*arg, mode='trace')

def debug(*arg):
    if _DEBUG == True:
        _log(*arg, mode = 'debug')

def filter_auto(dataframe, target='', model='tree'):
    """
    Integrated API to Filter or select features with multi methods as infinite 
    confine, Filter constant features, Filter duplicated features, Filter highly 
    correlated features. This API also Filter feature according to weight from 
    from different models as linear models and random forest.
    Parameters
    ----------
    dataframe : pandas.Dataframe
        dataframe to explore importance

    target : string        
        Feature name, as predict result.

    model : 'tree', 'linear', 'none'
        Filter features according to weights from model types. Random forest is
        adopted if select 'tree'; linear regression/ logistic regression is
        adopted if select 'linear'; none model is adopted if select 'none'.

    Return
    -------
    dataframe after Filter redundant features
    """
    model_type = 'classifier'
    if dataframe.loc[1:500,target].nunique() > 3:
        model_type = 'regressor'
    df = dataframe.copy(deep=True)
    df = confine_infinite(df)
    df = filter_constant(df)
    df = filter_duplication(df, target)
    df = filter_hi_corr(df, target)
    
    if model == 'linear':
        if model_type == 'classifier':
            df = filter_info_ratio(df, target)
            df = filter_logistic_regression(df, target)
        else:
            df = filter_linear_regression(df, target)
    elif model == 'tree':
            df = filter_forest_importance(df, target,model_type)

    return df


@numba.jit
def confine_infinite(dataframe):
    """
    Replace infinite value with an available one.
    'boundary' policy is applied to replace infinite value with boundary
    value. In another word, replace infinite with max value and replace 
    -infinite with min value.
    Parameters
    ----------
    dataframe : pandas.Dataframe
        dataframe to explore importance

    Return
    -------
    dataframe after process
    """
#    number_features = [f_ for f_ in dataframe.columns \
#                       if dataframe[f_].dtype != 'object']
    number_features = dataframe.select_dtypes('number').columns.tolist()
    for f in number_features:
        col = dataframe[f]
        col_inf_n = np.isneginf(col)
        col_inf_p = np.isposinf(col)
        col[col_inf_n]=np.nanmin(col) 
        col[col_inf_p]=np.nanmax(col)
        
        debug('confine_infinite: '+f)
        debug(np.sum(col_inf_n))
        debug(np.sum(col_inf_p))
        debug(np.nanmin(col))
        debug(np.nanmax(col))
    return dataframe
    

def filter_constant(dataframe):
    """
    Filter constant features which contain single value or all value is nan.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
        dataframe to process
        
    Return
    -------
    dataframe after process
    """
    df = dataframe
    const_columns = []
    for c in df.columns:
        if df[c].dtype != 'object':
            if np.nansum(df[c]) == 0 and np.nanvar(df[c]) == 0:
                const_columns.append(c)
        elif df[c].nunique() == 1:
            const_columns.append(c)
            
    df.drop(const_columns, axis = 1, inplace = True)
    df.dropna(axis=1, how='all', inplace = True)
    debug('filter_constant')
    debug(const_columns)
    return df
    

#@numba.jit
def filter_duplication(dataframe, target=''):
    """
    Filter duplicated features, support numeric type feature only.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
        dataframe to process
    target : string        
        Feature name, as predict result.
    
    Return
    -------
    dataframe after process
    """
    df = dataframe
    numeric_feats = df.select_dtypes('number').columns.tolist()
#    numeric_feats = [f for f in df.columns if df[f].dtype!='object']
    if target in numeric_feats:
        numeric_feats.remove(target)
    numeric_feats2 = numeric_feats[:]
    #    df_corr = df[numeric_feats].corr()
    duplicate_feats = []
    for f_1 in numeric_feats:
        numeric_feats2.remove(f_1)        
        if f_1 in duplicate_feats:
            continue
        for f_2 in numeric_feats2:
            diff = df[f_1]-df[f_2]
            if np.nansum(diff)==0 and np.nanvar(diff)==0:
                debug('Duplicate '+f_1+' '+f_2)
                duplicate_feats.append(f_2)
                
    remain_feats = list(set(df.columns).difference(set(duplicate_feats)))
    debug(duplicate_feats)
    debug(len(remain_feats))
    return df[remain_feats]

#@numba.jit
def filter_hi_corr(dataframe, target='', corr_threshold=0.95):
    """
    Filter highly correlated features, support numeric type feature only.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
        dataframe to process
    target : string        
        Feature name, as predict result.
    corr_threshold : float
        Between 0 and 1. Filter features with correlation above the value.
    Return
    -------
    dataframe after process
    """
    df = dataframe
    numeric_feats = df.select_dtypes('number').columns.tolist()
    if target in numeric_feats:
        numeric_feats.remove(target)
    numeric_feats2 = numeric_feats[:]
    
    df_corr = df[numeric_feats].corr()
    hi_corr_features = []
    for f_1 in numeric_feats:        
        numeric_feats2.remove(f_1)
        # avoid chain effect. ex, corr(f1,f2)>corr_threshold,
        # corr(f2,f3)>corr_threshold, but corr(f1,f3)<corr_threshold. 
        # Thus keep f1, f3 and Filter f2
        if f_1 in hi_corr_features:
            continue
        for f_2 in numeric_feats2:
            trace('filter_hi_corr '+f_1+'&'+f_2+' '+str( df_corr.loc[f_1,f_2]))
            if abs(df_corr.loc[f_1,f_2]) >= corr_threshold:
                hi_corr_features.append(f_2)

    df.drop(hi_corr_features, axis=1, inplace=True)
    trace(hi_corr_features)
    return df

#@numba.jit
def filter_info_ratio(dataframe, target, threshold=0.01, retain_num_feature=True):
    """
    Filter features with low information ratio, support category type feature 
    only.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
        dataframe to process
    target : string        
        Feature name, as predict result.
    threshold : float
        Between 0 and 1. Filter features with information above the value.
    retain_num_feature: Boolean
        Keep number type features if True and drop them otherwise.
    
    Return
    -------
    dataframe after process
    """
#    NUMBER2NOMINAL_NUM = 100
    df = dataframe
    features = df.columns.tolist()
    features.remove(target)
    info_ratio_low_features = []
    for feature in features:
        if df[feature].dtype !='object':
            if retain_num_feature == False:
                info_ratio_low_features.append(feature)
            else:
                continue
        else: 
            iv = _calc_iv(df, feature, target)
            trace(feature+" info value:"+str(iv))
            if iv < threshold:
                info_ratio_low_features.append(feature)

    df.drop(info_ratio_low_features, axis=1, inplace=True)
    trace('filter_info_ratio')
    trace('drop features')
    trace(info_ratio_low_features)
    return df

@numba.jit
def _calc_iv(df, feature, target):

    lst = []

    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature, val, df[df[feature] == val].count()[feature], \
                    df[(df[feature] == val) & (df[target] == 1)].count()[feature]])

    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Bad'])
    data = data[data['Bad'] > 0]

    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])
    data['IV'] = np.multiply(data['WoE'], (data['Distribution Good'] - data['Distribution Bad'])).sum()

    data = data.sort_values(by=['Variable', 'Value'], ascending=True)

    return data['IV'].values[0]

def filter_forest_importance(dataframe, target, model_type='classifier', threshold=0.0001):
    """
    Filter features with low weight when estimate with random forest model.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
        dataframe to process
    target : string        
        Feature name, as predict result.
    model_type: 'classifier', 'regressor'
        Model_type specifies forest classifier or forest regressor.
    threshold : float
        Between 0 and 1. Filter features with weight below the value.
    
    Return
    -------
    dataframe after process
    """
    from sklearn import preprocessing
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor
    categorical_feats = dataframe.select_dtypes('object').columns.tolist()

    for col in categorical_feats:
        lb = preprocessing.LabelEncoder()
        lb.fit(list(dataframe[col].values.astype('str')))
        dataframe[col] = lb.transform(list(dataframe[col].values.astype('str')))

    dataframe.fillna(-999, inplace = True)
    
    rf = 0
    if dataframe[target].dtype == 'object':
        rf = RandomForestClassifier(n_estimators=2000, max_depth=6, \
                                min_samples_leaf=10, max_features=0.5, \
                                n_jobs=-1 , random_state=2018, oob_score=True)

    else:
        rf = RandomForestRegressor(n_estimators=2000, max_depth=6, \
                                min_samples_leaf=10, max_features=0.5, \
                                n_jobs=-1 , random_state=2018, oob_score=True)

    rf.fit(dataframe.drop([target],axis=1), dataframe[target])
    features = dataframe.drop([target],axis=1).columns.values

    importance_df = pd.DataFrame()
    importance_df['feature'] = features
    importance_df['importance'] = rf.feature_importances_
    importance_df.sort_values('importance',inplace=True,ascending=False)
    less_important_features = importance_df.loc[importance_df['importance'] < threshold,'feature']
    dataframe.drop(less_important_features, axis = 1, inplace = True)
    score = rf.oob_score_
    trace('filter_forest_importance')
    trace(importance_df)
    trace('category features')
    trace(categorical_feats)
    trace('score')
    trace(score)
    trace('drop features')
    trace(less_important_features)

    return dataframe

def filter_LBGM_importance(dataframe, target, threshold= 1):
    from lightgbm import LGBMClassifier
    from sklearn import preprocessing
    categorical_feats = dataframe.select_dtypes('object').columns.tolist()

    for col in categorical_feats:
        lb = preprocessing.LabelEncoder()
        lb.fit(list(dataframe[col].values.astype('str')))
        dataframe[col] = lb.transform(list(dataframe[col].values.astype('str')))
    
    valid_size = int(dataframe.shape[0]/4)
    valid = dataframe.sample(valid_size)
    train = dataframe.drop(valid.index, axis=0)
    train_x = train.drop([target],axis=1)
    train_y = train[target]
    valid_x = valid.drop([target],axis=1)
    valid_y = valid[target]

    clf = LGBMClassifier(
        nthread=4,
        n_estimators=10000,
        learning_rate=0.02,
        num_leaves=34,
        colsample_bytree=0.95,
        subsample=0.9,
        max_depth=8,
        reg_alpha=0.04,
        reg_lambda=0.07,
        min_split_gain=0.025,
        min_child_weight=40,
#        importance_type='split',
        silent=-1,
        verbose=-1, )

    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
        eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)
#    oof_preds = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
    
    feats = train_x.columns.tolist()                  
    importance_df = pd.DataFrame()
    importance_df["feature"] = feats
    importance_df["importance"] = clf.feature_importances_
    importance_df.sort_values('importance',inplace=True,ascending=False)
    less_important_features = importance_df.loc[importance_df['importance'] < threshold,'feature']
    dataframe.drop(less_important_features, axis = 1, inplace = True)  
    score = clf.score(train_x, train_y)
    trace('filter_LBGM_importance')
    trace(importance_df)
    trace('category features')
    trace(categorical_feats)
    trace('score')
    trace(score)
    trace('drop features')
    trace(less_important_features)
    return dataframe

    
def filter_logistic_regression(dataframe, target, threshold=0.001):
    """
    Filter features with low weight from compution with logistic regression model.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
        dataframe to process
    target : string        
        Feature name, as predict result.
    threshold : float
        Between 0 and 1. Filter features with weight below the value.
    
    Return
    -------
    dataframe after process
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn import preprocessing

    categorical_feats = dataframe.select_dtypes('object').columns.tolist()
    if target in categorical_feats:
        categorical_feats.remove(target)

    for col in categorical_feats:
        lb = preprocessing.LabelEncoder()
        lb.fit(list(dataframe[col].values.astype('str')))
        dataframe[col] = lb.transform(list(dataframe[col].values.astype('str')))

    import fast_fill_na
    for feature in dataframe.columns:
        dataframe, acc = fast_fill_na.fill_nan_mean(dataframe,feature)

    X = dataframe.drop([target],axis=1)    
    Y = dataframe[target]
    column_names = X.columns
    X = preprocessing.scale(X)

#    lr = LogisticRegression(penalty='l1', max_iter=1000, C=lambd)
    lr = LogisticRegressionCV(cv=5, penalty='l1',solver='saga',n_jobs=-1, \
                              max_iter=1000, Cs=[0.03,0.05,0.1,0.3])
    lr.fit(X, Y)
    score = lr.score(X,Y)
    coef_df = pd.DataFrame()
    coef_df["feature"] = column_names
    coef_df["coef"] = lr.coef_.T
    coef_df["abs_coef"] = np.abs(lr.coef_.T)
    coef_df.sort_values('abs_coef',inplace=True,ascending=False)
    less_coef_features = coef_df.loc[coef_df['abs_coef'] < threshold,'feature']
    dataframe.drop(less_coef_features, axis = 1, inplace = True)
    trace('filter_logistic_regression')
    trace('category features')
    trace(categorical_feats)
    trace('coefficence list')
    trace(coef_df)
    trace('score')
    trace(score)
    trace('regularization C_')
    c = lr.C_
    trace(str(c))
    trace('drop features')
    trace(less_coef_features)
    return dataframe

def filter_linear_regression(dataframe, target, threshold=0.001):
    """
    Filter features with low weight when estimate with linear regression model.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
        dataframe to process
    target : string        
        Feature name, as predict result.
    threshold : float
        Between 0 and 1. Filter features with weight below the value.
    
    Return
    -------
    dataframe after process
    """
    from sklearn.linear_model import LassoCV
#    from sklearn.linear_model import MultiTaskLassoCV
    from sklearn import preprocessing

    categorical_feats = dataframe.select_dtypes('object').columns.tolist()
    if target in categorical_feats:
        categorical_feats.remove(target)

    for col in categorical_feats:
        lb = preprocessing.LabelEncoder()
        lb.fit(list(dataframe[col].values.astype('str')))
        dataframe[col] = lb.transform(list(dataframe[col].values.astype('str')))

    import fast_fill_na
    for feature in dataframe.columns:
        dataframe,acc = fast_fill_na.fill_nan_mean(dataframe,feature)

    x_train = dataframe.drop([target],axis=1)    
    y_train = dataframe[target]
    column_names = x_train.columns.tolist()
    x_train = preprocessing.scale(x_train)
    
    lr = LassoCV(cv=5, alphas =[0.01,0.03,0.05,0.1,0.3,0.5,1], n_jobs=-1)
    lr.fit(x_train, y_train)
    coef_df = pd.DataFrame()
    coef_df["feature"] = column_names
    coef_df["coef"] = lr.coef_.T
    coef_df["abs_coef"] = np.abs(lr.coef_.T)
    coef_df.sort_values('abs_coef',inplace=True,ascending=False)
    less_coef_features = coef_df.loc[coef_df['abs_coef'] < threshold,'feature']
    score = lr.score(x_train,y_train)
    dataframe.drop(less_coef_features, axis=1, inplace=True)
    trace('filter_linear_regression')
    trace('category features')
    trace(categorical_feats)
    trace('coefficence list')
    trace(coef_df)
    trace('score')
    trace(score)
    trace('regularization alpha_')
    alpha = lr.alpha_
    trace(str(alpha))
    trace('drop features')
    trace(less_coef_features)
    return dataframe
