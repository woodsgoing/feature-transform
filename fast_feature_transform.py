# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 13:48:02 2018

@author: june
"""

import pandas as pd
import numpy as np
import math
import os
import time
import dask.dataframe as dd

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
    with open(outputfilepath+outputfilename+mode+timeline+'.transform', "a+") as text_file:
        print(*arg, file=text_file)

def trace(*arg):
    _log(*arg, mode='trace')

def debug(*arg):
    if _DEBUG == True:
        _log(*arg, mode = 'debug')


def one_hot_encoder(dataframe, nan_as_category = True):
    """
    Apply one hot encode to nominal features.
    To handle multi-colinearity, drop first generated column and delete const 
    nan column, while nan_as_category is always true.
    Parameters
    ----------
    dataframe : pandas.Dataframe
    
    Return
    -------
    dataframe after process
    feature list which is newly generated
    """
    df = dataframe
    original_columns = list(df.columns)
    df = pd.get_dummies(df, dummy_na= True,drop_first=True)
    new_columns = [c for c in df.columns if c not in original_columns]
    const_columns = [c for c in new_columns if df[c].dtype != 'object' and sum(df[c]) == 0 and np.std(df[c]) == 0]
    df.drop(const_columns, axis = 1, inplace = True)
#    new_columns = [c for c in new_columns if c not in const_columns]
    new_columns = list(set(new_columns).difference(set(const_columns)))
    return df, new_columns

    
def confine_infinite(dataframe):
    """
    Replace infinite value with an available one.
    'boundary' policy is applied to replace infinite value with boundary
    value. In another word, replace infinite with max value and replace 
    -infinite with min value.
    Parameters
    ----------
    dataframe : pandas.Dataframe
    Return
    -------
    dataframe after process
    """
    for f in dataframe.columns:
        col = dataframe[f]
        col_inf_n = np.isneginf(col)
        col_inf_p = np.isposinf(col)
        col[col_inf_n]=min(col) 
        col[col_inf_p]=max(col)
        
        debug('confine_infinite: '+f)
        debug(sum(col_inf_n))
        debug(sum(col_inf_p))
        debug(min(col))
        debug(max(col))
    return
    

def binning(dataframe, target, binning_features=[], method='auto', num_limit=6, 
                    num_ratio_limit=0.05):
    """
    Binning values for smooth, stable and descriptable prediction, especially 
    for regression methods.
    for category type features, the binnings whose ratio is less than 
    min_binning_ratio will be merged. If the binning number is larger than 
    max_binning_num, binning with lowest information value loss.
    for number type features, first split it to 100 categories with same 
    frequency and then binning as category type features. the only difference
    is number binning only merge neighbours.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
                dataframe to process
    target : string
                target feature name
    binning_features : string list
                feature name list, on which binning is processed    
    
    method : 'auto', 'iv' (more method will be added later).
    'auto', same as 'iv', binning featue value referring to information value. 

    num_limit : int, larger than 1
    Specify binning number limit, binning result will contain equal or less 
    binning number
    
    min_binning_ratio: int, between 0 and 1
    Specify mininum size ratio of each binning, which may lead to less binning
    num than max_binning_num. size ratio equals sample number from one binning
    divided by total sample number.
    
    Return
    -------
    dataframe after binning
    """
    if len(binning_features) == 0:
        binning_features = [f_ for f_ in dataframe.columns if f_ != target]
    for feature in binning_features:
        if dataframe[feature].dtype == 'object':
            dataframe = _binning_iv_nominal(dataframe, feature, target, num_ratio_limit, num_limit)
        else:
            dataframe = _binning_iv_number(dataframe, feature, target, num_ratio_limit, num_limit)
    
    return dataframe

    
def _build_iv_feature(df, feature,target):    
    lst = []
    for val in list(df[feature].unique()):
        lst.append([feature,                                                        # Variable
                    val,                                                            # Value
                    df[df[feature] == val].count()[feature],                        # All
                    df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Good (think: Fraud == 0)
                    df[(df[feature] == val) & (df[target] == 1)].count()[feature]]) # Bad (think: Fraud == 1)

    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])
    data = _add_iv_feature(data)
    debug(data.to_string())
    return data
    
# Calculate information value
def _binning_iv_nominal(df, feature, target, num_ratio_limit = 0.05, num_limit = 6):
    if df[feature].dtype != 'object':
        return

    FEATURE_IV_NAME = feature
    df[FEATURE_IV_NAME] = df[feature].fillna("NAN")
    data = _build_iv_feature(df, feature,target)

    merge_dict = {}
    while True:
        start_index = -1
        merge_index = -1
        data.sort_values('Share',inplace=True, ascending = True)
        first_index = data.iloc[[0]].index.values[0]
        if data.loc[first_index,'Share'] < num_ratio_limit:
            #locate merge rows
            start_index = first_index
            woe_diff = 100
            for index, row in data.iterrows():
                if index == start_index:
                    continue
                diff = abs(data.loc[start_index,'WoE'] - row['WoE'])
                if diff<woe_diff:
                    merge_index = index
                    woe_diff = diff
        elif data.shape[0] > num_limit:
            #locate merge rows
            index_list = [_index for _index, _ in data.iterrows()]
            woe_diff = 100
            for index in index_list:
                for index2 in index_list:
                    diff = abs(data.loc[index,'WoE']-data.loc[index2,'WoE'])
                    if index != index2 and diff < woe_diff:
                        start_index = index
                        merge_index = index2                        
        # if satisfy both num and ratio limit, break loop
        else:
            break
        
        
        debug('_binning_iv_nominal, merge '+FEATURE_IV_NAME+': '+ str(data.loc[start_index,'Value'])\
              +' with '+ str(data.loc[merge_index,'Value']))
        if merge_index != -1:
            # record map to merged row
            # create merged row
            # del original row
            merge_name = data.loc[start_index,'Value']+'_'+data.loc[merge_index,'Value']
            merge_dict[data.loc[start_index,'Value']] = merge_name
            merge_dict[data.loc[merge_index,'Value']] = merge_name
            merge_row = [[FEATURE_IV_NAME, 
                         merge_name, 
                         data.loc[start_index,'All']+data.loc[merge_index,'All'],
                         data.loc[start_index,'Good']+data.loc[merge_index,'Good'],
                         data.loc[start_index,'Bad']+data.loc[merge_index,'Bad']                  
                         ]]
            merge_data = pd.DataFrame(merge_row, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])
            merge_data = _add_iv_feature(merge_data)
            data.drop([start_index,merge_index], inplace=True)
            data = pd.concat([data, merge_data], axis = 0)
            data.reset_index(drop=True, inplace=True)
            data = _add_iv_feature(data)
            
    #while end
    debug(data.to_string())
    trace(data.to_string())
    df[feature]=df[feature].apply(lambda x: _loop_replace_with_dict(x, merge_dict))
    return df
    
def _add_iv_feature(df):
    df['Share'] = df['All'] / df['All'].sum()
    df['Bad Rate'] = df['Bad'] / (df['All']+0.0001)
    df['Distribution Good'] = (df['All'] - df['Bad']) / (df['All'].sum() - df['Bad'].sum()+0.0001)
    df['Distribution Bad'] = df['Bad'] / (df['Bad'].sum()+0.0001)
    df['WoE'] = np.log(df['Distribution Good'] / (df['Distribution Bad']+0.000001))
    df = df.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    df['IV'] = df['WoE'] * (df['Distribution Good'] - df['Distribution Bad'])
    return df

def _loop_replace_with_dict(x, diction):
    while True: 
        if x in diction.keys():
            x = diction[x]
        else:
            break
    return x

# binning number type feature with neighbour pooling
def _binning_iv_number(df, feature, target, num_ratio_limit = 0.05, num_limit = 6):

    if df[feature].dtype == 'object':
        return

    
    FEATURE_IV_NAME = feature
    
    df[FEATURE_IV_NAME] = df[feature].fillna(max(df[FEATURE_IV_NAME])*2)
    df[FEATURE_IV_NAME], bins = pd.qcut(df[feature], q = 100, duplicates='drop',retbins=True)
    left_values = [a for a in bins[:-1]]
    df[FEATURE_IV_NAME].cat.rename_categories(left_values, inplace = True)
    df[FEATURE_IV_NAME].astype(float)
    
    data = _build_iv_feature(df, feature,target)

    merge_dict = {}
    while True:
        pre_index = -1
        merge_index = -1
        post_index = -1
        # look for smallest and neighbour to merge
        data_share_ratio = data.sort_values('Share', ascending = True)
        first_index = data_share_ratio.iloc[[0]].index.values[0]
        if data_share_ratio.loc[first_index, 'Share'] < num_ratio_limit:
            merge_index = first_index
            woe_diff1 = 100
            woe_diff2 = 100
            if merge_index > 0:
                pre_index = merge_index-1
                woe_diff1 = abs(data.loc[pre_index,'WoE']-data.loc[merge_index,'WoE'])
            if merge_index < data.shape[0]-1:
                post_index = merge_index+1
                woe_diff2 = abs(data.loc[post_index,'WoE']-data.loc[merge_index,'WoE'])
            if woe_diff2 > woe_diff1:
                #select nearest neighbour and only merge_index and post_index are needed
                post_index = merge_index
                merge_index = pre_index
                pre_index = -1
                
        elif data.shape[0] > num_limit:
            woe_diff = 100
            for index, row in data.iterrows():
                if index == data.shape[0]-1:
                    break
                index2 = index+1
                diff = abs(data.loc[index2,'WoE'] - row['WoE'])
                if diff<woe_diff:
                    merge_index = index
                    post_index = index2
                    woe_diff = diff
        else:
            break

        debug('_binning_iv_number, merge '+FEATURE_IV_NAME+': '+ str(data.loc[merge_index,'Value'])\
              +' with '+ str(data.loc[post_index,'Value']))
        if merge_index != -1:
            # record map to merged row
            # create merged row
            # del original row
            merge_name = data.loc[merge_index,'Value']
            merge_dict[data.loc[post_index,'Value']] = merge_name
            merge_row = [FEATURE_IV_NAME, 
                         merge_name, 
                         data.loc[post_index,'All']+data.loc[merge_index,'All'],
                         data.loc[post_index,'Good']+data.loc[merge_index,'Good'],
                         data.loc[post_index,'Bad']+data.loc[merge_index,'Bad'],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0
                         ]
            data.loc[merge_index] = merge_row
            data.drop([post_index], inplace=True)
            data.reset_index(drop=True, inplace=True)
            data = _add_iv_feature(data)
            
    #while end
    debug(data.to_string())
    trace(data.to_string())
    df[feature]=df[feature].apply(lambda x: _loop_replace_with_dict(x, merge_dict))
    
    df[FEATURE_IV_NAME].astype(str)
    return df


# Useful for regression
# No use for ensemble tree
def generate_on_solo_feature(dataframe, target=''):
    """
    Add normalize / lognormal / sqrt / square to number type features.
    It's useful to deeply explore relation between target and features.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
                dataframe to process
    target : string
                target feature name
    
    Return
    -------
    dataframe after transform
    """
    df = dataframe
    numeric_feats = [f for f in df.columns if df[f].dtype != 'object' \
                     and f != target]
    for f_ in numeric_feats:
        tail = min(df[f_])-1
        base = max(df[f_])-tail
        avg = np.average(df[f_])
        df[f_+'.'+'norm'] = df[f_].apply(lambda x: (x-tail)/base)
        df[f_+'.'+'diff'] = df[f_] - avg
        df[f_+'.'+'log'] = df[f_].apply(lambda x: np.log((x-tail)/base))
        df[f_+'.'+'sqrt'] = df[f_].apply(lambda x: np.sqrt((x-tail)/base))
        df[f_+'.'+'square'] = np.square(df[f_])

    return df

def generate_bias_features(dataframe, target='', corr_threshold=0.85, significant_alpha=0.5):
    """
    Calculate different values between similar features. Similar features have
    high correlation and similiar value range.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
                dataframe to process
    target : string, option
                target feature name
    corr_threshold : float, option
                between 0 and 1. Define pearson correlation value above that 
                the two features are considered as similar
    significant_alpha : float, option
                positive value, always between 0 and 1, meansure distance of the 
                two features different comparing with one feature's std deviation       
    
    Return
    -------
    dataframe after process
    """
    CORR_THRESHOLD = corr_threshold
    SIGNIFICANT_ALPHA = significant_alpha
#    SIGNIFICANT_ALPHA = 2.33
    df = dataframe
    
    numeric_feats = [f for f in df.columns if df[f].dtype != 'object' and target != f]
    numeric_feats2 = [f for f in df.columns if df[f].dtype != 'object' and target != f]
    df_corr = df[numeric_feats].corr()
    df_mean = df[numeric_feats].mean(axis = 0)
    df_std = df[numeric_feats].std(axis = 0)
    for f_1 in numeric_feats:
        numeric_feats2.remove(f_1)
        for f_2 in numeric_feats2:
#            if df_corr.loc[f_1,f_2] < CORR_THRESHOLD:
#                continue
            min_std = df_std[f_1] if df_std[f_1]<=df_std[f_2] else df_std[f_2]
            #TODO expect better way to define similiar features
#            test = abs(df_mean[f_1]-df_mean[f_2]) / (min_std/np.sqrt(df.shape[0]))
            test = abs(df_mean[f_1]-df_mean[f_2]) / min_std
            
            debug('generate_diff_on_accordance_features: '+\
                  f_1+'&'+f_2+' corr: '+str(df_corr.loc[f_1,f_2])+\
                  ' test mean/std '+ str(test))
            
            if df_corr.loc[f_1,f_2] < CORR_THRESHOLD or test > SIGNIFICANT_ALPHA:
                continue
            df['diff.'+f_1+'.'+f_2] = df[f_1]-df[f_2]
        
    return df
    
def generate_polynomial_feature(dataframe, one_hot_encode=True):
    """
    Generate new features with binary polynomial method.
    For category type features, intersection features are generated.
    For numeric type features, multiple values from each feature.
    To create cross features from category type features with numeric type 
    feautes, one_hot_encode can be used to transform category type features 
    to numeric dummy features.
    To create cross features from numeric type feautues with category type 
    feature, some numeric type features are transformed to category type.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
    
    one_hot_encode : Boolean, option
    If True, create dummy features from category features before generate 
    polynomial features. 
    In Addition, transform some number features to category features if 
    it has limited count of number value, defined as NUMBER2NOMINAL_NUM.
    
    Return
    -------
    dataframe after transform
    """
    NUMBER2NOMINAL_NUM = 30
    NUMBER2NOMINAL_RATIO = 1/10
    df = dataframe

    # convert nominal type feature to number type feature
    if one_hot_encode == True:
        cat_feats = [f for f in df.columns if df[f].dtype == object]
        df,new_cols = one_hot_encoder(dataframe, True)
        df = pd.concat([df, dataframe[cat_feats]],axis=1)
    # convert number type feature to nominal type feature 
        numeric_feats = [f for f in df.columns if df[f].dtype != object and f not in new_cols]
        for f_ in numeric_feats:
            if df[f_].nunique() <= NUMBER2NOMINAL_NUM \
            and df[f_].nunique()/df.shape[0] <= NUMBER2NOMINAL_RATIO:
                df[f_+'_cat'] = df[f_].astype(str)
        
                
    cat_feats = [f for f in df.columns if df[f].dtype == object]
    cat_feats2 = [f for f in df.columns if df[f].dtype == object]
    for f_1 in cat_feats:
        for f_2 in cat_feats2:
            if f_1!=f_2:
                df[f_1+'_'+f_2] = df[f_1]+'_'+df[f_2]
        cat_feats2.remove(f_1)
    
    numeric_feats = [f for f in df.columns if df[f].dtype != object]
    numeric_feats2 = [f for f in df.columns if df[f].dtype != object]
    for f_1 in numeric_feats:
        for f_2 in numeric_feats2:
            df[f_1+'.'+f_2] = df[f_1]*df[f_2]
        numeric_feats2.remove(f_1)

    return df

#TODO consider more, which depends on biz logic deeply
def generate_cross_feature(dataframe, relation_map):
    """
    Generate new features with relation map.
    Relation map is devised to describe feature relation ship.
    It's based on directed graph. connection specified relation and node 
    specify feature.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
    
    relation_map : internal defined data type
    
    Return
    -------
    dataframe after transform
    """
    
    return


def aggregrate_with_baseline(dataframe, baseline_feature, aggr_feature=[], \
                             target='', interval='equal', segment=10, fast_mode = True):
    """
    It's useful to analyze feature change along some continuous baseline,  like 
    time serials.the analyzed data are aggregated as mean/ size/ min/ max/std. 
    The baseline feature is divided to several segments with different interval
    type, like equal interval, square root interval or log internal.
    Dask module are introduced as fast mode to support quick calculate huge
    amount of data.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
            dataframe to process        
    baseline_feature : string
            name of the feature which is numeric type
    aggr_feature : string list, option
            feature list which are aggregated, supporting numric type features.
            all number features are aggregated if not specified.
    target : string, option
            target feature name
    interval : string 'exp', 'sqrt', 'equal' or float, option
            when string type, it specify interval togather with segment.
            it means transform baseline with exponential(log), square root or 
            equal algorithm.
            when float type, it specify a fixed interval, without concern of 
            segment
    segment : int, option
            always large than 1. specify the number of segments when divide 
            baseline feature.
    fast_mode : boolean, option
            support dask for fast calculation. If true, dask module is adopted.
    
    
    Return
    -------
    dataframe after process.
    """
    if np.size(aggr_feature) == 0:
        aggr_feature = [col for col in dataframe.columns if col != target]
    else:
        aggr_feature.append(baseline_feature)
        
    
    df = dataframe[aggr_feature]
    if interval=='sqrt':
        head = max(df[baseline_feature])
        tail = min(df[baseline_feature])-1
        inter_sqrt = math.sqrt(head-tail)/segment
        df[baseline_feature] = df[baseline_feature].apply(lambda x: \
          round(np.sqrt(x-tail)/inter_sqrt))

    elif interval == 'exp':
        head = max(df[baseline_feature])
        tail = min(df[baseline_feature])-1
        inter_e = math.log(head-tail)/segment
        df[baseline_feature] = df[baseline_feature].apply(lambda x: \
          0 if np.isnan(x) else round(math.log(x-tail)/inter_e))
    
    elif interval == 'equal':
        head = max(df[baseline_feature])
        tail = min(df[baseline_feature])-1
        inter_e = (head-tail)/segment
        df[baseline_feature] = df[baseline_feature].apply(lambda x: \
          0 if np.isnan(x) else round((x-tail)/inter_e))  
    elif interval > 0:
        df[baseline_feature] = round(df[baseline_feature]/interval)
    # if aggr_feature.count > 10:
    #     return
    df_enc = df[aggr_feature]
    df_enc, col_cat = one_hot_encoder(df_enc, True)

    aggregations = {}
    numeric_cols = [col for col in df_enc.columns \
                    if df_enc[col].dtype != 'object' and col != baseline_feature]

    for col in numeric_cols:
        if col!=baseline_feature:
            aggregations[col] = ['min', 'max', 'size','mean','sum','var','std']
#            aggregations[col] = ['min', 'max', 'size','mean','sum','var','skew','kur','std']

    df_agg = _group_agg(df_enc, baseline_feature, aggregations, fast_mode)
    df_agg.columns = pd.Index([e[0] + "_" + e[1].upper() \
                                      for e in df_agg.columns.tolist()])
    df_agg.reset_index(inplace=True)
    df_agg.sort_values(by = baseline_feature,inplace=True)
    
    debug(df_agg)
    return df_agg


def aggregrate_on_key_with_baseline(dataframe,  key_feature, baseline_feature, \
                                      aggr_feature=[], target='', interval=1, \
                                      segment=10, fast_mode=False):
    """
    It's useful to analyze feature change along some continuous baseline,  like 
    time serials.the analyzed data are aggregated as mean/ size/ min/ max/std.
    Key feature is specified to slice dataframe to multi-partitions, on which 
    baseline and aggregation works.
    The baseline feature is divided to several segments with different interval
    type, like equal interval, square root interval or log internal.
    Dask module are introduced as fast mode to support quick calculate huge
    amount of data.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
            dataframe to process        
    baseline_feature : string
            name of the feature which is numeric type
    aggr_feature : string list, option
            feature list which are aggregated, supporting numric type features.
            all number features are aggregated if not specified.
    target : string, option
            target feature name
    interval : string 'exp', 'sqrt', 'equal' or float, option
            when string type, it specify interval togather with segment.
            it means transform baseline with exponential(log), square root or 
            equal algorithm.
            when float type, it specify a fixed interval, without concern of 
            segment
    segment : int, option
            always large than 1. specify the number of segments when divide 
            baseline feature. It works when interval is string type.
    fast_mode : boolean, option
            support dask for fast calculation. If true, dask module is adopted.
    
    
    Return
    -------
    dataframe after process.
    """
    if len(aggr_feature) == 0:
        aggr_feature = [col for col in dataframe.columns]
    else:
        aggr_feature.append(baseline_feature)
        aggr_feature.append(key_feature)
        
    
    df = dataframe[aggr_feature]
    if interval=='sqrt':
        head = max(df[baseline_feature])
        tail = min(df[baseline_feature])-1
        inter_sqrt = math.sqrt(head-tail)/segment
        df[baseline_feature] = df[baseline_feature].apply(lambda x: round(np.sqrt(x-tail)/inter_sqrt))

    elif interval == 'exp':
        head = max(df[baseline_feature])
        tail = min(df[baseline_feature])-1
        inter_e = math.log(head-tail)/segment
        df[baseline_feature] = df[baseline_feature].apply(lambda x: int(\
                  math.log(x-tail)/inter_e))
        
    elif interval > 0:
        df[baseline_feature] = round(df[baseline_feature]/interval)

    df_enc = df[aggr_feature]
    df_enc, col_cat = one_hot_encoder(df_enc, True)

    aggregations = {}
    numeric_cols = [col for col in df_enc.columns \
                    if df_enc[col].dtype != 'object' 
                    and col != baseline_feature and col != target]

    for col in numeric_cols:
        if col!=baseline_feature and col!=key_feature:
            aggregations[col] = ['min', 'max', 'size','mean','sum','var','std']
#            aggregations[col] = ['min', 'max', 'size','mean','sum','var','skew','kurt']

    df_agg = pd.DataFrame(columns=['null'])
    keylist = dataframe[key_feature].unique()
    for _key in keylist:
        df_on_key = df_enc.loc[dataframe[key_feature]==_key]
        df_check = df_on_key.drop([key_feature,baseline_feature],axis=1).dropna(axis=0,how='all')
        if df_check.shape[0] == 0:
            debug('aggregrate_on_key_with_baseline: '+key_feature+'::'+str(_key)+' is all of null')
            continue        

        agg = _group_agg(df_on_key, [key_feature,baseline_feature], aggregations, fast_mode)
        agg.columns = pd.Index([e[0] + "_" + e[1].upper() \
                                      for e in agg.columns.tolist()])

        agg.reset_index(inplace=True)
        one_row = pd.DataFrame({key_feature:[_key]})
        for base in df_on_key[baseline_feature].unique(): 
            one_row_piece = agg[agg[baseline_feature]==base].drop([key_feature, baseline_feature],axis=1)
            one_row_piece.reset_index(drop=True, inplace=True)
            one_row_piece.columns = pd.Index([baseline_feature+'_'+str(base)\
                        +'_'+e for e in one_row_piece.columns.tolist()])
            one_row = pd.concat([one_row,one_row_piece],axis = 1)

        if df_agg.empty:
            df_agg = one_row
        else:
            df_agg = pd.concat([df_agg,one_row],axis = 0)
    df_agg.fillna(0)
    debug(df_agg)
    return df_agg


def _group_agg(dataframe, group_base,aggregation_list, fast_mode=False):
    agg = []
    if fast_mode == False:
        aggregation_list
        agg = dataframe.groupby(group_base).agg(aggregation_list)
    else:
        dask_df = dd.from_pandas(dataframe, npartitions=4)
        agg = dask_df.groupby(group_base).agg(aggregation_list).compute()
    return agg
