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
    with open(outputfilepath+outputfilename+mode+timeline+'.transform', "a+") as text_file:
        print(*arg, file=text_file)

def trace(*arg):
    _log(*arg, mode='trace')

def debug(*arg):
    if _DEBUG == True:
        _log(*arg, mode = 'debug')

def transform_auto(dataframe, target='', key_feature = '', axis='', \
                   mode='tree', fast_mode = True):
    """
    Integrated API to transfrom features with multi methods, like infinite 
    confine, solo transform, bias features, binning and aggregate.
    To handle multi-colinearity, drop first generated column and delete const 
    nan column, while nan_as_category is always true.
    Parameters
    ----------
    dataframe : pandas.Dataframe
    mode : 'tree', 'regression', 'others'
        Generate cross variable with binning, polynomial, solo features when 
        mode is regression. In other case, these cross variable is skipped.
    Return
    -------
    dataframe after process
    feature list which is newly generated
    """
    df = dataframe.copy(deep=True)
    df = confine_infinite(df)
    df = filt_constant_feature(df)
    features_orig = df.columns
    df = generate_bias_features(df, target)
    df = filt_constant_feature(df)
    if mode == 'regression':
        df = generate_on_solo_feature(df, target)
        df = filt_constant_feature(df)
        if target !='':
            df = binning(df, target)
            df = filt_constant_feature(df)
        df = generate_polynomial_feature(df, features=features_orig, target=target)    
        df = filt_constant_feature(df)
    if key_feature != '' and axis == '':
        df = aggregrate_on_key(df, target=target, key_feature=key_feature,\
                               fast_mode=fast_mode)
    elif key_feature == '' and axis != '':
        df = aggregrate_along_axis(df, target=target,\
                                       axis=axis,\
                                       fast_mode=fast_mode)        
    elif key_feature != '' and axis != '':
        aggregrate_on_key_along_axis(df,  key_feature=key_feature, \
                                         axis = axis, \
                                         target=target, interval='equal')
    df = filt_constant_feature(df)
    if outputfilename != '' or outputfilepath != '':
        timeline = time.strftime("%Y_%m_%d", time.localtime()) 
        df.to_csv(outputfilepath+outputfilename+timeline+'.out.csv', index= False)
    return df

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
    debug(df.info(memory_usage='deep'))
    df = df.loc[:,~df.columns.duplicated()]
    debug(df.info(memory_usage='deep'))
    new_columns = [c for c in df.columns if c not in original_columns]
    const_columns = [c for c in new_columns if df[c].dtype!='object' \
                     and np.sum(df[c]) == 0 and np.std(df[c]) == 0]
    df.drop(const_columns, axis = 1, inplace = True)
    new_columns = list(set(new_columns).difference(set(const_columns)))
    return df, new_columns

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
#@numba.jit
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
#            index_list = [_index for _index, _ in data.iterrows()]
            woe_diff = 100
            for index, _row in data.iterrows():
                for index2, _row2 in data.iterrows():
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
#@numba.jit
def _binning_iv_number(df, feature, target, num_ratio_limit = 0.05, num_limit = 6):

    if df[feature].dtype == 'object':
        return

    FEATURE_IV_NAME = feature
    df[FEATURE_IV_NAME] = df[feature].fillna(np.nanmax(df[FEATURE_IV_NAME])*2)
    value_num = min(len(np.unique(df[feature])),100)
    if value_num > num_limit:
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
#@numba.jit
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
#    numeric_feats = [f for f in df.columns if df[f].dtype != 'object' \
#                     and f != target]
    
    numeric_feats = []
    for f in df.columns:
        if df[f].dtype != 'object' and f != target:
            numeric_feats.append(f)

    for f_ in numeric_feats:
        tail = np.nanmin(df[f_])-1
        base = np.less(np.nanmax(df[f_]),tail)
        avg = np.average(df[f_])
        df[f_+'.'+'solo_norm'] = df[f_].apply(lambda x: (x-tail)/base)
        df[f_+'.'+'solo_diff'] = df[f_] - avg
        df[f_+'.'+'solo_log'] = df[f_].apply(lambda x: np.log((x-tail)/base))
        df[f_+'.'+'solo_sqrt'] = df[f_].apply(lambda x: np.sqrt((x-tail)/base))
        df[f_+'.'+'solo_square'] = np.square(df[f_])

    return df

@numba.jit
def generate_bias_features(dataframe, target='', corr_threshold=0.85, significant_alpha=0.5):
    """
    Calculate different values between similar features. 'Similar features' here
    means high correlation and similiar value range.
    
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

#    numeric_feats = [f for f in df.columns if df[f].dtype != 'object' and target != f]
#    numeric_feats2 = [f for f in df.columns if df[f].dtype != 'object' and target != f]
    numeric_feats = []
    for f in df.columns:
        if df[f].dtype != 'object' and target != f:
            numeric_feats.append(f)
    numeric_feats2 = numeric_feats[:]
    
#    df_corr = df[numeric_feats].corr()
#    df_mean = df[numeric_feats].mean(axis = 0)
#    df_std = df[numeric_feats].std(axis = 0)
    df_corr = df[numeric_feats].corr()
    df_mean = np.mean(df[numeric_feats],axis = 0)
    df_std = np.std(df[numeric_feats],axis = 0)
    for f_1 in numeric_feats:
        numeric_feats2.remove(f_1)
        for f_2 in numeric_feats2:
#            if df_corr.loc[f_1,f_2] < CORR_THRESHOLD:
#                continue
            min_std = min(df_std[f_1], df_std[f_2])
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

@numba.jit
def generate_polynomial_feature(dataframe, features=[], target='', one_hot_encode=True):
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
            dataframe to process
    features : string list
            feature names which is involved in polynominal derivation
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
    if len(features)==0:
        features = df.columns
    if one_hot_encode == True:
#        cat_feats = [f for f in features if df[f].dtype == object and f!=target]
        cat_feats = df.select_dtypes('object').columns.tolist()
        cat_feats = list(set(cat_feats).intersection(set(features)))
        if target in cat_feats:
            cat_feats.remove(target)

        df,new_cols = one_hot_encoder(dataframe, True)
        df = pd.concat([df, dataframe[cat_feats]],axis=1)
    # convert number type feature to nominal type feature
#        numeric_feats = [f for f in features if df[f].dtype != object \
#                         and f not in new_cols and f!=target]
        numeric_feats = df.select_dtypes('number').columns.tolist()
        numeric_feats = list(set(numeric_feats).intersection(set(features)))
        if target in numeric_feats:
            numeric_feats.remove(target)
        numeric_feats = list(set(numeric_feats).difference(set(new_cols)))  
            
        if target in cat_feats:
            cat_feats.remove(target)
            
        unique = df[numeric_feats].nunique()
        for f_ in numeric_feats:
            if unique[f_] <= NUMBER2NOMINAL_NUM \
            and unique[f_]/df.shape[0] <= NUMBER2NOMINAL_RATIO:
                df[f_+'_cat'] = df[f_].astype(str)

#    cat_feats = [f for f in features if df[f].dtype == object and f!=target]
#    cat_feats2 = [f for f in features if df[f].dtype == object and f!=target]
    cat_feats = df.select_dtypes('object').columns.tolist()
    cat_feats = list(set(cat_feats).intersection(set(features)))
    if target in cat_feats:
        cat_feats.remove(target)
    cat_feats2 = cat_feats[:]

    for f_1 in cat_feats:
        for f_2 in cat_feats2:
            if f_1!=f_2:
                df[f_1+'_'+f_2] =df[f_1]+'_'+df[f_2]
        cat_feats2.remove(f_1)
    
#    numeric_feats = [f for f in features if df[f].dtype != object]
#    numeric_feats2 = [f for f in features if df[f].dtype != object]
    numeric_feats = df.select_dtypes('number').columns.tolist()
    numeric_feats = list(set(numeric_feats).intersection(set(features)))
    if target in numeric_feats:
        numeric_feats.remove(target)
    numeric_feats2 = numeric_feats[:]      
    for f_1 in numeric_feats:
        for f_2 in numeric_feats2:
            df[f_1+'x'+f_2] = np.multiply(df[f_1],df[f_2])
            if f_1 != f_2:
                df[f_1+'/'+f_2] = np.divide(df[f_1], df[f_2])
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


def aggregrate_on_key(dataframe,  key_feature, aggr_feature=[], target='', fast_mode=False):
    """
    Aggregated as mean/ size/ min/ max/std according to key feature which is
    indexed for multi row.
    Key feature is specified to slice dataframe to multi-partitions, on which 
    aggregation works.
    Dask module are introduced as fast mode to support quick calculate huge
    amount of data.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
            dataframe to process        
    key_feature : string
            name of the feature which slice dataframe to multi partitions
    aggr_feature : string list, option
            feature list which are aggregated, supporting numric type features.
            all number features are aggregated if not specified.
    target : string, option
            target feature name
    fast_mode : boolean, option
            support dask for fast calculation. If true, dask module is adopted.
    
    
    Return
    -------
    dataframe after process.
    """
    if len(aggr_feature) == 0:
        aggr_feature = dataframe.columns.tolist()
    else:
        aggr_feature.append(key_feature)   
    
    df = dataframe[aggr_feature]
    df_enc = _filter_same_features(df, target=target)
    df_enc, col_cat = one_hot_encoder(df_enc, True)
    df_enc = _filter_same_features(df_enc, target=target)

    aggregations = {}
#    numeric_cols = [col for col in df_enc.columns \
#                    if df_enc[col].dtype != 'object' and col != target \
#                    and col != key_feature]
    numeric_cols = df_enc.select_dtypes('number').columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    if key_feature in numeric_cols:
        numeric_cols.remove(key_feature)    

    for col in numeric_cols:
        aggregations[col] = ['min', 'max', 'size','mean','sum','var','std']
#        aggregations[col] = ['min', 'max', 'size','mean','sum','var','skew','kurt']

    df_agg = _group_agg(df_enc, key_feature, aggregations, fast_mode)
    df_agg.columns = pd.Index([e[0] + "_" + e[1].upper() \
                                      for e in df_agg.columns.tolist()])
    df_agg.reset_index(inplace=True)
    df_agg.sort_values(by = key_feature,inplace=True)

    df_agg.fillna(0)
    debug(df_agg)
    return df_agg

def aggregrate_along_axis(dataframe, axis, aggr_feature=[], \
                             target='', interval='equal', segment=10, fast_mode = True):
    """
    It's useful to analyze feature change along some continuous axis,  like 
    time serials.the analyzed data are aggregated as mean/ size/ min/ max/std. 
    The axis feature is divided to several segments with different interval
    type, like equal interval, square root interval or log internal.
    Dask module are introduced as fast mode to support quick calculate huge
    amount of data.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
            dataframe to process        
    axis : string
            name of the feature which is numeric type
    aggr_feature : string list, option
            feature list which are aggregated, supporting numric type features.
            all number features are aggregated if not specified.
    target : string, option
            target feature name
    interval : string 'exp', 'sqrt', 'equal' or float, option
            when string type, it specify interval togather with segment.
            it means transform axis with exponential(log), square root or 
            equal algorithm.
            when float type, it specify a fixed interval, without concern of 
            segment
    segment : int, option
            always large than 1. specify the number of segments when divide 
            axis feature.
    fast_mode : boolean, option
            support dask for fast calculation. If true, dask module is adopted.
    
    
    Return
    -------
    dataframe after process.
    """
    if np.size(aggr_feature) == 0:
        aggr_feature = [col for col in dataframe.columns if col != target]
    else:
        aggr_feature.append(axis)
        
    
    df = dataframe[aggr_feature]
    if str(interval) == 'sqrt':
        head = np.nanmax(df[axis])
        tail = np.nanmin(df[axis])-1
        inter_sqrt = math.sqrt(head-tail)/segment
        df[axis] = df[axis].apply(lambda x: \
          round(np.sqrt(x-tail)/inter_sqrt))

    elif str(interval) == 'exp':
        head = np.nanmax(df[axis])
        tail = np.nanmin(df[axis])-1
        inter_e = math.log(head-tail)/segment
        df[axis] = df[axis].apply(lambda x: \
          0 if np.isnan(x) else round(math.log(x-tail)/inter_e))
    
    elif str(interval) == 'equal':
        head = np.nanmax(df[axis])
        tail = np.nanmin(df[axis])-1
        inter_e = (head-tail)/segment
        df[axis] = df[axis].apply(lambda x: \
          0 if np.isnan(x) else round((x-tail)/inter_e))  
    elif int(interval) > 0:
        df[axis] = round(df[axis]/interval)
    # if aggr_feature.count > 10:
    #     return
    df_enc = df[aggr_feature]
    df_enc = _filter_same_features(df_enc, target=target)
    df_enc, col_cat = one_hot_encoder(df_enc, True)
    df_enc = _filter_same_features(df_enc, target=target)

    aggregations = {}
    numeric_cols = [col for col in df_enc.columns \
                    if df_enc[col].dtype != 'object' and col != axis]

    for col in numeric_cols:
        if col!=axis:
            aggregations[col] = ['min', 'max', 'size','mean','sum','var','std']
#            aggregations[col] = ['min', 'max', 'size','mean','sum','var','skew','kur','std']

    df_agg = _group_agg(df_enc, axis, aggregations, fast_mode)
    df_agg.columns = pd.Index([e[0] + "_" + e[1].upper() \
                                      for e in df_agg.columns.tolist()])
    df_agg.reset_index(inplace=True)
    df_agg.sort_values(by = axis,inplace=True)
    
    df_agg.fillna(0)
    debug(df_agg)
    return df_agg



def aggregrate_on_key_along_axis(dataframe,  key_feature, axis, \
                                      aggr_feature=[], target='', interval=1, \
                                      segment=10, fast_mode=False):
    """
    It's useful to analyze feature change along some continuous axis,  like 
    time serials.the analyzed data are aggregated as mean/ size/ min/ max/std.
    Key feature is specified to slice dataframe to multi-partitions, on which 
    axis and aggregation works.
    The axis feature is divided to several segments with different interval
    type, like equal interval, square root interval or log internal.
    Dask module are introduced as fast mode to support quick calculate huge
    amount of data.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
            dataframe to process
    key_feature : string
            name of the feature which slice dataframe to multi partitions
    axis : string
            name of the feature which is numeric type
    aggr_feature : string list, option
            feature list which are aggregated, supporting numric type features.
            all number features are aggregated if not specified.
    target : string, option
            target feature name
    interval : string 'exp', 'sqrt', 'equal' or float, option
            when string type, it specify interval togather with segment.
            it means transform axis with exponential(log), square root or 
            equal algorithm.
            when float type, it specify a fixed interval, without concern of 
            segment
    segment : int, option
            always large than 1. specify the number of segments when divide 
            axis feature. It works when interval is string type.
    fast_mode : boolean, option
            support dask for fast calculation. If true, dask module is adopted.
    
    
    Return
    -------
    dataframe after process.
    """
    if len(aggr_feature) == 0:
        aggr_feature = [col for col in dataframe.columns]
    else:
        aggr_feature.append(axis)
        aggr_feature.append(key_feature)
        
    
    df = dataframe[aggr_feature]
    if str(interval)=='sqrt':
        head = np.nanmax(df[axis])
        tail = np.nanmin(df[axis])-1
        inter_sqrt = math.sqrt(head-tail)/segment
        df[axis] = df[axis].apply(lambda x: round(np.sqrt(x-tail)/inter_sqrt))

    elif str(interval) == 'exp':
        head = np.nanmax(df[axis])
        tail = np.nanmin(df[axis])-1
        inter_e = math.log(head-tail)/segment
        df[axis] = df[axis].apply(lambda x: round(\
                  math.log(x-tail)/inter_e))
        
    elif str(interval) == 'equal':
        head = np.nanmax(df[axis])
        tail = np.nanmin(df[axis])-1
        inter_e = (head-tail)/segment
        df[axis] = df[axis].apply(lambda x: \
          0 if np.isnan(x) else round((x-tail)/inter_e)) 

    elif int(interval) > 0:
        df[axis] = round(df[axis]/interval)

    df_enc = df[aggr_feature]
    df_enc = _filter_same_features(df_enc, target=target)
    df_enc, col_cat = one_hot_encoder(df_enc, True)
    df_enc = _filter_same_features(df_enc, target=target)

    aggregations = {}
    numeric_cols = [col for col in df_enc.columns \
                    if df_enc[col].dtype != 'object' 
                    and col != axis and col != target]

    for col in numeric_cols:
        if col!=axis and col!=key_feature:
            aggregations[col] = ['min', 'max', 'size','mean','sum','var','std']
#            aggregations[col] = ['min', 'max', 'size','mean','sum','var','skew','kurt']

    df_agg = pd.DataFrame(columns=['null'])
    keylist = dataframe[key_feature].unique()
    for _key in keylist:
        df_on_key = df_enc.loc[dataframe[key_feature]==_key]
        df_check = df_on_key.drop([key_feature,axis],axis=1).dropna(axis=0,how='all')
        if df_check.shape[0] == 0:
            debug('aggregrate_on_key_along_axis: '+key_feature+'::'+str(_key)+' is all of null')
            continue        

        agg = _group_agg(df_on_key, [key_feature,axis], aggregations, fast_mode)
        agg.columns = pd.Index([e[0] + "_" + e[1].upper() \
                                      for e in agg.columns.tolist()])

        agg.reset_index(inplace=True)
        one_row = pd.DataFrame({key_feature:[_key]})
        for base in df_on_key[axis].unique(): 
            one_row_piece = agg[agg[axis]==base].drop([key_feature, axis],axis=1)
            one_row_piece.reset_index(drop=True, inplace=True)
            one_row_piece.columns = pd.Index([axis+'_'+str(base)\
                        +'_'+e for e in one_row_piece.columns.tolist()])
            one_row = pd.concat([one_row,one_row_piece],axis = 1)

        if df_agg.empty:
            df_agg = one_row
        else:
            df_agg = pd.concat([df_agg,one_row],axis = 0)
    df_agg.fillna(0)
    debug(df_agg)
    return df_agg

@numba.jit
def _group_agg(dataframe, group_base,aggregation_list, fast_mode=False):
    agg = []
    if fast_mode == False:
        aggregation_list
        agg = dataframe.groupby(group_base).agg(aggregation_list)
    else:
        dask_df = dd.from_pandas(dataframe, npartitions=4)
        agg = dask_df.groupby(group_base).agg(aggregation_list).compute()
    return agg

@numba.jit
def filt_constant_feature(dataframe):
    df = dataframe
#    const_columns = [c for c in df if df[c].dtype != 'object' and np.nansum(df[c]) == 0 and np.nanvar(df[c]) == 0]
    const_columns = []
    for c in df.columns:
        if df[c].dtype != 'object' and np.nansum(df[c]) == 0 \
        and np.nanvar(df[c]) == 0:
            const_columns.append(c)
    
    df.drop(const_columns, axis = 1, inplace = True)
    df.dropna(axis=1,how='all', inplace = True)
    return df
    

@numba.jit
def _filter_same_features(dataframe, target=''):
    df = dataframe
    numeric_feats = df.select_dtypes('number').columns.tolist()
#    numeric_feats = [f for f in df.columns if df[f].dtype!='object']
    if target in numeric_feats:
        numeric_feats.remove(target)
    numeric_feats2 = numeric_feats[:]
#    df_corr = df[numeric_feats].corr()
    feats_same = []
    for f_1 in numeric_feats:
        numeric_feats2.remove(f_1)
        for f_2 in numeric_feats2:
            diff = df[f_1]-df[f_2]
            if np.nansum(diff)==0 and np.nanvar(diff)==0:
                feats_same.append(f_2)
                
    feats_remain = list(set(df.columns).difference(set(feats_same)))
    debug(feats_same)
    debug(len(feats_remain))
    return df[feats_remain]