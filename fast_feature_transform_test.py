import pandas as pd
import fast_feature_transform as transform
import matplotlib.pyplot as plt # for plotting
import seaborn as sns 
import gc
import time
from contextlib import contextmanager
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


in_file_path = 'E:/python/credit/input/'
out_file_path = 'E:/python/credit/output/'

def test_transform_general():
    transform.setEnvInfo(out_file_path,'application_train.log')
    table = pd.read_csv(in_file_path+'application_train_sample.csv')
    table.reset_index(drop=True,inplace=True)
    test_polynomial_feature(table)
    test_process_binning(table)
    test_generate_bias_features(table)

def test_aggr_general():
    transform.setEnvInfo(out_file_path,'installments_payments.log')
    table = pd.read_csv(in_file_path+'installments_payments_sample.csv')
        
#    with timer("test_aggregrate_along_baseline"):
#        test_aggregrate_along_baseline(table)
    
    with timer("test_aggregrate_on_key_along_baseline"):
        test_aggregrate_on_key_along_baseline(table)
#    with timer("test_generate_bias_features2"):
#    test_generate_bias_features2(table)
        
#    with timer("test_aggregrate_along_baseline_fast"):
#        test_aggregrate_along_baseline_fast(table)
    
    with timer("test_aggregrate_on_key_along_baseline_fast"):
        test_aggregrate_on_key_along_baseline_fast(table)

def test_transform_auto():
    with timer("transform aggregrate_on_key"):
        transform.setEnvInfo(out_file_path,'previous_application.log')
        table = pd.read_csv(in_file_path+'previous_application_sample.csv')
#        test_aggregrate_on_key(table)  
        df = transform.transform_auto(table, key_feature='SK_ID_CURR')
        df.to_csv(out_file_path+'previous_application_sample.1205.output.csv')
        del df
        gc.collect()
    with timer("transform aggregrate_along_baseline"):
        transform.setEnvInfo(out_file_path,'installments_payments.log')
        table = pd.read_csv(in_file_path+'installments_payments_sample.csv')
        df = transform.transform_auto(table, baseline='DAYS_ENTRY_PAYMENT')
        df.to_csv(out_file_path+'installments_payments_sample.1205.output.csv')
        del df
        gc.collect()
#        test_aggregrate_along_baseline(table)
        
    with timer("transform aggregrate_on_key_along_baseline"):
        transform.setEnvInfo(out_file_path,'installments_payments.log')
        table = pd.read_csv(in_file_path+'installments_payments_sample.csv')
        df = transform.transform_auto(table,  key_feature = 'SK_ID_CURR', base_feature='DAYS_ENTRY_PAYMENT', \
                                      aggr_feature=['AMT_PAYMENT'])
#        test_aggregrate_on_key_along_baseline(table)
        df.to_csv(out_file_path+'installments_payments_sample2.1205.output.csv')
        del df
        gc.collect()
        


test_transform_auto()    

def test_polynomial_feature(table):
    df = table[['TARGET','CODE_GENDER','FLAG_OWN_CAR','AMT_CREDIT','AMT_ANNUITY','OWN_CAR_AGE']]
    df = transform.generate_polynomial_feature(df,True)
    print(df.columns.tolist())

def test_process_binning(table):
    print(table['AMT_CREDIT'].unique())
    print(table['EXT_SOURCE_1'].unique())
    print(table['OCCUPATION_TYPE'].unique())
    print(table['CODE_GENDER'].unique())
    
    df_ret=transform.binning(table, binning_features=['AMT_CREDIT','EXT_SOURCE_1','OCCUPATION_TYPE','CODE_GENDER'], target='TARGET', num_ratio_limit = 0.05, num_limit = 6)
    print(df_ret['AMT_CREDIT'].unique())
    print(df_ret['EXT_SOURCE_1'].unique())
    print(df_ret['OCCUPATION_TYPE'].unique())
    print(df_ret['CODE_GENDER'].unique())


def test_aggregrate_on_key(table):
    out_file_name = 'previous_application.output.1205'
    dataframe = transform.aggregrate_on_key(table, base_feature='SK_ID_CURR')  
    dataframe.to_csv(out_file_path+out_file_name)

def test_aggregrate_along_baseline(table):
    out_file_name = 'installments_payments.output.1205'
    dataframe_agg = transform.aggregrate_along_baseline(table, base_feature='DAYS_ENTRY_PAYMENT',aggr_feature=['AMT_PAYMENT'], interval = 'exp', segment=10, fast_mode=False)  
    
    for col in dataframe_agg.columns:
        plt.figure(figsize=(12,6))
        plt.title("SQRT Time series of "+ 'SK_ID_CURR'+'_2'+col)
        sns.lineplot(x='DAYS_ENTRY_PAYMENT', y=col, data=dataframe_agg)
        plt.savefig(out_file_path+out_file_name.replace('/','_')+'.'+col.replace('/','_')+'.line.png')

    dataframe_agg = transform.aggregrate_along_baseline(table, base_feature='DAYS_ENTRY_PAYMENT',aggr_feature=['AMT_PAYMENT'], interval = 365, fast_mode=False)  
    dataframe_agg.to_csv(out_file_path+out_file_name)

    for col in dataframe_agg.columns:
        plt.figure(figsize=(12,6))
        plt.title("Equal time series of "+ 'SK_ID_CURR'+'_'+col)
        sns.lineplot(x='DAYS_ENTRY_PAYMENT', y=col, data=dataframe_agg)
        plt.savefig(out_file_path+out_file_name.replace('/','_')+'.'+col.replace('/','_')+'.line.png')

def test_aggregrate_on_key_along_baseline(table):
    out_file_name = 'installments_payments2.output.1205'
    df=transform.aggregrate_on_key_along_baseline(table,  key_feature = 'SK_ID_CURR', base_feature='DAYS_ENTRY_PAYMENT', \
                                      aggr_feature=['AMT_PAYMENT'], interval='sqrt', segment=3, fast_mode=False)
    df.to_csv(out_file_path+out_file_name)


def test_aggregrate_along_baseline_fast(table):
    out_file_name = 'installments_payments'
    dataframe_agg = transform.aggregrate_along_baseline(table, base_feature='DAYS_ENTRY_PAYMENT',aggr_feature=['AMT_PAYMENT'], interval = 'exp', segment=10,fast_mode=True)  
    
    for col in dataframe_agg.columns:
        plt.figure(figsize=(12,6))
        plt.title("SQRT Time series of "+ 'SK_ID_CURR'+'_2'+col)
        sns.lineplot(x='DAYS_ENTRY_PAYMENT', y=col, data=dataframe_agg)
        plt.savefig(out_file_path+out_file_name.replace('/','_')+'.'+col.replace('/','_')+'.line.png')

    dataframe_agg = transform.aggregrate_along_baseline(table, base_feature='DAYS_ENTRY_PAYMENT',aggr_feature=['AMT_PAYMENT'], interval = 365,fast_mode=True)  
    
    for col in dataframe_agg.columns:
        plt.figure(figsize=(12,6))
        plt.title("Equal time series of "+ 'SK_ID_CURR'+'_'+col)
        sns.lineplot(x='DAYS_ENTRY_PAYMENT', y=col, data=dataframe_agg)
        plt.savefig(out_file_path+out_file_name.replace('/','_')+'.'+col.replace('/','_')+'.line.png')

def test_aggregrate_on_key_along_baseline_fast(table):
    out_file_name = 'installments_payments.csv'
    df=transform.aggregrate_on_key_along_baseline(table,  key_feature = 'SK_ID_CURR', base_feature='DAYS_ENTRY_PAYMENT', \
                                      aggr_feature=['AMT_PAYMENT'], interval='sqrt', segment=3,fast_mode=True)
    df.to_csv(out_file_path+out_file_name)


def test_generate_bias_features(table):
    df = transform.generate_bias_features(table[['AMT_CREDIT','AMT_GOODS_PRICE','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY','REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON']])
    print(df.columns)

def test_generate_bias_features2(table):
    df = transform.generate_bias_features(table[['DAYS_INSTALMENT','DAYS_ENTRY_PAYMENT','AMT_INSTALMENT','AMT_PAYMENT']])
    print(df.columns)    
    
#test_transform_general()
#test_aggr_general()
test_transform()