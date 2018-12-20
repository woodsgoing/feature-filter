import pandas as pd
import fast_feature_filter as filt
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

def test_filter_general():
    filt.setEnvInfo(out_file_path,'application_train.log')
    table = pd.read_csv(in_file_path+'application_train_sample.csv')
    table.reset_index(drop=True,inplace=True)
#    test_filter_infomation_ratio(table)
#    test_filter_LBGM_importance(table)
#    test_filter_forest_importance(table)
    test_filter_logistic_regression(table.copy(deep=True))
#    test_filter_hi_corr(table)
    test_filter_linear_regression(table)
#    test_filter_auto(table)


def test_filter_auto(table):
    
    with timer("filt application_train regression"):
        df = table.copy(deep=True)
        df = filt.filter_auto(df, target='TARGET', model='linear')
        df.to_csv(out_file_path+'application_train_sample.regression.1219.output.csv')
        del df
        gc.collect()
    
    with timer("filt application_train forest"):
        df = table.copy(deep=True)
        df = filt.filter_auto(df, target='TARGET',mode='tree')
        df.to_csv(out_file_path+'application_train_sample.RF.1219.output.csv')
        del df
        gc.collect()
        
    with timer("filt application_train regression"):
        df = table.copy(deep=True)
        df = filt.filter_auto(df, target='AMT_CREDIT',mode='linear')
        df.to_csv(out_file_path+'application_train_sample.1219.output.csv')
        del df
        gc.collect()


def test_filter_hi_corr(table):
    filt.filter_hi_corr(table,'TARGET',0.9)
    
def test_filter_infomation_ratio(table):
    filt.filter_info_ratio(table,'TARGET')
  
def test_filter_LBGM_importance(table):
    filt.filter_LBGM_importance(table,'TARGET')

def test_filter_forest_importance(table):
    filt.filter_forest_importance(table,'TARGET')

def test_filter_logistic_regression(table):
    filt.filter_logistic_regression(table, 'TARGET')

def test_filter_linear_regression(table):
    filt.filter_linear_regression(table, 'AMT_CREDIT')

test_filter_general()
#test_filter_auto()  