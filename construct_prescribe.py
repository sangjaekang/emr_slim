# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import sys
import os
import argparse
import time

DEBUG_PRINT=True

## time period among data
START_TIME = 20100101
END_TIME  = 20161231

os_path = os.path.abspath('./')
find_path = re.compile('emr_slim')
BASE_PATH = os_path[:find_path.search(os_path).span()[1]]

## data directory
DATA_DIR  = BASE_PATH + '/data/'
PREP_OUTPUT_DIR  = DATA_DIR + 'prep/'
PRESCRIBE_OUTPUT_PATH = 'prescribe_df.h5'
prescribe_output_path = PREP_OUTPUT_DIR + PRESCRIBE_OUTPUT_PATH

def check_time_format(x):
    # timeformat(ex: 20170801)의　형태로　되어있는지　확인하는　함수
    re_time = re.compile('^\d{8}$')
    if re_time.match(str(x)):
        return int(str(x)[2:6])
    else:
        raise ValueError("wrong type time format : {}".format(x))

def yield_time_column(time,end_t):
    # time-series column을　만드는데　있어서　순차적으로　생성하는　함수X
    # 1210 1211 1212 1301 1302 이런식으로　생성
    while time <= end_t:
        yield time
        year = time // 100 ; month = time % 100
        if month >= 12 :
            year = year + 1
            month = 1
        else : 
            month = month + 1
        time = (year*100 + month) 
        
def get_timeseries_column():
    global START_TIME, END_TIME
    start_time = check_time_format(START_TIME)
    end_time  = check_time_format(END_TIME)

    col_list = []
    for time in yield_time_column(start_time,end_time):
        col_list.append(time)

    return col_list

def set_prescribe_row():
    '''
    약품코드를row_index_name으로나열
    OFFSET_PRESICRIBE_COUNTS 기준에　따라서，drop 할　row을　결정
    drop하고　남은　row를　metadata/usecol에　저장
    '''
    global prescribe_output_path
    prescribe_store = pd.HDFStore(prescribe_output_path,mode='r')
    try:
        if not '/data' in prescribe_store.keys():
            raise ValueError("There is no data in prescribe data")
        pres_df = prescribe_store.select('data')
        value_counts_diag = pres_df['mapping_code'].value_counts()
        total_pres =  pres_df.shape[0]
        min_value_counts = int(total_pres * 0.001)
        use_index_df = value_counts_diag[value_counts_diag>min_value_counts].sort_index().reset_index()
        use_index_df.columns = ['col','value_counts']
    finally:
        prescribe_store.close()
    use_index_df.to_hdf(prescribe_output_path,'metadata/usecol',format='table',data_columns=True,mode='a')
    del use_index_df

def get_index_name_map():
    '''
    mapping_table에서　row_index에 사용될　name 가져오는　함수
    '''
    global prescribe_output_path
    prescribe_store = pd.HDFStore(prescribe_output_path,mode='r')
    class_map_df=prescribe_store.select('metadata/mapping_table',columns=['ingd_name','mapping_code']).drop_duplicates()
    return class_map_df.set_index('mapping_code').to_dict()['ingd_name']

def get_prescribe_df(no):
    '''
    환자번호를　넣으면　column은　KCDcode, row는　time-serial의　형태인　dataframe이　나오는　함수
    '''
    global prescribe_output_path
    prescribe_store = pd.HDFStore(prescribe_output_path,mode='r')
    if not '/metadata/usecol' in prescribe_store.keys():
        set_prescribe_row()
        prescribe_store = pd.HDFStore(prescribe_output_path,mode='r')
    try:
        col_list = get_timeseries_column()
        # create empty dataframe
        use_prescribe_values = prescribe_store.select('metadata/usecol').col.values
        result_df = pd.DataFrame(columns=col_list,index=use_prescribe_values)
        # target patient dataframe
        target_df = prescribe_store.select('data',where='no=={}'.format(no),columns=['no','mapping_code','date','count'])
    finally:
        prescribe_store.close()
    
    for value in target_df.values:
        _ , _medi_code, _date, _times = value
        if _medi_code in use_prescribe_values:
            start_index = result_df.columns.get_loc(_date)
            end_index = start_index + _times + 1
            result_df.loc[_medi_code].iloc[start_index:end_index] = 1
    
    index_name_dict = get_index_name_map()
    result_df.index = result_df.index.map(index_name_dict.get)  
    result_df.fillna(0,inplace=True)
    del target_df
    return result_df