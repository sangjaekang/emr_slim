# -*- coding: utf-8 -*-
import sys, os, re
import numpy as np
import pandas as pd
import time
import argparse
import multiprocessing
import re

os_path = os.path.abspath('./') ; find_path = re.compile('emr_slim')
BASE_PATH = os_path[:find_path.search(os_path).span()[1]]
sys.path.append(BASE_PATH)

from construct_labtest import get_labtest_df, get_labtest_aggregated_df

DEBUG_PRINT=True

## time period among data
START_TIME = 20100101
END_TIME  = 20161231

## data directory
DATA_DIR  = BASE_PATH + '/data/'
PREP_OUTPUT_DIR  = DATA_DIR + 'prep/'
INPUT_DIR = DATA_DIR +'input/'
LABEL_PATIENT_PATH = 'label_patient_df.h5'

def save_patient_mean_min_max(no_range,label_name,aug_target=None,save_dir=None,time_length=6,gap_length=1,target_length=3,offset_min_counts=50,offset_max_counts=100):
    global DEBUG_PRINT, PREP_OUTPUT_DIR, LABEL_PATIENT_PATH
    if save_dir is None:
        save_dir = INPUT_DIR

    for label_value in range(0,3):
        o_path = save_dir +'{}/'.format(label_value)
        if not os.path.isdir(o_path):
            os.makedirs(o_path)

    # syntax checking existence for directory
    PREP_OUTPUT_DIR = check_directory(PREP_OUTPUT_DIR)
    output_path = PREP_OUTPUT_DIR + LABEL_PATIENT_PATH
    label_store = pd.HDFStore(output_path,mode='r')
    label_df = label_store.select('label/{}'.format(label_name))
    label_store.close()
    colist = get_timeseries_column()

    if aug_target is not None:
        #라벨의　비대칭성을　해소하기위해　특정　라벨을　포함한　것을　위주로　no_range를　추려냄
        no_range = set(no_range)&set(label_df[label_df.label == aug_target].no.values)
        
    for idx, no in enumerate(no_range):
        if DEBUG_PRINT and idx %1000 == 0:  print("process({}){} th start".format(os.getpid(),idx))
        sys.stdout.flush()
        mean_df,min_df,max_df = get_labtest_aggregated_df(no)
        label_series = get_patient_timeseries_label(no,label_df)

        if mean_df.count().sum() <= offset_min_counts : continue
        for i in range(0,len(colist) - (time_length+gap_length+target_length)):
            window = mean_df.loc[:,colist[i]:colist[i+time_length-1]]
            counts_in_window = window.count().sum()
            if counts_in_window >= offset_min_counts and counts_in_window <= offset_max_counts:
                target_stime = colist[time_length-1+gap_length+i]
                label_value = check_label(label_series.loc[target_stime:get_add_interval(target_stime,target_length-1)])
                if np.isnan(label_value): continue
                file_path = save_dir +'{}/{}_{}.npy'.format(label_value,no,i)
                avg_mat = window.as_matrix()
                min_mat = min_df.loc[:,colist[i]:colist[i+time_length-1]].as_matrix()
                max_mat = max_df.loc[:,colist[i]:colist[i+time_length-1]].as_matrix()
                np.save(file_path,np.stack((avg_mat,min_mat,max_mat)))


def get_patient_timeseries_label(no,label_df):
    global  PREP_OUTPUT_DIR,  LABEL_PATIENT_PATH
    result_series = pd.Series(index=get_timeseries_column())    
    for _,label_df_date,label_df_label in label_df[label_df.no==no].values:
        result_series[label_df_date]=label_df_label    
    return result_series


def check_directory(directory_path):
    # syntax checking for directory
    if not (directory_path[-1] is '/'):
        directory_path  = directory_path + '/'
    # not exists in directory
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    return directory_path        


def yield_time_column(time,end_t):
    # time-series column을　만드는데　있어서　순차적으로　생성하는　함수
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

def check_time_format(x):
    # timeformat(ex: 20170801)의　형태로　되어있는지　확인하는　함수
    re_time = re.compile('^\d{8}$')
    if re_time.match(str(x)):
        return int(str(x)[2:6])
    else:
        raise ValueError("wrong type time format : {}".format(x))


def get_time_interval(t_1,t_2):
    #t_1 t_2 두 시각의 시간 차 계산
    def _get_time(t):
        year = t//100; month = t%100
        return (year*12+month)
    if t_1<t_2:
        return _get_time(t_2)-_get_time(t_1)
    else :
        return _get_time(t_1)-_get_time(t_2)

def get_add_interval(t,interval):
    #t에다가　interval을　더한　시간
    year = t//100; month = t%100
    times = (year*12+month+interval)
    return (times//12*100+ times%12)

def write_metadata_README(path,label_name,np_stacked,time_length,gap_length,target_length,offset_min_counts,offset_max_counts):
    metadata_README = '''
### parameter Setting
| parameter             | value |
| ---------------------    | ----- |
| label_name            | {}    |
| avg_min_max  stacked  |{}     |
| time_length           | {}    |
| gap_length            | {}    |
| target_length         | {}    |
| offset_min_counts | {}    |
| offset_max_counts | {}    |
| Created date          | {}    |
'''.format(label_name,np_stacked,time_length,gap_length,target_length,offset_min_counts,offset_max_counts,time.ctime())
    
    file_path = path + 'README.md'    
    with open(file_path,'w') as f:
        f.write(metadata_README)

def check_label(x):
    NORMAL_FLAG = False
    for _,y in x.items():
        if y>0:  return int(y)
            # hyper : y==2
            # hypo  : y==1
        if y == 0.0 : NORMAL_FLAG = True
            # normal : y==0
    if NORMAL_FLAG:
        return 0
    else:
        return np.nan

def _set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='save path')
    parser.add_argument('label', help='label_name')
    parser.add_argument('np_stacked', help='numpy stacked or not, 1 : true, 0: false')
    parser.add_argument('set_type', help='train test validation')
    parser.add_argument('chunk_size', help='the number of patients using per one process')
    parser.add_argument('time_length',help='time_length')
    parser.add_argument('gap_length', help='gap_length')
    parser.add_argument('target_length',help='target_length')
    parser.add_argument('offset_min_counts',help='offset_min_counts')
    parser.add_argument('offset_max_counts',help='offset_max_counts')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    #argument
    args = _set_parser()
    label_name = args.label 
    np_stacked = int(args.np_stacked)
    set_type = args.set_type
    chunk_size = int(args.chunk_size)
    time_length = int(args.time_length) 
    gap_length = int(args.gap_length)
    target_length = int(args.target_length)
    offset_min_counts = int(args.offset_min_counts)
    offset_max_counts = int(args.offset_max_counts) 
    
    o_path = check_directory(args.path)
    data_path = o_path + set_type
    data_path = check_directory(data_path)
    write_metadata_README(o_path, label_name,np_stacked,time_length,gap_length,target_length,offset_min_counts,offset_max_counts)
    
    PREP_OUTPUT_DIR = check_directory(PREP_OUTPUT_DIR)
    output_path = PREP_OUTPUT_DIR + LABEL_PATIENT_PATH

    sample_store = pd.HDFStore(output_path,mode='r')
    data_set = sample_store.select('label/{}/{}'.format(label_name,set_type)).no.unique()
    sample_store.close()

    ## data_set generating
    print("Creating pool with 8 workers")
    start_time = time.time()

    pool = multiprocessing.Pool(processes=8)
    print("Invoking apply {}_set".format(set_type))
    for divider in np.array_split(data_set,8):
        if np_stacked == 1:
            pool.apply_async(save_patient_mean_min_max,[divider[:chunk_size],label_name,None,data_path,time_length,
                         gap_length,target_length,offset_min_counts,offset_max_counts])
        else :
            pool.apply_async(save_patient_input,[divider[:chunk_size],label_name,None,data_path,time_length,
                         gap_length,target_length,offset_min_counts,offset_max_counts])
    pool.close()
    pool.join()    

    if set_type == 'train':
    #train 경우에만　augment 함
        pool2 = multiprocessing.Pool(processes=8)
        print("Invoking apply {}_set to augment label 2".format(set_type))
        for divider in np.array_split(data_set,8):
            if np_stacked == 1:
                pool2.apply_async(save_patient_mean_min_max,[divider[chunk_size:],label_name,2,data_path,time_length,
                             gap_length,target_length,offset_min_counts,offset_max_counts])
            else :
                pool2.apply_async(save_patient_input,[divider[chunk_size:],label_name,2,data_path,time_length,
                             gap_length,target_length,offset_min_counts,offset_max_counts])
        pool2.close()
        pool2.join()

    print("{}_set Finished--consumed time : {}".format(set_type, time.time()-start_time))
