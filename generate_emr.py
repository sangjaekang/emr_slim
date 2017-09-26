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
from config import *

def save_patient_mean_min_max(no_range,label_name,aug_target=None,save_dir=None,time_length=12,gap_length=1,target_length=3,offset_min_counts=100,offset_max_counts=2000):
    global DEBUG_PRINT, PREP_OUTPUT_DIR, LABEL_PATIENT_PATH
    if save_dir is None:
        save_dir = INPUT_DIR

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
                
                o_path = save_dir +'{}/'.format(label_value)
                if not os.path.isdir(o_path):
                    os.makedirs(o_path)

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


def write_metadata_README(path,label_name,time_length,gap_length,target_length,offset_min_counts,offset_max_counts):
    metadata_README = '''
### parameter Setting
| parameter             | value |
| ---------------------    | ----- |
| label_name            | {}    |
| time_length           | {}    |
| gap_length            | {}    |
| target_length         | {}    |
| offset_min_counts | {}    |
| offset_max_counts | {}    |
| Created date          | {}    |
'''.format(label_name,time_length,gap_length,target_length,offset_min_counts,offset_max_counts,time.ctime())
    
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
    write_metadata_README(o_path, label_name,time_length,gap_length,target_length,offset_min_counts,offset_max_counts)
    
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
        pool.apply_async(save_patient_mean_min_max,[divider[:chunk_size],label_name,None,data_path,time_length,
                         gap_length,target_length,offset_min_counts,offset_max_counts])
    pool.close()
    pool.join()    

    # if set_type == 'train':
    # #train 경우에만　augment 함
    #     pool2 = multiprocessing.Pool(processes=8)
    #     print("Invoking apply {}_set to augment label 2".format(set_type))
    #     for divider in np.array_split(data_set,8):
    #         pool2.apply_async(save_patient_mean_min_max,[divider[chunk_size:],label_name,2,data_path,time_length,
    #                          gap_length,target_length,offset_min_counts,offset_max_counts])
    #     pool2.close()
    #     pool2.join()

    print("{}_set Finished--consumed time : {}".format(set_type, time.time()-start_time))
