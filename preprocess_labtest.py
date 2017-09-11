# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import sys
import os
import argparse
import time

DELIM = ','
LAB_COL_NAME = ['no','lab_code','date','result']
USE_LAB_COL_NAME = ['no','date','result']
DEBUG_PRINT = True

## data directory
os_path = os.path.abspath('./')
find_path = re.compile('emr_slim')
BASE_PATH = os_path[:find_path.search(os_path).span()[1]]

DATA_DIR  = BASE_PATH + '/data/'
PREP_OUTPUT_DIR =  DATA_DIR + 'prep/'
LABTEST_OUTPUT_PATH = 'labtest_df.h5'


def divide_per_test(lab_test_path):
    #labtest 별로　나누어서，　HDFStore에　저장하는　함수
    global  PREP_OUTPUT_DIR, LABTEST_OUTPUT_PATH, DELIM, LAB_COL_NAME, USE_LAB_COL_NAME
    
    PREP_OUTPUT_DIR = check_directory(PREP_OUTPUT_DIR)
    output_path = PREP_OUTPUT_DIR + LABTEST_OUTPUT_PATH

    # remove temp file in output_path
    if os.path.isfile(output_path):
        os.remove(output_path)

    lab_store = pd.HDFStore(output_path,mode='a')
    labtest_df = pd.read_csv(lab_test_path, delimiter=DELIM, header=None,
                                                names=LAB_COL_NAME ,error_bad_lines=False)
    
    if DEBUG_PRINT:
        print('labtest_df is stored in RAM memory successfully')
    
    try:
        for lab_name in labtest_df.lab_code.unique():
            temp_save_df= labtest_df[labtest_df.lab_code.isin([lab_name])]
            save_name = 'data/{}'.format(lab_name)
            
            lab_store.append(key=save_name, value=temp_save_df[USE_LAB_COL_NAME],data_columns=True)

            if DEBUG_PRINT:
                print('{} completed'.format(lab_name))
    finally:
        lab_store.close()
        
    del labtest_df


def set_mapping_table():
    '''
    labtest의 mapping table을　생성하는　함수
    평균/ 최솟값/최댓값으로　구성
    이를　hdf5파일의　metadata에　저장
    '''
    global DELIM, LAB_COL_NAME, PREP_OUTPUT_DIR, LABTEST_OUTPUT_PATH
    
    check_directory(PREP_OUTPUT_DIR)
    
    output_path = PREP_OUTPUT_DIR + LABTEST_OUTPUT_PATH
    lab_store = pd.HDFStore(output_path,mode='r')
    result_df = pd.DataFrame(columns=['labtest','AVG','MIN','MAX'])
    lab_node = lab_store.get_node('data')
    
    try:
        for lab_name in lab_node._v_children.keys():
            per_lab_df = lab_store.select('data/'+lab_name,columns=['no','result'])
            # 1. 숫자로　치환하기
            per_lab_df.result = per_lab_df.result.map(change_number)
            # 2. 이상 값 처리 시 대응되는 값
            r_avg   = revise_avg(per_lab_df.result)
            r_min  = revise_min(per_lab_df.result)
            r_max = revise_max(per_lab_df.result)
            # 3. save
            result_df = result_df.append({'labtest':lab_name,'AVG':r_avg,'MIN':r_min,'MAX':r_max}, ignore_index=True)
            
            if DEBUG_PRINT:
                print("write {} completed".format(lab_name))
    finally:
        lab_store.close()

    result_df.to_hdf(output_path,'metadata/mapping_table',format='table',data_columns=True,mode='a')
    del result_df


def set_labtest_count():
    '''
    labtest의 코드별 총 갯수를 저장해놓은 것
    col : labtest code
    value_counts : 총　횟수 
    '''
    global PREP_OUTPUT_DIR, LABTEST_OUTPUT_PATH
    PREP_OUTPUT_DIR = check_directory(PREP_OUTPUT_DIR)
    output_path = PREP_OUTPUT_DIR + LABTEST_OUTPUT_PATH

    lab_store = pd.HDFStore(output_path,mode='r')

    try:
        data_node = lab_store.get_node('data')
        if not data_node:
            raise ValueError("There is no data in labtest data")

        use_index_df = pd.DataFrame(columns=['col','value_counts'])
        lab_node = lab_store.get_node('data')
        for lab_name in lab_node._v_children.keys():
            value_counts = lab_store.select('data/{}'.format(lab_name)).shape[0]
            use_index_df = use_index_df.append({'col':lab_name,'value_counts':value_counts},ignore_index=True)
    finally:
        lab_store.close()

    use_index_df['value_counts'] = pd.to_numeric(use_index_df['value_counts'])
    use_index_df.to_hdf(output_path,'metadata/usecol',format='table',data_columns=True,mode='a')
    del use_index_df


def preprocess_per_test():
    global PREP_OUTPUT_DIR, LABTEST_OUTPUT_PATH
    
    # syntax checking existence for directory
    PREP_OUTPUT_DIR = check_directory(PREP_OUTPUT_DIR)
    output_path = PREP_OUTPUT_DIR + LABTEST_OUTPUT_PATH

    lab_store = pd.HDFStore(output_path,mode='a')
    labtest_mapping_df = lab_store.select('metadata/mapping_table')
    try:
        for lab_name in lab_store.get_node('data')._v_children.keys():
            labtest_df=lab_store.select('data/{}'.format(lab_name))
            
            r_avg, r_min, r_max=get_labtest_value(labtest_mapping_df,lab_name)
            labtest_df.result = labtest_df.result.map(normalize_number(r_avg,r_min,r_max))
            labtest_df.date = labtest_df.date.map(convert_month)

            save_key = 'prep/' + lab_name
            labtest_df = labtest_df.apply(pd.to_numeric,errors='ignore')
            
            lab_store.append(key=save_key, value=labtest_df,data_columns=True)

            if DEBUG_PRINT:
                print('{} completed'.format(lab_name))
    finally:
        lab_store.close()
    

    del labtest_mapping_df


def check_directory(directory_path):
    # syntax checking for directory
    if not (directory_path[-1] is '/'):
        directory_path  = directory_path + '/'
    # not exists in directory
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    return directory_path        

def revise_avg(x):
    # 10~90% 내에 있는 값을 이용해서 평균 계산
    quan_min = x.quantile(0.10)
    quan_max = x.quantile(0.90)
    return x[(x>quan_min) & (x<quan_max)].mean()

def revise_std(x):
    # 1~99% 내에 있는 값을 이용해서 표준편차 계산
    quan_min = x.quantile(0.01)
    quan_max = x.quantile(0.99)
    return x[(x>quan_min) & (x<quan_max)].std()

def revise_min(x):
    # 3시그마 바깥 값과 quanter 값의 사이값으로 결정
    std_min = revise_avg(x)-revise_std(x)*3 # 3 시그마 바깥 값
    q_min = x.quantile(0.01)
    if std_min<0 :
        # 측정값중에서 음수가 없기 때문에, 음수인 경우는 고려안함
        return q_min
    else :
        return np.mean((std_min,q_min))

def revise_max(x):
    # 3시그마 바깥 값과 quanter 값의 사이값으로 결정
    std_max = revise_avg(x)+revise_std(x)*3
    q_max = x.quantile(0.99)
    return np.mean((std_max,q_max))

def change_number(x):
    '''
    숫자　표현을　통일
    （범위　쉼표　등　표현을　단일표현으로　통일）
    '''
    str_x = str(x).replace(" ","")

    re_num   = re.compile('^[+-]{0,1}[\d\s]+[.]{0,1}[\d\s]*$') #숫자로 구성된 데이터를 float로 바꾸어 줌
    re_comma = re.compile('^[\d\s]*,[\d\s]*[.]{0,1}[\d\s]*$') # 쉼표(,)가 있는 숫자를 선별
    re_range = re.compile('^[\d\s]*[~\-][\d\s]*$') # 범위(~,-)가 있는 숫자를 선별

    if re_num.match(str_x):
        return float(str_x)
    else:
        if re_comma.match(str_x):
            return change_number(str_x.replace(',',""))
        elif re_range.match(str_x):
            if "~" in str_x:
                a,b = str_x.split("~")
            else:
                a,b = str_x.split("-")
            return np.mean((change_number(a),change_number(b)))
        else :
            return np.nan


def get_labtest_value(df,lab_name):
    _temp = df[df.labtest.isin([lab_name])]
    return _temp['AVG'].values[0], _temp['MIN'].values[0], _temp['MAX'].values[0]


def get_labtest_map():
    global MAPPING_DIR, LAB_MAPPING_PATH, DELIM

    MAPPING_DIR = check_directory(MAPPING_DIR)
    lab_mapping_path = MAPPING_DIR + LAB_MAPPING_PATH

    if not os.path.isfile(lab_mapping_path):
        raise ValueError("There is no labtest_OUTPUT file!")

    labtest_mapping_df = pd.read_csv(lab_mapping_path,delimiter=DELIM)
    return labtest_mapping_df


def convert_month(x):
    '''
    datetype을　month 단위로　바꾸어주는　함수
    ex) 20110132 -> 1101
    '''
    re_date = re.compile('^\d{8}$') 

    str_x = str(x)
    if re_date.match(str_x):
        return int(str_x[2:6])
    else : 
        raise ValueError("wrong number in date : {}".format(str_x))


def normalize_number(mean_x,min_x,max_x):
    '''
    dataframe 내 이상값을 전처리하는 함수.
    dataframe.map 을 이용할 것이므로, 함수 in 함수 구조 사용
    '''
    def _normalize_number(x):
        str_x = str(x).strip()

        re_num = re.compile('^[+-]?[\d]+[.]?[\d]*$')
        re_lower = re.compile('^<[\d\s]*[.]{0,1}[\d\s]*$')
        re_upper = re.compile('^>[\d\s]*[.]{0,1}[\d\s]*$')
        re_star = re.compile('^[\s]*[*][\s]*$')
        if re_num.match(str_x):
            # 숫자형태일경우
            float_x = np.float(str_x)
            if float_x > max_x:
                return 1
            elif float_x < min_x:
                return 0
            else:
                return (np.float(str_x) - min_x)/(max_x-min_x)
        else:
            if re_lower.match(str_x):
                return np.float(0)
            elif re_upper.match(str_x):
                return  np.float(1)
            elif re_star.match(str_x):
                return np.float( (mean_x-min_x)/(max_x-min_x) )
            else:
                return np.nan

    return _normalize_number


def _set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help="lab_test path")
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = _set_parser()

    # HDF5 dir : data/
    print("Divide_per_test method start!")
    start_time = time.time()
    divide_per_test(args.i) # divide dataframe per labtest
    print("consumed time : {}".format(time.time()-start_time))

    # HDF5  dir : metadata/mapping_table
    print("set_mapping_table start!")
    start_time = time.time()
    set_mapping_table() #  set the mapping table for labtest
    print("consumed time : {}".format(time.time()-start_time))

    # HDF5  dir : metadata/usecol
    print("set_labtest_count start!")
    start_time = time.time()
    set_labtest_count() # set the name and total counts of the labtest 
    print("consumed time : {}".format(time.time()-start_time))    

    # HDF5 dir : prep/
    print("preprocess_per_test start!")
    start_time = time.time()
    preprocess_per_test()
    print("consumed time : {}".format(time.time()-start_time))    