# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import sys
import os
import argparse
import time

os_path = os.path.abspath('./')
find_path = re.compile('emr_slim')
BASE_PATH = os_path[:find_path.search(os_path).span()[1]]

from config import *

def get_na_label_df():
    global PREP_OUTPUT_DIR, LABEL_PATIENT_PATH, DEBUG_PRINT
    # syntax checking existence for directory
    PREP_OUTPUT_DIR = check_directory(PREP_OUTPUT_DIR)
    lab_output_path = PREP_OUTPUT_DIR + LABTEST_OUTPUT_PATH
    output_path = PREP_OUTPUT_DIR + LABEL_PATIENT_PATH

    # remove temp file in output_path
    if os.path.isfile(output_path):
        os.remove(output_path)
    
    lab_store = pd.HDFStore(lab_output_path,mode='r')
    try:
        na_df = lab_store.select('data/L3041')    
        na_e_df = lab_store.select('data/L8041')
    finally:
        lab_store.close()

    na_df.date = na_df.date.map(convert_month)
    na_e_df.date = na_e_df.date.map(convert_month)

    na_df.result = na_df.result.map(convert_to_numeric)
    na_e_df.result = na_e_df.result.map(convert_to_numeric)

    na_df.loc[na_df.result<135,'result'] = 1
    na_df.loc[(na_df.result>=135)&(na_df.result<=145),'result'] = 0
    na_df.loc[na_df.result>145,'result'] = 2

    na_e_df.loc[na_e_df.result<135,'result'] = 1
    na_e_df.loc[(na_e_df.result>=135)&(na_e_df.result<=145),'result'] = 0
    na_e_df.loc[na_e_df.result>145,'result'] = 2

    total_df = pd.concat([na_df,na_e_df])
    have_problem_df = total_df[(total_df.result==2) | (total_df.result==1)].sort_values('date')
    have_problem_df.drop_duplicates(subset=['no','date'],keep='first',inplace=True)

    no_problem_df = total_df[(total_df.result==0)].sort_values('date')
    
    label_df = pd.concat([have_problem_df,no_problem_df])
    label_df = label_df.groupby(['no','date','result']).size().unstack(fill_value=0)
    label_df = (2*(label_df[2.0]>0)) + (1*(label_df[1.0]>0))

    label_df = label_df.reset_index()
    label_df.columns=['no','date','label']

    label_df.to_hdf(output_path,"label/na",format='table',data_columns=True,mode='a')
    del total_df, have_problem_df, no_problem_df, na_df, na_e_df
    divide_test_train_set_from_label_per_patient('label/na')

def get_ka_label_df():
    global PREP_OUTPUT_DIR, LABEL_PATIENT_PATH, DEBUG_PRINT
    # syntax checking existence for directory
    PREP_OUTPUT_DIR = check_directory(PREP_OUTPUT_DIR)
    lab_output_path = PREP_OUTPUT_DIR + LABTEST_OUTPUT_PATH
    output_path = PREP_OUTPUT_DIR + LABEL_PATIENT_PATH

    lab_store = pd.HDFStore(lab_output_path,mode='r')
    try:
        ka_df = lab_store.select('data/L3042')
        ka_e_df = lab_store.select('data/L8042')
    finally:
        lab_store.close()

    ka_df.date = ka_df.date.map(convert_month)
    ka_e_df.date = ka_e_df.date.map(convert_month)

    ka_df.result = ka_df.result.map(convert_to_numeric)
    ka_e_df.result = ka_e_df.result.map(convert_to_numeric)

    ka_df.loc[ka_df.result<3.5,'result'] = 1
    ka_df.loc[(ka_df.result>=3.5)&(ka_df.result<=5.5),'result'] = 0
    ka_df.loc[ka_df.result>5.5,'result'] = 2

    ka_e_df.loc[ka_e_df.result<3.5,'result'] = 1
    ka_e_df.loc[(ka_e_df.result>=3.5)&(ka_e_df.result<=5.5),'result'] = 0
    ka_e_df.loc[ka_e_df.result>5.5,'result'] = 2

    total_df = pd.concat([ka_df,ka_e_df])
    have_problem_df = total_df[(total_df.result==2) | (total_df.result==1)].sort_values('date')
    have_problem_df.drop_duplicates(subset=['no','date'],keep='first',inplace=True)

    no_problem_df = total_df[(total_df.result==0)].sort_values('date')

    label_df = pd.concat([have_problem_df,no_problem_df])
    label_df = label_df.groupby(['no','date','result']).size().unstack(fill_value=0)
    label_df = (2*(label_df[2.0]>0)) + (1*(label_df[1.0]>0))

    label_df = label_df.reset_index()
    label_df.columns=['no','date','label']

    label_df.to_hdf(output_path,"label/ka",format='table',data_columns=True,mode='a')
    del total_df, have_problem_df, no_problem_df, ka_df, ka_e_df
    divide_test_train_set_from_label_per_patient('label/ka')


def divide_test_train_set_from_label_per_patient(label_name):
    global PREP_OUTPUT_DIR, LABEL_PATIENT_PATH, DEBUG_PRINT
    # syntax checking existence for directory
    PREP_OUTPUT_DIR = check_directory(PREP_OUTPUT_DIR)
    output_path = PREP_OUTPUT_DIR + LABEL_PATIENT_PATH

    lab_store = pd.HDFStore(output_path,mode='r')
    label_df = lab_store.select(label_name)
    lab_store.close()

    patient_no_arr = label_df.no.unique()
    np.random.shuffle(patient_no_arr)
    train_patient, validate_patient, test_patient = np.split(patient_no_arr,[int(.7*len(patient_no_arr)), int(.8*len(patient_no_arr))])

    train_df = label_df[label_df.no.isin(train_patient)]
    validate_df = label_df[label_df.no.isin(validate_patient)]
    test_df = label_df[label_df.no.isin(test_patient)]
    
    train_df.to_hdf(output_path,label_name+'/train',format='table',data_columns=True,mode='a')
    validate_df.to_hdf(output_path,label_name+'/validation',format='table',data_columns=True,mode='a')
    test_df.to_hdf(output_path,label_name+'/test',format='table',data_columns=True,mode='a')


def get_labtest_df(no): 
    '''
    환자번호를　넣으면　column은　KCDcode, row는　time-serial의　형태인　dataframe이　나오는　함수
    '''
    global PREP_OUTPUT_DIR, LABTEST_OUTPUT_PATH
    lab_output_path = PREP_OUTPUT_DIR + LABTEST_OUTPUT_PATH
    lab_store = pd.HDFStore(lab_output_path,mode='r')
    
    col_list = get_timeseries_column()
    # create empty dataframe
    try:
        lab_node = lab_store.get_node('prep')
        use_labtest_values = list(lab_node._v_children.keys())
        result_df = pd.DataFrame(columns=col_list, index=use_labtest_values)

        target_df=lab_store.select('total',where='no=={}'.format(no),columns=['date','lab_name','result'])\
                                          .groupby(['date','lab_name'])\
                                          .mean().reset_index() 
        for value in target_df.values:
            date,lab_name,result = value
            result_df.loc[lab_name,date]= result
    finally:
        lab_store.close()
    
    del target_df
    return result_df


def get_labtest_aggregated_df(no): 
    '''
    환자번호를　넣으면　column은　KCDcode, row는　time-serial의　형태인　dataframe이　나오는　함수
    겹쳤을때　３가지　matrix(min max mean)가 생성．　
    '''
    global PREP_OUTPUT_DIR, LABTEST_OUTPUT_PATH
    lab_output_path = PREP_OUTPUT_DIR + LABTEST_OUTPUT_PATH
    lab_store = pd.HDFStore(lab_output_path,mode='r')
    
    col_list = get_timeseries_column()
    # create empty dataframe
    try:
        lab_node = lab_store.get_node('prep')
        use_labtest_values = list(lab_node._v_children.keys())
        mean_df = pd.DataFrame(columns=col_list, index=use_labtest_values)
        min_df  = pd.DataFrame(columns=col_list, index=use_labtest_values)
        max_df = pd.DataFrame(columns=col_list, index=use_labtest_values)

        target_df=lab_store.select('total',where='no=={}'.format(no),columns=['date','lab_name','result'])

        temp = target_df.groupby(['date','lab_name']).mean().reset_index() 
        for value in temp.values:
            date,lab_name,result = value
            mean_df.loc[lab_name,date]= result
        temp = target_df.groupby(['date','lab_name']).min().reset_index() 
        for value in temp.values:
            date,lab_name,result = value
            min_df.loc[lab_name,date]= result
        temp = target_df.groupby(['date','lab_name']).max().reset_index() 
        for value in temp.values:
            date,lab_name,result = value
            max_df.loc[lab_name,date]= result
    finally:
        lab_store.close()
    
    del target_df
    return mean_df, min_df, max_df

def convert_to_numeric(x):
    str_x = str(x)
    return float(str_x.replace(">","").replace("<",""))


if __name__=='__main__':
    get_na_label_df()
    get_ka_label_df()
