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

def check_age(x):
    if not isinstance(x,float):
        return np.nan
    elif x<0.0 : 
        return np.nan
    else:
        return x


def construct_demo(demographic_path):
    global PREP_OUTPUT_DIR, DEMOGRAPHIC_OUTPUT_PATH
    demographic_output_path = PREP_OUTPUT_DIR + DEMOGRAPHIC_OUTPUT_PATH
        
    demo_df = pd.read_excel(demographic_path)
    demo_df.columns = ['no','sex','age']
    demo_df['sex'] = demo_df['sex'].map(lambda x : 1 if x == 'F' else 0)
    demo_df['age'] = demo_df['age'].map(check_age)

    demo_df.to_hdf(demographic_output_path, 'data/original',
                                                format='table',data_columns=True,mode='a')
    
    #나이를　０～１００세까지　１０단위로　쪼갬．　
    #범위　바깥은　０세，　１００세에　몰아넣음
    AGE_BREAK_POINTS = list(range(0,100,10))
    AGE_BREAK_POINTS[0] = -10
    AGE_BREAK_POINTS.append(160)
    AGE_LABELS = ['Y10','Y1020','Y2030','Y3040','Y4050','Y5060','Y6070','Y7080','Y8090','Y90100']

    demo_df.age = pd.cut(demo_df.age,AGE_BREAK_POINTS,labels=AGE_LABELS)
    demo_df.age = demo_df.age.cat.add_categories(['notknown'])
    demo_df.loc[demo_df.age.isnull(),'age'] = 'notknown'
    prep_df = pd.concat([demo_df[['no','sex']],pd.get_dummies(demo_df.age)],axis=1)
    
    prep_df.to_hdf(demographic_output_path,'data/prep',
        format='table',data_columns=True,mode='a')


def get_np_array_demo(no):
    global PREP_OUTPUT_DIR, DEMOGRAPHIC_OUTPUT_PATH

    demographic_output_path = PREP_OUTPUT_DIR + DEMOGRAPHIC_OUTPUT_PATH
    demo_store=pd.HDFStore(demographic_output_path,mode='r')
    result = demo_store.select('/data/prep',where='no=={}'.format(no)).values[0,1:]
    demo_store.close()
    return result
