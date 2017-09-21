# -*- coding: utf-8 -*-
import sys, os, re

os_path = os.path.abspath('./') ; find_path = re.compile('emr_slim')
BASE_PATH = os_path[:find_path.search(os_path).span()[1]]
sys.path.append(BASE_PATH)
from converter.config import *
from converter.convert_common import check_directory

def strip_space(x):
    # 띄어쓰기 날려버리는 함수
    if isinstance(x,str):
        return x.strip()
    else :
        return str(x).strip()

def apply_strip(df):
    # dataframe에　strip_space를　적용시킴
    for column in df.columns:
        df[column] = df[column].map(strip_space, na_action='ignore')
    return df

def remove_reg(from_regs,to_regs=" ",except_case=None):
    '''
    from_regs 패턴을 to_regs의 형태로 바꾸어주는 함수
    map을 위해서, 함수 in 함수 형태로 디자인
    '''
    if not hasattr(None,"__iter__"):
        # non
        except_case = " " + str(except_case) + " "
        except_case = [except_case]
    else :
        _temp = []
        for except_c in except_case :
            except_c = " " + str(except_c) + " "
            _temp.append(except_c)
        except_case = _temp
    def _remove_reg(x):
        x = " " + str(x) + " " # padding space 추가
        re_pattern = re.compile(from_regs)
        str_pattern = re_pattern.search(x)
        if str_pattern:
            if x in except_case:
                return x.strip()
            revised_str = x.replace(str_pattern.group(),to_regs)
            return revised_str.strip()
        else :
            return x.strip()
    return _remove_reg

def unify_reg(from_regs,to_regs):
    '''
    from_regs 패턴을　to_reg의　형태로　통일시키는　함수
    map을 위해서, 함수 in 함수 형태로 디자인
    '''
    def _unify_reg(x):
        x = " " + str(x) +" "
        re_pattern = re.compile(from_regs)
        str_pattern = re_pattern.search(x)
        if str_pattern:
            revised_str = to_regs
            return to_regs
        else :
            return x.strip()
    return _unify_reg

def remove_useless_expr(df):
    '''
    성분과　무관한　표현　삭제
    '''
    meaningless_list = ["™","외","만","종",","," etc "," and ","human"," ophth "," inactivated "]
    for word_pattern in meaningless_list:
        df.ingd = df.ingd.map(remove_reg(word_pattern," "))
    return df

def remove_unit_expr(df):
    '''
    단위　표현　삭제
    '''
    # 1. 퍼센트(%) 삭제
    df.ingd = df.ingd.map(remove_reg(("\\d+(?:\\.\\d+)?%"),""),na_action="ignore")
    # 2. ( : ) 삭제
    df.ingd = df.ingd.map(remove_reg(("\\d+[:]\d+"),""),na_action="ignore")
    # 3. 슬래쉬(/) 삭제
    df.ingd = df.ingd.map(remove_reg("\/"," ","n/s"),na_action="ignore")
    for _ in range(3):
        df.ingd = df.ingd.map(remove_reg("\s\d*(m|mg|g|ml|l|mcg|mtc|u|tab|via|cap|bag|dose|gemini|amp|btl|syr|cm3|ta|pack|btlpk|iu|dw|mci|inj)\s"," "),na_action="ignore")
    df.ingd = df.ingd.map(remove_reg("\\d+(?:\\.\\d+)?(m|mg|g|ml|l|mcg|mtc|u|tab|via|cap|bag|dose|gemini|amp|btl|syr|cm3|ta|pack|btlpk|iu|dw)\s"," "),na_action="ignore")
    return df

def change_abbreviation(df):
    '''
    축약　표현　통일
    '''
    # 1. calcium 바꾸기
    df.ingd = df.ingd.map(remove_reg("\s(ca|cal)\s"," calcium "),na_action="ignore")
    df.ingd = df.ingd.map(remove_reg("\sca\."," calcium "),na_action="ignore")
    df.ingd = df.ingd.map(remove_reg("\scal\."," calcium "),na_action="ignore")
    # 2. sodium 바꾸기
    df.ingd = df.ingd.map(remove_reg("\ssod\s"," sodium "),na_action="ignore")
    df.ingd = df.ingd.map(remove_reg("\ssod\."," sodium "),na_action="ignore")
    # 3. potassium 바꾸기
    df.ingd = df.ingd.map(remove_reg("\spot\."," potassium "),na_action="ignore")
    # 4. hydrobromide 바꾸기
    df.ingd = df.ingd.map(remove_reg("\shbr\s"," hydrobromide "),na_action="ignore")
    return df

def revise_misprint_expr(df):
    '''
    오타　통일
    '''
    change_dict = {"acids"  : "acid",  "caffeine" : "caffein", "cyclosporine" : "cyclosporin",  
                            "estrogens"  : "estrogen",  "i-131mibg" : "i-131 mibg",  "kabiven peripheral" : "kabiven",  
                            "levan-h" : "levan h",  "levothyroxin sodium" : "levothyroxin",  "levothyroxine": "levothyroxin",  
                            "lidocaine hcl": "lidocaine",  "medilac-dc"   : "medilac-ds",  "methyprednisolone" : "methylprednisolone",  
                            "multivitamin" : "multi vitamin",  "multi-vitamin": "multi vitamin",  "nalbuphine"   : "nalbuphin", 
                             "piroxicam-b-cyclodextrin" : "piroxicam",  "premell cycle" : "premell",  "progesterone micronized" : "progesterone",  
                             "rabeprazole" : "rabeprazol",  "ritodrine hcl" : "ritodrine",  "rosiglitazone maleate ta" : "rosiglitazone maleate",  
                             "rosuvastatin ezetimibe" : "rosuvastatin",  "sevelamer carbonate" : "sevelamer",  "sevelamer hcl" : "sevalamer",  
                             "silodenafil" : "sildenafil",  "stay safe balance" : "stay safe",  "theophylline anhydrous" : "theophylline",  
                             "tisseel kit"  : "tisseel",  "tisseel duo quick"   : "tisseel",  "tissucol duo quick"  : "tisseel",  
                             "venlafaxine xr" : "venlafaxine",  "velafaxine" : "venlafaxine",  "tripot\." : "tripotassium", 
                             "dianela" : "dianeal", "inteferon" : "interferon", "cefdnir" : "cefidnir", "sevalemer" : "sevalamer", 
                             "gallamine" : 'galamine', "trifluorothimidine"  : "trifluorothymidine", "intralipose" : "intralipos", 
                             "acetyl-l-carnitine"  : "acetylcarnitine", "alprostsadil"  : "alprostadil", "citratre"  : "citrate", 
                             " beriplast " : " beriplast-p ", "biphenyl diethyl dicarboxylate" : "biphenyl diehtyl dicarboxylate", 
                             "buspirone" : "buspiron", "ciprofolxacin" : "ciprofloxacin",  "danazole" : "danazol", "dantrolen "  : "dantrolene ", 
                             "dihydroergocriptin " : "dihydroergocriptine ", "doxazocin" : "doxazosin", "hci" : "hcl", 
                             "famcyclovir" : "famciclovir", " hbr "    : "hydrobromide", "gingko" : "ginkgo", "\.\."  : "", 
                             "hydroxypropylmethylcellulose" : "hydroxypropylmethyl cellulose", "itraconazole"  : "itraconazol", 
                             "trometamine" : "tromethamine", "metformine" : "metformin", "methylpenidate" : "methylphenidate", 
                             "microemulsoin" : "microemulsion", "oxybutinin" : "oxybutynin", " raloxifen "  : " raloxifene ", 
                             " pyridoxin " : " pyridoxine ", "sevalamer "   : "sevelamer ", "simenthicone" : "simethicone", 
                             "terazocin" : "terazosin", "ursodesoxycholic" : "ursodeoxycholic", "5mg*20포낭" : ""}
    for key,item in change_dict.items():
        df.ingd = df.ingd.map(remove_reg(key,item),na_action="ignore")
    return df

def unify_medicine_expr(df):
    # capd 정리
    df.ingd = df.ingd.map(unify_reg("\scapd\d?\s","capd"),na_action="ignore")
    # tpn 정리
    df.ingd = df.ingd.map(unify_reg("^\s*tpn(-)?.*\s","tpn"),na_action="ignore")
    # dianeal 정리
    df.ingd = df.ingd.map(unify_reg("\sdianeal\s","dianeal"),na_action="ignore")
    return df


def remove_surplus_expr(df):
    for _ in range(3):
        # 숫자만 있는 표현 삭제
        df.ingd = df.ingd.map(remove_reg("\s\d+\s"," "),na_action="ignore")
        # 더블스페이스 공간 변경
        df.ingd = df.ingd.map(remove_reg("\s\s"," "),na_action="ignore")
    return df

def get_mapping_table(context_path):
    medicine_context_df = pd.read_excel(context_path)
    medicine_context_df.columns = ['medi_code','medi_name','s_date','e_date','ingd','ATC_code','ATC_desc']
    medicine_context_df.drop(medicine_context_df.columns[[1, 2, 3, 5, 6]], axis=1,inplace=True)
    # ['약품코드', '약품명', '시작일자', '종료일자', '성분명', 'ATC분류코드', 'ATC분류설명']
    # 1. strip() & 소문자화
    medicine_context_df.ingd = medicine_context_df.ingd.str.strip().str.lower()
    # 2. 불필요 표현 삭제
    medicine_context_df = remove_useless_expr(medicine_context_df)
    # 3. (~~~) 패턴 삭제
    medicine_context_df.ingd = medicine_context_df.ingd.map(remove_reg(("\(.*\)"),""),na_action="ignore")
    # 4. 단위 삭제
    medicine_context_df = remove_unit_expr(medicine_context_df)
    # 5. 제조형태 의미하는 것 제외
    medicine_context_df.ingd = medicine_context_df.ingd.map(remove_reg("\s(sol|sol.|soln.|soln|extract|ext|ext.|syrup|tab|conjugated|복합|strip|disol|cream|elixir|clavulanated|dry|dried|lotion|capsule|reagent)\s"," "),na_action="ignore")
    # 6. 오탈자 수정
    medicine_context_df = revise_misprint_expr(medicine_context_df)
    # 7. 특정 약품 표현 통일medicine_context_df.성분명 = medicine_context_df.성분명.map(remove_reg("\s(sol|sol.|soln.|soln|extract|ext|ext.|syrup|tab|conjugated|복합|strip|disol|cream|elixir|clavulanated|dry|dried|lotion|capsule|reagent)\s"," "),na_action="ignore")
    medicine_context_df = unify_medicine_expr(medicine_context_df)
    # 8. 전처리 정리
    medicine_context_df = remove_surplus_expr(medicine_context_df)
    # mapping series 만들기
    code_to_name = pd.Series(medicine_context_df.ingd.values, index=medicine_context_df.medi_code)
    unique_name_set = medicine_context_df.ingd.unique()
    name_to_int    = pd.Series(range(1,len(unique_name_set)+1), index=unique_name_set)
    
    mapping_df = pd.concat([code_to_name,code_to_name.map(name_to_int)],axis=1,keys=code_to_name)
    mapping_df.index.name = 'medi_code'
    mapping_df.columns = ['ingd_name','mapping_code']
    mapping_df.reset_index(inplace=True)
    mapping_df.drop_duplicates(inplace=True)
    mapping_df.to_hdf(BASE_PATH+'data/prep/prescribe_df.h5','metadata/mapping_table',format='table',data_columns=True,mode='a')
    print(mapping_df.head())
    del mapping_df

def get_prescribe_map():
    prescribe_store = pd.HDFStore(BASE_PATH+'data/prep/prescribe_df.h5')
    try:
        prescribe_map_df = prescribe_store.select("metadata/mapping_table")
    finally:
        prescribe_store.close()
    mapping_dict = pd.Series(prescribe_map_df.mapping_code.values, index=prescribe_map_df.medi_code ).to_dict()
    del prescribe_map_df
    return mapping_dict

def preprocess_prescirbe(preprocess_path):
    medicine_df = pd.read_csv(preprocess_path)
    mapping_dict = get_prescribe_map()
    medicine_df['mapping_code'] = medicine_df.medi_code.map(mapping_dict)
    medicine_df.dropna(inplace=True)

    medicine_df.date = medicine_df.date.astype(int)
    medicine_df['count'] = medicine_df['count'].astype(int)
    medicine_df.no = medicine_df.no.astype(int)
    medicine_df.mapping_code = medicine_df.mapping_code.astype(int)

    medicine_df.date = medicine_df.date.map(convert_month)
    medicine_df.count = medicine_df['count'].map(convert_times_per_month)

    medicine_df.to_hdf(BASE_PATH+'data/prep/prescribe_df.h5','data/',format='table',data_columns=True,mode='a')
    del medicine_df

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

def convert_times_per_month(x):
    return float(x) // 30

def _set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('context_path', help="medicine context dataframe path")
    parser.add_argument('prescribe_path', help='prescribe dataframe path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args=_set_parser()
    get_mapping_table(args.context_path)
    preprocess_prescribe(args.prescribe_path)