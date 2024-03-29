import pandas as pd
import torch
from utils import *

DEVICE = utils.get_device()

def get_colunm_info() -> (list, list, str, list):
    RAW_data = pd.read_csv(utils.get_dataset_path())
    NUM, CAT, LABEL = utils.get_dataset_attributes()
    if len(CAT) == 0:
        cat_num = []
    elif len(CAT) == 1:
        cat_num = [RAW_data[CAT[0]].nunique()]
    else:
        cat_num = [len(RAW_data[col].unique()) for col in CAT]
    return NUM, CAT, LABEL, cat_num

def get_data() -> (torch.tensor, torch.tensor, (list, list, str, list)):

    # RAW_data = pd.read_csv('data/compass_old.csv')
    # CAT=['sex','age_cat','race','c_charge_degree','decile_score.1','score_text','v_type_of_assessment','v_decile_score','v_score_text']
    # NUM=['age','juv_fel_count','juv_misd_count','juv_other_count','priors_count','days_b_screening_arrest','c_days_from_compas','end']
    # LABEL = 'is_recid'
    RAW_data = pd.read_csv(utils.get_dataset_path())
    NUM, CAT, LABEL = utils.get_dataset_attributes()
    
    # convert categorical data to ordinal data
    from sklearn.preprocessing import OrdinalEncoder
    enc = OrdinalEncoder()
    data_pd = RAW_data.copy()
    data_pd[CAT] = enc.fit_transform(RAW_data[CAT])
    # data_pd = pd.get_dummies(RAW_data, columns=CAT, dtype=float)
    # label to category
    data_pd[LABEL] = data_pd[LABEL].astype('category').cat.codes

    # realign data to num + cat
    data_pd = data_pd[NUM + CAT + [LABEL]]

    # caculate unique value of each categorical feature
    if len(CAT) == 0:
        cat_num = []
    elif len(CAT) == 1:
        cat_num = [data_pd[CAT[0]].nunique()]
    else:
        cat_num = [len(data_pd[col].unique()) for col in CAT]

    # normalize numerical data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_pd[NUM] = scaler.fit_transform(data_pd[NUM])

    # convert data to tensor
    x = torch.tensor(data_pd.drop(columns=[LABEL]).values, dtype=torch.float, device=DEVICE)  # [48842, 108]
    y = torch.tensor(data_pd[LABEL].values, dtype=torch.long, device=DEVICE) # [48842]
    
    return x, y, (NUM, CAT, LABEL, cat_num)
