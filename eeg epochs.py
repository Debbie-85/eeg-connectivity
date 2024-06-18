import json
import os
import random

import numpy as np
import scipy.io
from functions import *

seed = 2023  # 固定随机数
random.seed(seed)
np.random.seed(seed)


AD_path ='eeg_mat\AD'
MCI_path ='eeg_mat\MCI'
HC_path = 'eeg_mat\CONTROL'



AD_file_temp = get_files(AD_path)
AD_file = []
for file_i in AD_file_temp:
    if file_i[-1] != 't':
        temp_file = get_files(file_i)
        AD_file = AD_file + temp_file
    else:
        AD_file = AD_file + file_i


MCI_file_temp = get_files(MCI_path)
MCI_file = []
for file_i in MCI_file_temp:
    if file_i[-1] != 't':
        temp_file = get_files(file_i)
        MCI_file = MCI_file + temp_file
    else:
        MCI_file = MCI_file + file_i

HC_file_temp = get_files(HC_path)
HC_file = []
for file_i in HC_file_temp:
    if file_i[-1] != 't':
        temp_file = get_files(file_i)
        HC_file = HC_file + temp_file
    else:
        HC_file = HC_file + [file_i]


AD_file_temp = get_files(AD_path)
AD_file = []
AD_eeg = {}
sum_n = 0
for file_i in AD_file_temp:
    if file_i[-1] != 't':
        temp_folder = get_files(file_i)
        if temp_folder == []: continue
        temp_name = file_i.split('\\')[-1]
        eeg_sub = []
        for j in range(len(temp_folder)):
            temp_file = scipy.io.loadmat(temp_folder[j])
            temp_eeg = temp_file['export']
            if temp_eeg.shape[-1] != 19:
                continue
            temp_eeg_segmention = eeg_segmention(temp_eeg,n=2)
            eeg_sub.append(temp_eeg_segmention)
        eeg_sub = np.vstack(eeg_sub)
        sum_n = sum_n + len(eeg_sub)
        AD_eeg[temp_name] = eeg_sub.tolist()
print(sum_n)  # 590


MCI_file_temp = get_files(MCI_path)
MCI_file = []
MCI_eeg = {}
sum_n = 0
for file_i in MCI_file_temp:
    if file_i[-1] != 't':
        temp_folder = get_files(file_i)
        if temp_folder == []: continue
        temp_name = file_i.split('\\')[-1]
        eeg_sub = []
        for j in range(len(temp_folder)):
            temp_file = scipy.io.loadmat(temp_folder[j])
            temp_eeg = temp_file['export']
            if temp_eeg.shape[-1] != 19:
                continue
            temp_eeg_segmention = eeg_segmention(temp_eeg,n=10)
            eeg_sub.append(temp_eeg_segmention)
        eeg_sub = np.vstack(eeg_sub)
        sum_n = sum_n + len(eeg_sub)
        MCI_eeg[temp_name] = eeg_sub.tolist()
print(sum_n)  # 560


HC_file_temp = get_files(HC_path)
HC_file = []
HC_eeg = {}
sum_n = 0
for file_i in HC_file_temp:
    temp_name = file_i.split('\\')[-1]
    if file_i[-1] != 't':
        temp_folder = get_files(file_i)
        if temp_folder == []: continue
        eeg_sub = []
        for j in range(len(temp_folder)):
            temp_file = scipy.io.loadmat(temp_folder[j])
            temp_eeg = temp_file['export']
            if temp_eeg.shape[-1] != 19:
                continue
            temp_eeg_segmention = eeg_segmention(temp_eeg,n=6)
            eeg_sub.append(temp_eeg_segmention)
        eeg_sub = np.vstack(eeg_sub)
        HC_eeg[temp_name] = eeg_sub.tolist()
    else:
        temp_file = scipy.io.loadmat(file_i)
        temp_eeg = temp_file['segmenty']
        temp_eeg = temp_eeg.transpose()[:,0:19]
        if temp_eeg.shape[-1] != 19:
            continue
        temp_eeg_segmention = eeg_segmention(temp_eeg, n=5)
        HC_eeg[temp_name] = temp_eeg_segmention.tolist()

eeg = {
    'AD': AD_eeg,  # 590
    'MCI': MCI_eeg,  # 560
    'HC': HC_eeg  # 550
}
filename = 'EEG_segment.json'
with open(filename,'w') as f:
    json.dump(eeg,f)







