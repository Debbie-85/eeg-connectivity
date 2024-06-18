import json
from nilearn import plotting

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.stats import stats
from functions import *

file_path = 'EEG_segment.json'
with open(file_path, 'r') as file:
    js_file = json.load(file)


AD_eeg = js_file['AD']
MCI_eeg = js_file['MCI']
HC_eeg = js_file['HC']

type = 'correlation'  # correlation, coherence, PLI, PLV
AD_eeg_Correlation = get_cor(AD_eeg, type = type)
MCI_eeg_Correlation = get_cor(MCI_eeg, type = type)
HC_eeg_Correlation = get_cor(HC_eeg, type = type)

eeg_cor = {
    'AD': AD_eeg_Correlation,  # 590
    'MCI': MCI_eeg_Correlation,  # 560
    'HC': HC_eeg_Correlation # 550
}
filename = 'EEG_correlation.json'
with open(filename,'w') as f:
    json.dump(eeg_cor,f)

print(1)

