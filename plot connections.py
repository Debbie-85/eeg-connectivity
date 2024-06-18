import json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_chord_diagram import chord_diagram
import numpy as np

from functions import *
tick_label=['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']


file_path = 'EEG_Correlation.json'
with open(file_path, 'r') as file:
    js_file = json.load(file)

AD_eeg_Correlation = js_file['AD']
MCI_eeg_Correlation = js_file['MCI']
HC_eeg_Correlation = js_file['HC']

Corr_AD = []
for name, value in AD_eeg_Correlation.items():
    value = np.array(value)
    Corr_AD.append(value)

Corr_MCI = []
for name, value in MCI_eeg_Correlation.items():
    value = np.array(value)
    Corr_MCI.append(value)

Corr_HC = []
for name, value in HC_eeg_Correlation.items():
    value = np.array(value)
    Corr_HC.append(value)

Corr_AD = np.concatenate(Corr_AD, axis=0)
Corr_MCI = np.concatenate(Corr_MCI, axis=0)
Corr_HC = np.concatenate(Corr_HC, axis=0)
Corr_AD[np.isnan(Corr_AD)] = 0
Corr_MCI[np.isnan(Corr_MCI)] = 0
Corr_HC[np.isnan(Corr_HC)] = 0


#　－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－
T = 0.4
T_save = '0_4'
Corr_AD[abs(Corr_AD)<T]=0
Corr_MCI[abs(Corr_MCI)<T]=0
Corr_HC[abs(Corr_HC)<T]=0


#　－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－
Corr_AD = np.mean(Corr_AD,axis = 0)
Corr_MCI = np.mean(Corr_MCI,axis = 0)
Corr_HC = np.mean(Corr_HC,axis = 0)

for i in range(len(Corr_AD)):
    Corr_AD[i,i]=0
for i in range(len(Corr_MCI)):
    Corr_MCI[i,i]=0
for i in range(len(Corr_HC)):
    Corr_HC[i,i]=0
# plot_leads_with_weights(abs(Corr_AD), title='AD',threshold=0.4)
# plot_cor(abs(Corr_AD), title='AD')
# plot_leads_with_weights(abs(Corr_MCI), title='MCI',threshold=0.4)
# plot_cor(abs(Corr_MCI), title='MCI')
# plot_leads_with_weights(abs(Corr_HC), title='HC',threshold=0.4)
# plot_cor(abs(Corr_HC), title='HC')

chord_diagram(Corr_AD, tick_label)
plt.savefig('chord_diagram_AD_{}.png'.format(T_save),dpi=1000)

chord_diagram(Corr_MCI, tick_label)
plt.savefig('chord_diagram_MCI_{}.png'.format(T_save),dpi=1000)

chord_diagram(Corr_HC, tick_label)
plt.savefig('chord_diagram_HC_{}.png'.format(T_save),dpi=1000)



label = ["Fp1", "Fp2", "F7", "Fz", "F3", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1",
         "O2"]
fig, ax = plt.subplots(1, 3,figsize=(9,3))  # figsize=(15,6)
ax = ax.flatten()
im = ax[0].imshow(abs(Corr_AD),cmap='Blues')
ax[0].set_title('AD')
ax[0].set_yticks(range(len(label)))
ax[0].set_yticklabels(label)
ax[0].set_xticks(range(len(label)))
ax[0].set_xticklabels(label,rotation=90)

im = ax[1].imshow(abs(Corr_MCI),cmap='Blues')
ax[1].set_title('MCI')
ax[1].set_yticks(range(len(label)))
ax[1].set_yticklabels(label)
ax[1].set_xticks(range(len(label)))
ax[1].set_xticklabels(label,rotation=90)

im = ax[2].imshow(abs(Corr_HC),cmap='Blues')
ax[2].set_title('HC')
ax[2].set_yticks(range(len(label)))
ax[2].set_yticklabels(label)
ax[2].set_xticks(range(len(label)))
ax[2].set_xticklabels(label,rotation=90)

fig.colorbar(im, ax=[ax[i] for i in range(0,3)], fraction=0.02, pad=0.05)
plt.show()



# 绘制余弦图
fig, ax = plt.subplots(1, 3, figsize=(26,6))  # figsize=(15,6)
ax = ax.flatten()
chord_diagram_AD = mpimg.imread('chord_diagram_AD_{}.png'.format(T_save)) #
im = ax[0].imshow(chord_diagram_AD)
ax[0].set_title('AD', fontsize = 24)
ax[0].axis('off')
chord_diagram_AD = mpimg.imread('chord_diagram_MCI_{}.png'.format(T_save)) #
im = ax[1].imshow(chord_diagram_AD)
ax[1].set_title('MCI', fontsize = 24)
ax[1].axis('off')
chord_diagram_AD = mpimg.imread('chord_diagram_HC_{}.png'.format(T_save)) #
im = ax[2].imshow(chord_diagram_AD)
ax[2].set_title('HC', fontsize = 24)
ax[2].axis('off')
plt.tight_layout()
plt.show()


# 绘制脑网络图
fig = plt.figure(figsize=(18,6))
plt.subplot(131)
plot_leads_with_weights_for_subplot(abs(Corr_AD), title='AD',threshold=0.2)
plt.title('AD',fontsize=26)

plt.subplot(132)
plot_leads_with_weights_for_subplot(abs(Corr_MCI), title='MCI',threshold=0.2)
plt.title('MCI',fontsize=26)

plt.subplot(133)
pc = plot_leads_with_weights_for_subplot(abs(Corr_HC), title='HC',threshold=0.2)
plt.title('HC',fontsize=26)

cax = fig.add_axes([0.95, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
plt.colorbar(pc, cax)

plt.savefig('brain_connectivity_{}.png'.format(T_save), dpi=1000)
plt.savefig('brain_connectivity_{}.pdf'.format(T_save), dpi=1000)
plt.show()

