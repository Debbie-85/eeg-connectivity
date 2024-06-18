import matplotlib.pyplot as plt
import numpy as np
from functions import *

file_path = 'EEG_correlation.json'
with open(file_path, 'r') as file:
    js_file = json.load(file)

AD_eeg_Correlation = js_file['AD']
MCI_eeg_Correlation = js_file['MCI']
HC_eeg_Correlation = js_file['HC']


threshold = 0.2
AD_degrees, AD_degree_centrality, AD_betweenness_centrality = get_degree_features(AD_eeg_Correlation, threshold = threshold , mean_flag=False)
MCI_degrees, MCI_degree_centrality, MCI_betweenness_centrality = get_degree_features(MCI_eeg_Correlation, threshold = threshold, mean_flag=False )
HC_degrees, HC_degree_centrality, HC_betweenness_centrality = get_degree_features(HC_eeg_Correlation, threshold = threshold, mean_flag=False)


from scipy.stats import kruskal
p = np.zeros(19)
for i in range(len(AD_degrees[0])):
    statistic, p_value = kruskal(AD_degrees[:,i],MCI_degrees[:,i],HC_degrees[:,i])
    if p_value <0.05:
        p[i] = 100
    else:
        p[i] = 10
        print(i+1)
plot_nodes(degree_centrality = p, N=10, title='kruskal')


# n_bars = 50
# plt.hist(AD_degrees, bins=n_bars)
# plt.ylim([0,1350])
# plt.show()
# plt.hist(MCI_degrees, bins=n_bars)
# plt.ylim([0,1350])
# plt.show()
# plt.hist(HC_degrees, bins=n_bars)
# plt.ylim([0,1350])
# plt.show()


# 绘制平均度分布
AD_degrees, AD_degree_centrality, AD_betweenness_centrality = get_degree_features(AD_eeg_Correlation, threshold = threshold )
MCI_degrees, MCI_degree_centrality, MCI_betweenness_centrality = get_degree_features(MCI_eeg_Correlation, threshold = threshold )
HC_degrees, HC_degree_centrality, HC_betweenness_centrality = get_degree_features(HC_eeg_Correlation, threshold = threshold)


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
mpl.rcParams["font.sans-serif"]=["SimHei"]
mpl.rcParams["axes.unicode_minus"]=False

plt.figure(figsize=(8,4))
x = np.arange(len(AD_degree_centrality))
bar_width = 0.2
tick_label=['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
# create bar
plt.bar(x,AD_degree_centrality,bar_width, label="AD")
plt.bar(x+bar_width,MCI_degree_centrality,bar_width, label = 'MCI' )
plt.bar(x+bar_width*2,HC_degree_centrality,bar_width, label = 'HC' )
plt.xlabel("channels")
plt.ylabel("AD degree centrality")
plt.xticks(x+bar_width,tick_label)
plt.legend(loc="upper center", ncol=3)
plt.show()


plt.figure(figsize=(8,4))
x = np.arange(len(AD_betweenness_centrality))
plt.bar(x,AD_betweenness_centrality,bar_width, label="AD")
plt.bar(x+bar_width,MCI_betweenness_centrality,bar_width, label = 'MCI' )
plt.bar(x+bar_width*2,HC_betweenness_centrality,bar_width, label = 'HC' )
plt.xlabel("channels")
plt.ylabel("AD betweenness centrality")
plt.xticks(x+bar_width,tick_label)  # x+bar_width/2 只有两列
plt.legend(loc="upper center", ncol=3)
plt.show()

plot_nodes(degree_centrality = AD_degrees, N=100, title='AD')
plot_nodes(degree_centrality = MCI_degrees,N=100, title='MCI')
plot_nodes(degree_centrality = HC_degrees, N=100, title='HC')

plot_nodes(degree_centrality = AD_degree_centrality, title='AD')
plot_nodes(degree_centrality = MCI_degree_centrality, title='MCI')
plot_nodes(degree_centrality = HC_degree_centrality, title='HC')

plot_nodes(degree_centrality = AD_betweenness_centrality, N=20000, title='AD')
plot_nodes(degree_centrality = MCI_betweenness_centrality, N=20000, title='MCI')
plot_nodes(degree_centrality = HC_betweenness_centrality, N=20000, title='HC')



