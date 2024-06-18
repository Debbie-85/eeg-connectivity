import json
import math
import os
import random


import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy import signal

tick_label=['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']

network_feature_name = ['Diameter',
 'Average shortest path length',
 'Average dgree',
 'Local efficiency',
 'Global efficiency',
 'Average clustering',
 'Transitivity',
 'Degree centrality',
 'Closeness centrality',
 'Betweenness centrality',
 ]

def get_files(base_dir):
    files = [os.path.join(base_dir, file) for file in os.listdir(base_dir)]
    return files

def eeg_segmention(eeg, n, len_s = 5, fs=128):  # n,19, 5s,fs=128Hz
    length = len(eeg)
    eeg_seg = []
    for i in range(n):
        start = random.randint(fs*2, length-fs*len_s)
        eeg_seg.append(eeg[start:start+fs*len_s])
    eeg_seg = np.stack((eeg_seg)) #　n,640,19
    return eeg_seg

def normalization(eeg_seg):  # n,640,19
    mean = np.mean(eeg_seg,axis=1,keepdims=1)
    std = np.std(eeg_seg,axis=1,keepdims=1)
    eeg_seg = (eeg_seg-mean)/(std + 0.00000001)
    return eeg_seg


def calculate_conn(a,b, sample_rate=128, type = 'correlation'):
    r = None
    if type == 'correlation':
        r, p_value = (stats.pearsonr(a,b))
    if type == 'coherence':
        f, Cxy = signal.coherence(a, b, sample_rate)
        r = Cxy.mean()
    if type == 'PLI':
        a_angle = instantaneous_phase(a)
        b_angle = instantaneous_phase(b)
        phase_diff = a_angle - b_angle
        phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
        pli = abs(np.mean(np.sign(phase_diff)))
        r = pli
    if type == 'PLV':
        a_phi = np.angle(signal.hilbert(a))
        b_phi = np.angle(signal.hilbert(b))
        plv = np.abs(np.sum(np.exp(1j * (a_phi - b_phi))) / len(a_phi))
        r = plv
    return r

def instantaneous_phase(s):
    s_hilbert = signal.hilbert(s)
    s_angle = np.angle(s_hilbert)
    return s_angle


def make_psd_plot(signal_list,  fs = 128):
    plt.figure(figsize=(12,6))
    x = np.array(signal_list)
    f, Psd = signal.welch(x, fs, nperseg=1024, scaling='density')
    Psd = 10*np.log10(Psd)
    plt.plot(f, Psd)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()

def get_cor(AD_eeg : dict, type = 'correlation'):
    AD_eeg_Correlation = {}
    for name, value in AD_eeg.items():
        value = np.array(value)
        corr = []
        for k in range(len(value)):
            temp = value[k]  # 640,19
            temp = temp.transpose()
            corr_matrix = np.zeros([len(temp), len(temp)])  # 19,19
            # make_psd_plot(temp[0],fs=128)

            for i in range(len(temp)):
                for j in range(len(temp)):
                    temp_corr = calculate_conn(temp[i], temp[j], type = type)
                    corr_matrix[i, j] = temp_corr
            corr.append(corr_matrix)
        corr = np.stack(corr)
        AD_eeg_Correlation[name] = corr.tolist()
    return AD_eeg_Correlation



def plot_cor(correlation_matrix, title, if_show=True):

    label = ["Fp1", "Fp2", "F7", "Fz", "F3", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yticks(range(len(label)))
    ax.set_yticklabels(label)
    ax.set_xticks(range(len(label)))
    ax.set_xticklabels(label)
    im = ax.imshow(correlation_matrix, cmap='Blues')
    plt.colorbar(im)

    plt.title(title)
    # show
    if if_show:
        plt.show()
    return fig





def adjacency_matrix(eeg, threshold =0.5):
    bool_matrix = (eeg >= threshold)
    bool_matrix = bool_matrix.astype(int)
    eeg = eeg * bool_matrix
    return eeg

def plot_leads(correlation_matrix, title, if_show = True):
    df = pd.read_excel('channel19.xlsx', engine='openpyxl')
    names = df.channel
    Names = []
    for i in range(len(names)):
        # print(names[i])
        temp_name = names[i]
        temp_name = temp_name.replace('.', '')
        Names.append(temp_name)
    list1 = np.zeros([19], dtype=np.int)
    list2 = np.zeros([19], dtype=np.float)
    for i in range(19):
        list1[i] = int(df.x[i])
        list2[i] = float(df.y[i])
    xy = np.zeros([19, 2])  # （19,2)
    for i in range(19):
        xy[i, 1] = math.cos(math.radians(list1[i])) * list2[i]
        xy[i, 0] = math.sin(math.radians(list1[i])) * list2[i]
    pos = {}
    pos_mapping = {}
    for i in range(len(xy)):
        pos[i] = tuple(xy[i])
        pos_mapping[Names[i]] = tuple(xy[i])

    np.fill_diagonal(correlation_matrix, 0)
    correlation_matrix = adjacency_matrix(correlation_matrix)
    correlation_matrix = np.matrix(correlation_matrix)
    G = nx.from_numpy_matrix(correlation_matrix)

    As = nx.adjacency_matrix(G)
    weight = As.data
    edge_color = ['#043476'] * len(weight)

    options = {
        'font_size': 9,
        'node_size': 300,
        'node_color': 'white',
        'edgecolors': '#043476',
        'linewidths': 1,
        'width': 1,
        "with_labels": True,
        'edge_color': edge_color
    }


    mapping = {}
    for i in range(len(Names)):
        mapping[i] = Names[i]
    G = nx.relabel_nodes(G, mapping)
    nx.draw(G, pos_mapping, **options)

    plt.axis('off')
    plt.title(title)
    if if_show:   plt.show()


def plot_leads_with_weights(correlation_matrix, title = None, threshold = 0.4,if_show = True, Colorbar=True):
    fig, ax = plt.subplots()

    df = pd.read_excel('channel19.xlsx', engine='openpyxl')
    names = df.channel
    Names = []
    for i in range(len(names)):

        temp_name = names[i]
        temp_name = temp_name.replace('.', '')
        Names.append(temp_name)

    list1 = np.zeros([19], dtype=np.int)
    list2 = np.zeros([19], dtype=np.float)
    for i in range(19):
        list1[i] = int(df.x[i])
        list2[i] = float(df.y[i])

    xy = np.zeros([19, 2])  # （19,2)
    for i in range(19):
        xy[i, 1] = math.cos(math.radians(list1[i])) * list2[i]
        xy[i, 0] = math.sin(math.radians(list1[i])) * list2[i]
    pos = {}
    pos_mapping = {}
    for i in range(len(xy)):
        pos[i] = tuple(xy[i])
        pos_mapping[Names[i]] = tuple(xy[i])

    np.fill_diagonal(correlation_matrix, 0)
    correlation_matrix = adjacency_matrix(correlation_matrix, threshold)
    correlation_matrix = np.matrix(correlation_matrix)
    G = nx.from_numpy_matrix(correlation_matrix)

    mapping = {}
    for i in range(len(Names)):
        mapping[i] = Names[i]
    G = nx.relabel_nodes(G, mapping)

    edge_color_list = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
    weight_list = [edge[2] for edge in edge_color_list]
    # edge_colors = [plt.cm.autumn((1. / max(weight_list)) * weight) for weight in weight_list]
    edge_colors = [plt.cm.Blues((1. / max(weight_list)) * (weight)) for weight in weight_list]

    options = {
        'font_size': 12,
        'node_size': 600,
        'node_color': 'white',
        'edgecolors': '#043476',
        'linewidths': 1,
        'width': 2,
        "with_labels": True,
        'edge_color': edge_colors,

    }
    nx.draw(G, pos_mapping, **options)
    if title != None:
        plt.title(title)

    pc = mpl.collections.PathCollection(weight_list, cmap=plt.cm.Blues)
    pc.set_array(edge_colors)
    if Colorbar:
        cax = fig.add_axes([0.92, 0.1, 0.02, ax.get_position().height])  # [left, bottom, width, height]
        plt.colorbar(pc,  cax)

    # plt.colorbar()
    if if_show:   plt.show()

def plot_leads_with_weights_for_subplot(correlation_matrix, title, threshold = 0.4):
    df = pd.read_excel('channel19.xlsx', engine='openpyxl')
    names = df.channel
    Names = []
    for i in range(len(names)):
        # print(names[i])
        temp_name = names[i]
        temp_name = temp_name.replace('.', '')
        Names.append(temp_name)
    # print('\n')
    # print(Names)
    list1 = np.zeros([19], dtype=np.int)
    list2 = np.zeros([19], dtype=np.float)
    for i in range(19):
        list1[i] = int(df.x[i])
        list2[i] = float(df.y[i])
    xy = np.zeros([19, 2])  # （19,2)
    for i in range(19):
        xy[i, 1] = math.cos(math.radians(list1[i])) * list2[i]
        xy[i, 0] = math.sin(math.radians(list1[i])) * list2[i]
    pos = {}
    pos_mapping = {}
    for i in range(len(xy)):
        pos[i] = tuple(xy[i])
        pos_mapping[Names[i]] = tuple(xy[i])

    np.fill_diagonal(correlation_matrix, 0)
    correlation_matrix = adjacency_matrix(correlation_matrix, threshold)
    correlation_matrix = np.matrix(correlation_matrix)
    G = nx.from_numpy_matrix(correlation_matrix)
    mapping = {}
    for i in range(len(Names)):
        mapping[i] = Names[i]
    G = nx.relabel_nodes(G, mapping)

    edge_color_list = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
    weight_list = [edge[2] for edge in edge_color_list]
    # edge_colors = [plt.cm.autumn((1. / max(weight_list)) * weight) for weight in weight_list]
    edge_colors = [plt.cm.Blues((1. / max(weight_list)) * (weight)) for weight in weight_list]

    options = {
        'font_size': 12,
        'node_size': 600,
        'node_color': 'white',
        'edgecolors': '#043476',
        'linewidths': 1,
        'width': 2,
        "with_labels": True,
        'edge_color': edge_colors,

    }
    nx.draw(G, pos_mapping, **options)

    pc = mpl.collections.PathCollection(weight_list, cmap=plt.cm.Blues)
    pc.set_array(edge_colors)
    return pc



def get_T_range(matrix):
    min_T = None
    max_T = None
    for T_i in np.linspace(0.01, 1, 100):
        correlation_matrix = adjacency_matrix(abs(matrix),threshold=T_i)
        np.fill_diagonal(correlation_matrix, 0)
        correlation_matrix = np.matrix(correlation_matrix)
        G = nx.from_numpy_matrix(correlation_matrix)
        f = nx.is_connected(G)

        if 1:
            G.remove_nodes_from(nx.isolates(G.copy()))
            d = dict(nx.degree(G))
            if len(d) == 0:
                break
            mean_node = sum(d.values()) / len(G.nodes)
            density = 2 * len(G.edges) / (len(G.nodes) * (len(G.nodes) - 1))
            if mean_node >= 2*math.log(len(matrix)) and density<= 0.5:
                if not min_T:
                    min_T = T_i
                else:
                    min_T = min(min_T, T_i)

                if not max_T:
                    max_T = T_i
                else:
                    max_T = max(max_T, T_i)
        else:
            break

    return min_T,max_T


def get_graph_features(G):
    G.remove_nodes_from(nx.isolates(G.copy()))
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        average_shortest_path_length = nx.average_shortest_path_length(G)
    else:
        diameter = 0
        average_shortest_path_length = 0

    d = dict(nx.degree(G))
    average_dgree = sum(d.values()) / len(G.nodes)
    local_efficiency = nx.local_efficiency(G)
    global_efficiency = nx.global_efficiency(G)
    average_clustering = nx.average_clustering(G)
    transitivity = nx.transitivity(G)
    # rich_club_coefficient = nx.rich_club_coefficient(G)

    closeness_centrality = nx.closeness_centrality(G)
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    degree_centrality = list(degree_centrality.values())
    closeness_centrality = list(closeness_centrality.values())
    betweenness_centrality = list(betweenness_centrality.values())

    degree_centrality = np.average(degree_centrality)
    closeness_centrality = np.average(closeness_centrality)
    betweenness_centrality = np.average(betweenness_centrality)


    return [
            diameter,
            average_shortest_path_length,
            average_dgree,
            local_efficiency,
            global_efficiency,
            average_clustering,
            transitivity,
            # rich_club_coefficient,
            degree_centrality,
            closeness_centrality,
            betweenness_centrality,
            # small_world
            ]

def get_degree_features(eeg_Correlation, threshold, mean_flag=True):
    degree_centrality_all = []
    betweenness_centrality_all = []
    degrees_all = []
    for name, value in eeg_Correlation.items():
        value = np.array(value)
        for i in range(len(value)):
            correlation_matrix = adjacency_matrix(abs(value[i]), threshold=threshold)
            np.fill_diagonal(correlation_matrix, 0)
            correlation_matrix = np.matrix(correlation_matrix)
            G = nx.from_numpy_matrix(correlation_matrix)
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            degrees = [val for (node, val) in G.degree()]
            # nodes = [node for (node, val) in G.degree()]
            degrees_all.append(degrees)
            temp_degree_centrality = [i for i in degree_centrality.values()]
            temp_betweenness_centrality = [i for i in betweenness_centrality.values()]
            degree_centrality_all.append(temp_degree_centrality)
            betweenness_centrality_all.append(temp_betweenness_centrality)
    degrees_all = np.array(degrees_all)
    degree_centrality_all = np.array(degree_centrality_all)
    betweenness_centrality_all = np.array(betweenness_centrality_all)
    if mean_flag:
        degrees_all = np.mean(degrees_all, axis=0)
        degree_centrality_all = np.mean(degree_centrality_all, axis=0)
        betweenness_centrality_all = np.mean(betweenness_centrality_all, axis=0)

    return degrees_all, degree_centrality_all, betweenness_centrality_all


def plot_nodes(degree_centrality,N=1400, title = 'title', show = True):
    node_list = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    G = nx.Graph()
    G.add_nodes_from(node_list)

    df = pd.read_excel('channel19.xlsx', engine='openpyxl')
    names = df.channel
    Names = []
    for i in range(len(names)):
        temp_name = names[i]
        temp_name = temp_name.replace('.', '')
        Names.append(temp_name)
    # print('\n')
    # print(Names)
    list1 = np.zeros([19], dtype=np.int)
    list2 = np.zeros([19], dtype=np.float)
    for i in range(19):
        list1[i] = int(df.x[i])
        list2[i] = float(df.y[i])
    xy = np.zeros([19, 2])
    for i in range(19):
        xy[i, 1] = math.cos(math.radians(list1[i])) * list2[i]
        xy[i, 0] = math.sin(math.radians(list1[i])) * list2[i]
    pos = {}
    pos_mapping = {}
    for i in range(len(xy)):
        pos[i] = tuple(xy[i])
        pos_mapping[Names[i]] = tuple(xy[i])

    sizes = N * degree_centrality
    options = {
        'font_size': 12, # 16
        'node_size': sizes,
        'node_color': 'white',
        'edgecolors': '#043476',
    }
    # plt.figure(figsize=(8,6))
    nx.draw_networkx(G,pos_mapping,**options)
    plt.axis('off')
    plt.title(title,fontsize=24,x=0.5,y=1.05)
    if show: plt.show()



def plot_cm(confusion, classes, title=None, normalize = False, show=True):
    '''
    eg:
    confusion = np.array([[97, 2,  0,  0, 1, 0],
                     [ 4, 94,  1,  21, 0, 0],
                     [ 3,  2, 95,  0, 0, 0],
                     [ 0,  0,  0, 98, 2, 0],
                     [ 3,  1,  0,  0,96, 0],
                     [ 0,  1,  3,  0, 6,90]])
    classes = ['A', 'B', 'C', 'D', 'E', 'F']     # 设置坐标轴显示列表
    '''
    plt.figure(figsize=(5, 5))

    plt.imshow(confusion, cmap=plt.cm.Blues)
    plt.colorbar(shrink=0.8)
    indices = range(len(confusion))
    plt.xticks(indices, classes, rotation=45)
    plt.yticks(indices, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = confusion.max() / 2.

    for i in range(len(confusion)):
        for j in range(len(confusion[i])):
            plt.text(j, i, format(confusion[i][j], fmt),
            fontsize=12,
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if confusion[i, j] > thresh else "black")

    if title: plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    if show: plt.show()

def get_upper_triangular_element(matrix):
    temp=[]
    n=len(matrix)
    for i in range(0,n):
        for j in range(i+1,n):
            temp.append(matrix[i,j])
    return temp
