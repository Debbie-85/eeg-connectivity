from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn import metrics
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.ensemble import RandomForestClassifier # RF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #LDA
from sklearn.naive_bayes import GaussianNB  # NB
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler  # 0-1
from functions import *

import random
random.seed(2023)
np.random.seed(2023)

def load_features_for_best_threshold(file_path = 'EEG_correlation.json', threshold = 0.4, mode = '123'):

    with open(file_path, 'r') as file:
        js_file = json.load(file)
    AD_eeg_Correlation = js_file['AD']
    MCI_eeg_Correlation = js_file['MCI']
    HC_eeg_Correlation = js_file['HC']

    # 读取数据以上三角元素作为特征值
    X_AD = []
    for name, value in AD_eeg_Correlation.items():
        value = np.array(value)
        for i in range(len(value)):
            # matrix_temp = get_upper_triangular_element(value[i])  # 上三角元素

            correlation_matrix = adjacency_matrix(abs(value[i]), threshold=threshold)  # 单个片段的邻接矩阵
            matrix_temp = get_upper_triangular_element(correlation_matrix)  # 上三角元素  171
            np.fill_diagonal(correlation_matrix, 0)  # fill_diagonal 没有返回值
            correlation_matrix = np.matrix(correlation_matrix)
            G = nx.from_numpy_matrix(correlation_matrix)

            node_feature_temp = [float(val) for (node, val) in G.degree()]  # 每个节点的度 19
            graph_feature_temp = get_graph_features(G)   # 网络特征 10

            # 需要所有都是list
            if mode == '1':
                feature_temp = matrix_temp   # 171
            if mode == '2':
                feature_temp = graph_feature_temp  # 10
            if mode == '3':
                feature_temp = node_feature_temp  # 19
            if mode == '12':
                feature_temp = matrix_temp + graph_feature_temp
            if mode == '13':
                feature_temp = matrix_temp + node_feature_temp
            if mode == '23':
                feature_temp = graph_feature_temp + node_feature_temp
            if mode == '123':
                feature_temp = matrix_temp + graph_feature_temp + node_feature_temp
            # feature_temp = np.array(feature_temp)
            X_AD.append(feature_temp)

    # X_AD = np.array(X_AD)
    # X_AD = np.concatenate(X_AD, axis=1)

    X_MCI = []
    for name, value in MCI_eeg_Correlation.items():
        value = np.array(value)
        for i in range(len(value)):
            # matrix_temp = get_upper_triangular_element(value[i])  # 上三角元素

            correlation_matrix = adjacency_matrix(abs(value[i]), threshold=threshold)  # 单个片段的邻接矩阵
            matrix_temp = get_upper_triangular_element(correlation_matrix)  # 上三角元素

            np.fill_diagonal(correlation_matrix, 0)  # fill_diagonal 没有返回值
            correlation_matrix = np.matrix(correlation_matrix)
            G = nx.from_numpy_matrix(correlation_matrix)
            node_feature_temp = [float(val) for (node, val) in G.degree()]  # 每个节点的度
            graph_feature_temp = get_graph_features(G)  # 网络特征


            # 需要所有都是list
            if mode == '1':
                feature_temp = matrix_temp
            if mode == '2':
                feature_temp = graph_feature_temp
            if mode == '3':
                feature_temp = node_feature_temp
            if mode == '12':
                feature_temp = matrix_temp + graph_feature_temp
            if mode == '13':
                feature_temp = matrix_temp + node_feature_temp
            if mode == '23':
                feature_temp = graph_feature_temp + node_feature_temp
            if mode == '123':
                feature_temp = matrix_temp + graph_feature_temp + node_feature_temp
            # feature_temp = np.array(feature_temp)
            X_MCI.append(feature_temp)
    # X_MCI = np.array(X_MCI)

    X_HC = []
    for name, value in HC_eeg_Correlation.items():
        value = np.array(value)
        for i in range(len(value)):
            # matrix_temp = get_upper_triangular_element(value[i])  # 上三角元素

            correlation_matrix = adjacency_matrix(abs(value[i]), threshold=threshold)  # 单个片段的邻接矩阵
            matrix_temp = get_upper_triangular_element(correlation_matrix)  # 上三角元素

            np.fill_diagonal(correlation_matrix, 0)  # fill_diagonal 没有返回值
            correlation_matrix = np.matrix(correlation_matrix)
            G = nx.from_numpy_matrix(correlation_matrix)
            node_feature_temp = [float(val) for (node, val) in G.degree()]  # 每个节点的度
            graph_feature_temp = get_graph_features(G)  # 网络特征

            # 需要所有都是list
            if mode == '1':
                feature_temp = matrix_temp
            if mode == '2':
                feature_temp = graph_feature_temp
            if mode == '3':
                feature_temp = node_feature_temp
            if mode == '12':
                feature_temp = matrix_temp + graph_feature_temp
            if mode == '13':
                feature_temp = matrix_temp + node_feature_temp
            if mode == '23':
                feature_temp = graph_feature_temp + node_feature_temp
            if mode == '123':
                feature_temp = matrix_temp + graph_feature_temp + node_feature_temp
            # feature_temp = np.array(feature_temp)
            X_HC.append(feature_temp)
    # X_HC = np.array(X_HC)

    np.save('X_AD_features_all_t_{}.npy'.format(threshold), np.array(X_AD))
    np.save('X_MCI_features_all_t_{}.npy'.format(threshold), np.array(X_MCI))
    np.save('X_HC_features_all_t_{}.npy'.format(threshold), np.array(X_HC))

    # Y
    Y_AD = np.ones(len(X_AD)) * 2
    Y_MCI = np.ones(len(X_MCI)) * 1
    Y_HC = np.ones(len(X_HC)) * 0

    X = X_AD + X_MCI + X_HC
    # X = np.concatenate(X,axis=1)
    X=np.array(X)
    Y = np.concatenate([Y_AD, Y_MCI, Y_HC], axis = 0)
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0
    return X, Y


def load_X_Y(threshold, mode = '123'):
    AD_path = 'X_AD_features_all_t_{}.npy'.format(str(threshold))
    MCI_path = 'X_MCI_features_all_t_{}.npy'.format(str(threshold))
    HC_path = 'X_HC_features_all_t_{}.npy'.format(str(threshold))

    X_AD = np.load(AD_path)
    X_MCI = np.load(MCI_path)
    X_HC = np.load(HC_path)
    if mode == '123':
        pass
    if mode == '1':
        X_AD = X_AD[:,0:171]
        X_MCI = X_MCI[:,0:171]
        X_HC = X_HC[:,0:171]
    if mode == '2':
        X_AD = X_AD[:,171:181]
        X_MCI = X_MCI[:,171:181]
        X_HC = X_HC[:,171:181]
    if mode == '3':
        X_AD = X_AD[:,181:]
        X_MCI = X_MCI[:,181:]
        X_HC = X_HC[:,181:]
    if mode == '12':
        X_AD = X_AD[:,0:181]
        X_MCI = X_MCI[:,0:181]
        X_HC = X_HC[:,0:181]
    if mode == '23':
        X_AD = X_AD[:,171:]
        X_MCI = X_MCI[:,171:]
        X_HC = X_HC[:,171:]
    if mode == '13':
        X_AD = np.concatenate([X_AD[:,0:171], X_AD[:,181:]], axis = 1)
        X_MCI = np.concatenate([X_MCI[:,0:171], X_MCI[:,181:]], axis = 1)
        X_HC = np.concatenate([X_HC[:,0:171], X_HC[:,181:]], axis = 1)

    # Y
    Y_AD = np.ones(len(X_AD)) * 2
    Y_MCI = np.ones(len(X_MCI)) * 1
    Y_HC = np.ones(len(X_HC)) * 0

    # X = X_AD + X_MCI + X_HC
    X = np.concatenate([X_AD, X_MCI, X_HC], axis=0)
    Y = np.concatenate([Y_AD, Y_MCI, Y_HC], axis = 0)
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0
    return X, Y

#### sort features ####
# mode = '123'
# for i in tqdm(np.linspace(0.0, 0.6, 13)):  # 以0.05为步长
#     # for i in tqdm(np.linspace(0.4, 0.6, 3)):  # 以0.01为步长
#     # if i == 0:
#     #     continue
#     # X, Y = load_features_for_best_threshold(threshold=i, mode = mode) # generate features
#     X, Y = load_X_Y(threshold=i, mode=mode)  # load features
#
#     index = np.arange(Y.shape[0])
#     np.random.shuffle(index)
#     np.random.shuffle(index)
#     X = X[index]
#     Y = Y[index]
#
#     scaler = MinMaxScaler()
#     X = scaler.fit_transform(X)
#     selector = SelectKBest(mutual_info_classif, k='all')
#     selector.fit(X, Y)
#     scores = selector.scores_
#     pvalues = selector.pvalues_
#
#     sorted_idx = np.argsort(scores)
#     np.save('sorted_idx_mode_{}_threshold_{}.npy'.format(mode, i), sorted_idx)


# model = SVC()
# model = KNeighborsClassifier()
# model = RandomForestClassifier()
model = LinearDiscriminantAnalysis()
# model = GaussianNB()

mode = '1'
Accuracy = []
Precision = []
Recall = []
F1 = []
print("\n feature：", mode, '，model：LDA')
for i in tqdm(np.linspace(0.0, 0.6, 13)):
    # if i == 0:
    #     continue  # just LDA
    # X, Y = load_features_for_best_threshold(threshold=i, mode = mode) # generate features
    X, Y = load_X_Y(threshold=i, mode=mode)  # load features
    sorted_idx_path = 'sorted_idx_mode_{}_threshold_{}.npy'.format(mode, i)
    sorted_idx = np.load(sorted_idx_path)
    X = X[:, sorted_idx]

    index = np.arange(Y.shape[0])
    np.random.shuffle(index)
    np.random.shuffle(index)
    X = X[index]
    Y = Y[index]


    from sklearn.preprocessing import MinMaxScaler  # 0-1
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    Accuracy_temp = []
    Precision_temp = []
    Recall_temp = []
    F1_temp = []
    for j in range(len(X[0])):
        X_temp = X.copy()[:, :j+1]
        y_pre = cross_val_predict(model, X_temp, Y, cv=5)
        accuracy = metrics.accuracy_score(y_true=Y, y_pred=y_pre)
        precision = metrics.precision_score(y_true=Y, y_pred=y_pre, average='weighted')
        recall = metrics.recall_score(y_true=Y, y_pred=y_pre, average='weighted')
        f1 = metrics.f1_score(y_true=Y, y_pred=y_pre, average='weighted')
        Accuracy_temp.append(accuracy)
        Precision_temp.append(precision)
        Recall_temp.append(recall)
        F1_temp.append(f1)
    Accuracy.append(Accuracy_temp)
    Precision.append(Precision_temp)
    Recall.append(Recall_temp)
    F1.append(F1_temp)

metric = {
    'Accuracy': Accuracy,
    'Precision': Precision,
    'Recall': Recall,
    'F1': F1,
}
filename = 'metric_mode_{}_LDA.json'.format(mode)
with open(filename,'w') as f:
    json.dump(metric,f)



