import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

##############################################################
print(50 * '#')
print("Step 1: Get datasets")
print(50 * '#')

# get dataset from http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz
NET_DATA_PATH = os.path.join("../datasets")


# get data from file
def load_datasets(data_path=NET_DATA_PATH):
    # 41 features , 1 label
    feature_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
                     "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                     "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                     "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                     "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                     "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                     "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                     "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                     "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                     "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
    data_file = os.path.join(data_path, "datasets_10")
    return pd.read_csv(data_file, header=None, names=feature_names)


def plot_features(X):
    # plot data
    # for index, classes in X.groupby('label'):
    #     print('class_{}'.format(index))
    #     print(classes)

    # attack_data = X[X['label'] == 1]                                  #for dataframe
    # normal_data = X[X['label'] == -1]                                 #for dataframe
    attack_data = X[X[:, -1] > 0]  # numpy
    normal_data = X[X[:, -1] < 0]  # numpy
    # print(attack_data)
    # print(normal_data)
    #
    # plt.scatter(attack_data.values[:200, 0], attack_data.values[:200, 1],         #dataframe
    #             color='red', marker='o', label='attack')
    # plt.scatter(normal_data.values[:200, 0], normal_data.values[:200, 1],            #dataframe
    #             color='blue', marker='x', label='normal')

    plt.scatter(attack_data[:, 0], attack_data[:, 1],  # numpy
                color='red', marker='o', label='attack traffics')
    plt.scatter(normal_data[:, 0], normal_data[:, 1],  # numpy
                color='blue', marker='x', label='normal traffics')

    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.legend(loc='upper left')

    plt.show()


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=cl)


data = load_datasets()
data = data.iloc[:25000, :]

# select top 2 features: srv_count, num_compromised
# data = datasets[['srv_count', 'num_compromised', 'label']]


# print(data.head())
# print(data.tail())

# print(data.info())

print(data['label'].value_counts())

##############################################################

print(50 * '#')
print("Step 2: Preprocessing")
print(50 * '#')

print(50 * '#')
print("Step 2.1: Binarization for label")
print(50 * '#')

# get label, last column, convert it to binary classification label
ori_labels = data.iloc[:,-1].values
data.iloc[:, -1] = np.where(data.iloc[:, -1] == 'normal.', -1, 1)
labels = data.iloc[:, -1].values
# labels = last_column.copy()
# labels[labels != 'normal.'] = 'attack.'               # dataframe
# print(labels.value_counts())                          # dataframe
# print(labels)

print(50 * '#')
print("Step 2.2: Encoding Categorical Features")
print(50 * '#')

# get features, Encoding Categorical Features for 3 features
data.sort_values('label')
enc = LabelEncoder()
data['service'] = enc.fit_transform(data['service'])
data['flag'] = enc.fit_transform(data['flag'])
data['protocol_type'] = enc.fit_transform(data['protocol_type'])

# features.astype(float)
# print(features.info())
# print(features['service'])

print(50 * '#')
print("Step 2.3: Standardization")
print(50 * '#')
std_data = StandardScaler().fit(data).transform(data)
# print(std_features)

print(50 * '#')
print("Step 2.4: Normalization")
print(50 * '#')
normalized_data = Normalizer().fit(std_data).transform(std_data)

# plot_features(normalized_data)

print(50 * '#')
print("Step 2.5: Training And Test Data")
print(50 * '#')
# features = normalized_data.drop('label', axis=1)                  #dataframe
features = np.delete(normalized_data, -1, axis=1)
train_data, test_data, train_labels, test_labels = train_test_split(features, labels, test_size=0.3,
                                                                    random_state=24)

print(50 * '#')
print("Step 3: Training")
print(50 * '#')

print(50 * '#')
print("Step 3.1: Training with LR")
print(50 * '#')

lr = LogisticRegression(C=1.0, penalty='l1', random_state=0)
lr.fit(train_data, train_labels)

# plot_decision_regions(train_data, train_labels, classifier=lr)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend(loc='upper left')
#
# plt.tight_layout()
# plt.show()

print(50 * '#')
print("Step 3.2: Training with linear SVC")
print(50 * '#')

svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=100.0)
# svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(train_data, train_labels)

#
# plot_decision_regions(train_data, train_labels, classifier=svm)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend(loc='upper left')
#
# plt.tight_layout()
# plt.show()

print(50 * '#')
print("Step 3.1: Feature selection")
print(50 * '#')

print(50 * '#')
print("Step 3.1.1: Feature selection with L1 regularization")
print(50 * '#')
# print('Training accuracy:', lr.score(train_data, train_labels))
# print('Test accuracy:', lr.score(train_data, train_labels))
# print('Intercept:', lr.intercept_)
# print('Model weights:', lr.coef_)

print(50 * '#')
print("Step 3.1.2: Feature importance with Random Forest")
print(50 * '#')
names = data.columns[1:]
forest = RandomForestClassifier(n_estimators=10,
                                random_state=0,
                                n_jobs=-1)
forest.fit(train_data, train_labels)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(train_data.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            names[indices[f]],
                            importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(train_data.shape[1]),
        importances[indices],
        color='lightblue',
        align='center')

plt.xticks(range(train_data.shape[1]),
           names[indices], rotation=90)
plt.xlim([-1, train_data.shape[1]])
plt.tight_layout()
plt.show()

print(50 * '#')
print("Step 4: Prediction")
print(50 * '#')

pred_labels = lr.predict(test_data)
# print(pred_labels)

pred_labels_rf = forest.predict(test_data)
pred_labels_svc = svm.predict(test_data)

print(50 * '#')
print("Step 5: Evaluation")
print(50 * '#')

print(50 * '#')
print("Step 5.1: Evaluation for LogisticRegression")
print(50 * '#')

print('Accuracy for LogisticRegression: %.2f' % accuracy_score(test_labels, pred_labels))
print(classification_report(test_labels, pred_labels))
print(confusion_matrix(test_labels, pred_labels))

print(50 * '#')
print("Step 5.2: Evaluation for RandomForest")
print(50 * '#')

print('Accuracy for Random Forest: %.2f' % accuracy_score(test_labels, pred_labels_rf))
print(classification_report(test_labels, pred_labels_rf))
confmat = confusion_matrix(test_labels, pred_labels_rf)
print(confmat)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.show()

print(50 * '#')
print("Step 5.3: Evaluation for SVC")
print(50 * '#')

print('Accuracy for Random Forest: %.2f' % accuracy_score(test_labels, pred_labels_rf))
print(classification_report(test_labels, pred_labels_rf))
print(confusion_matrix(test_labels, pred_labels_svc))

print(50 * '#')
print("Step 6: Dimensional reduction")
print(50 * '#')

pca = PCA(n_components=2)
pca_train_data = pca.fit_transform(train_data)
pca_test_data = pca.transform(test_data)  # no need fit again ?
print('Variance explained ratio:\n', pca.explained_variance_ratio_)

print(50 * '#')
print("Step 7: Re-training")
print(50 * '#')

print(50 * '#')
print("Step 7.1: Re-training: LR")
print(50 * '#')

lr = LogisticRegression(C=0.1, random_state=1)
lr.fit(pca_train_data, train_labels)

plot_decision_regions(pca_train_data, train_labels, classifier=lr)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

print(50 * '#')
print("Step 7.2: Re-training: RF")
print(50 * '#')

forest = RandomForestClassifier(n_estimators=10,
                                random_state=1,
                                n_jobs=-1)
forest.fit(pca_train_data, train_labels)

plot_decision_regions(pca_train_data, train_labels, classifier=forest)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

print(50 * '#')
print("Step 7.3: Re-training: SVC")
print(50 * '#')

svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=0.1)
svm.fit(pca_train_data, train_labels)

plot_decision_regions(pca_train_data, train_labels, classifier=svm)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

print(50 * '#')
print("Step 8: Re-Prediction")
print(50 * '#')

print(50 * '#')
print("Step 8.1: Re-Prediction: LR")
print(50 * '#')

pca_pred_labels = lr.predict(pca_test_data)

print(50 * '#')
print("Step 8.2: Re-Prediction: RF")
print(50 * '#')

pca_pred_labels_rf = forest.predict(pca_test_data)

print(50 * '#')
print("Step 8.3: Re-Prediction: SVC")
print(50 * '#')

pca_pred_labels_svc = svm.predict(pca_test_data)

print(50 * '#')
print("Step 9: Re-Evaluation")
print(50 * '#')

print(50 * '#')
print("Step 9.1: Re-Evaluation for LR")
print(50 * '#')
print('Accuracy for LR: %.2f' % accuracy_score(test_labels, pca_pred_labels))
print(classification_report(test_labels, pca_pred_labels))
print(confusion_matrix(test_labels, pca_pred_labels))

print(50 * '#')
print("Step 9.2: Re-Evaluation for RF")
print(50 * '#')

print('Accuracy for RF: %.2f' % accuracy_score(test_labels, pca_pred_labels_rf))
print(classification_report(test_labels, pca_pred_labels_rf))
print(confusion_matrix(test_labels, pca_pred_labels_rf))

print(50 * '#')
print("Step 9.3: Re-Evaluation for SVC")
print(50 * '#')

print('Accuracy for RF: %.2f' % accuracy_score(test_labels, pca_pred_labels_svc))
print(classification_report(test_labels, pca_pred_labels_svc))
print(confusion_matrix(test_labels, pca_pred_labels_svc))

print(50 * '#')
print("Step 10: Eigendecomposition")
print(50 * '#')

# cov_mat = np.cov(train_data.T)
# eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
#
# print('\nEigenvalues \n%s' % eigen_vals)
#
# tot = sum(eigen_vals)
# var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
# cum_var_exp = np.cumsum(var_exp)
#
# plt.bar(range(1, 42), var_exp, alpha=0.5, align='center',
#         label='individual explained variance')
# plt.step(range(1, 42), cum_var_exp, where='mid',
#          label='cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal components')
# plt.legend(loc='best')
# plt.show()

print(50 * '#')
print("Step 11: Clustering")
print(50 * '#')

# plot_features(pca_train_data)

k = 20
km = KMeans(n_clusters=k)
km.fit(train_data)
print(pd.Series(km.labels_).value_counts())

label_names = list(map(
    lambda x: pd.Series([ori_labels[i] for i in range(len(km.labels_)) if km.labels_[i] == x]),
    range(k)))

for i in range(k):
    print("Cluster {} labels:".format(i))
    print(label_names[i].value_counts())
