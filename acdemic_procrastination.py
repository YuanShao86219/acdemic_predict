
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import  recall_score, accuracy_score, f1_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import AdaBoostClassifier
import matplotlib as mpl

# read data
path = r'/Users/mllab/Desktop/DATA/整理資料/procrastination_MAX_3_dummy.csv'
seed = 55688
df =  pd.read_csv(path,
                  header = None,
                  encoding = 'utf-8')
df.head()
x = df.iloc[1:500,[3,4,5,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29]] 
y = df.iloc[1:500,[20]]
print("class labels:", np.unique(y)) #0=low learnig risk, 1=high learning risk

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=seed, shuffle=True, stratify=y)

# change to np array
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)



# build base learner of hard voting
# get parameter by RandomSearchCV 
clf1 = LogisticRegression(penalty='l2', C=10, solver='lbfgs', tol = 1e-5, random_state=seed,max_iter=1000)
clf2 = RandomForestClassifier(max_depth=8, criterion='entropy', random_state=seed, n_estimators=300, max_features=6, min_samples_split=2)
clf3 = KNeighborsClassifier(n_neighbors=3, metric='cosine',algorithm="auto", leaf_size=30, weights="distance")
clf4 = SVC(kernel='rbf', random_state=seed, gamma=0.01, C=50000, tol=1e-4)
clf5 = LDA(n_components=1, tol=1e-5, solver='svd')
clf1.fit(x_train,y_train.ravel())
clf2.fit(x_train,y_train.ravel())
clf3.fit(x_train,y_train.ravel())
clf4.fit(x_train,y_train.ravel())
clf5.fit(x_train,y_train.ravel())

# build base learner of soft voting 
soft_clf1 = LogisticRegression(penalty='l2', C=10, solver='lbfgs', tol = 1e-5, random_state=seed,max_iter=1000)
soft_clf2 = RandomForestClassifier(max_depth=8, criterion='entropy', random_state=seed, n_estimators=300, max_features=6, min_samples_split=2)
soft_clf3 = KNeighborsClassifier(n_neighbors=3, metric='cosine',algorithm="auto", leaf_size=30, weights="distance")
soft_clf4 = SVC(kernel='rbf', random_state=seed, gamma=0.01, C=50000, tol=1e-4,probability=True)
soft_clf5 = LDA(n_components=1, tol=1e-5, solver='svd')
soft_clf1.fit(x_train,y_train.ravel())
soft_clf2.fit(x_train,y_train.ravel())
soft_clf3.fit(x_train,y_train.ravel())
soft_clf4.fit(x_train,y_train.ravel())
soft_clf5.fit(x_train,y_train.ravel())

# hard voting
voting = VotingClassifier(estimators=[('LR', clf1),
                            ('Random Forest', clf2),
                            ('KNN', clf3),
                            ('SVM', clf4),                            
                            ('LDA',clf5)
                            ])


voting.fit(x_train, y_train.ravel())


prediction_1=clf1.predict(x_test)
prediction_2=clf2.predict(x_test)
prediction_3=clf3.predict(x_test)
prediction_4=clf4.predict(x_test)
prediction_5=clf5.predict(x_test)
hard_predictions = voting.predict(x_test)

# Virsualize hard voting training set classitify error
mpl.style.use('seaborn-paper')
train_prediction_1=clf1.predict(x_train)
train_prediction_2=clf2.predict(x_train)
train_prediction_3=clf3.predict(x_train)
train_prediction_4=clf4.predict(x_train)
train_prediction_5=clf5.predict(x_train)

error_1 = error = pd.to_numeric(np.asarray(y_train).ravel()) - pd.to_numeric(train_prediction_1)
error_2 = error = pd.to_numeric(np.asarray(y_train).ravel()) - pd.to_numeric(train_prediction_2)
error_3 = error = pd.to_numeric(np.asarray(y_train).ravel()) - pd.to_numeric(train_prediction_3)
error_4 = error = pd.to_numeric(np.asarray(y_train).ravel()) - pd.to_numeric(train_prediction_4)
error_5 = error = pd.to_numeric(np.asarray(y_train).ravel()) - pd.to_numeric(train_prediction_5)
x_axis1=[]
y_axis1=[]
for i in range(len(error_1)):
    if not error_1[i] == 0:
        x_axis1.append(i)
        y_axis1.append(error_1[i])
plt.scatter(x_axis1, y_axis1, marker='*', s=60, label='LR', alpha=.8)
x_axis1=[]
y_axis1=[]
for i in range(len(error_2)):
    if not error_2[i] == 0:
        x_axis1.append(i)
        y_axis1.append(error_2[i])
plt.scatter(x_axis1, y_axis1,  marker='o', s=60, label='RF', alpha=.5)
x_axis1=[]
y_axis1=[]
for i in range(len(error_3)):
    if not error_3[i] == 0:
        x_axis1.append(i)
        y_axis1.append(error_3[i])
plt.scatter(x_axis1, y_axis1, marker='D', s=60, label='KNN', alpha=.4)
x_axis1=[]
y_axis1=[]
for i in range(len(error_4)):
    if not error_4[i] == 0:
        x_axis1.append(i)
        y_axis1.append(error_4[i])
plt.scatter(x_axis1, y_axis1,  marker='x', s=60, label='SVM', alpha=.5)
x_axis1=[]
y_axis1=[]
for i in range(len(error_5)):
    if not error_5[i] == 0:
        x_axis1.append(i)
        y_axis1.append(error_5[i])
plt.scatter(x_axis1, y_axis1,  marker='^', s=60, label='LDA', alpha=.5)
plt.title('Learner error')
plt.xlabel('Test sample')
plt.ylabel('Error')
plt.legend()
plt.show()



# Soft Voting
SoftVoting = VotingClassifier(estimators=[('LR', soft_clf1),
                            ('Random Forest', soft_clf2),
                            ('KNN', soft_clf3),
                            ('SVM', soft_clf4),                           
                            ('LDA',soft_clf5)
                            ],
                           voting='soft'
                           )
SoftVoting.fit(x_train,y_train.ravel())
soft_predictions = SoftVoting.predict(x_test)


# print result
print('-----------------training set-----------------')
print('-----------------accuracy-----------------')
print('Logistic Regression acc:', accuracy_score(y_train, clf1.predict(x_train)))
print('Random Forest acc:', accuracy_score(y_train, clf2.predict(x_train)))
print('KNN acc:', accuracy_score(y_train, clf3.predict(x_train)))
print('SVM acc:', accuracy_score(y_train, clf4.predict(x_train)))
print('LDA acc:', accuracy_score(y_train, clf5.predict(x_train)))
print('Hard Voting acc:', accuracy_score(y_train, voting.predict(x_train)))
print('Soft Voting acc:', accuracy_score(y_train, SoftVoting.predict(x_train)))
print('-----------------recall-----------------')
print('Logistic Regression recall:', recall_score(y_train, clf1.predict(x_train),pos_label='1'))
print('Random Forest recall:', recall_score(y_train, clf2.predict(x_train),pos_label='1'))
print('KNN recall:', recall_score(y_train, clf3.predict(x_train) ,pos_label='1'))
print('SVM recall:', recall_score(y_train, clf4.predict(x_train),pos_label='1'))
print('LDA recall:', recall_score(y_train, clf5.predict(x_train),pos_label='1'))
print('Hard Voting recall:', recall_score(y_train, voting.predict(x_train), pos_label='1'))
print('Soft Voting recall:', recall_score(y_train, SoftVoting.predict(x_train), pos_label='1'))
print('-----------------precision-----------------')
print('Logistic Regression precision:', precision_score(y_train, clf1.predict(x_train),pos_label='1'))
print('Random Forest precision:', precision_score(y_train, clf2.predict(x_train),pos_label='1'))
print('KNN precision:', precision_score(y_train, clf3.predict(x_train) ,pos_label='1'))
print('SVM precision:', precision_score(y_train, clf4.predict(x_train),pos_label='1'))
print('LDA precision:', precision_score(y_train, clf5.predict(x_train),pos_label='1'))
print('Hard Voting precision:', precision_score(y_train, voting.predict(x_train), pos_label='1'))
print('Soft Voting precision:', precision_score(y_train, SoftVoting.predict(x_train), pos_label='1'))
print('-----------------F1-----------------')
print('Logistic Regression F1:', f1_score(y_train, clf1.predict(x_train),pos_label='1'))
print('Random Forest F1:', f1_score(y_train, clf2.predict(x_train),pos_label='1'))
print('KNN F1:', f1_score(y_train, clf3.predict(x_train) ,pos_label='1'))
print('SVM F1:', f1_score(y_train, clf4.predict(x_train),pos_label='1'))
print('LDA F1:', f1_score(y_train, clf5.predict(x_train),pos_label='1'))
print('Hard Voting F1:', f1_score(y_train, voting.predict(x_train), pos_label='1'))
print('Soft Voting F1:', f1_score(y_train, SoftVoting.predict(x_train), pos_label='1'))
print('-----------------testing set-----------------')
print('-----------------accuracy-----------------')
print('Logistic Regression acc:', accuracy_score(y_test, clf1.predict(x_test)))
print('Random Forest acc:', accuracy_score(y_test, clf2.predict(x_test)))
print('KNN acc:', accuracy_score(y_test, clf3.predict(x_test)))
print('SVM acc:', accuracy_score(y_test, clf4.predict(x_test)))
print('LDA acc:', accuracy_score(y_test, clf5.predict(x_test)))
print('Hard Voting acc:', accuracy_score(y_test, voting.predict(x_test)))
print('Soft Voting acc:', accuracy_score(y_test, SoftVoting.predict(x_test)))
print('-----------------recall-----------------')
print('Logistic Regression recall:', recall_score(y_test, clf1.predict(x_test),pos_label='1'))
print('Random Forest recall:', recall_score(y_test, clf2.predict(x_test),pos_label='1'))
print('KNN recall:', recall_score(y_test, clf3.predict(x_test) ,pos_label='1'))
print('SVM recall:', recall_score(y_test, clf4.predict(x_test),pos_label='1'))
print('LDA recall:', recall_score(y_test, clf5.predict(x_test),pos_label='1'))
print('Hard Voting recall:', recall_score(y_test, voting.predict(x_test), pos_label='1'))
print('Soft Voting recall:', recall_score(y_test, SoftVoting.predict(x_test), pos_label='1'))
print('-----------------precision-----------------')
print('Logistic Regression precision:', precision_score(y_test, clf1.predict(x_test),pos_label='1'))
print('Random Forest precision:', precision_score(y_test, clf2.predict(x_test),pos_label='1'))
print('KNN precision:', precision_score(y_test, clf3.predict(x_test) ,pos_label='1'))
print('SVM precision:', precision_score(y_test, clf4.predict(x_test),pos_label='1'))
print('LDA precision:', precision_score(y_test, clf5.predict(x_test),pos_label='1'))
print('Hard Voting precision:', precision_score(y_test, voting.predict(x_test), pos_label='1'))
print('Soft Voting precision:', precision_score(y_test, SoftVoting.predict(x_test), pos_label='1'))
print('-----------------F1-----------------')
print('Logistic Regression F1:', f1_score(y_test, clf1.predict(x_test),pos_label='1'))
print('Random Forest F1:', f1_score(y_test, clf2.predict(x_test),pos_label='1'))
print('KNN F1:', f1_score(y_test, clf3.predict(x_test) ,pos_label='1'))
print('SVM F1:', f1_score(y_test, clf4.predict(x_test),pos_label='1'))
print('LDA F1:', f1_score(y_test, clf5.predict(x_test),pos_label='1'))
print('Hard Voting F1:', f1_score(y_test, voting.predict(x_test), pos_label='1'))
print('Soft Voting F1:', f1_score(y_test, SoftVoting.predict(x_test), pos_label='1'))


# print cofusion matrix
confmat_train_HardVoting = confusion_matrix(y_true=y_train, y_pred=voting.predict(x_train),labels=['1','0'])
print('confusion matrix of HardVoting \n',confmat_train_HardVoting)
confmat_test_HardVoting = confusion_matrix(y_true=y_test, y_pred=hard_predictions,labels=['1','0'])
print('confusion matrix of HardVoting \n',confmat_test_HardVoting)
confmat_train_SoftVoting = confusion_matrix(y_true=y_train, y_pred=SoftVoting.predict(x_train),labels=['1','0'])
print('confusion matrix of SoftVoting \n',confmat_train_SoftVoting)
confmat_test_SoftVoting = confusion_matrix(y_true=y_test, y_pred=soft_predictions,labels=['1','0'])
print('confusion matrix of SoftVoting \n',confmat_test_SoftVoting)




# Virsualize soft voting classitify error
soft_clf1.fit(x_train,y_train.ravel())
soft_clf2.fit(x_train,y_train.ravel())
soft_clf3.fit(x_train,y_train.ravel())
soft_clf4.fit(x_train,y_train.ravel())
soft_clf5.fit(x_train,y_train.ravel())
error = pd.to_numeric(np.asarray(y_test).ravel()) - pd.to_numeric(soft_predictions)
probabiliities_LR = soft_clf1.predict_proba(x_train)
probabiliities_RF = soft_clf2.predict_proba(x_train)
probabiliities_KNN = soft_clf3.predict_proba(x_train)
probabiliities_SVM = soft_clf4.predict_proba(x_train)
probabiliities_LDA = soft_clf5.predict_proba(x_train)


x_1=[]
y_1=[]
y_2=[]
y_3=[]
y_4=[]
y_5=[]
y_6=[]
y_7=[]

y_avg=[]

for i in range (len(error)):
    if not error[i] == 0:
        x_1.append(i)
        y_1.append(probabiliities_LR[i][0])
        y_2.append(probabiliities_RF[i][0])
        y_3.append(probabiliities_KNN[i][0])
        y_4.append(probabiliities_SVM[i][0])
        y_5.append(probabiliities_LDA[i][0])
        y_s = probabiliities_LR[i][0] + probabiliities_RF[i][0]
        y_s = y_s + probabiliities_KNN[i][0]
        y_s = y_s + probabiliities_SVM[i][0]
        y_s = y_s + probabiliities_LDA[i][0]
        y_avg.append(y_s/5)

plt.figure(figsize=(10,10))
plt.scatter(x_1, y_1, marker='*', c='k', label='LR', zorder = 10)
plt.scatter(x_1, y_2, marker='o', c='k', label='RF', zorder = 10)
plt.scatter(x_1, y_3, marker='.', c='k', label='KNN', zorder = 10)
plt.scatter(x_1, y_4, marker='x', c='k', label='SVM', zorder = 10)
plt.scatter(x_1, y_5, marker='1', c='k', label='LDA', zorder = 10)
plt.scatter(x_1, y_avg, marker='3', c='red' , label='Average Positive', zorder = 10, s=150)
y_axis=[0.5 for x in range(len(error))]
plt.plot(y_axis, c='k',)
plt.title('Positive Probability')
plt.xlabel('Test sample')
plt.ylabel('probability')
plt.legend()
plt.show()












# Stacking
# append base learner
base_learner = []
base_learner.append(clf1) #LR
base_learner.append(clf2)
base_learner.append(clf3)
base_learner.append(soft_clf4) #SVM
base_learner.append(clf5)

# build meta learner
meta_learner =  AdaBoostClassifier(
                              n_estimators = 6500,
                              random_state = seed, learning_rate=0.1
                              )
meta_data = np.zeros((len(base_learner), len(x_train)))
meta_targets = np.zeros(len(x_train))
KF = KFold(n_splits=10)
index = 0

for train_indices, test_indices in KF.split(x_train):
    for i in range(len(base_learner)):
        learner = base_learner[i]
        learner.fit(x_train[train_indices], y_train[train_indices].ravel())
        p = learner.predict_proba(x_train[test_indices])[:,0]

        meta_data[i][index:index + len(test_indices)] = p

    meta_targets[index:index + len(test_indices)] = y_train[test_indices].ravel()
    index += len(test_indices)


meta_data = meta_data.transpose()


test_meta_data = np.zeros((len(base_learner), len(x_test)))
base_acc = []

for i in range(len(base_learner)):
    b=base_learner[i]
    b.fit(x_train, y_train.ravel())
    predictions = b.predict_proba(x_test)[:,0]
    test_meta_data[i]=predictions
    acc = metrics.accuracy_score(y_test, b.predict(x_test))
    
    base_acc.append(acc)
test_meta_data = test_meta_data.transpose()



meta_learner.fit(meta_data, meta_targets)
ensemble_predictions = meta_learner.predict(test_meta_data)
ensemble_train = meta_learner.predict(meta_data)
acc_meta_train = metrics.accuracy_score(pd.to_numeric(y_train.ravel()), ensemble_train)
acc_meta = metrics.accuracy_score(pd.to_numeric(y_test.ravel()), ensemble_predictions)


# print stacking result
print('STACKING')
print('-'*20)
print(f'{acc_meta_train: .3f} Meta Ensemble train')
print('stacking train recall: ',recall_score(pd.to_numeric(y_train.ravel()), y_pred = meta_learner.predict(meta_data),pos_label=1))
print('stacking train precision: ',precision_score(pd.to_numeric(y_train.ravel()), y_pred = meta_learner.predict(meta_data),pos_label=1))
print('stacking train f1: ',f1_score(pd.to_numeric(y_train.ravel()), y_pred = meta_learner.predict(meta_data),pos_label=1))
print(f'{acc_meta: .3f} Meta Ensemble test')
print('stacking test recall: ',recall_score(pd.to_numeric(y_test.ravel()),  y_pred=ensemble_predictions, pos_label=1))
print('stacking test precision: ',precision_score(pd.to_numeric(y_test.ravel()), y_pred=ensemble_predictions, pos_label=1))
print('stacking test f1: ',f1_score(pd.to_numeric(y_test.ravel()), y_pred=ensemble_predictions, pos_label=1))
confmat_train_Stacking = confusion_matrix(pd.to_numeric(y_train.ravel()), y_pred = meta_learner.predict(meta_data))
confmat_test_Stacking = confusion_matrix(pd.to_numeric(y_test.ravel()), y_pred=ensemble_predictions)
print('confusion matrix of Ensemble(train) \n',confmat_train_Stacking)
print('confusion matrix of Ensemble(test) \n',confmat_test_Stacking) 



# feature selection
# correlation 
# 這邊跑出來是共變異數矩陣，後來我匯出data(x_train與y_train合併)用R跑相關係數
# data = df.iloc[1:500,[20,3,4,5,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29]]
data = np.hstack([y_train,x_train])
data = data.astype(float)
cor = np.corrcoef(data.T)
print('-----------------cor-----------------')  
print(cor)

# MI
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
def mi_function(x,y):
    return mutual_info_classif(x, y, random_state=seed, n_neighbors=3, discrete_features='auto')
select_k = 10
selection_MI = SelectKBest(mi_function,k=select_k).fit(x_train,y_train)
features_select_MI = selection_MI.scores_
feature_name_MI = selection_MI.get_feature_names_out()
print('-----------------MI-----------------')
print(feature_name_MI)
print(features_select_MI)

# gini
from sklearn.ensemble import GradientBoostingClassifier
gini = GradientBoostingClassifier(random_state=seed)
gini.fit(x_train,y_train)
feature_importance = list(gini.feature_importances_)

print('-----------------gini-----------------')
print(feature_importance)

# f-test
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest

select_k = 10

selection_f = SelectKBest(f_classif, k=select_k).fit(x_train, y_train)
features_select_f = selection_f.scores_
feature_name_f = selection_f.get_feature_names_out()
print('-----------------F-test-----------------')
print(feature_name_f)
print(features_select_f)

