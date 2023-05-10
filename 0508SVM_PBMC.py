#以下是以PBMC data預測是否罹患SLE的分類式機器學習模型的範例程式碼，
#包含資料分割、模型建立、梯度下降法訓練、以及overfitting處理
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn import metrics
import matplotlib.pyplot as plt

### Data preprocessing 
## 1.Data Cleaning> 處理缺失值、異常值
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error
from math import sqrt

# 嘗試將Th, Tc, Treg皆納入> 欄數擴增 / 加入新資料列>列數擴增
df_T1 = pd.read_excel("E:\國防醫學院 醫學系\課業\lab\風城資料T cell-processed.xlsx", sheet_name='T_SLE',index_col=None, header=None)
df_T0 = pd.read_excel("E:\國防醫學院 醫學系\課業\lab\風城資料T cell-processed.xlsx", sheet_name='T_HC', header=None)

df_T1 = df_T1.drop(df_T1.columns[0], axis=1)
df_T0 = df_T0.drop(df_T0.columns[0], axis=1)
# print(df_T0.shape, df_T1.shape)

df_T1 = df_T1.transpose().values # df轉置!  ??轉置後header無法變??
df_T0 = df_T0.T.values
df_T1 = pd.DataFrame(df_T1[1:], columns = df_T1[0])  #將nparray轉成df
df_T0 = pd.DataFrame(df_T0[1:], columns = df_T0[0])
# print(df_T1)
# print(df_T0)
df_T_both = pd.concat([df_T1, df_T0], axis=0)  #資料列合併
# 將缺少標示 '是否SLE' 的人剔除**
# print(df_T_both.columns)

#這裡的 SLE? 參數應該不能用差捕的，因為是監督式學習，必須知道他有沒有SLE
df_T_both[df_T_both['SLE?'].notnull()]  #或df_T_both[df_T_both['SLE?'].notna()]，但是不能df_T_both[df_T_both['SLE?']  is not None] QQ

# excel dataset資料清洗 1.缺資料者把該行特徵刪除(上次只用27參數) 2.將雜亂標記轉換為id, SLE, features
# print(df_T[1])
column = ['姓名', 'patient_ID', '收件日期', 'SLE?']
df_Tx = df_T_both.drop(column, axis=1)
df_Ty = df_T_both['SLE?'] 
# print('X', X.head())

### Data preprocessing 
## 1.Data Cleaning> 處理缺失值、異常值
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error
from math import sqrt

# 刪除缺失值較多的特徵或病人
df_Tx = df_Tx.dropna(thresh=len(df_Tx)*0.5, axis=1) # 刪除缺失值佔一半以上的特徵
df_Tx = df_Tx.dropna(thresh=len(df_Tx.columns)*0.5, axis=0) # 刪除缺失值佔一半以上的病人

# 線性回歸填補缺失值??
lr_imputer = SimpleImputer(strategy='mean')
df_Tx_lr = lr_imputer.fit_transform(df_Tx)

# MICE方法填補缺失值 (多重差補法?)
mice_imputer = IterativeImputer()
df_Tx_mice = mice_imputer.fit_transform(df_Tx)

# 評估線性回歸填補後的數據
X_lr = df_Tx_lr[:, :]
y_lr = df_Ty 
lr_model = LinearRegression()
lr_model.fit(X_lr, y_lr)
y_pred_lr = lr_model.predict(X_lr)
rmse_lr = sqrt(mean_squared_error(y_lr, y_pred_lr))
print('線性回歸填補後的數據 RMSE:', rmse_lr)

# 評估MICE方法填補後的數據
X_mice = df_Tx_mice[:, :]
y_mice = df_Ty
mice_model = LinearRegression()
mice_model.fit(X_mice, y_mice)
y_pred_mice = mice_model.predict(X_mice)
rmse_mice = sqrt(mean_squared_error(y_mice, y_pred_mice))
print('MICE方法填補後的數據 RMSE:', rmse_mice)

# 選擇較好的方法作為最終填補資料方式
if rmse_lr < rmse_mice:
    df_Tx_filled = df_Tx_lr
    print('選擇線性回歸填補後的數據作為最終填補資料方式')
else:
    df_Tx_filled = df_Tx_mice
    print('選擇MICE方法填補後的數據作為最終填補資料方式')


## 2.Data Transformation> 特徵縮放、特徵擴展、特徵選擇等
# 轉換目標變數為二元類別
# merged_data['sle_disease'] = np.where(merged_data['sle_disease'] == 'Yes', 1, 0)

# 分割特徵和目標??
df_Tx_filled = pd.DataFrame(df_Tx_filled[:], columns = df_Tx.columns)
X = df_Tx_filled
y = np.ravel(df_Ty)  

# 資料標準化
scaler = StandardScaler()
X = scaler.fit_transform(X)

## 3.Data Integration  4.Data Reduction(PCA??降為 去除耍費向量)

#拆分訓練集和測試集/ 重要: The 'stratify' parameter is used to ensure that the classes are evenly represented in each set.
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)


# X_train_val = np.ravel(X_train_val)
# y_train_val = np.ravel(y_train_val)
y_train = np.ravel(y_train)   #為何需要這樣??
y_val = np.ravel(y_val)  
y_test = np.ravel(y_test)  

from sklearn.preprocessing import LabelEncoder #使用 LabelEncoder 將 y_train 中的類別標籤轉換為整數標籤ㄎㄎ
le = LabelEncoder()
y_train = le.fit_transform(y_train)  # datatype轉成int相當重要，花了我一個晚上>>ㄎㄎ
y_val = le.fit_transform(y_val)
y_test = le.fit_transform(y_test)


# #用GridSearchCV來進行交叉驗證和參數調整。設置一個參數網格，用於調整SVM的C值和gamma值。
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import math

# 將訓練集和驗證集合併成一個大的資料集
X_train_val = np.vstack((X_train, X_val))
y_train_val = np.hstack((y_train, y_val))

# 檢查資料筆數是否能夠被 5 整除，如果不能，切割資料，以符合資料能夠被 5 整除的條件
n_samples = X_train_val.shape[0]
if n_samples % 5 != 0:
    n_samples = math.floor(n_samples / 5) * 5
    X_train_val = X_train_val[:n_samples]
    y_train_val = y_train_val[:n_samples]

# 使用 GridSearchCV 進行交叉驗證和參數調整
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100]}
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, return_train_score=True)
grid_search.fit(X_train_val, y_train_val)
print("Best parameters: ", grid_search.best_params_)

#使用最佳的參數來訓練模型
# linear: 线性核函数，用于线性可分问题。 poly: 多项式核函数，用于多项式可分问题。 
# rbf: 径向基核函数，用于非线性可分问题。 sigmoid: Sigmoid核函数，用于非线性可分问题。
svm = SVC(kernel='linear', C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'])
svm2 = SVC(kernel='poly', C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'])
svm3 = SVC(kernel='rbf', C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'])
svm4 = SVC(kernel='sigmoid', C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'])
svm.fit(X_train, y_train)
svm2.fit(X_train, y_train)  #rbf、linear?
svm3.fit(X_train, y_train)
svm4.fit(X_train, y_train)


# # 創建SVM模型2
# svm_model = SVC(kernel='linear', C=1, random_state=42)
# # 使用交叉驗證法對模型進行評估
# scores = cross_val_score(svm_model, X_train, y_train, cv=5)

# # 打印模型在每個折疊上的得分和平均得分
# print("Cross-validation scores:", scores)
# print("Average score:", np.mean(scores))

# 在測試集上進行預測
y_pred = svm.predict(X_test)  #最佳
y_pred2 = svm2.predict(X_test)
y_pred3 = svm3.predict(X_test)
y_pred4 = svm4.predict(X_test)

# 打印模型的準確率
print("linear_Accuracy:", svm.score(X_test, y_test))
print("poly_Accuracy:", svm2.score(X_test, y_test))
print("rbf_Accuracy:", svm3.score(X_test, y_test))
print("sigmoid_Accuracy:", svm4.score(X_test, y_test))

y_train_pred = svm.predict(X_train)  #最佳
y_val_pred = svm.predict(X_val)  #最佳
y_test_pred = svm.predict(X_test)  #最佳
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
# 計算混淆矩陣
test_cm = confusion_matrix(y_test, y_test_pred)
print("Testing Confusion Matrix:\n", test_cm)

# 計算分類報告>> 包含精準度、召回率、F1-score!! 真貼心
train_cr = classification_report(y_train, y_train_pred)
val_cr = classification_report(y_val, y_val_pred)
test_cr = classification_report(y_test, y_test_pred)

print("Training Classification Report:\n", train_cr)
print("Validation Classification Report:\n", val_cr)
print("Testing Classification Report:\n", test_cr)

# 計算 ROC 曲線和AUC值
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_pred)
roc_auc_train = roc_auc_score(y_train, y_train_pred)

fpr_val, tpr_val, thresholds_val = roc_curve(y_val, y_val_pred)
roc_auc_val = roc_auc_score(y_val, y_val_pred)

fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_pred)
roc_auc_test = roc_auc_score(y_test, y_test_pred)
# 印出結果
print("Training auc:", roc_auc_train)
print("Validation auc:", roc_auc_val)
print("Testing auc:", roc_auc_test)

print("ROC Curve:")
plt.plot(fpr_test, tpr_test, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

## 用使用者的角度（給最重要的參數
weights = np.abs(svm.coef_).ravel()  # 因為clf.coef_是2D-array(1, 20)，所以...要用revel將ndarray降維成(20,)
weights1 = svm.coef_.ravel()    # 考量原始正負值!
importances = weights / np.sum(weights)
importances1 = weights1 / np.sum(weights)
print("importances:\n", importances)
# 獲取前 5 個特徵的索引.
# top = np.argsort(importances)[::-1]
# print(top.shape, top)  #shape = (1,5)!!!
top_5 = np.argsort(importances)[::-1][:5]   #結果是把索引值對應的重要性由高到低做排序，再挑前五!!

# 根據特徵的索引從數據集中獲取對應的名稱和重要性值
feature_names = df_Tx_filled.columns
top_5_names = feature_names[top_5]
top_5_importances1 = importances1[top_5]
# 依次列印出前 5 個特徵的名稱和重要性值
print("以下為本預測模型重要性前5名的參數:\n註: 正負值分別表示與結果(是否罹患SLE)的正負相關!")  #??不知道這樣解讀可不可以??
for i in range(len(top_5_importances1)):
    print(f"{i+1}. {top_5_names[i]}: {top_5_importances1[i]}")