#以下是以PBMC data預測是否罹患SLE的分類式機器學習模型的範例程式碼，
#包含資料分割、模型建立、梯度下降法訓練、以及overfitting處理
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

y_train = np.ravel(y_train)   #為何需要這樣??
y_val = np.ravel(y_val)  
y_test = np.ravel(y_test)  

from sklearn.preprocessing import LabelEncoder #使用 LabelEncoder 將 y_train 中的類別標籤轉換為整數標籤ㄎㄎ
le = LabelEncoder()
y_train = le.fit_transform(y_train)  # datatype轉成int相當重要，花了我一個晚上>>ㄎㄎ
y_val = le.fit_transform(y_val)
y_test = le.fit_transform(y_test)

# 訓練模型
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
clf.fit(X_train, y_train)


# 驗證模型
val_preds = clf.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)

# 測試模型
test_preds = clf.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# 驗證集表現評估
y_val_pred = clf.predict(X_val)
y_val_prob = clf.predict_proba(X_val)[:, 1]   # 最後面[:, 1]是甚麼意思

print("Validation Set Metrics:")
print("Accuracy: {:.4f}".format(accuracy_score(y_val, y_val_pred)))
print("Precision: {:.4f}".format(precision_score(y_val, y_val_pred)))
print("Recall: {:.4f}".format(recall_score(y_val, y_val_pred)))
print("F1 Score: {:.4f}".format(f1_score(y_val, y_val_pred)))
print("ROC AUC Score: {:.4f}".format(roc_auc_score(y_val, y_val_prob)))

# 測試集表現評估
y_test_pred = clf.predict(X_test)
y_test_prob = clf.predict_proba(X_test)[:, 1]

print("\nTest Set Metrics:")
print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_test_pred)))
print("Precision: {:.4f}".format(precision_score(y_test, y_test_pred)))
print("Recall: {:.4f}".format(recall_score(y_test, y_test_pred)))
print("F1 Score: {:.4f}".format(f1_score(y_test, y_test_pred)))
print("ROC AUC Score: {:.4f}".format(roc_auc_score(y_test, y_test_prob)))

# confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print(f'Confusion Matrix: \n{cm}')

# # 預測測試集
# y_pred = clf.predict(X_test)

# # 評估模型
# accuracy = accuracy_score(y_test, y_pred)
# AUC = roc_auc_score(y_test, y_pred)
# confusion = confusion_matrix(y_test, y_pred)

# print("Accuracy:", accuracy)
# print("AUC:", AUC)
# print('Precision:', metrics.precision_score(y_test, y_pred))
# print('Recall:', metrics.recall_score(y_test, y_pred))
# print('F1:', metrics.f1_score(y_test, y_pred))
# print('混淆矩陣(Confusion Matrix):\n', confusion)


# #呈現ROC curve
# fpr, tpr, thersholds = roc_curve(y_test, y_pred)
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, 'k--', label = 'ROC (area = {0:2f})'.format(roc_auc), lw=2)
# plt.xlim([-0.05, 1.05]) #＃设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel("True Positive Rate") #可以使用中文，但需要导入一些库即宁体
# plt.title('ROC Curve')
# plt.legend(loc="lower right")
# plt.show()

## 用使用者的角度（給最重要的參數
# 取得特徵重要性
importances = clf.feature_importances_
feature_names = df_Tx_filled.columns  
print(importances)
# 找出前五個重要的特徵及其重要性  (但是RF的feature_importances_好像沒有正負?)
top_n = 5
indices = importances.argsort()[::-1][:top_n]
print("Top %d features:" % top_n)
for f in range(top_n):
    print(f"{f + 1}. {feature_names[indices[f]]}: {importances[indices[f]]}")
# 在上述程式碼中，load_data() 是用來載入您的資料集的函式。RandomForestClassifier 則是建立隨機森林模型的類別，其中 n_estimators 參數表示您想要建立幾棵決策樹，random_state 參數則是隨機種子，讓您的模型訓練結果可以重現。fit() 函式則是用來訓練模型的。
# 當模型訓練完畢後，您可以使用 feature_importances_ 屬性來取得每個特徵的重要性。接著，我們可以使用 argsort() 函式來將特徵重要性由小到大排序，並使用 [::1] 來將其反轉，以便取得前五個重要的特徵。最後，我們可以使用 print() 函式來印出每個特徵的重要性排名。