##EarlyStopping + NN
# 載入相關套件
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping 
from keras.models import Sequential
from keras.layers import Dense

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

# 分割特徵與目標
# X = data.iloc[:, :-1]
# y = data.iloc[:, -1]
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

# # Scale the data using StandardScaler(前面已做過)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
# X_test = scaler.transform(X_test)

from sklearn.preprocessing import LabelEncoder #使用 LabelEncoder 將 y_train 中的類別標籤轉換為整數標籤ㄎㄎ
le = LabelEncoder()
y_train = le.fit_transform(y_train)  # datatype轉成int相當重要，花了我一個晚上>>ㄎㄎ
y_val = le.fit_transform(y_val)
y_test = le.fit_transform(y_test)

# 建立模型
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 編譯模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 設定早停法
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# 訓練模型
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stop])

# # 評估模型
# train_loss, train_acc = model.evaluate(X_train, y_train)
# val_loss, val_acc = model.evaluate(X_val, y_val)
# test_loss, test_acc = model.evaluate(X_test, y_test)

# print('Train accuracy:', train_acc)
# print('Validation accuracy:', val_acc)
# print('Test accuracy:', test_acc)

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
# 預測訓練集
y_train_pred = model.predict(X_train).ravel()
for i in range(len(y_train_pred)):
    if y_train_pred[i] > 0.5:
        y_train_pred[i] = 1
    else:
        y_train_pred[i] = 0
# 預測驗證集
y_val_pred = model.predict(X_val).ravel()
for i in range(len(y_val_pred)):
    if y_val_pred[i] > 0.5:
        y_val_pred[i] = 1
    else:
        y_val_pred[i] = 0
# 預測測試集
y_test_pred = model.predict(X_test).ravel()
for i in range(len(y_test_pred)):
    if y_test_pred[i] > 0.5:
        y_test_pred[i] = 1
    else:
        y_test_pred[i] = 0
y_train_pred = y_train_pred.astype(int)
y_val_pred = y_val_pred.astype(int)
y_test_pred = y_test_pred.astype(int)
from sklearn.preprocessing import LabelEncoder #使用 LabelEncoder 將 y_train 中的類別標籤轉換為整數標籤ㄎㄎ
le = LabelEncoder()
y_train = le.fit_transform(y_train_pred)  # datatype轉成int相當重要，花了我一個晚上>>ㄎㄎ
y_val = le.fit_transform(y_val_pred)
y_test = le.fit_transform(y_test_pred)
print("y_val_pred", type(y_val_pred[0]), y_val_pred)    #預測結果為float，該如何轉為binary?>>以0.5為分界嗎??
print("y_val", type(y_val[0]), y_val)
print("y_test_pred", type(y_test_pred[0]), y_test_pred)
print("y_test", type(y_test[0]), y_test)

# 計算混淆矩陣
train_cm = confusion_matrix(y_train, y_train_pred)
val_cm = confusion_matrix(y_val, y_val_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

print("Training Confusion Matrix:\n", train_cm)
print("Validation Confusion Matrix:\n", val_cm)
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


feature_names = df_Tx_filled.columns
# 計算每個特徵的權重
weights = model.layers[0].get_weights()[0]  #同個參數在每次計算時weights都各有正負
importances = np.mean(np.abs(weights), axis=1)        #不知道絕對值用或不用哪個好??
importances1 = np.mean(weights, axis=1)

# 獲取前5個重要特徵的索引
n = 5
top_n = np.argsort(importances)[::-1][:n]

# 根據特徵的索引從數據集中獲取對應的名稱
top_n_names = feature_names[top_n]

# 列印結果
print(f"Top {n} features:")
for i in range(n):
    # print("Feature {}: {} ({})".format(i+1, top_5_names[i], importances[top_5[i]]))
    print(f"{i+1}. {top_n_names[i]}: {importances[top_n[i]]}")