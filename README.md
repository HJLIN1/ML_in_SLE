# ML_in_SLE
# Model_Objection: 利用T cell(Tc、Th、Memory) marker表現量預測目前是否罹患SLE
# 臨床情境
希望透過基因、PBMC biomarker data進行分析預測，進行SLE早期診斷，在病患臨床表徵明顯呈現之前，就進行早期診斷，避免病情惡化
# 目前模型表現
![image](https://github.com/StephenROY6/ML_in_SLE/assets/107903023/e9279f25-48b2-4d8e-816b-3839e682f614) 
# 實踐步驟大致如下:
1. Data preprocessing 
 a. Data Cleaning> 處理缺失值、異常值(刪除缺失值較多的特徵或病人，比較線性回歸與MICE多重差補並擇優)
 b. Data Transformation> 特徵縮放、特徵擴展、特徵選擇等
 c. Data Integration  
 d. Data Reduction(PCA??降維 去除耍廢向量)
2. 定義模型、特徵選擇
3. 訓練模型
4. 模型評估(驗證集、測試集；混淆矩陣；ROC_AUC；classification_report)
5. 用使用者的角度（獲取前 5 個特徵的索引、給最重要的參數)

# 0328RF_PBMC
# 0422Logreg2_PBMC_SLE
# 0423Logreg3_L1
# 0422keras_NN
# 0508SVM_PBMC
