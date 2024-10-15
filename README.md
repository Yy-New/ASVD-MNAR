# ASVD-MNAR

### Step 1:
Run `python LogData.py` can logarithmically transform the data in the PublicData folder and obtain a LogPublicData folder. The data file name corresponds to the data file name of the PublicData folder.

### Step 2:
Run `python MissingGeneration.py` can generate a series of missing data from the data set in LogPublicData. The missing data is saved in the MMData folder.

### Step 3:
Run `python ./ImputationTest/MultipleImputationTest.py` to execute imputation.It will yield KNN, SVD, iter_svd, RF, NS_KNN, and QRILC imputation results. The results are saved in the FillData\MultipleImputation folder.
#### Tips：
When executing the QRILC interpolation, you need to call the local R environment, so please modify the fourth line of the ImputationAlgorithm.py file as follows: "os.environ['R_HOME'] = "D:/PROGRA~1/R/R-xxx.xx/", and comment out the corresponding code in the MultipleImputationTest.py file if you do not need to execute QRILC.

### Step 4:
Run `python ./ImputationTest/ASVDImputationTest.py` to execute imputation. Estimates of ASVD-MNAR and ASVD-MNAR for a range of different parameters will be produced. The results are saved in the FillData\ASVDImputation folder.


### Step 5:
Run `python ./Evaluation/NRMSE/NRMSE.py` to get the NRMSE of imputation data. The results are saved in the Results folder.
Run `python ./Evaluation/NRMSE/AvergaeNRMSE.py` to get the Avergae NRMSE of imputation data. The results are saved in the AverageResult folder.

Run `python ./Evaluation/SOR/SOR.py` to get the SOR of imputation data. The results are saved in the Results folder.
Run `python ./Evaluation/SOR/AvergaeSOR.py` to get the Avergae SOR of imputation data. The results are saved in the AverageResult folder.

### Step 5:
Run `python ./Draw/NRMSELineChart.py` to get KNN, SVD, iter_svd, RF, NS_KNN, QRILC and ASVD-MNAR NRMSE results comparison graph.

Run `python ./Draw/NRMSELineChart.py` to get KNN, SVD, iter_svd, RF, NS_KNN, QRILC and ASVD-MNAR SOR results comparison graph.

Run `python ./Draw/VSLineChart/VSRandomW.py` to get ASVD-MNAR result in a comparison graph of NRMSE results for different ranges of randomW.

Run `python ./Draw/VSLineChart/VSThresholdRatio.py` to get ASVD-MNAR result in a comparison graph of NRMSE results for different ranges of ThresholdRatio.

Run `python ./Draw/VSLineChart/VSW1.py` to get ASVD-MNAR result in a comparison graph of NRMSE results for different ranges of W1.
