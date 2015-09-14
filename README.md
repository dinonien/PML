---
title: "Practical Machine Learning - Course Project"
author: "D.N."
date: "14th September 2015"
output: html_document
---

#Requirements
Please load the following libraries.
```{r}
library(dplyr)
library(e1071)
library(caret)
```

#Data
Download the files from the given URL. Please note that the data comes from [1].
```{r}
download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile="./Data/pml-test.csv")
download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile="./Data/pml-test.csv")
```
#Data Processing
Load data from CSV files.
```{r, echo=FALSE}
rawTrainingData <- read.csv("./Data/pml-training.csv")
rawTestData <- read.csv("./Data/pml-test.csv")
```

#Feature Selection
Only roll, pitch and yaw raw data is used as these are sufficient to recognise activities [2],[3],[4].
```{r}
trainingData <- rawTrainingData[,colnames(rawTrainingData)[
        c(grep("^roll_", colnames(rawTrainingData)),grep("^pitch_", colnames(rawTrainingData)),grep("^yaw",                                                 colnames(rawTrainingData)),grep("^classe",colnames(rawTrainingData)))]]

trainingData <- trainingData[complete.cases(trainingData),]

testData <- rawTestData[,colnames(rawTestData)[c(grep("^roll_", colnames(rawTestData)),grep("^pitch_", 
        colnames(rawTestData)),grep("^yaw",colnames(rawTestData)))]]
```

#Train SVM Algorithm with Cross Validation
The SVM Algorithm is used with a radial kernel and cross validation to find the best values for gamma and cost. SVM has shown good performance for activity recognition [5],[6].
```{r}
tune.out <- tune(svm, classe~., data = trainingData, kernel="radial", ranges=list(cost=c(.1,1,10,100,1000),gamma=c(.5,1,2,3,4))) 
summary(tune.out)
```
![Cross Validation Summary](https://raw.githubusercontent.com/dinonien/PML/master/Assets/tune_out.png)

#Confusionmatrix
Plot the confusion matrix.
```{r}
confusionMatrix(predict(tune.out$best.model,newx=testData[1,]),trainingData[,"classe"])
```
![Confusion Matrix](https://raw.githubusercontent.com/dinonien/PML/master/Assets/conf_matrix.png)

#Predict TestData
Predict classes for the test data.
```{r}
predict(tune.out$best.model, newdata = testData[,])
```
![Test Results](https://raw.githubusercontent.com/dinonien/PML/master/Assets/test.png)

#Sample Error
Out of sample error is expected to higher than the one from the training set. However, all submissions are correct.

#References
[1] Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings         of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

[2] Ayub, Shahid, Alireza Bahraminisaab, and Bahram Honary. "A sensor fusion method for smart phone orientation estimation." Proceedings of         the 13th Annual Post Graduate Symposium on the Convergence of Telecommunications, Networking and Broadcasting, Liverpool. 2012.

[3] Kwon, Doo Young, and Markus Gross. "Combining body sensors and visual sensors for motion training." Proceedings of the 2005 ACM SIGCHI          International Conference on Advances in computer entertainment technology. ACM, 2005.

[4] Lementec, Jean-Christophe, and Peter Bajcsy. "Recognition of arm gestures using multiple orientation sensors: gesture classification."          Intelligent Transportation Systems, 2004. Proceedings. The 7th International IEEE Conference on. IEEE, 2004.

[5] Singla, Geetika, Diane J. Cook, and Maureen Schmitter-Edgecombe. "Recognizing independent and joint activities among multiple residents         in smart environments." Journal of ambient intelligence and humanized computing 1.1 (2010): 57-63.

[6] Fleury, Anthony, Michel Vacher, and Norbert Noury. "SVM-based multimodal classification of activities of daily living in health smart           homes: sensors, algorithms, and first experimental results." Information Technology in Biomedicine, IEEE Transactions on 14.2                   (2010): 274-283.