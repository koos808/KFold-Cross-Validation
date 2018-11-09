# K-fold cross validation example code
# K-fold 교차 검증 여러가지 예제 코드

################################################################
# 첫번째 Example code
# iris데이터 - 5개의 열과 150개의 행
# 사용 모델링 알고리즘 : Random forest
# iris의 Sepal.Length 예측 - iris Sepal.Length Predict

library(plyr)
library(dplyr)
library(randomForest)

data <- iris
glimpse(data)# 데이터 확인. dplyr 패키지에 내장

#random forest를 사용하여 sepal.length 예측.
#cross validation, using rf to predict sepal.length
k = 5
data$id <- sample(1:k, nrow(data), replace = TRUE)
list <- 1:k

# 예측 및 테스트는 폴드를 반복 할 때마다 추가되는 데이터 프레임을 설정합니다.
# prediction and test set data frames that we add to with each iteration over the folds.
# 데이터 프레임 초기화(data frame reset)
prediction <- testsetCopy <- data.frame()

# 코드 실행 시 작업진행률을 보여주는 progress.bar
#Creating a progress bar to know the status of CV
progress.bar <- create_progress_bar("text") # plyr 패키지안에 내장
progress.bar$init(k)

#function for k fold
#i는 1부터 5로 나눈후에 5번을 진행하도록 합니다.

for(i in 1:k){
  # remove rows with id i from dataframe to create training set
  # ex) id가 1인 것을 제외하고 나머지 id 2~5를 training set으로 사용
  # select rows with id i to create test set
  # ex) id가 1인 것만 test set으로 사용
  trainset <- subset(data, id %in% list[-i])
  testset <- subset(data, id %in% c(i))
  
  # 랜덤포레스트 모델을 생성.
  #run a random forest model
  model <- randomForest(trainset$Sepal.Length ~ .-id, data = trainset, ntree = 100)
  temp <- as.data.frame(predict(model, testset))

  # 예측값을 예측 데이터 프레임의 끝에 추가.
  # append this iteration's predictions to the end of the prediction data frame
  prediction <- rbind(prediction, temp)
  
  # 실제값(testset) testsetCopy에 추가.
  # append this iteration's test set to the test set copy data frame
  testsetCopy <- rbind(testsetCopy, as.data.frame(testset[,1]))
  
  progress.bar$step()
}

# 예측값과 실제값 데이터프레임.
# add predictions and actual Sepal Length values
result <- cbind(prediction, testsetCopy[, 1])
names(result) <- c("Predicted", "Actual")
result$Difference <- abs(result$Actual - result$Predicted)

# 모델 평가로 MAE[Mean Absolute Error] 사용.
# As an example use Mean Absolute Error as Evalution
summary(result$Difference)

################################################################
# 두번째 Example code
# iris의 Species 분류 - iris Specieies classification

data <- iris
k = 5; list <- 1:k
data$id <- sample(1:k, nrow(data), replace = TRUE)
prediction <- testsetCopy <- data.frame()

#function for k fold
for(i in 1:k){
  trainset <- subset(data, id %in% list[-i])
  testset <- subset(data, id %in% c(i))
  model <- randomForest(trainset$Species~.-id, data = trainset, ntree = 100)
  temp <- as.data.frame(predict(model, testset))
  prediction <- rbind(prediction, temp)
  testsetCopy <- rbind(testsetCopy, as.data.frame(testset[,5]))
}

result <- cbind(prediction, testsetCopy[, 1])
names(result) <- c("Predicted", "Actual")
library(e1071) ; library(caret) # confusion matrix 내장
confusionMatrix(result$Predicted, result$Actual)


