#### FilenName :Project for Practical machine learning

install.packages("lattice")
install.packages("ggplot2")
install.packages("caret")
install.packages("randomForest")
install.packages("corrplot")
install.packages("parallel")
install.packages("gbm")
install.packages("plyr")
install.packages("survival")
install.packages("splines")

set.seed(1234)
library(caret); library(kernlab); library(randomForest); library(corrplot)

### read the csv file for training 
data_training <- read.csv("pml-training.csv", na.strings= c("NA",""," "))

### remove the columns with NA from training data
data_training_NA <- apply(data_training, 2, function(x){
        sum(is.na(x))
})
data_training_new1 <- data_training[, which(data_training_NA == 0)]
#### remove the comlumns of user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window
data_training_new2 <- data_training_new1[8:length(data_training_new1)]

### split the cleaned testing data into training and cross validation
### 90 percent subsample is used to train the modle, and 10 percent sample is used for cross-validation.
inTrain <- createDataPartition(y = data_training_new2$classe, p = 0.7, list = FALSE)
training <- data_training_new2[inTrain, ]
crossval <- data_training_new2[-inTrain, ]

### plot a correlation matrix
cor <- abs(sapply(colnames(training[, -ncol(training)]), function(x) cor(as.numeric(training[, x]), as.numeric(training$classe), method = "spearman")))
summary(cor)
png("plot1.png", width = 480, height = 480)
plot(training[, names(which.max(cor))], training[, names(which.max(cor[-which.max(cor)]))], col = training$classe, pch = 19, cex = 0.1, xlab = names(which.max(cor)), ylab = names(which.max(cor[-which.max(cor)])))
dev.off()

correlMatrix <- cor(training[, -length(training)])
correlMatrix 
png("corrplot.png", width = 480, height = 480)
corrplot(correlMatrix, order = "FPC", method = "circle", type = "lower", tl.cex = 0.8,  tl.col = rgb(0, 0, 0))
dev.off()

### Boosting Model
# Fit model with boosting algorithm.
# Plot accuracy of this model on the scale [0.9, 1].

boostFit <- train(classe ~ ., method = "gbm", data = training, verbose = FALSE)
print(boostFit)

png("boostFit.png", width = 480, height = 480)
plot(boostFit, ylim = c(0.9, 1))
dev.off()

### Random Forest Model
rfFit <- train(classe ~ ., method = "rf", data = training, importance = T)
rfFit
plot(rfFit, ylim = c(0.9, 1))

png("rfFit.png", width = 480, height = 480)
plot(rfFit, ylim = c(0.9, 1))
dev.off()

imp <- varImp(rfFit)$importance 
imp$Max <- apply(imp, 1, max)
impNew <- imp[order(imp$Max, decreasing = T), ] 


#### crossvalidate the model using the remaining 30% of data
# using random forest as finan model
predictCrossVal <- predict(rfFit, crossval)
confusionMatrix(crossval$classe, predictCrossVal)

### read the csv file for testing
data_testing <- read.csv("pml-testing.csv", na.strings= c("NA",""," "))

### clean testing data by removing columns with NA and the first 8 identifier cloumns
data_testing_NA <- apply(data_testing, 2, function(x){sum(is.na(x))})
data_testing_new1 <- data_testing[, which(data_testing_NA == 0)]
testing <- data_testing_new1[8:length(data_testing_new1)]

### predict the classes of the testing data
prediction <- predict(rfFit, testing)
prediction

