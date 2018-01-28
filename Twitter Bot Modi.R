###Project Brief
# We have extracted the tweets from Prime Minister Modi's twitter handle.
# We apply Natural Language Processing(NLP) to gauge the receptionof his tweets among twitterusers 
# and use it to build a model to predict the popularity of any new tweets based on NLP and Machine Learning.
# We will call it - A Twitter Bot for Modi


#Loading Data
modi_tweets1 = read.csv("D:/EDA/narendramodi_tweets.csv")
modi_tweets = modi_tweets1[,c(2,3,5)]

#Loading libraries
library(dplyr)
library(tm)
library(SnowballC)
library(caTools)

#Cleaning Data
modi_tweets$text = gsub(x= modi_tweets$text,pattern = '[^[:alpha:] ]', replacement = '')

modi_tweets$text = gsub(x= modi_tweets$text,pattern = 'à', replacement = '')

modi_tweets$text = gsub(x= modi_tweets$text,pattern = 'Â', replacement = '')

modi_tweets$text = gsub(x= modi_tweets$text,pattern = 'â', replacement = '')

modi_tweets$text = gsub(x= modi_tweets$text,pattern = 'http.*', replacement = '')

# To remove white spaces

a = which(nchar(gsub(" ",'', modi_tweets$text))==0)
#nrow(modi_tweets[a,])
modi_tweets = modi_tweets[-c(a),]

#nrow(modi_tweets)
#View(modi_tweets)
#View(modi_tweets[grep("UIDAI",modi_tweets$text),])

modi = modi_tweets
library(dplyr)

#hist(modi$favorite_count) # positive skew
quantile(modi$favorite_count)
# median = 4658.50
modi$Liked = if_else(modi$favorite_count>4658,1,0)
# We set this as the threshold to determine whether a particluar tweet was liked by twitter users.


text = VCorpus(VectorSource(modi$text))

text = tm_map(text,content_transformer(tolower))  #to lower case
#as.character(text[[242]])
text = tm_map(text,removeNumbers)                #to remove numbers

text = tm_map(text,removePunctuation)            #to remove punctuations



text = tm_map(text,removeWords,stopwords())      #to remove common words

text = tm_map(text,stemDocument)                 # to convert to root words

text = tm_map(text,stripWhitespace)              #to remove white spaces
#as.character(text[[6]])
#creating sparse matrix
dtm = DocumentTermMatrix(text)
dtm = removeSparseTerms(dtm,0.998)

dataset = as.data.frame(as.matrix(dtm))

dataset$Liked = modi$Liked

dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

#classification - random forrest

split = sample.split(dataset$Liked,SplitRatio = 0.8)
#set.seed(123)
training_set = subset(dataset,split == TRUE)
test_set = subset(dataset, split == FALSE)


#random forrest
library(randomForest)
classifier = randomForest(x = training_set[-896],
                          y = training_set$Liked,
                          ntree = 50)

y_pred = predict(classifier, newdata = test_set[-896])

results = cbind(test_set$Liked,y_pred)
colnames(results)= c('Real','pred')
results = as.data.frame(results)
results$prediction = if_else(results$Real == results$pred,"Correct","Incorrect")
a = length(which(results$prediction=='Correct'))
b = length(which(results$prediction=='Incorrect'))
accuracy = a /(a + b)*100
accuracy 

#accuracy = 71.68

important_words = data.frame(Words = rownames(classifier$importance),classifier$importance)
important_words1 = important_words %>% arrange(-MeanDecreaseGini)

# Here we can see which words influence the no. of likes the most.

# XGBoost

library(xgboost)
library(caret)
library(Matrix)
sparse_matrix_full = sparse.model.matrix(Liked~.-1, data = dataset)

#set.seed(123)
sparse_train =sparse_matrix_full[split == TRUE,]
sparse_test= sparse_matrix_full[split == FALSE,]

output_vector = dataset[split ==T,"Liked"]== "1"
dtrain = xgb.DMatrix(sparse_train, label = output_vector)
dtest = xgb.DMatrix(sparse_test)

params <- list(booster = "gblinear", objective = "binary:logistic", eta=0.1, gamma=50, max_depth=100, min_child_weight=1, subsample=0.8, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 2000, nfold = 5, showsd = T, stratified = T, maximize = F)

plot(xgbcv$evaluation_log$test_error_mean)
mean(xgbcv$evaluation_log$test_error_mean)
which.min(xgbcv$evaluation_log$test_error_mean)


xgb1 = xgb.train(params = params, data = dtrain, nrounds =1,verbose = 1,maximize = F)
xgbpred = predict(xgb1,dtest)
xgbpred = ifelse(xgbpred>0.5,1,0)
hist(xgbpred)

test = as.numeric(as.character(dataset[split == FALSE,'Liked']))
confusionMatrix(xgbpred,test)
#Accuracy = 70 %
