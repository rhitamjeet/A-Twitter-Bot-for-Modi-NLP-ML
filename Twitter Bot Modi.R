###Project Brief
# We have extracted the tweets from Prime Minister Modi twitter handle. We apply Natural Language Processing(NLP) to gauge the reception of his tweets among twitter users and use it to build a model to predict the popularity of any new tweets based on NLP and Machine Learning. We will call it - A Twitter Bot for Modi

modi_tweets1 = read.csv("D:/EDA/narendramodi_tweets.csv")
modi_tweets = modi_tweets1[,c(2,3,5)]
library(dplyr)


modi_tweets$text = gsub(x= modi_tweets$text,pattern = '[^[:alpha:] ]', replacement = '')

modi_tweets$text = gsub(x= modi_tweets$text,pattern = 'à', replacement = '')

modi_tweets$text = gsub(x= modi_tweets$text,pattern = 'Â', replacement = '')

modi_tweets$text = gsub(x= modi_tweets$text,pattern = 'â', replacement = '')

modi_tweets$text = gsub(x= modi_tweets$text,pattern = 'http.*', replacement = '')

# have to remove white spaces

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

library(tm)
text = VCorpus(VectorSource(modi$text))

text = tm_map(text,content_transformer(tolower))  #to lower case
#as.character(text[[242]])
text = tm_map(text,removeNumbers)                #to remove numbers

text = tm_map(text,removePunctuation)            #to remove punctuations

library(SnowballC)

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

library(caTools)

split = sample.split(dataset$Liked,SplitRatio = 0.8)
#set.seed(123)
training_set = subset(dataset,split == TRUE)
test_set = subset(dataset, split == FALSE)


#random forrest
library(randomForest)
classifier = randomForest(x = training_set[-896],
                          y = training_set$Liked,
                          ntree = 50)

#library(e1071)
#classifier = naiveBayes(x=training_set[-640],
#                        y=training_set$Liked)
y_pred = predict(classifier, newdata = test_set[-896])

results = cbind(test_set$Liked,y_pred)
colnames(results)= c('Real','pred')
results = as.data.frame(results)
results$prediction = if_else(results$Real == results$pred,"Correct","Incorrect")
a = length(which(results$prediction=='Correct'))
b = length(which(results$prediction=='Incorrect'))
accuracy = a /(a + b)*100
accuracy 

