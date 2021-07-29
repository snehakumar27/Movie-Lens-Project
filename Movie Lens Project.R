##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

## Create Train & Test Sets
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train <- edx[-test_index,]
temp <- edx[test_index,]

test <- temp%>%
  semi_join(train,by="movieId")%>%
  semi_join(train,by="userId")
train <- rbind(train, anti_join(temp,test))

rm(test_index, temp)

##########################################################
################## Exploratory Analysis ##################
##########################################################
#useful packages
library(ggplot2)
library(lubridate)

head(edx)
str(edx)
summary(edx)  

## Number of movies & users
n_distinct(edx$movieId)
n_distinct(edx$userId)

## Popular Movies
edx%>%
  group_by(title)%>%
  summarize(count=n())%>%
  arrange(desc(count))

## Popular genres
edx%>%
  group_by(genres)%>%
  summarize(count=n())%>%
  arrange(desc(count))

## Ratings distribution
edx%>%
  ggplot(aes(rating,..prop..))+
  geom_bar(color="black",fill="royalblue3")+
  labs(x="Ratings",y="Proportion")

## Users bias
edx%>%
  group_by(userId)%>%
  summarize(count=n())%>%
  ggplot(aes(count))+
  geom_histogram(color="black",fill="orchid2",bins=50)+
  scale_x_log10()+
  labs(x="Ratings",y="Users")

## Movies bias 
edx%>%
  group_by(movieId)%>%
  summarize(count=n())%>%
  ggplot(aes(count))+
  geom_histogram(color="black",fill="palegreen4",bins=50)+
  scale_x_log10()+
  labs(x="Ratings",y="Movies")


##########################################################
##################### Data Modeling ######################
##########################################################

## Evaluation of Model using RMSE method 
rmse<-function(actual,predicted){
  sqrt(mean((actual-predicted)^2))
}

##1. Mean Ratings 
mu_hat<-mean(train$rating)

#RMSE
mean_rmse<-rmse(test$rating,mu_hat)

#Storing results into a tibble
results<-tibble(Method="Model 1: Mean",RMSE=mean_rmse)


##2. Movie Bias 
bi<-train%>%
  group_by(movieId)%>%
  summarize(bi=mean(rating-mu_hat))

#Distribution
bi %>% 
  ggplot(aes(bi)) +
  geom_histogram(color = "black", fill = "palegreen4",bins=15)

#Predicted Ratings
pred1<-mu_hat+test %>%
  left_join(bi, by = "movieId") %>%
  pull(bi)

#RMSE
mov_bias_rmse<-rmse(test$rating,pred1)

#Storing results into a tibble
results<-bind_rows(results,tibble(Method="Model 2: Mean & Movie Bias",RMSE=mov_bias_rmse))


##3. User Bias 
bu<-train%>%
  left_join(bi,by ="movieId")%>%
  group_by(userId)%>%
  summarize(bu=mean(rating-mu_hat-bi))

#Predicted Ratings 
pred2<-test%>%
  left_join(bi,by="movieId")%>%
  left_join(bu,by="userId")%>%
  mutate(pred=mu_hat+bi+bu)%>%
  pull(pred)

#RMSE
user_bias_rmse<-rmse(test$rating,pred2)

#Storing results into a tibble
results<-bind_rows(results,tibble(Method="Model 3: Mean, Movie Bias, & User Bias",RMSE=user_bias_rmse))


##4. Regularization (Movies & Users)
lambdas<-seq(0,10,0.25)

#RMSE
rmses <- sapply(lambdas, function(x){
  bi<-train%>%
    group_by(movieId)%>%
    summarize(bi=sum(rating-mu_hat)/(n()+x))
  bu<-train%>%
    left_join(bi,by="movieId")%>%
    group_by(userId)%>%
    summarize(bu=sum(rating-bi-mu_hat)/(n()+x))
  pred3<-test%>%
    left_join(bi,by="movieId")%>%
    left_join(bu,by="userId")%>%
    mutate(pred=mu_hat+bi+bu) %>%
    pull(pred)
  return(rmse(pred3, test$rating))
})

# Plot
qplot(lambdas,rmses)
#Choose the best lambda
lambda <- lambdas[which.min(rmses)]

#Storing results into a tibble
results<-bind_rows(results,tibble(Method="Model 4: Regularization: Movies & Users",RMSE=min(rmses)))


##5. Matrix Factorization (recosystem)
library(recosystem)
set.seed(1, sample.kind="Rounding")
train_rec<-with(train,data_memory(userId,movieId,rating))
test_rec<-with(test,data_memory(userId,movieId,rating))
rec <- Reco()

#Tuning
tune_reco<-rec$tune(train_rec,opts=list(dim=c(20, 30),
                                        costp_l2=c(0.01, 0.1),
                                        costq_l2=c(0.01, 0.1),
                                        lrate=c(0.01, 0.1),
                                        nthread=4,
                                        niter=10))

#Training
rec$train(train_rec,opts=c(tune_reco$min,nthread = 4,niter = 30))

#Predicted Ratings 
pred4<-rec$predict(test_rec,out_memory())

#RMSE
mat_fact_rmse<-rmse(test$rating,pred4)

#Storing results into a tibble
results<-bind_rows(results,tibble(Method="Model 5: Matrix Factorization",RMSE=mat_fact_rmse))

## Results from our Models 
results%>%knitr::kable()


##########################################################
################### Final Validation #####################
##########################################################
## Use the model with lowest RMSE for validation 
set.seed(1, sample.kind="Rounding")
edx_reco<-with(edx,data_memory(user_index=userId,item_index=movieId,rating=rating))
val_rec<-with(validation, data_memory(user_index=userId,item_index=movieId,rating=rating))
r<-Reco()

par_reco <- r$tune(edx_reco,opts=list(dim=c(20,30),
                                      costp_l2=c(0.01, 0.1),
                                      costq_l2=c(0.01, 0.1),
                                      lrate=c(0.01, 0.1),
                                      nthread=4,
                                      niter=10))

r$train(edx_reco, opts = c(par_reco$min, nthread = 4, niter = 30))
final_rec<- r$predict(val_rec, out_memory())

#Final RMSE
final_rmse<-rmse(validation$rating,final_rec)

#Store final value to tibble
results<-bind_rows(results,tibble(Method ="Final validation: Matrix factorization ",RMSE=final_rmse))

results%>%knitr::kable()