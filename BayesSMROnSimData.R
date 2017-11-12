## Bayesian Logistic Regression in R / Stan

Test_BayesSMR_On_Sim_Data <- function()
{
  synth_data <- Create_GM_Data()
  Plot_Synth_Data(synth_data)
  Xs <- synth_data$Xs
  ys <- synth_data$ys
  
  Xs_new <- cbind(array(1,dim=c(nrow(Xs))),Xs)
  head(Xs_new)
  sf <- Fit_With_SMR(Xs_new,ys)
  comp_list <- list(Xs=Xs_new,ys=ys,sf=sf)
  return(comp_list)
}

Predict_Dataset <- function(comp_list)
{
  Xs_new <- comp_list$Xs
  ys <- comp_list$ys
  sf <- comp_list$sf
  mat <- as.matrix(sf)
  preds <- array(0,dim=c(nrow(Xs_new)))
  probs <- array(0,dim=c(nrow(Xs_new),2))
  num_classes <- max(ys)
  print(num_classes)
  num_features <- ncol(Xs_new)-1
  
  thetas <- array(0,dim=c(num_classes,num_features+1))
  thetas[1,1] <- mean(mat[,1])
  thetas[1,2] <- mean(mat[,3])
  thetas[1,3] <- mean(mat[,5])
  thetas[2,1] <- mean(mat[,2])
  thetas[2,2] <- mean(mat[,4])
  thetas[2,3] <- mean(mat[,6])
  
  for(i in 1:nrow(Xs_new))
  {
    lin_comb1 <- Xs_new[i,1]*thetas[1,1]+Xs_new[i,2]*thetas[1,2]+Xs_new[i,3]*thetas[1,3]
    lin_comb2 <- Xs_new[i,1]*thetas[2,1]+Xs_new[i,2]*thetas[2,2]+Xs_new[i,3]*thetas[2,3]
    
    exp1 <- exp(lin_comb1)
    exp2 <- exp(lin_comb2)
    
    p1 <- exp1/(exp1+exp2)
    p2 <- exp2/(exp1+exp2)
    
   # print("sample:")
    #print(i)
    #print("prob1: ")
    #print(p1)
    #print("prob2:")
    #print(p2)
    #print("correct:")
    preds[i] <- which.max(c(p1,p2))
    
    print("prediction:")
    print(which.max(c(p1,p2)))
    print("correct:")
    print(ys[i])
    print("")
  }
  
  ## do 2 plots: correct values and predictions:
  
  plot(Xs_new[ys==1,2],Xs_new[ys==1,3],col="red",pch=19,xlab="feature1",ylab="feature2",xlim=c(-1,4),ylim=c(0,4))
  points(Xs_new[ys==2,2],Xs_new[ys==2,3],col="blue",pch=19)
  
  plot(Xs_new[preds==1,2],Xs_new[preds==1,3],col="red",pch=19,xlab="feature1",ylab="feature2",xlim=c(-1,4),ylim=c(0,4))
  points(Xs_new[preds==2,2],Xs_new[preds==2,3],col="blue",pch=19)

}


Fit_With_SMR <- function(Xs_with_1,ys)
{
  require("rstan")
  num_classes <- max(ys)
  print(num_classes)
  num_data_points <- nrow(Xs_with_1)
  print(num_data_points)
  num_features <- ncol(Xs_with_1) - 1
  data_stan <- list(K=num_classes,N=num_data_points,D=num_features,y=ys,x=Xs_with_1)
  beta_matrix_init <- matrix( rnorm(num_classes*(num_features+1),mean=0,sd=1),nrow=num_classes,ncol=num_features+1)
  init_stan <- list(list(beta=beta_matrix_init))
  stan_fit <- stan(file="LogisticRegression2.stan",data=data_stan,init=init_stan,iter=200,chain=1)
  return(stan_fit)
}

#data 
#{
#  int K; // number of classes
#  int N; // number of data points (examples)
 # int D; // number of features 
#  int y[N]; // classes
#  vector[D+1] x[N]; // feature-values for each example 
#}

#parameters 
#{
##  matrix[K,D+1] beta;
#}



Create_GM_Data <- function()
{
  require("MASS")
  num_classes <- 2
  dim <- 2 ## 2-dimensional data
  
  num_data_points <- 300
  
  ## class-membership-probabilities:
  probs <- c(0.7,0.3)
  
  
  ## mu's for the individual classes:
  mus <- array(0,dim=c(num_classes,dim))
  mus[1,] <- c(1,2)
  mus[2,] <- c(2,1)
  
  ## covariance matrices for the classes:
  sigmas <- array(0,dim=c(num_classes,dim,dim))
  sigmas[1,,] <- matrix(c(0.15,0,0,0.15),nrow=2,ncol=2)
  sigmas[2,,] <- matrix(c(0.15,0,0,0.15),nrow=2,ncol=2)
  print(sigmas[1,,])
  
  
  class_labels <- sample(x=num_classes, size=num_data_points, replace = TRUE, prob = probs)
  
  Xs <- array(0,dim=c(num_data_points,dim))
  for(i in 1:num_data_points)
  {
    Xs[i,] <- mvrnorm(n=1,mu=mus[class_labels[i],],Sigma=sigmas[class_labels[i],,])  
  }
  
  list_synth_data <- list(Xs=Xs,ys=class_labels)
  return(list_synth_data)
}

## Plot Data colored by class-membership..
## list_synth_data is supposed to contain an element
## "Xs" (n_samples x n_features) and an element 
## "ys" (the class-membership index for each sample)
Plot_Synth_Data <- function(list_synth_data)
{
  colors <- c("red","blue","green","orange","yellow")
  num_classes <- max(list_synth_data$ys)
  Xs <- list_synth_data$Xs
  ys <- list_synth_data$ys
  plot(Xs[ys==1,1],Xs[ys==1,2],col=colors[1],xlab="feature1",ylab="feature2",pch=19,xlim=c(-1,3),ylim=c(0,4))
  for(i in 2:num_classes)
  {
    points(Xs[ys==i,1],Xs[ys==i,2],col=colors[i],pch=19)
  }
}