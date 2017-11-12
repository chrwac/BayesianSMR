data 
{
  int K; // number of classes
  int N; // number of data points (examples)
  int D; // number of features 
  int y[N]; // classes
  vector[D+1] x[N]; // feature-values for each example 
}

parameters 
{
  matrix[K,D+1] beta;
}

model 
{
  for (k in 1:K)
    beta[k] ~ normal(0, 5);
  for (n in 1:N)
    y[n] ~ categorical(softmax(beta * x[n]));
}
