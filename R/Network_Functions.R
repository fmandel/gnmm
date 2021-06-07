require(lme4)


### if sigmoid package not installed, the following functions are from it:
relu = function (x){
  ifelse(x >= 0, x, 0)
} 

logit = function (x){ 
  log(x/(1 - x))
}

logistic=function (x, k = 1, x0 = 0){ 
  1/(1 + exp(-k * (x - x0)))
}

sigmoid=function (x, method = c("logistic", "Gompertz", "tanh", "ReLU", 
                                "leakyReLU"), inverse = FALSE, SoftMax = FALSE, ...) 
{
  method <- match.arg(method)
  if (SoftMax == TRUE) 
    x <- SoftMax(x)
  if (method == "logistic" && inverse == FALSE) {
    return(logistic(x, ...))
  }
  else if (method == "Gompertz" && inverse == FALSE) {
    return(Gompertz(x, ...))
  }
  else if (method == "tanh") {
    return(tanh(x))
  }
  else if (method == "logistic" && inverse == TRUE) {
    return(logit(x))
  }
  else if (method == "Gompertz" && inverse == TRUE) {
    return(inverse_Gompertz(x))
  }
  else if (method == "ReLU" && inverse == FALSE) {
    return(relu(x))
  }
  else if (method == "leakyReLU" && inverse == FALSE) {
    return(leakyrelu(x))
  }
}
############### end of sigmoid functions



#### Activation Functions ####

# g0: inverse link function #
g0 <- function(d, family){
  if(family == 'binomial'){
    exp(d)/(1+exp(d))
  }
  else if (family == 'poisson'){
    exp(d)
  }
  else if (family == 'gaussian') {
    d
  }
}

g0.prime <- function(d, family){
  if(family == 'binomial'){
    exp(d)/(1+exp(d))^2
  }
  else if (family == 'poisson'){
    exp(d)
  }
  else if (family == 'gaussian') {
    1
  }
}

# g1: activation function - sigmoid or ReLU #
g1 <- function(d, type){
  if(type == 'sigmoid'){
    sigmoid(d)
  }
  else if (type == 'relu'){
    relu(d)
  }
}

g1.prime <- function(d, type){
  if(type == 'sigmoid'){
    sigmoid(d)*(1-sigmoid(d))
  }
  else if (type == 'relu'){
    ifelse(d>0, 1, 0)
  }
}



#### Derivatives ####

# Layer 0
weight0 <- function(theta, j, y, x, mu, a, var.fun, type, z, family, phi, m, n, lambda, b.vec){
  sumw0 <- vector(length = sum(n))
  for (i in 1:(sum(n))) {
    sumw0[i] <- ((y[i]-mu[i])/(a[i]*var.fun(mu[i], family = family)))*g0.prime(theta[[1]][[1]]%*%g1(theta[[1]][[2]]%*%x[i,] + 
                                                                                                      theta[[2]][[2]], type = type) + 
                                                                                 theta[[2]][[1]] + t(z[i,])%*%b.vec, family = family)*g1(theta[[1]][[2]][j,]%*%x[i,] + 
                                                                                                                                           theta[[2]][[2]][j], type = type)
  }
  full.sum <- sum(sumw0)
  (1/phi)*(1/(sum(n)))*full.sum - 2*lambda*theta[[1]][[3]][j]
}

bias0 <- function(theta, y, x, mu, a, var.fun, type, z, family, phi, m, n, lambda, b.vec){
  sumb0 <- vector(length = sum(n))
  for (i in 1:(sum(n))) {
    sumb0[i] <- ((y[i]-mu[i])/(a[i]*var.fun(mu[i], family = family)))*g0.prime(theta[[1]][[1]]%*%g1(theta[[1]][[2]]%*%x[i,] + theta[[2]][[2]], type = type) + 
                                                                                 theta[[2]][[1]] + t(z[i,])%*%b.vec, family = family)
  }
  full.sum <- sum(sumb0)
  (1/phi)*(1/(sum(n)))*full.sum - (2*lambda*theta[[2]][[3]][1])
}

# Layer 1
weight1 <- function(theta, j, k, y, x, mu, a, var.fun, type, z, family, phi, m, n, lambda, k1, b.vec){
  sumw1 <- vector(length = sum(n))
  for (i in 1:(sum(n))) {
    sumw1[i] <- ((y[i]-mu[i])/(a[i]*var.fun(mu[i], family = family)))*g0.prime(theta[[1]][[1]]%*%g1(theta[[1]][[2]]%*%x[i,] + theta[[2]][[2]], type = type) + 
                                                                                 theta[[2]][[1]] + t(z[i,])%*%b.vec, family = family)*theta[[1]][[1]][k]*g1.prime(theta[[1]][[2]][k,]%*%x[i,] + theta[[2]][[2]][k,], type = type)*x[i,j]
  }
  full.sum <- sum(sumw1)
  (1/phi)*(1/(sum(n)))*full.sum - (2*lambda*theta[[1]][[3]][j*k1+k])
}

bias1 <- function(theta, k, y, x, mu, a, var.fun, type, z, family, phi, m, n, lambda, b.vec){
  sumb1 <- vector(length = sum(n))
  for (i in 1:(sum(n))) {
    sumb1[i] <- ((y[i]-mu[i])/(a[i]*var.fun(mu[i], family = family)))*g0.prime(theta[[1]][[1]]%*%g1(theta[[1]][[2]]%*%x[i,] + theta[[2]][[2]], type = type) + theta[[2]][[1]] + 
                                                                                 t(z[i,])%*%b.vec, family = family)*theta[[1]][[1]][k]*g1.prime(theta[[1]][[2]][k,]%*%x[i,] + 
                                                                                                                                                  theta[[2]][[2]][k,], type = type)
  }
  full.sum <- sum(sumb1)
  (1/phi)*(1/(sum(n)))*full.sum - (2*lambda*theta[[2]][[3]][k+1])
}

# Layer 0 - stochastic
weight0sgd <- function(obs,theta, j, y, x, mu, a, var.fun, type, z, family, phi, m, n, lambda, b.vec){
  w0 <- ((y[obs]-mu[obs])/(a[obs]*var.fun(mu[obs], family = family)))*g0.prime(theta[[1]][[1]]%*%g1(theta[[1]][[2]]%*%x[obs,] + 
                                                                                                      theta[[2]][[2]], type = type) + 
                                                                                 theta[[2]][[1]] + t(z[obs,])%*%b.vec, family = family)*g1(theta[[1]][[2]][j,]%*%x[obs,] + 
                                                                                                                                             theta[[2]][[2]][j], type = type)
  (1/phi)*w0 - 2*lambda*theta[[1]][[1]][1,j]
}

bias0sgd <- function(obs,theta, y, x, mu, a, var.fun, type, z, family, phi, m, n, lambda, b.vec){
  b0 <- ((y[obs]-mu[obs])/(a[obs]*var.fun(mu[obs], family = family)))*g0.prime(theta[[1]][[1]]%*%g1(theta[[1]][[2]]%*%x[obs,] + theta[[2]][[2]], type = type) + 
                                                                                 theta[[2]][[1]] + t(z[obs,])%*%b.vec, family = family)
  (1/phi)*b0 - (2*lambda*theta[[2]][[1]][1,1])
}

# Layer 1 - stochastic
weight1sgd <- function(obs,theta, j, k, y, x, mu, a, var.fun, type, z, family, phi, m, n, lambda, k1, b.vec){
  w1 <- ((y[obs]-mu[obs])/(a[obs]*var.fun(mu[obs], family = family)))*g0.prime(theta[[1]][[1]]%*%g1(theta[[1]][[2]]%*%x[obs,] + theta[[2]][[2]], type = type) + 
                                                                                 theta[[2]][[1]] + t(z[obs,])%*%b.vec, family = family)*theta[[1]][[1]][k]*g1.prime(theta[[1]][[2]][k,]%*%x[obs,] + theta[[2]][[2]][k,], type = type)*x[obs,j]
  (1/phi)*w1 - (2*lambda*theta[[1]][[2]][k,j])
}

bias1sgd <- function(obs,theta, k, y, x, mu, a, var.fun, type, z, family, phi, m, n, lambda, b.vec){
  b1 <- ((y[obs]-mu[obs])/(a[obs]*var.fun(mu[obs], family = family)))*g0.prime(theta[[1]][[1]]%*%g1(theta[[1]][[2]]%*%x[obs,] + theta[[2]][[2]], type = type) + theta[[2]][[1]] + 
                                                                                 t(z[obs,])%*%b.vec, family = family)*theta[[1]][[1]][k]*g1.prime(theta[[1]][[2]][k,]%*%x[obs,] + 
                                                                                                                                                    theta[[2]][[2]][k,], type = type)
  (1/phi)*b1 - (2*lambda*theta[[2]][[2]][k,1])
}


# kappa
k.prime <- function(theta, y, x, mu, a, var.fun, type, z, family, phi, m, n, D){

  kp.matrix <- matrix(nrow = m, ncol = sum(n))
  for(i in 1:(sum(n))){
    kp.matrix[,i] <- ((y[i] - mu[i])*g0.prime(theta[[1]][[1]]%*%g1(theta[[1]][[2]]%*%x[i,] + theta[[2]][[2]], type = type)
                                              + theta[[2]][[1]] + t(z[i,])%*%theta[[3]], family = family)%*%z[i,])/(phi*a[i]*var.fun(mu[i], family = family))
  }

  kp.vector <- vector(length = m)
  for(j in 1:m){
    kp.vector[j] <- sum(kp.matrix[j,])
  }
  -kp.vector + theta[[3]]/D
}



#### GNMM training function ####
# Arguments:
# - formula: response ~ predictors + (1|subjectID)
# - family: 'gaussian' or 'binomial'
# - penalization: lambda for squared L2 penalty term
# - nodes: number of nodes in the hidden layer
# - tolerance: 
# - step_size: network learning rate
# - act_fun: activation function, can be 'relu' or 'sigmoid'
# - weights, biases: initial network parameters; if NA, randomly intialized
# - nepochs: number of training epochs
# - incl_ranef: TRUE/FALSE to include a random intercept term
gnmm.sgd <- function(formula, family, penalization, nodes, tolerance, step_size, 
                      act_fun, weights = NA, biases = NA, nepochs=10,incl_ranef=TRUE, ...){
  
  ## parse formula: y is response, x is the data frame, and lab is the identifier
  ## formula in the form: response ~ df + (1|ID)
  if(length(all.vars(formula))>3) stop("incorrect formula specified")
  
  p.env	<- environment(formula)
  # extract y component
  y <- all.vars(formula)[1]
  y	<- eval(parse(text = y), envir = p.env)
  
  # extract x component
  x	<- all.vars(formula)[2]
  x	<- eval(parse(text = x), envir = p.env)
  
  # extract ID vector
  lab <- all.vars(formula)[3]
  lab	<- eval(parse(text = lab), envir = p.env)
  
  
  # set network conditions
  p <- ncol(x)
  k1 <- nodes
  m <- length(unique(lab))
  df1 <- data.frame(y, x, lab)
  n <- aggregate(x~lab, data = df1, length)[,2]
  
  y <- y
  x <- x
  
  phi <- 1
  lambda <- penalization
  eta <- step_size
  lfamily <- family
  atype <- act_fun
  tol <- tolerance
  a <- rep(1, sum(n))
  
  
  # variance function
  var.fun <- function(x, family){
    if(family == 'binomial'){
      x*(1-x)
    }
    else if (family == 'poisson'){
      x
    }
    else if (family == 'gaussian') {
      1
    }
  }
  
  # Create z matrix
  z <- matrix(rep(0, sum(n)*m), nrow = sum(n), ncol = m)
  for (i in 1:m) {
    z[(sum(n[1:i-1])+1):sum(n[1:i]), i] <- 1
  }
  z <- z
  
  # weight matrices
  suppressWarnings(if(is.na(weights)){
    w0 <- matrix(rnorm(k1), nrow = 1, ncol = k1)
    w1 <- matrix(rnorm(k1*p), nrow = k1, ncol = p)
    joint.wv <- c(w0, w1)
    weight.mat <- list(w0, w1, joint.wv)
  } else{
    weight.mat <- weights
  })
  
  # bias vectors
  suppressWarnings(if(is.na(biases)){
    b0 <- matrix(rnorm(1), nrow = 1, ncol = 1)
    b1 <- matrix(rnorm(k1), nrow = k1, ncol = 1)
    joint.bv <- c(b0, b1)
    bias.vec <- list(b0, b1, joint.bv)
  } else{
    bias.vec <- biases
  })
  
  # b vector
  rname <- noquote(all.vars(formula)[3])
  suppressWarnings(glmm1 <- glmer(formula, family = family))
  b <- as.vector(c(ranef(glmm1)[[rname]]))
  b <- unname(b)
  b <- unlist(b)
  D <- as.data.frame(summary(glmm1)$varcor)$vcov[1]
  if(D<0.05){
    incl_ranef=FALSE
  }
  if(!incl_ranef){
    b <- rep(0,length(b))
  }
  
  # initial theta
  theta <- list(weight.mat, bias.vec, b)
  

  ### maximization ###
  # run through network / calculate derivatives / update theta
  nepoch=0
  while(TRUE){
    nepoch=nepoch+1
    if(nepoch>nepochs){
      break
    }
    obs_v = sample(1:(sum(n)),replace=F)
    for(iii in 1:(sum(n))){
      ## run through network with x[i,] and get mu
      alpha1=rep(NA,k1*sum(n))
      mu <- rep(NA,sum(n))
      for(i in 1:m){
        curinds=(sum(n[0:(i-1)])+1):sum(n[1:i])
        full_curinds2 = (sum(n[0:(i-1)])*k1 + 1):(k1*sum(n[1:i]))
        for(j in curinds){
          trj = j-curinds[1]+1
          curinds2 = full_curinds2[((trj-1)*k1 +1):(trj*k1)]
          alpha.i1 <- vector(length = k1)
          alpha.i1 <- g1(theta[[1]][[2]]%*%x[j,] + theta[[2]][[2]], type = atype)
          alpha1[curinds2] <- alpha.i1
          mu.i <- g0(theta[[1]][[1]]%*%alpha1[curinds2] + theta[[2]][[1]] + 
                       t(z[j,])%*%theta[[3]], family = lfamily)
          mu[(sum(n[0:(i-1)])+1)+(trj-1)*1] <- mu.i
        }
      }
      
      # calculate derivatives and put into matrices
      obs = obs_v[iii]
      
      # row vector of weight0 derivatives 
      delta.weight0 <- vector(length = k1)
      for (j in 1:k1){
        delta.weight0[j] <- t(weight0sgd(obs,theta, j = j, y=y, x=x, mu=mu, a=a, 
                                         var.fun=var.fun, type = atype, z=z,
                                         family = lfamily, phi = phi, m=m, n=n,
                                         lambda=lambda, b.vec = theta[[3]]))
      }
      
      # matrix of weight1 derivatives
      delta.weight1 <- matrix(nrow = k1, ncol = p)
      for (k in 1:k1) {
        for (j in 1:p) {
          delta.weight1[k,j] <- weight1sgd(obs,theta, j=j, k=k, y=y, x=x, mu=mu,
                                           a=a, var.fun=var.fun, type = atype, 
                                           z=z, family = lfamily, phi = phi, m=m, 
                                           n=n, lambda=lambda, k1=k1, b.vec = theta[[3]])
        }
      }
      
      # scalar bias0 derivative
      delta.bias0 <- bias0sgd(obs,theta, y=y, x=x, mu=mu, a=a, var.fun=var.fun, type = atype,
                              z=z, family = lfamily, phi = phi, m=m, n=n, lambda=lambda,
                              b.vec = theta[[3]])
      
      # column vector of bias1 derivatives
      delta.bias1 <- vector(length = k1)
      for (k in 1:k1) {
        delta.bias1[k] <- bias1sgd(obs,theta, k = k, y=y, x=x, mu=mu, a=a, var.fun=var.fun,
                                   type = atype, z=z, family = lfamily, phi = phi, m=m,
                                   n=n, lambda=lambda, b.vec = theta[[3]])
      }
      
      # column vector of kappa prime 
      delta.b <- k.prime(theta, y=y, x=x, mu=mu, a=a, var.fun=var.fun, type = atype,
                         z=z, family = lfamily, phi = phi, m=m, n=n, D = D)
      
      
      ## update weights, biases, and b
      theta[[1]][[1]] <- theta[[1]][[1]] + (eta*delta.weight0)
      theta[[1]][[2]] <- theta[[1]][[2]] + (eta*delta.weight1)
      theta[[1]][[3]] <- c(theta[[1]][[1]], theta[[1]][[2]])
      theta[[2]][[1]] <- theta[[2]][[1]] + (eta*delta.bias0)
      theta[[2]][[2]] <- theta[[2]][[2]] + (eta*delta.bias1)
      theta[[2]][[3]] <- c(theta[[2]][[1]], theta[[2]][[2]])
      if(incl_ranef){
        theta[[3]]      <- theta[[3]] - (eta*delta.b)
      }
      
      
      ## print training progress
      if(iii==sum(n)){
        cat('Epoch ',nepoch,': ',
            'MSE = ',mean((y-mu)^2, na.rm = T), '\n', sep = '')
      }
      
    }
  }
  
  ## function return
  fit			<- list()
  fit$weights		<- theta[[1]]
  fit$biases		<- theta[[2]]
  fit$ranef		<- theta[[3]]
  fit$ID <- lab
  fit$obspp <- n
  fit$zmat <- z
  fit$activation <- atype
  fit$family <- lfamily
  class(fit)		<- "gnmm"
  return(fit)
}


#### GNMM prediction function ####
# Arguments:
# - new_data: matrix of predictors (must be the same as gnmm.fit)
# - id: vector of subject IDs for new_data (must have at least one obs in training data)
# - gnmm.fit: object of type 'gnmm'
gnmm.predict <- function(new_data, id, gnmm.fit){
  if(!class(gnmm.fit)=='gnmm') stop("must use object of class gnmm")
  
  # put estimated weights, biases, and random effects into theta
  theta <- list(gnmm.fit$weights, gnmm.fit$biases, gnmm.fit$ranef)
  
  # save network settings
  k1 <- length(gnmm.fit$biases[[2]])
  N <- nrow(new_data)
  z <- gnmm.fit$zmat
  zind <- gnmm.fit$obspp
  atype <- gnmm.fit$activation
  lfamily <- gnmm.fit$family
  
  # run through network to get response estimates
  mu <- rep(NA,N)
  for(i in 1:N){
    subj <- which(unique(gnmm.fit$ID)==id[i])
    alpha.i1 <- vector(length = k1)
    alpha.i1 <- g1(theta[[1]][[2]]%*%new_data[i,] + theta[[2]][[2]], type = atype)
    mu.i <- g0(theta[[1]][[1]]%*%alpha.i1 + theta[[2]][[1]] + t(z[(sum(zind[1:subj])),])%*%theta[[3]], family = lfamily)
    mu[i] <- mu.i
  }
  return(mu)
}
