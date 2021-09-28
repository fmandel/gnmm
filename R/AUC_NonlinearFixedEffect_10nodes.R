## Simulation 6: AUC, nonlinear fixed effect, 10 nodes ##

library(foreach)
library(doSNOW)
library(doRNG)

source("Network_Functions.R")

nsim <- 500
ncores <- 10
cl <- makeCluster(ncores)
registerDoSNOW(cl)
getDoParWorkers()

pb <- txtProgressBar(min = 0, max = nsim, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

set.seed(120412387)

#################################################################################################################
psims <- foreach(a = 1:nsim, .options.snow = opts) %dorng% {
  library(lme4)
  tau_v <- seq(0,20,4)
  m <- 100
  
  # Simulation 6: nonlinear, 10 nodes
  pred.matrix <- matrix(NA, nrow = m*6, ncol = 6)
  for(bb in 1:length(tau_v)){
    tau <- tau_v[bb]
    
    #### Simulate nonlinear binary data
    p <- 6
    m <- 100
    n <- rpois(m,6)+2
    nepochs <- 20
    betas <- rnorm(3,3,1)
    sigma <- .05
    hidnodes1 <- 10
    hidnodes2 <- 5
    
    lab <- rep(NA,sum(n))
    Y <- rep(NA,sum(n))
    pp <- rep(NA,sum(n))
    X <- matrix(NA,ncol=p,nrow=sum(n))
    for(i in 1:m){
      curinds <- (sum(n[0:(i-1)])+1):sum(n[1:i])
      lab[curinds] <- i
      X[curinds,] <- rbinom(n[i]*p,1,0.5)
      ranef <- rnorm(1,mean=0,sd=sqrt(tau))
      for(j in curinds){
        y1 <- ifelse(X[j,1]==1 & X[j,2]==0 | X[j,1]==0 & X[j,2]==1, 1, 0)
        y2 <- ifelse(X[j,3]==1 & X[j,4]==1 | X[j,3]==0 & X[j,4]==0, 1, 0)
        y3 <- ifelse(X[j,5]==1 & X[j,6]==0 | X[j,5]==0 & X[j,6]==1, 1, 0)
        s.comp <- ranef + betas[1]*y1 + betas[2]*y2 + betas[3]*y3 + rnorm(1, mean = 0, sd=sigma)
        pp[j] <- exp(s.comp)/(1+exp(s.comp))
        Y[j] <- rbinom(1,1,prob = pp[j])
      }
    }
    
    
    #### Partition data
    cum.n <- rep(NA, length(n))
    for (i in 1:length(n)) {
      cum.n[i] <- sum(n[1:i])
    }
    
    ## Training data: Y.train ~ X.train + (1|lab.train)
    Y.train <- Y[-cum.n]
    X.train <- X[-cum.n,]
    lab.train <- lab[-cum.n]
    
    # Last observation Data: Y.everyn ~ X.everyn
    Y.everyn <- Y[cum.n]
    X.everyn <- X[cum.n,]
    lab.everyn <- lab[cum.n]
    
    ## set up data for maity-pal method
    subject.indicator <- matrix(rep(0, m*sum(n)), nrow = sum(n), ncol = m)
    for (i in 1:m) {
      subject.indicator[(sum(n[0:(i-1)])+1):(sum(n[1:i])),i] <- rep(1, n[i])
    }
    X.mp <- cbind(X, subject.indicator)
    X.mp.train <- X.mp[-cum.n,]
    X.mp.everyn <- X.mp[cum.n,]
    
    #### Run 5 models
    
    ### Run GNMM and predict last observation
    m1 <- gnmm.sgd(formula = Y.train ~ X.train + (1|lab.train), family = 'binomial', penalization = 0.001,
                   nodes1 = hidnodes1, nodes2 = NULL, step_size = 0.025, act_fun = 'relu', nepochs = nepochs, incl_ranef = TRUE)
    pred1 <- gnmm.predict(new_data = X.everyn, id = lab.everyn, gnmm.fit = m1)
    
    ### Run GNMM and predict last observation
    m2 <- gnmm.sgd(formula = Y.train ~ X.train + (1|lab.train), family = 'binomial', penalization = 0.005,
                   nodes1 = hidnodes1, nodes2 = hidnodes2, step_size = 0.01, act_fun = 'relu', nepochs = nepochs, incl_ranef = TRUE)
    pred2 <- gnmm.predict(new_data = X.everyn, id = lab.everyn, gnmm.fit = m2)

    ### Run ANN and predict last observation
    m3 <- gnmm.sgd(formula = Y.train ~ X.train + (1|lab.train), family = 'binomial', penalization = 0.001,
                   nodes1 = hidnodes1, nodes2 = NULL, step_size = 0.025, act_fun = 'relu', nepochs = nepochs, incl_ranef = FALSE)
    pred3 <- gnmm.predict(new_data = X.everyn, id = lab.everyn, gnmm.fit = m3)
    
    ### Run GLMER and predict last observation
    glmm.out <- glmer(Y.train ~ X.train + (1|lab.train), family = 'binomial')
    s1 <- summary(glmm.out)
    glmm.coef <- matrix(s1$coefficients[2:(p+1),1])
    fef.pred <- X.everyn%*%glmm.coef + rep(matrix(s1$coefficients[1,1]),m)     ## fixed effect prediction
    glm.ref <- as.vector(c(ranef(glmm.out)$lab.train))
    glm.ref <- unname(glm.ref)
    glm.ref <- unlist(glm.ref)
    pred4 <- fef.pred + glm.ref             ## fixed effect + random effect prediction
    
    ### Run ANN with inputs for each individual (Maity-Pal)
    m5 <- gnmm.sgd(formula = Y.train ~ X.mp.train + (1|lab.train), family = 'binomial', penalization = 0.001,
                   nodes1 = hidnodes1, nodes2 = NULL, step_size = 0.025, act_fun = 'relu', nepochs = nepochs, incl_ranef = FALSE)
    pred5 <- gnmm.predict(new_data = X.mp.everyn, id = lab.everyn, gnmm.fit = m5)
    
    ### Fill in prediction values
    pred.matrix[((bb-1)*m+1):(bb*m),1] <- Y.everyn
    pred.matrix[((bb-1)*m+1):(bb*m),2] <- pred1
    pred.matrix[((bb-1)*m+1):(bb*m),3] <- pred2
    pred.matrix[((bb-1)*m+1):(bb*m),4] <- pred3
    pred.matrix[((bb-1)*m+1):(bb*m),5] <- pred4
    pred.matrix[((bb-1)*m+1):(bb*m),6] <- pred5
    
  }
  
  rbind(pred.matrix[(1:(6*m)),])
  
}


#### Calculate AUC
require(pROC)
bin_n10 <- psims
m <- 100
auc_gnmm1_n10 <- list(); auc_gnmm2_n10 <- list()
auc_ann_n10 <- list(); auc_glmm_n10 <- list(); auc_mp_n10 <- list()

## tau = 0
auc_gnmm1_n10[[1]] <- rep(NA, 500); auc_gnmm2_n10[[1]] <- rep(NA, 500)
auc_ann_n10[[1]] <- rep(NA, 500); auc_glmm_n10[[1]] <- rep(NA, 500); auc_mp_n10[[1]] <- rep(NA, 500)
for (i in 1:500) {
  auc_gnmm1_n10[[1]][i] <- auc(roc(bin_n10[[i]][1:m,1] ~ bin_n10[[i]][1:m,2]))
  if(!is.na(bin_n10[[i]][1:m,3][1])) 
    auc_gnmm2_n10[[1]][i] <- auc(roc(bin_n10[[i]][1:m,1] ~ bin_n10[[i]][1:m,3]))
  auc_ann_n10[[1]][i] <- auc(roc(bin_n10[[i]][1:m,1] ~ bin_n10[[i]][1:m,4]))
  auc_glmm_n10[[1]][i] <- auc(roc(bin_n10[[i]][1:m,1] ~ bin_n10[[i]][1:m,5]))
  auc_mp_n10[[1]][i] <- auc(roc(bin_n10[[i]][1:m,1] ~ bin_n10[[i]][1:m,6]))
}

## tau = 4
auc_gnmm1_n10[[2]] <- rep(NA, 500); auc_gnmm2_n10[[2]] <- rep(NA, 500)
auc_ann_n10[[2]] <- rep(NA, 500); auc_glmm_n10[[2]] <- rep(NA, 500); auc_mp_n10[[2]] <- rep(NA, 500)
for (i in 1:500) {
  auc_gnmm1_n10[[2]][i] <- auc(roc(bin_n10[[i]][(m+1):(2*m),1] ~ bin_n10[[i]][(m+1):(2*m),2]))
  if(!is.na(bin_n10[[i]][(m+1):(2*m),3][1])) 
    auc_gnmm2_n10[[2]][i] <- auc(roc(bin_n10[[i]][(m+1):(2*m),1] ~ bin_n10[[i]][(m+1):(2*m),3]))
  auc_ann_n10[[2]][i] <- auc(roc(bin_n10[[i]][(m+1):(2*m),1] ~ bin_n10[[i]][(m+1):(2*m),4]))
  auc_glmm_n10[[2]][i] <- auc(roc(bin_n10[[i]][(m+1):(2*m),1] ~ bin_n10[[i]][(m+1):(2*m),5]))
  auc_mp_n10[[2]][i] <- auc(roc(bin_n10[[i]][(m+1):(2*m),1] ~ bin_n10[[i]][(m+1):(2*m),6]))
}

## tau = 8
auc_gnmm1_n10[[3]] <- rep(NA, 500); auc_gnmm2_n10[[3]] <- rep(NA, 500) 
auc_ann_n10[[3]] <- rep(NA, 500); auc_glmm_n10[[3]] <- rep(NA, 500); auc_mp_n10[[3]] <- rep(NA, 500)
for (i in 1:500) {
  auc_gnmm1_n10[[3]][i] <- auc(roc(bin_n10[[i]][(2*m+1):(3*m),1] ~ bin_n10[[i]][(2*m+1):(3*m),2]))
  if(!is.na(bin_n10[[i]][(2*m+1):(3*m),3][1])) 
    auc_gnmm2_n10[[3]][i] <- auc(roc(bin_n10[[i]][(2*m+1):(3*m),1] ~ bin_n10[[i]][(2*m+1):(3*m),3]))
  auc_ann_n10[[3]][i] <- auc(roc(bin_n10[[i]][(2*m+1):(3*m),1] ~ bin_n10[[i]][(2*m+1):(3*m),4]))
  auc_glmm_n10[[3]][i] <- auc(roc(bin_n10[[i]][(2*m+1):(3*m),1] ~ bin_n10[[i]][(2*m+1):(3*m),5]))
  auc_mp_n10[[3]][i] <- auc(roc(bin_n10[[i]][(2*m+1):(3*m),1] ~ bin_n10[[i]][(2*m+1):(3*m),6]))
}

## tau = 12
auc_gnmm1_n10[[4]] <- rep(NA, 500); auc_gnmm2_n10[[4]] <- rep(NA, 500)
auc_ann_n10[[4]] <- rep(NA, 500); auc_glmm_n10[[4]] <- rep(NA, 500); auc_mp_n10[[4]] <- rep(NA, 500)
for (i in 1:500) {
  auc_gnmm1_n10[[4]][i] <- auc(roc(bin_n10[[i]][(3*m+1):(4*m),1] ~ bin_n10[[i]][(3*m+1):(4*m),2]))
  if(!is.na(bin_n10[[i]][(3*m+1):(4*m),3][1])) 
    auc_gnmm2_n10[[4]][i] <- auc(roc(bin_n10[[i]][(3*m+1):(4*m),1] ~ bin_n10[[i]][(3*m+1):(4*m),3]))
  auc_ann_n10[[4]][i] <- auc(roc(bin_n10[[i]][(3*m+1):(4*m),1] ~ bin_n10[[i]][(3*m+1):(4*m),4]))
  auc_glmm_n10[[4]][i] <- auc(roc(bin_n10[[i]][(3*m+1):(4*m),1] ~ bin_n10[[i]][(3*m+1):(4*m),5]))
  auc_mp_n10[[4]][i] <- auc(roc(bin_n10[[i]][(3*m+1):(4*m),1] ~ bin_n10[[i]][(3*m+1):(4*m),6]))
}

## tau = 16
auc_gnmm1_n10[[5]] <- rep(NA, 500); auc_gnmm2_n10[[5]] <- rep(NA, 500)
auc_ann_n10[[5]] <- rep(NA, 500); auc_glmm_n10[[5]] <- rep(NA, 500); auc_mp_n10[[5]] <- rep(NA, 500)
for (i in 1:500) {
  auc_gnmm1_n10[[5]][i] <- auc(roc(bin_n10[[i]][(4*m+1):(5*m),1] ~ bin_n10[[i]][(4*m+1):(5*m),2]))
  if(!is.na(bin_n10[[i]][(4*m+1):(5*m),3][1])) 
    auc_gnmm2_n10[[5]][i] <- auc(roc(bin_n10[[i]][(4*m+1):(5*m),1] ~ bin_n10[[i]][(4*m+1):(5*m),3]))
  auc_ann_n10[[5]][i] <- auc(roc(bin_n10[[i]][(4*m+1):(5*m),1] ~ bin_n10[[i]][(4*m+1):(5*m),4]))
  auc_glmm_n10[[5]][i] <- auc(roc(bin_n10[[i]][(4*m+1):(5*m),1] ~ bin_n10[[i]][(4*m+1):(5*m),5]))
  auc_mp_n10[[5]][i] <- auc(roc(bin_n10[[i]][(4*m+1):(5*m),1] ~ bin_n10[[i]][(4*m+1):(5*m),6]))
}

## tau = 20
auc_gnmm1_n10[[6]] <- rep(NA, 500); auc_gnmm2_n10[[6]] <- rep(NA, 500) 
auc_ann_n10[[6]] <- rep(NA, 500); auc_glmm_n10[[6]] <- rep(NA, 500); auc_mp_n10[[6]] <- rep(NA, 500)
for (i in 1:500) {
  auc_gnmm1_n10[[6]][i] <- auc(roc(bin_n10[[i]][(5*m+1):(6*m),1] ~ bin_n10[[i]][(5*m+1):(6*m),2]))
  if(!is.na(bin_n10[[i]][(5*m+1):(6*m),3][1])) 
    auc_gnmm2_n10[[6]][i] <- auc(roc(bin_n10[[i]][(5*m+1):(6*m),1] ~ bin_n10[[i]][(5*m+1):(6*m),3]))
  auc_ann_n10[[6]][i] <- auc(roc(bin_n10[[i]][(5*m+1):(6*m),1] ~ bin_n10[[i]][(5*m+1):(6*m),4]))
  auc_glmm_n10[[6]][i] <- auc(roc(bin_n10[[i]][(5*m+1):(6*m),1] ~ bin_n10[[i]][(5*m+1):(6*m),5]))
  auc_mp_n10[[6]][i] <- auc(roc(bin_n10[[i]][(5*m+1):(6*m),1] ~ bin_n10[[i]][(5*m+1):(6*m),6]))
}

# calculate average AUC
AUC_mat_n10 <- matrix(NA, nrow = 6, ncol = 6)
AUC_mat_n10[1,] <- seq(0,20,4)
AUC_mat_n10[2,] <- sapply(auc_gnmm1_n10, function(x) mean(x, na.rm = T))
AUC_mat_n10[3,] <- sapply(auc_gnmm2_n10, function(x) mean(x, na.rm = T))
AUC_mat_n10[4,] <- sapply(auc_ann_n10, function(x) mean(x, na.rm = T))
AUC_mat_n10[5,] <- sapply(auc_glmm_n10, function(x) mean(x, na.rm = T))
AUC_mat_n10[6,] <- sapply(auc_mp_n10, function(x) mean(x, na.rm = T))


# AUC plot (top right panel of Figure 3)
plot(AUC_mat_n10[1,], AUC_mat_n10[2,], type='l', ylim=c(0.5,1), xlab=expression(tau), ylab='AUC', 
     main='Nonlinear fixed effect, 10 Nodes', lwd = 2)
lines(AUC_mat_n10[1,], AUC_mat_n10[3,], col='Gray', lty = 2, lwd = 2)
lines(AUC_mat_n10[1,], AUC_mat_n10[4,], col='Blue', lty = 3, lwd = 2)
lines(AUC_mat_n10[1,], AUC_mat_n10[5,], col='Red', lty = 4, lwd = 2)
lines(AUC_mat_n10[1,], AUC_mat_n10[6,], col='green3', lty = 5, lwd = 2)
