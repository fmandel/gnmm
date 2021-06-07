## Simulation 5: AUC, nonlinear fixed effect, 5 nodes ##

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
  tau_v = seq(0,20,4)
  m=100
  
  # Simulation 5: nonlinear, 5 nodes
  pred.matrix <- matrix(NA, nrow = m*6, ncol = 5)
  for(bb in 1:length(tau_v)){
    tau=tau_v[bb]
    
    #### Simulate nonlinear binary data
    p = 6
    m = 100
    n = rpois(m,6)+2
    nepochs=5
    betas = rnorm(3,3,1)
    sigma=.05
    hidnodes=5
    
    lab = rep(NA,sum(n))
    Y = rep(NA,sum(n))
    pp = rep(NA,sum(n))
    X = matrix(NA,ncol=p,nrow=sum(n))
    for(i in 1:m){
      curinds=(sum(n[0:(i-1)])+1):sum(n[1:i])
      lab[curinds]=i
      X[curinds,]= rbinom(n[i]*p,1,0.5)
      ranef=rnorm(1,mean=0,sd=sqrt(tau))
      for(j in curinds){
        y1 <- ifelse(X[j,1]==1 & X[j,2]==0 | X[j,1]==0 & X[j,2]==1, 1, 0)
        y2 <- ifelse(X[j,3]==1 & X[j,4]==1 | X[j,3]==0 & X[j,4]==0, 1, 0)
        y3 <- ifelse(X[j,5]==1 & X[j,6]==0 | X[j,5]==0 & X[j,6]==1, 1, 0)
        s.comp <- ranef + betas[1]*y1 + betas[2]*y2 + betas[3]*y3 + rnorm(1, mean = 0, sd=sigma)
        pp[j] = exp(s.comp)/(1+exp(s.comp))
        Y[j]= rbinom(1,1,prob = pp[j])
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
    
    #### Run 4 models
    
    ### Run GNMM and predict last observation
    m1 <- gnmm.sgd(formula = Y.train ~ X.train + (1|lab.train), family = 'binomial', penalization = 0.001,
                   nodes = hidnodes, tolerance = 10^-8, step_size = 0.05, act_fun = 'relu', nepochs = nepochs, incl_ranef = TRUE)
    pred1 <- gnmm.predict(new_data = X.everyn, id = lab.everyn, gnmm.fit = m1)
    
    ### Run ANN and predict last observation
    m2 <- gnmm.sgd(formula = Y.train ~ X.train + (1|lab.train), family = 'binomial', penalization = 0.001,
                   nodes = hidnodes, tolerance = 10^-8, step_size = 0.05, act_fun = 'relu', nepochs = nepochs, incl_ranef = FALSE)
    pred2 <- gnmm.predict(new_data = X.everyn, id = lab.everyn, gnmm.fit = m2)
    
    ### Run GLMER and predict last observation
    glmm.out <- glmer(Y.train ~ X.train + (1|lab.train), family = 'binomial')
    s1 <- summary(glmm.out)
    glmm.coef <- matrix(s1$coefficients[2:(p+1),1])
    fef.pred <- X.everyn%*%glmm.coef + rep(matrix(s1$coefficients[1,1]),m)     ## fixed effect prediction
    glm.ref <- as.vector(c(ranef(glmm.out)$lab.train))
    glm.ref <- unname(glm.ref)
    glm.ref <- unlist(glm.ref)
    pred3 <- fef.pred + glm.ref             ## fixed effect + random effect prediction
    
    ### Run ANN with inputs for each individual (Maity-Pal)
    m4 <- gnmm.sgd(formula = Y.train ~ X.mp.train + (1|lab.train), family = 'binomial', penalization = 0.001,
                   nodes = hidnodes, tolerance = 10^-8, step_size = 0.05, act_fun = 'relu', nepochs = nepochs, incl_ranef = FALSE)
    pred4 <- gnmm.predict(new_data = X.mp.everyn, id = lab.everyn, gnmm.fit = m4)
    
    ### Fill in prediction values
    pred.matrix[((bb-1)*m+1):(bb*m),1] <- Y.everyn
    pred.matrix[((bb-1)*m+1):(bb*m),2] <- pred1
    pred.matrix[((bb-1)*m+1):(bb*m),3] <- pred2
    pred.matrix[((bb-1)*m+1):(bb*m),4] <- pred3
    pred.matrix[((bb-1)*m+1):(bb*m),5] <- pred4
    
  }
  
  rbind(pred.matrix[(1:(6*m)),])
  
}

#### Calculate AUC
require(pROC)
bin_n5 <- psims
m=100

## tau = 0
auc_gnmm_0_n5 <- rep(NA, 500); auc_ann_0_n5 <- rep(NA, 500)
auc_glmm_0_n5 <- rep(NA, 500); auc_mp_0_n5 <- rep(NA, 500)
for (i in 1:500) {
  auc_gnmm_0_n5[i] <- auc(roc(bin_n5[[i]][1:m,1] ~ bin_n5[[i]][1:m,2]))
  auc_ann_0_n5[i] <- auc(roc(bin_n5[[i]][1:m,1] ~ bin_n5[[i]][1:m,3]))
  auc_glmm_0_n5[i] <- auc(roc(bin_n5[[i]][1:m,1] ~ bin_n5[[i]][1:m,4]))
  auc_mp_0_n5[i] <- auc(roc(bin_n5[[i]][1:m,1] ~ bin_n5[[i]][1:m,5]))
}

## tau = 4
auc_gnmm_4_n5 <- rep(NA, 500); auc_ann_4_n5 <- rep(NA, 500)
auc_glmm_4_n5 <- rep(NA, 500); auc_mp_4_n5 <- rep(NA, 500)
for (i in 1:500) {
  auc_gnmm_4_n5[i] <- auc(roc(bin_n5[[i]][(m+1):(2*m),1] ~ bin_n5[[i]][(m+1):(2*m),2]))
  auc_ann_4_n5[i] <- auc(roc(bin_n5[[i]][(m+1):(2*m),1] ~ bin_n5[[i]][(m+1):(2*m),3]))
  auc_glmm_4_n5[i] <- auc(roc(bin_n5[[i]][(m+1):(2*m),1] ~ bin_n5[[i]][(m+1):(2*m),4]))
  auc_mp_4_n5[i] <- auc(roc(bin_n5[[i]][(m+1):(2*m),1] ~ bin_n5[[i]][(m+1):(2*m),5]))
}

## tau = 8
auc_gnmm_8_n5 <- rep(NA, 500); auc_ann_8_n5 <- rep(NA, 500)
auc_glmm_8_n5 <- rep(NA, 500); auc_mp_8_n5 <- rep(NA, 500)
for (i in 1:500) {
  auc_gnmm_8_n5[i] <- auc(roc(bin_n5[[i]][(2*m+1):(3*m),1] ~ bin_n5[[i]][(2*m+1):(3*m),2]))
  auc_ann_8_n5[i] <- auc(roc(bin_n5[[i]][(2*m+1):(3*m),1] ~ bin_n5[[i]][(2*m+1):(3*m),3]))
  auc_glmm_8_n5[i] <- auc(roc(bin_n5[[i]][(2*m+1):(3*m),1] ~ bin_n5[[i]][(2*m+1):(3*m),4]))
  auc_mp_8_n5[i] <- auc(roc(bin_n5[[i]][(2*m+1):(3*m),1] ~ bin_n5[[i]][(2*m+1):(3*m),5]))
}

## tau = 12
auc_gnmm_12_n5 <- rep(NA, 500); auc_ann_12_n5 <- rep(NA, 500)
auc_glmm_12_n5 <- rep(NA, 500); auc_mp_12_n5 <- rep(NA, 500)
for (i in 1:500) {
  auc_gnmm_12_n5[i] <- auc(roc(bin_n5[[i]][(3*m+1):(4*m),1] ~ bin_n5[[i]][(3*m+1):(4*m),2]))
  auc_ann_12_n5[i] <- auc(roc(bin_n5[[i]][(3*m+1):(4*m),1] ~ bin_n5[[i]][(3*m+1):(4*m),3]))
  auc_glmm_12_n5[i] <- auc(roc(bin_n5[[i]][(3*m+1):(4*m),1] ~ bin_n5[[i]][(3*m+1):(4*m),4]))
  auc_mp_12_n5[i] <- auc(roc(bin_n5[[i]][(3*m+1):(4*m),1] ~ bin_n5[[i]][(3*m+1):(4*m),5]))
}

## tau = 16
auc_gnmm_16_n5 <- rep(NA, 500); auc_ann_16_n5 <- rep(NA, 500)
auc_glmm_16_n5 <- rep(NA, 500); auc_mp_16_n5 <- rep(NA, 500)
for (i in 1:500) {
  auc_gnmm_16_n5[i] <- auc(roc(bin_n5[[i]][(4*m+1):(5*m),1] ~ bin_n5[[i]][(4*m+1):(5*m),2]))
  auc_ann_16_n5[i] <- auc(roc(bin_n5[[i]][(4*m+1):(5*m),1] ~ bin_n5[[i]][(4*m+1):(5*m),3]))
  auc_glmm_16_n5[i] <- auc(roc(bin_n5[[i]][(4*m+1):(5*m),1] ~ bin_n5[[i]][(4*m+1):(5*m),4]))
  auc_mp_16_n5[i] <- auc(roc(bin_n5[[i]][(4*m+1):(5*m),1] ~ bin_n5[[i]][(4*m+1):(5*m),5]))
}

## tau = 20
auc_gnmm_20_n5 <- rep(NA, 500); auc_ann_20_n5 <- rep(NA, 500)
auc_glmm_20_n5 <- rep(NA, 500); auc_mp_20_n5 <- rep(NA, 500)
for (i in 1:500) {
  auc_gnmm_20_n5[i] <- auc(roc(bin_n5[[i]][(5*m+1):(6*m),1] ~ bin_n5[[i]][(5*m+1):(6*m),2]))
  auc_ann_20_n5[i] <- auc(roc(bin_n5[[i]][(5*m+1):(6*m),1] ~ bin_n5[[i]][(5*m+1):(6*m),3]))
  auc_glmm_20_n5[i] <- auc(roc(bin_n5[[i]][(5*m+1):(6*m),1] ~ bin_n5[[i]][(5*m+1):(6*m),4]))
  auc_mp_20_n5[i] <- auc(roc(bin_n5[[i]][(5*m+1):(6*m),1] ~ bin_n5[[i]][(5*m+1):(6*m),5]))
}

# calculate average AUC
AUC_mat_n5 <- matrix(NA, nrow = 5, ncol = 6)
AUC_mat_n5[1,] <- seq(0,20,4)
AUC_mat_n5[2,] <- c(mean(auc_gnmm_0_n5, na.rm = TRUE), mean(auc_gnmm_4_n5, na.rm = TRUE),
                    mean(auc_gnmm_8_n5, na.rm = TRUE), mean(auc_gnmm_12_n5, na.rm = TRUE),
                    mean(auc_gnmm_16_n5, na.rm = TRUE), mean(auc_gnmm_20_n5, na.rm = TRUE))
AUC_mat_n5[3,] <- c(mean(auc_ann_0_n5, na.rm = TRUE), mean(auc_ann_4_n5, na.rm = TRUE),
                    mean(auc_ann_8_n5, na.rm = TRUE), mean(auc_ann_12_n5, na.rm = TRUE),
                    mean(auc_ann_16_n5, na.rm = TRUE), mean(auc_ann_20_n5, na.rm = TRUE))
AUC_mat_n5[4,] <- c(mean(auc_glmm_0_n5, na.rm = TRUE), mean(auc_glmm_4_n5, na.rm = TRUE),
                    mean(auc_glmm_8_n5, na.rm = TRUE), mean(auc_glmm_12_n5, na.rm = TRUE),
                    mean(auc_glmm_16_n5, na.rm = TRUE), mean(auc_glmm_20_n5, na.rm = TRUE))
AUC_mat_n5[5,] <- c(mean(auc_mp_0_n5, na.rm = TRUE), mean(auc_mp_4_n5, na.rm = TRUE),
                    mean(auc_mp_8_n5, na.rm = TRUE), mean(auc_mp_12_n5, na.rm = TRUE),
                    mean(auc_mp_16_n5, na.rm = TRUE), mean(auc_mp_20_n5, na.rm = TRUE))

# AUC plot (top left panel of Figure 3)
plot(AUC_mat_n5[1,], AUC_mat_n5[2,], type="l", ylim=c(0.5,1), xlab=expression(tau), ylab="AUC", main="Nonlinear fixed effect, 5 Nodes", lwd = 2)
lines(AUC_mat_n5[1,], AUC_mat_n5[3,], col="Blue",lty = 2, lwd = 2)
lines(AUC_mat_n5[1,], AUC_mat_n5[4,], col="Red", lty = 3, lwd = 2)
lines(AUC_mat_n5[1,], AUC_mat_n5[5,], col="green3", lty = 4, lwd = 2)
