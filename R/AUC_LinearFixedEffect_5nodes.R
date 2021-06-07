## Simulation 7: AUC, linear fixed effect, 5 nodes ##

library(foreach)
library(doSNOW)
library(doRNG)

source("Network_Functions2.R")

nsim <- 500
ncores <- 10
cl <- makeCluster(ncores)
registerDoSNOW(cl)
getDoParWorkers()

pb <- txtProgressBar(min = 0, max = nsim, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

set.seed(1204123878)

#################################################################################################################
psims <- foreach(a = 1:nsim, .options.snow = opts, .errorhandling = 'pass') %dorng% {
  library(lme4)
  tau_v = seq(0,20,4)
  m=100
  
  # Simulation 7: linear, 5 nodes
  pred.matrix <- matrix(NA, nrow = m*6, ncol = 5)
  for(bb in 1:length(tau_v)){
    tau=tau_v[bb]
    
    #### Simulate linear binary data
    p = 6
    m = 100
    n = rpois(m,6)+2
    nepochs=5
    betas <- rnorm(6,0.3,3)
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
        s.comp <- ranef + betas %*% X[j,] + rnorm(1, mean = 0, sd=sigma)
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
    
    ## Last observation data: Y.everyn ~ X.everyn
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
bin_l5 <- psims     # iteration 428 did not converge
m=100

## tau = 0
auc_gnmm_0_l5 <- rep(NA, 500); auc_ann_0_l5 <- rep(NA, 500)
auc_glmm_0_l5 <- rep(NA, 500); auc_mp_0_l5 <- rep(NA, 500)
# iteration 8 has all 1's in response
for (i in c(1:7,9:427,429:500)){
  auc_gnmm_0_l5[i] <- auc(roc(bin_l5[[i]][1:m,1] ~ bin_l5[[i]][1:m,2]))
  auc_ann_0_l5[i] <- auc(roc(bin_l5[[i]][1:m,1] ~ bin_l5[[i]][1:m,3]))
  auc_glmm_0_l5[i] <- auc(roc(bin_l5[[i]][1:m,1] ~ bin_l5[[i]][1:m,4]))
  auc_mp_0_l5[i] <- auc(roc(bin_l5[[i]][1:m,1] ~ bin_l5[[i]][1:m,5]))
}

## tau = 4
auc_gnmm_4_l5 <- rep(NA, 500); auc_ann_4_l5 <- rep(NA, 500)
auc_glmm_4_l5 <- rep(NA, 500); auc_mp_4_l5 <- rep(NA, 500)
for (i in c(1:427,429:500)) {
  if(i==14){  # gnmm doesn't converge
    auc_ann_4_l5[i] <- auc(roc(bin_l5[[i]][(m+1):(2*m),1] ~ bin_l5[[i]][(m+1):(2*m),3]))
    auc_glmm_4_l5[i] <- auc(roc(bin_l5[[i]][(m+1):(2*m),1] ~ bin_l5[[i]][(m+1):(2*m),4]))
    auc_mp_4_l5[i] <- auc(roc(bin_l5[[i]][(m+1):(2*m),1] ~ bin_l5[[i]][(m+1):(2*m),5]))
  } else{
    auc_gnmm_4_l5[i] <- auc(roc(bin_l5[[i]][(m+1):(2*m),1] ~ bin_l5[[i]][(m+1):(2*m),2]))
    auc_ann_4_l5[i] <- auc(roc(bin_l5[[i]][(m+1):(2*m),1] ~ bin_l5[[i]][(m+1):(2*m),3]))
    auc_glmm_4_l5[i] <- auc(roc(bin_l5[[i]][(m+1):(2*m),1] ~ bin_l5[[i]][(m+1):(2*m),4]))
    auc_mp_4_l5[i] <- auc(roc(bin_l5[[i]][(m+1):(2*m),1] ~ bin_l5[[i]][(m+1):(2*m),5]))
  }
}

## tau = 8
auc_gnmm_8_l5 <- rep(NA, 500); auc_ann_8_l5 <- rep(NA, 500)
auc_glmm_8_l5 <- rep(NA, 500); auc_mp_8_l5 <- rep(NA, 500)
# iteration 459 has all 1's in response
for (i in c(1:427,429:458,460:500)) {
  if(i==130 | i==178){  # gnmm doesn't converge
    auc_ann_8_l5[i] <- auc(roc(bin_l5[[i]][(2*m+1):(3*m),1] ~ bin_l5[[i]][(2*m+1):(3*m),3]))
    auc_glmm_8_l5[i] <- auc(roc(bin_l5[[i]][(2*m+1):(3*m),1] ~ bin_l5[[i]][(2*m+1):(3*m),4]))
    auc_mp_8_l5[i] <- auc(roc(bin_l5[[i]][(2*m+1):(3*m),1] ~ bin_l5[[i]][(2*m+1):(3*m),5]))
  } else{
    auc_gnmm_8_l5[i] <- auc(roc(bin_l5[[i]][(2*m+1):(3*m),1] ~ bin_l5[[i]][(2*m+1):(3*m),2]))
    auc_ann_8_l5[i] <- auc(roc(bin_l5[[i]][(2*m+1):(3*m),1] ~ bin_l5[[i]][(2*m+1):(3*m),3]))
    auc_glmm_8_l5[i] <- auc(roc(bin_l5[[i]][(2*m+1):(3*m),1] ~ bin_l5[[i]][(2*m+1):(3*m),4]))
    auc_mp_8_l5[i] <- auc(roc(bin_l5[[i]][(2*m+1):(3*m),1] ~ bin_l5[[i]][(2*m+1):(3*m),5]))
  }
}

## tau = 12
auc_gnmm_12_l5 <- rep(NA, 500); auc_ann_12_l5 <- rep(NA, 500)
auc_glmm_12_l5 <- rep(NA, 500); auc_mp_12_l5 <- rep(NA, 500)
for (i in c(1:427,429:500)) {
  if(i==18){  # gnmm doesn't converge
    auc_ann_12_l5[i] <- auc(roc(bin_l5[[i]][(3*m+1):(4*m),1] ~ bin_l5[[i]][(3*m+1):(4*m),3]))
    auc_glmm_12_l5[i] <- auc(roc(bin_l5[[i]][(3*m+1):(4*m),1] ~ bin_l5[[i]][(3*m+1):(4*m),4]))
    auc_mp_12_l5[i] <- auc(roc(bin_l5[[i]][(3*m+1):(4*m),1] ~ bin_l5[[i]][(3*m+1):(4*m),5]))
  } else {
    auc_gnmm_12_l5[i] <- auc(roc(bin_l5[[i]][(3*m+1):(4*m),1] ~ bin_l5[[i]][(3*m+1):(4*m),2]))
    auc_ann_12_l5[i] <- auc(roc(bin_l5[[i]][(3*m+1):(4*m),1] ~ bin_l5[[i]][(3*m+1):(4*m),3]))
    auc_glmm_12_l5[i] <- auc(roc(bin_l5[[i]][(3*m+1):(4*m),1] ~ bin_l5[[i]][(3*m+1):(4*m),4]))
    auc_mp_12_l5[i] <- auc(roc(bin_l5[[i]][(3*m+1):(4*m),1] ~ bin_l5[[i]][(3*m+1):(4*m),5])) 
  }
}

## tau = 16
auc_gnmm_16_l5 <- rep(NA, 500); auc_ann_16_l5 <- rep(NA, 500)
auc_glmm_16_l5 <- rep(NA, 500); auc_mp_16_l5 <- rep(NA, 500)
for (i in c(1:427,429:500)) {
  if(i==19){  # gnmm doesn't converge
    auc_ann_16_l5[i] <- auc(roc(bin_l5[[i]][(4*m+1):(5*m),1] ~ bin_l5[[i]][(4*m+1):(5*m),3]))
    auc_glmm_16_l5[i] <- auc(roc(bin_l5[[i]][(4*m+1):(5*m),1] ~ bin_l5[[i]][(4*m+1):(5*m),4]))
    auc_mp_16_l5[i] <- auc(roc(bin_l5[[i]][(4*m+1):(5*m),1] ~ bin_l5[[i]][(4*m+1):(5*m),5]))
  } else{
    auc_gnmm_16_l5[i] <- auc(roc(bin_l5[[i]][(4*m+1):(5*m),1] ~ bin_l5[[i]][(4*m+1):(5*m),2]))
    auc_ann_16_l5[i] <- auc(roc(bin_l5[[i]][(4*m+1):(5*m),1] ~ bin_l5[[i]][(4*m+1):(5*m),3]))
    auc_glmm_16_l5[i] <- auc(roc(bin_l5[[i]][(4*m+1):(5*m),1] ~ bin_l5[[i]][(4*m+1):(5*m),4]))
    auc_mp_16_l5[i] <- auc(roc(bin_l5[[i]][(4*m+1):(5*m),1] ~ bin_l5[[i]][(4*m+1):(5*m),5]))
  }
}

## tau = 20
auc_gnmm_20_l5 <- rep(NA, 500); auc_ann_20_l5 <- rep(NA, 500)
auc_glmm_20_l5 <- rep(NA, 500); auc_mp_20_l5 <- rep(NA, 500)
for (i in c(1:427,429:500)) {
  if(i==67 | i==112 | i==245){  # gnmm doesn't converge
    auc_ann_20_l5[i] <- auc(roc(bin_l5[[i]][(5*m+1):(6*m),1] ~ bin_l5[[i]][(5*m+1):(6*m),3]))
    auc_glmm_20_l5[i] <- auc(roc(bin_l5[[i]][(5*m+1):(6*m),1] ~ bin_l5[[i]][(5*m+1):(6*m),4]))
    auc_mp_20_l5[i] <- auc(roc(bin_l5[[i]][(5*m+1):(6*m),1] ~ bin_l5[[i]][(5*m+1):(6*m),5]))
  } else{
    auc_gnmm_20_l5[i] <- auc(roc(bin_l5[[i]][(5*m+1):(6*m),1] ~ bin_l5[[i]][(5*m+1):(6*m),2]))
    auc_ann_20_l5[i] <- auc(roc(bin_l5[[i]][(5*m+1):(6*m),1] ~ bin_l5[[i]][(5*m+1):(6*m),3]))
    auc_glmm_20_l5[i] <- auc(roc(bin_l5[[i]][(5*m+1):(6*m),1] ~ bin_l5[[i]][(5*m+1):(6*m),4]))
    auc_mp_20_l5[i] <- auc(roc(bin_l5[[i]][(5*m+1):(6*m),1] ~ bin_l5[[i]][(5*m+1):(6*m),5]))
  }
}

# calculate average AUC
AUC_mat_l5 <- matrix(NA, nrow = 5, ncol = 6)
AUC_mat_l5[1,] <- seq(0,20,4)
AUC_mat_l5[2,] <- c(mean(auc_gnmm_0_l5, na.rm = TRUE), mean(auc_gnmm_4_l5, na.rm = TRUE),
                    mean(auc_gnmm_8_l5, na.rm = TRUE), mean(auc_gnmm_12_l5, na.rm = TRUE),
                    mean(auc_gnmm_16_l5, na.rm = TRUE), mean(auc_gnmm_20_l5, na.rm = TRUE))
AUC_mat_l5[3,] <- c(mean(auc_ann_0_l5, na.rm = TRUE), mean(auc_ann_4_l5, na.rm = TRUE),
                    mean(auc_ann_8_l5, na.rm = TRUE), mean(auc_ann_12_l5, na.rm = TRUE),
                    mean(auc_ann_16_l5, na.rm = TRUE), mean(auc_ann_20_l5, na.rm = TRUE))
AUC_mat_l5[4,] <- c(mean(auc_glmm_0_l5, na.rm = TRUE), mean(auc_glmm_4_l5, na.rm = TRUE),
                    mean(auc_glmm_8_l5, na.rm = TRUE), mean(auc_glmm_12_l5, na.rm = TRUE),
                    mean(auc_glmm_16_l5, na.rm = TRUE), mean(auc_glmm_20_l5, na.rm = TRUE))
AUC_mat_l5[5,] <- c(mean(auc_mp_0_l5, na.rm = TRUE), mean(auc_mp_4_l5, na.rm = TRUE),
                    mean(auc_mp_8_l5, na.rm = TRUE), mean(auc_mp_12_l5, na.rm = TRUE),
                    mean(auc_mp_16_l5, na.rm = TRUE), mean(auc_mp_20_l5, na.rm = TRUE))

# AUC plot (bottom left panel of Figure 3)
plot(AUC_mat_l5[1,], AUC_mat_l5[2,], type="l", ylim=c(0.5,1), xlab=expression(tau), ylab="AUC", main="Linear fixed effect, 5 Nodes", lwd = 2)
lines(AUC_mat_l5[1,], AUC_mat_l5[3,], col="Blue",lty = 2, lwd = 2)
lines(AUC_mat_l5[1,], AUC_mat_l5[4,], col="Red", lty = 3, lwd = 2)
lines(AUC_mat_l5[1,], AUC_mat_l5[5,], col="green3", lty = 4, lwd = 2)
