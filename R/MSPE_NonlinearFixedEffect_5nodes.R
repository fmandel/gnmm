## Simulation 1: MSPE, nonlinear fixed effect, 5 nodes ##

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

set.seed(1204123871)

#################################################################################################################
psims <- foreach(a = 1:nsim, .options.snow = opts) %dorng% {
  library(lme4)
  tau_v <- seq(0,.5,.1)
  
  # Simulation 1: nonlinear, 5 nodes
  MSEmat.n5 <- matrix(NA,nrow=6,ncol=length(tau_v))
  for(bb in 1:length(tau_v)){
    tau <- tau_v[bb]
    
    #### Simulate nonlinear data
    p <- 6
    m <- 100
    n <- rpois(m,6)+2
    nepochs <- 15
    betas <- rnorm(3,2,1)
    sigma <- .05
    hidnodes1 <- 5
    hidnodes2 <- 5
    
    lab <- rep(NA,sum(n))
    Y <- rep(NA,sum(n))
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
        Y[j] <- ranef + betas[1]*y1 + betas[2]*y2 + betas[3]*y3 + rnorm(1,mean=0,sd=sigma)
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
    
    ## Last observation Data: Y.everyn ~ X.everyn
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
    
    ### Run 1-layer GNMM and predict last observation
    m1 <- gnmm.sgd(formula = Y.train ~ X.train + (1|lab.train), family = 'gaussian', penalization = 0.001,
                   nodes1 = hidnodes1, nodes2 = NULL, step_size = 0.025, act_fun = 'relu', nepochs = nepochs, incl_ranef = TRUE)
    pred1 <- gnmm.predict(new_data = X.everyn, id = lab.everyn, gnmm.fit = m1)
    
    ### Run 2-layer GNMM and predict last observation
    m2 <- gnmm.sgd(formula = Y.train ~ X.train + (1|lab.train), family = 'gaussian', penalization = 0.005,
                   nodes1 = hidnodes1, nodes2 = hidnodes2, step_size = 0.01, act_fun = 'relu',nepochs=nepochs,incl_ranef=TRUE)
    pred2 <- gnmm.predict(new_data = X.everyn, id = lab.everyn, gnmm.fit = m2)
    
    
    ### Run ANN and predict last observation
    m3 <- gnmm.sgd(formula = Y.train ~ X.train + (1|lab.train), family = 'gaussian', penalization = 0.001,
                   nodes1 = hidnodes1, nodes2 = NULL, step_size = 0.025, act_fun = 'relu', nepochs = nepochs, incl_ranef = FALSE)
    pred3 <- gnmm.predict(new_data = X.everyn, id = lab.everyn, gnmm.fit = m3)
    
    ### Run GLMER and predict last observation
    glmm.out <- glmer(Y.train ~ X.train + (1|lab.train), family = 'gaussian')
    s1 <- summary(glmm.out)
    glmm.coef <- matrix(s1$coefficients[2:(p+1),1])
    fef.pred <- X.everyn%*%glmm.coef + rep(matrix(s1$coefficients[1,1]),m)     ## fixed effect prediction
    glm.ref <- as.vector(c(ranef(glmm.out)$lab.train))
    glm.ref <- unname(glm.ref)
    glm.ref <- unlist(glm.ref)
    pred4 <- fef.pred + glm.ref             ## fixed effect + random effect prediction
    
    ### Run ANN with inputs for each individual (Maity-Pal)
    m5 <- gnmm.sgd(formula = Y.train ~ X.mp.train + (1|lab.train), family = 'gaussian', penalization = 0.001,
                   nodes1 = hidnodes1, nodes2 = NULL, step_size = 0.025, act_fun = 'relu', nepochs = nepochs, incl_ranef = FALSE)
    pred5 <- gnmm.predict(new_data = X.mp.everyn, id = lab.everyn, gnmm.fit = m5)
    
    #### fill in MSE values
    MSEmat.n5[,bb] <- c(tau,mean((Y.everyn-pred1)^2),mean((Y.everyn-pred2)^2),mean((Y.everyn-pred3)^2),
                        mean((Y.everyn-pred4)^2),mean((Y.everyn-pred5)^2))
  }
  
  rbind(MSEmat.n5[1:6,])
  
}

# calculate average MSPE
outmat.n5 <- Reduce("+", psims) / nsim

# MSPE plot (top left panel of Figure 2)
tau_v <- outmat.n5[1,]
plot(tau_v, outmat.n5[2,], type='l', ylim=range(outmat.n5), xlab=expression(tau), ylab='MSPE', 
     main='Nonlinear fixed effect, 5 Nodes', lwd = 2)
lines(tau_v, outmat.n5[3,], col='Gray', lty = 2, lwd = 2)
lines(tau_v, outmat.n5[4,], col='Blue', lty = 3, lwd = 2)
lines(tau_v, outmat.n5[5,], col='Red', lty = 4, lwd = 2)
lines(tau_v, outmat.n5[6,], col='green3', lty = 5, lwd = 2)
