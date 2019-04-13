## Title: Robust Infinite Gaussian Mixture 5(a) ##
## Author: Abbas Zaidi                          ##
## Last Updated: 01/06/2017                     ##

## Load in the required packages and the data ##
require(MASS)
data(galaxies)


robustInfiniteGaussianMixture <- function(Y, K, v, r, c, m = 0, t = 1, a.0, 
  b.0, mu.j.init, phi.j.init, mu.0.init, sigma.0.init, a.init, tau.inv.init, iter, nburn){
	
	## Define some initial parameters ##
	n <- length(Y)
	Y.ind <- seq_along(Y)


	## Start by placing everything in its own cluster ##
	clusters <- as.list(Y.ind); names(clusters) <- seq_along(clusters)
	# Compute the cluster sizes #
	njs <- sapply(clusters, length, USE.NAMES = TRUE)



	## Allocate Memory ##
	mu.j <- rep(mu.j.init, length(clusters)); names(mu.j) <- names(clusters)
	phi.j <- rep(phi.j.init, length(clusters)); names(phi.j) <- names(clusters)
	iter.mean <- iter.phi <- vector("list", length = iter)
	a<- tau.inv <- mu.0 <- sigma.0 <- vector("numeric", length = iter)
	## Assign Initial Values ##
	iter.mean[[1]] <- mu.j
		iter.phi[[1]] <- phi.j
	a[1] <- a.init
		tau.inv[1] <- tau.inv.init
			mu.0[1] <- mu.0.init
				sigma.0[1] <- sigma.0.init
	
	## Create a matrix to hold observation specific results i.e. mean and variance ##
	clean.phi.results <- matrix(nrow = iter, ncol = n)
	clean.mu.results <- matrix(nrow = iter, ncol = n)
	clean.mu.results[1,] <- mu.j
	clean.phi.results[1,] <- phi.j
	names(clean.phi.results) <- names(clean.mu.results) <- paste0("obs",Y.ind)

	for(t in 2:iter){
		
		## Set the cluster specific parameters to their previous iteration values ##
		## These will be reassigned later.                                        ##
		mu.j <- iter.mean[[t-1]]
		phi.j <- iter.phi[[t-1]]
	
	
		## Update the cluster assignments ##
		for(i in 1:length(Y)){
			# Find which cluster currently contains the ith entry #
			contains <- names(clusters)[which(!is.na(as.numeric(sapply(clusters,
				function(x){which(x == i)}))))]
			# Drop the ith entry from the cluster that contains it #
			clusters[[contains]] <- 
				clusters[[contains]][-which(clusters[[contains]] == i)]
			# Determine where the ith data point should go #
			njs.minus <- sapply(clusters, length)
			# Check which clusters need to be dropped since they are of length 0 #
			zeroed <- which(njs.minus == 0)
			if(length(zeroed) != 0){
				# If there are clusters that are of size 0 we drop their parameters #
				drop <- which(names(njs.minus) == names(njs.minus)[zeroed])
				clusters[drop] <- NULL
				mu.j <- mu.j[-1*drop]
				phi.j <- phi.j[-1*drop]
				njs.minus <- njs.minus[-1*drop]
				}
	
			# compute the new sampling probabilities #
			# the existing probabilities #
			existing.probs <- njs.minus/(n-1+a[t-1]) * 
					sqrt(phi.j) * exp(-phi.j*((Y[i] - mu.j)^2)/2)
	 
			# The new probabilities #
			# Perform prior draws for the parameters as indicated by rasmussen #
			mu.draw <- rnorm(n = 1, mean = mu.0[t-1], 
					sd = sqrt(1/tau.inv[t-1])) 
			phi.draw <-  rgamma(n = 1, shape = v/2 , 
					rate = v*sigma.0[t-1]/2)
			new.probs <- (a[t-1]/length(names(clusters)))/(n-1+a[t-1]) * 
				dnorm(x = Y[i], mean = mu.draw, sd = sqrt(1/phi.draw))
			
			# Normalize these probabilities #
			probs <- c(existing.probs, new.probs) / 
					sum(c(existing.probs, new.probs))
	

			# Find the assignment #
			assignment <- sample(c(names(njs.minus), "new"), 
					size = 1, prob = probs)

			# Check if the cluster assignment is one of the existing ones or a new one #
			# and assign accordingly 												   #
			if(assignment == "new"){
				new.name <- as.character(max(as.integer(names(clusters))) + 1)
				clusters[[new.name]] <- i
				# Assign parameters to this new cluster                          #
				# update the cluster specific parameters with the values sampled #
				phi.j[new.name] <- phi.draw
				mu.j[new.name] <- mu.draw
		
				} else if (assignment != "new"){
			clusters[[assignment]] <- c(clusters[[assignment]],i)
			}
		}
	
		## Update the size of each cluster based on the new assignments ##
		njs <- sapply(clusters, length, USE.NAMES = TRUE)
	
	
		## Update the global parameters ##
		# The prior mean first #
		J <- length(names(clusters))
		mu.bar <- mean(mu.j)
		mu.0[t] <- rnorm(n = 1, mean = (J*mu.bar + K*m)/(K+J), 
			sd = sqrt((1/tau.inv[t-1])/(K+J)))
		# The prior precision #
		tau.inv[t]<- rgamma(n = 1, shape = (J+r+1)/2, 
			rate = (sum((mu.j - mu.0[t])^2)+(r*(t^2))+(K*(mu.0[t] - m)^2))/2)
		# The prior variance #
		sigma.0[t] <- rgamma(n = 1, shape = (J*v)/2 + c, rate = (v/2)*sum(phi.j) + c)
	
		## Update the cluster specific parameters ##
		# Update the mean #
		for(j in names(mu.j)){
			cluster.mean <- mean(Y[clusters[[j]]])
			var.par <- 1/(((tau.inv[t])) + phi.j[j]*njs[j])
			mean.par <- (njs[j] * phi.j[j] * cluster.mean + 
				mu.0[t]/(1/tau.inv[t]))*(var.par)
			mu.j[j] <- rnorm(n=1, mean = mean.par, 
				sd = sqrt(var.par))
			}
		# Update the phi #
		for (j in names(phi.j)){
			phi.j[j] <- rgamma(n = 1, shape = (v+njs[j])/2,
				rate = sum((Y[clusters[[j]]] - mu.j[j])^2)/2 + 
					(v*sigma.0[t])/2)
			}
		# Store the updated values #
		iter.mean[[t]] <- mu.j
		iter.phi[[t]] <- phi.j
		# Add these to the observation specific matrix #
		for(j in names(clusters)){
			c.ind <- clusters[[j]]
			clean.mu.results[t,as.integer(c.ind)] <- mu.j[j]
			clean.phi.results[t,as.integer(c.ind)] <- phi.j[j]
			}
	

		## Update the alpha parameter ##
		# Use an auxilary variable #
		aux <- rbeta(n = 1, shape1 = a[t-1], n)
		constant <- (a.0 + length(names(clusters)) -1) / (n*(b.0 - log(aux)))
		pi <- constant/(constant+1)
		a[t] <- pi * rgamma(n = 1, shape = a.0+length(names(clusters)),
			rate = b.0 - log(aux)) + (1-pi) * rgamma(n = 1, 
			shape = a.0+length(names(clusters))-1, rate = b.0 - log(aux))	
	}

	## Drop the burned entries for the parameters ##
	final.means <- clean.mu.results[-(1:nburn),]
	final.prec <- clean.phi.results[-(1:nburn),]
	final.a <- a[-(1:nburn)]
	final.tau.inv <- tau.inv[-(1:nburn)]
	final.mu.0 <- mu.0[-(1:nburn)]
	final.sigma <- sigma.0[-(1:nburn)]
	
	## Return the post burn in values ##
	return(list(mus = final.means, phis = final.prec, as = final.a,
		tau.invs = final.tau.inv, mu0s = final.mu.0, sigmas = final.sigma))

}

## The Mixture Density Estimate ##

infiniteGaussianMixtureDensity <- function(x, mean.val, prec.val, v, a.val, tau.inv.val,
  sigma.val, mu.0.val, ss, n){
  	densityVal <- (sum(dnorm(x = x, mean = mean.val, 
    	sd = sqrt(1/prec.val))) + 
      		a.val * mean(dnorm(x = x, 
        		mean = rnorm(n = ss, mean = mu.0.val, 
					sd = sqrt(1/tau.inv.val)), sd = sqrt(1/rgamma(n = ss, shape = v/2 , 
						rate = v*sigma.val/2)))))/(a.val+n)
	return(densityVal)
  }


## Fit the model ##

iter.num <- 20000; nburn.num <- 10000

modelFit <- robustInfiniteGaussianMixture(Y = galaxies, K = 0.1, r = 0.1, v = 0.1, 
  m = 0, c = 0.1, a.0 = 1, b.0 = 1, mu.j.init = mean(galaxies), phi.j.init = 
    1/var(galaxies), mu.0.init = 0, sigma.0.init = 1, a.init = 0.5, tau.inv.init = 0.1, 
      iter = iter.num, nburn = nburn.num)
      
      
## Fit the model (Rescaled) ##

modelFitII <- robustInfiniteGaussianMixture(Y = galaxies/1000, K = 0.1, r = 0.1, v = 0.1, 
  m = 0, c = 0.1, a.0 = 1, b.0 = 1, mu.j.init = mean(galaxies/1000), phi.j.init = 
    1/var(galaxies/1000), mu.0.init = 0, sigma.0.init = 1, a.init = 0.5, 
     tau.inv.init = 0.1, iter = iter.num, nburn = nburn.num)

## Use the fitted model in the scheme ##


## Allocate some storage memory ##

## Generate a sequence of points to estimate the density over ##
newY <- seq(from = min(galaxies), to = max(galaxies), length.out = 1000)
newYII <- seq(from = min(galaxies/1000), to = max(galaxies/1000), length.out = 1000)

## Evaluate the density ##
dmix <- sapply(seq_len(iter.num - nburn.num), function(k){
			sapply(newY, function(zeta){
		    infiniteGaussianMixtureDensity(x = zeta, 
			mean.val = modelFit$mus[k,], prec.val = modelFit$phis[k,], 
			v = 0.1, a.val = modelFit$as[k], tau.inv.val = modelFit$tau.invs[k],
			mu.0.val = modelFit$mu0s[k], sigma.val = modelFit$sigmas[k], ss = 500,
			n = length(galaxies))})

})


## Evaluate the density rescaled ##
dmixII <- sapply(seq_len(iter.num - nburn.num), function(k){
			sapply(newYII, function(zeta){
		    infiniteGaussianMixtureDensity(x = zeta, 
			mean.val = modelFitII$mus[k,], prec.val = modelFitII$phis[k,], 
			v = 0.1, a.val = modelFitII$as[k], tau.inv.val = modelFitII$tau.invs[k],
			mu.0.val = modelFitII$mu0s[k], sigma.val = modelFitII$sigmas[k], ss = 500,
			n = length(galaxies))})

})



## Find the requisite quantiles ##
posterior.means <- apply(dmix, 1, median, na.rm = TRUE)
posterior.quantiles <- apply(dmix, 1, quantile, 
  prob=c(0.025, 0.975), na.rm = TRUE)

pdf("Posterior Estimate.pdf")
plot(newY, posterior.means, type = "l", lwd = 2, 
  ylim = c(min(posterior.quantiles, na.rm = TRUE), 
  max(posterior.quantiles, na.rm = TRUE)), xlab = "Galaxies Data",
  ylab = "Mixture Density Estimate")
lines(newY, posterior.quantiles[1,], lty = 2, col = "orange", lwd = 2)
lines(newY, posterior.quantiles[2,], lty = 2, col = "orange", lwd = 2)
dev.off()


## Find the requisite quantiles ##
posterior.meansII <- apply(dmixII, 1, mean, na.rm = TRUE)
posterior.quantilesII <- apply(dmixII, 1, quantile, 
  prob=c(0.025, 0.975), na.rm = TRUE)

pdf("Scaled Posterior Estimate.pdf")
plot(newYII, posterior.meansII, type = "l", lwd = 2, 
  ylim = c(min(posterior.quantilesII, na.rm = TRUE), 
  max(posterior.quantilesII, na.rm = TRUE)), xlab = "Scaled Galaxies Data",
  ylab = "Mixture Density Estimate")
lines(newYII, posterior.quantilesII[1,], lty = 2, col = "orange", lwd = 2)
lines(newYII, posterior.quantilesII[2,], lty = 2, col = "orange", lwd = 2)
dev.off()

pdf("dataHist.pdf")
hist(galaxies, breaks = 50, prob = TRUE, xlab = "Galaxies Data", 
  col = "purple")
dev.off()
