
makeXMatrix <- function(l, b=NULL, s.vector=NULL, d.vector=NULL, epsilon.vector=NULL) {
  x.transpose <- matrix(l, nrow = 1, ncol = 1)
  if (!is.null(b)) {
    x.transpose <- cbind(x.transpose, matrix(b, nrow = 1, ncol = 1))
  }
  if (!is.null(s.vector)) {
    x.transpose <- cbind(x.transpose, matrix(s.vector, nrow = 1, ncol = length(s.vector)))
  }
  
  if (!is.null(d.vector)) {
    x.transpose <- cbind(x.transpose, matrix(d.vector, nrow = 1, ncol = length(d.vector)))
  }
  
  if (!is.null(epsilon.vector)) {
    x.transpose <- cbind(x.transpose, matrix(epsilon.vector, nrow = 1, ncol = length(epsilon.vector)))
  }
  
  x <- t(x.transpose)
  return(list(x = x, x.transpose = x.transpose))
}

parameterise <- function(alpha, beta.v=NULL, small.phi=1, gamma.v=NULL, lambda=NULL, ar.coefs=NULL, ma.coefs=NULL) {
  # print("urg")
  # print(lambda)
  if (!is.null(lambda)) {
    param.vector <- cbind(lambda, alpha)
    use.box.cox <- TRUE
  } else {
    # print("hello")
    param.vector <- alpha
    use.box.cox <- FALSE
    # print(use.box.cox)
  }
  if (!is.null(beta.v)) {
    use.beta <- TRUE
    if (is.null(small.phi)) {
      use.damping <- FALSE
    } else if (small.phi != 1) {
      param.vector <- cbind(param.vector, small.phi)
      use.damping <- TRUE
    } else {
      use.damping <- FALSE
    }
    param.vector <- cbind(param.vector, beta.v)
  } else {
    use.beta <- FALSE
    use.damping <- FALSE
  }
  if (!is.null(gamma.v)) {
    gamma.v <- matrix(gamma.v, nrow = 1, ncol = length(gamma.v))
    param.vector <- cbind(param.vector, gamma.v)
    length.gamma <- length(gamma.v)
  } else {
    length.gamma <- 0
  }
  if (!is.null(ar.coefs)) {
    ar.coefs <- matrix(ar.coefs, nrow = 1, ncol = length(ar.coefs))
    param.vector <- cbind(param.vector, ar.coefs)
    p <- length(ar.coefs)
  } else {
    p <- 0
  }
  if (!is.null(ma.coefs)) {
    ma.coefs <- matrix(ma.coefs, nrow = 1, ncol = length(ma.coefs))
    param.vector <- cbind(param.vector, ma.coefs)
    q <- length(ma.coefs)
  } else {
    q <- 0
  }
  # print(use.box.cox)
  control <- list(use.beta = use.beta, use.box.cox = use.box.cox, use.damping = use.damping, length.gamma = length.gamma, p = p, q = q)
  return(list(vect = as.numeric(param.vector), control = control))
}

makeParscale <- function(control) {
  # print(control)
  if (control$use.box.cox) {
    parscale <- c(.001, .01)
  } else {
    parscale <- .01
  }
  if (control$use.beta) {
    if (control$use.damping) {
      parscale <- c(parscale, 1e-2, 1e-2)
    } else {
      parscale <- c(parscale, 1e-2)
    }
  }
  if (control$length.gamma > 0) {
    parscale <- c(parscale, rep(1e-5, control$length.gamma))
  }
  
  if ((control$p != 0) | (control$q != 0)) {
    parscale <- c(parscale, rep(1e-1, (control$p + control$q)))
  }
  # print(parscale)
  return(parscale)
}

makeTBATSFMatrix <- function(alpha, beta=NULL, small.phi=NULL, seasonal.periods=NULL, k.vector=NULL, gamma.bold.matrix=NULL, ar.coefs=NULL, ma.coefs=NULL) {
  
  # 1. Alpha Row
  F <- matrix(1, nrow = 1, ncol = 1)
  if (!is.null(beta)) {
    F <- cbind(F, matrix(small.phi, nrow = 1, ncol = 1))
  }
  if (!is.null(seasonal.periods)) {
    tau <- sum(k.vector) * 2
    zero.tau <- matrix(0, nrow = 1, ncol = tau)
    F <- cbind(F, zero.tau)
  }
  if (!is.null(ar.coefs)) {
    p <- length(ar.coefs)
    ar.coefs <- matrix(ar.coefs, nrow = 1, ncol = p)
    alpha.phi <- alpha * ar.coefs
    F <- cbind(F, alpha.phi)
  }
  if (!is.null(ma.coefs)) {
    q <- length(ma.coefs)
    ma.coefs <- matrix(ma.coefs, nrow = 1, ncol = q)
    alpha.theta <- alpha * ma.coefs
    F <- cbind(F, alpha.theta)
  }
  
  # 2. Beta Row
  if (!is.null(beta)) {
    beta.row <- matrix(c(0, small.phi), nrow = 1, ncol = 2)
    if (!is.null(seasonal.periods)) {
      beta.row <- cbind(beta.row, zero.tau)
    }
    if (!is.null(ar.coefs)) {
      beta.phi <- beta * ar.coefs
      beta.row <- cbind(beta.row, beta.phi)
    }
    if (!is.null(ma.coefs)) {
      beta.theta <- beta * ma.coefs
      beta.row <- cbind(beta.row, beta.theta)
    }
    F <- rbind(F, beta.row)
  }
  
  # 3. Seasonal Row
  if (!is.null(seasonal.periods)) {
    seasonal.row <- t(zero.tau)
    if (!is.null(beta)) {
      seasonal.row <- cbind(seasonal.row, seasonal.row)
    }
    
    # Make the A matrix
    A <- matrix(0, tau, tau)
    last.pos <- 0
    for (i in 1:length(k.vector)) {
      if (seasonal.periods[i] != 2) {
        C <- .Call("makeCIMatrix", k_s = as.integer(k.vector[i]), m_s = as.double(seasonal.periods[i]), PACKAGE = "forecast")
      } else {
        C <- matrix(0, 1, 1)
      }
      S <- .Call("makeSIMatrix", k_s = as.integer(k.vector[i]), m_s = as.double(seasonal.periods[i]), PACKAGE = "forecast")
      
      # C <- matrix(0,k.vector[i],k.vector[i])
      # for(j in 1:k.vector[i]) {
      # 	l <- round((2*pi*j/seasonal.periods[i]), digits=15)
      # 	C[j,j] <- cos(l)
      # }
      # S <- matrix(0,k.vector[i],k.vector[i])
      # for(j in 1:k.vector[i]) {
      # 	S[j,j] <- sin(2*pi*j/seasonal.periods[i])
      # }
      # print(C)
      # print(S)
      Ai <- .Call("makeAIMatrix", C_s = C, S_s = S, k_s = as.integer(k.vector[i]), PACKAGE = "forecast")
      A[(last.pos + 1):(last.pos + (2 * k.vector[i])), (last.pos + 1):(last.pos + (2 * k.vector[i]))] <- Ai
      last.pos <- last.pos + (2 * k.vector[i])
    }
    seasonal.row <- cbind(seasonal.row, A)
    
    if (!is.null(ar.coefs)) {
      B <- t(gamma.bold.matrix) %*% ar.coefs
      seasonal.row <- cbind(seasonal.row, B)
    }
    if (!is.null(ma.coefs)) {
      C <- t(gamma.bold.matrix) %*% ma.coefs
      seasonal.row <- cbind(seasonal.row, C)
    }
    F <- rbind(F, seasonal.row)
  }
  
  # 4. AR() Rows
  if (!is.null(ar.coefs)) {
    p <- length(ar.coefs)
    ar.rows <- matrix(0, nrow = p, ncol = 1)
    if (!is.null(beta)) {
      ar.rows <- cbind(ar.rows, ar.rows)
    }
    if (!is.null(seasonal.periods)) {
      ar.seasonal.zeros <- matrix(0, nrow = p, ncol = tau)
      ar.rows <- cbind(ar.rows, ar.seasonal.zeros)
    }
    ident <- diag((p - 1))
    ident <- cbind(ident, matrix(0, nrow = (p - 1), ncol = 1))
    ar.part <- rbind(ar.coefs, ident)
    ar.rows <- cbind(ar.rows, ar.part)
    
    if (!is.null(ma.coefs)) {
      ma.in.ar <- matrix(0, nrow = p, ncol = q)
      ma.in.ar[1, ] <- ma.coefs
      ar.rows <- cbind(ar.rows, ma.in.ar)
    }
    
    F <- rbind(F, ar.rows)
  }
  
  # 5. MA() Rows
  if (!is.null(ma.coefs)) {
    ma.rows <- matrix(0, nrow = q, ncol = 1)
    if (!is.null(beta)) {
      ma.rows <- cbind(ma.rows, ma.rows)
    }
    if (!is.null(seasonal.periods)) {
      ma.seasonal <- matrix(0, nrow = q, ncol = tau)
      ma.rows <- cbind(ma.rows, ma.seasonal)
    }
    if (!is.null(ar.coefs)) {
      ar.in.ma <- matrix(0, nrow = q, ncol = p)
      ma.rows <- cbind(ma.rows, ar.in.ma)
    }
    ident <- diag((q - 1))
    ident <- cbind(ident, matrix(0, nrow = (q - 1), ncol = 1))
    ma.part <- rbind(matrix(0, nrow = 1, ncol = q), ident)
    ma.rows <- cbind(ma.rows, ma.part)
    F <- rbind(F, ma.rows)
  }
  return(F)
}

calcLikelihoodTBATS <- function(param.vector, opt.env, use.beta, use.small.phi, seasonal.periods, param.control, p=0, q=0, tau=0, bc.lower=0, bc.upper=1) {
  # param vector should be as follows: Box-Cox.parameter, alpha, beta, small.phi, gamma.vector, ar.coefs, ma.coefs
  # Put the components of the param.vector into meaningful individual variables
  paramz <- unParameteriseTBATS(param.vector, param.control) 
  #paramz <- unParameteriseTBATS(param.vector$vect, param.vector$control) # I added this line to execute function
  box.cox.parameter <- paramz$lambda
  alpha <- paramz$alpha
  beta.v <- paramz$beta
  small.phi <- paramz$small.phi
  gamma.one.v <- paramz$gamma.one.v
  gamma.two.v <- paramz$gamma.two.v
  ar.coefs <- paramz$ar.coefs
  ma.coefs <- paramz$ma.coefs
  if (!is.null(paramz$ar.coefs)) {
    p <- length(paramz$ar.coefs)
    ar.coefs <- matrix(paramz$ar.coefs, nrow = 1, ncol = p)
  } else {
    ar.coefs <- NULL
    p <- 0
  }
  if (!is.null(paramz$ma.coefs)) {
    q <- length(paramz$ma.coefs)
    ma.coefs <- matrix(paramz$ma.coefs, nrow = 1, ncol = q)
  } else {
    ma.coefs <- NULL
    q <- 0
  }
  x.nought <- BoxCox(opt.env$x.nought.untransformed, lambda = box.cox.parameter)
  lambda <- attr(x.nought, "lambda")
  
  .Call("updateWtransposeMatrix", wTranspose_s = opt.env$w.transpose, smallPhi_s = small.phi, tau_s = as.integer(tau), arCoefs_s = ar.coefs, maCoefs_s = ma.coefs, p_s = as.integer(p), q_s = as.integer(q), PACKAGE = "forecast")
  
  if (!is.null(opt.env$gamma.bold)) {
    .Call("updateTBATSGammaBold", gammaBold_s = opt.env$gamma.bold, kVector_s = opt.env$k.vector, gammaOne_s = gamma.one.v, gammaTwo_s = gamma.two.v)
  }
  .Call("updateTBATSGMatrix", g_s = opt.env$g, gammaBold_s = opt.env$gamma.bold, alpha_s = alpha, beta_s = beta.v, PACKAGE = "forecast")
  
  .Call("updateFMatrix", opt.env$F, small.phi, alpha, beta.v, opt.env$gamma.bold, ar.coefs, ma.coefs, tau, PACKAGE = "forecast")
  
  mat.transformed.y <- BoxCox(opt.env$y, box.cox.parameter)
  lambda <- attr(mat.transformed.y, "lambda")
  n <- ncol(opt.env$y)
  
  .Call("calcTBATSFaster", ys = mat.transformed.y, yHats = opt.env$y.hat, wTransposes = opt.env$w.transpose, Fs = opt.env$F, xs = opt.env$x, gs = opt.env$g, es = opt.env$e, xNought_s = x.nought, PACKAGE = "forecast")
  
  ##
  ####
  ####################################################################
  
  log.likelihood <- n * log(sum(opt.env$e ^ 2)) - 2 * (box.cox.parameter - 1) * sum(log(opt.env$y))
  
  if (is.na(log.likelihood)) { # Not sure why this would occur
    return(Inf)
  }
  
  assign("D", (opt.env$F - opt.env$g %*% opt.env$w.transpose), envir = opt.env)
  if (checkAdmissibility(opt.env, box.cox = box.cox.parameter, small.phi = small.phi, ar.coefs = ar.coefs, ma.coefs = ma.coefs, tau = sum(seasonal.periods), bc.lower = bc.lower, bc.upper = bc.upper)) {
    #print(log.likelihood)
    return(log.likelihood)
  } else {
    #print("hi")
    return(Inf)
  }
}

calcLikelihoodNOTransformedTBATS <- function(param.vector, opt.env, x.nought, use.beta, use.small.phi, seasonal.periods, param.control, p=0, q=0, tau=0) {
  # The likelihood function without the Box-Cox Transformation
  # param vector should be as follows: alpha, beta, small.phi, gamma.vector, ar.coefs, ma.coefs
  # Put the components of the param.vector into meaningful individual variables
  paramz <- unParameteriseTBATS(param.vector, param.control)
  box.cox.parameter <- paramz$lambda
  alpha <- paramz$alpha
  beta.v <- paramz$beta
  small.phi <- paramz$small.phi
  gamma.one.v <- paramz$gamma.one.v
  gamma.two.v <- paramz$gamma.two.v
  
  if (!is.null(paramz$ar.coefs)) {
    p <- length(paramz$ar.coefs)
    ar.coefs <- matrix(paramz$ar.coefs, nrow = 1, ncol = p)
  } else {
    ar.coefs <- NULL
    p <- 0
  }
  
  if (!is.null(paramz$ma.coefs)) {
    q <- length(paramz$ma.coefs)
    ma.coefs <- matrix(paramz$ma.coefs, nrow = 1, ncol = q)
  } else {
    ma.coefs <- NULL
    q <- 0
  }
  
  .Call("updateWtransposeMatrix", wTranspose_s = opt.env$w.transpose, smallPhi_s = small.phi, tau_s = as.integer(tau), arCoefs_s = ar.coefs, maCoefs_s = ma.coefs, p_s = as.integer(p), q_s = as.integer(q), PACKAGE = "forecast")
  
  if (!is.null(opt.env$gamma.bold)) {
    .Call("updateTBATSGammaBold", gammaBold_s = opt.env$gamma.bold, kVector_s = opt.env$k.vector, gammaOne_s = gamma.one.v, gammaTwo_s = gamma.two.v)
  }
  
  .Call("updateTBATSGMatrix", g_s = opt.env$g, gammaBold_s = opt.env$gamma.bold, alpha_s = alpha, beta_s = beta.v, PACKAGE = "forecast")
  
  .Call("updateFMatrix", opt.env$F, small.phi, alpha, beta.v, opt.env$gamma.bold, ar.coefs, ma.coefs, tau, PACKAGE = "forecast")
  
  n <- ncol(opt.env$y)
  
  .Call("calcTBATSFaster", ys = opt.env$y, yHats = opt.env$y.hat, wTransposes = opt.env$w.transpose, Fs = opt.env$F, xs = opt.env$x, gs = opt.env$g, es = opt.env$e, xNought_s = x.nought, PACKAGE = "forecast")
  ##
  ####
  ####################################################################
  
  log.likelihood <- n * log(sum(opt.env$e * opt.env$e))
  if (is.na(log.likelihood)) { # Not sure why this would occur
    return(Inf)
  }
  
  assign("D", (opt.env$F - opt.env$g %*% opt.env$w.transpose), envir = opt.env)
  
  if (checkAdmissibility(opt.env = opt.env, box.cox = NULL, small.phi = small.phi, ar.coefs = ar.coefs, ma.coefs = ma.coefs, tau = tau)) {
    return(log.likelihood)
  } else {
    return(Inf)
  }
}

unParameteriseTBATS <- function(param.vector, control) {
  # print(control)
  if (control$use.box.cox) {
    lambda <- param.vector[1]
    alpha <- param.vector[2]
    if (control$use.beta) {
      if (control$use.damping) {
        small.phi <- param.vector[3]
        beta <- param.vector[4]
        gamma.start <- 5
      } else {
        small.phi <- 1
        beta <- param.vector[3]
        gamma.start <- 4
      }
    } else {
      small.phi <- NULL
      beta <- NULL
      gamma.start <- 3
    }
    if (control$length.gamma > 0) {
      gamma.one.vector <- param.vector[gamma.start:(gamma.start + (control$length.gamma / 2) - 1)]
      gamma.two.vector <- param.vector[(gamma.start + (control$length.gamma / 2)):(gamma.start + (control$length.gamma) - 1)]
      final.gamma.pos <- gamma.start + control$length.gamma - 1
    } else {
      gamma.one.vector <- NULL
      gamma.two.vector <- NULL
      final.gamma.pos <- gamma.start - 1
    }
    if (control$p != 0) {
      ar.coefs <- param.vector[(final.gamma.pos + 1):(final.gamma.pos + control$p)]
    } else {
      ar.coefs <- NULL
    }
    if (control$q != 0) {
      ma.coefs <- param.vector[(final.gamma.pos + control$p + 1):length(param.vector)]
    } else {
      ma.coefs <- NULL
    }
  } else {
    lambda <- NULL
    alpha <- param.vector[1]
    if (control$use.beta) {
      if (control$use.damping) {
        small.phi <- param.vector[2]
        beta <- param.vector[3]
        gamma.start <- 4
      } else {
        small.phi <- 1
        beta <- param.vector[2]
        gamma.start <- 3
      }
    } else {
      small.phi <- NULL
      beta <- NULL
      gamma.start <- 2
    }
    if (control$length.gamma > 0) {
      gamma.one.vector <- param.vector[gamma.start:(gamma.start + (control$length.gamma / 2) - 1)]
      gamma.two.vector <- param.vector[(gamma.start + (control$length.gamma / 2)):(gamma.start + (control$length.gamma) - 1)]
      final.gamma.pos <- gamma.start + control$length.gamma - 1
    } else {
      gamma.one.vector <- NULL
      gamma.two.vector <- NULL
      final.gamma.pos <- gamma.start - 1
    }
    if (control$p != 0) {
      ar.coefs <- param.vector[(final.gamma.pos + 1):(final.gamma.pos + control$p)]
    } else {
      ar.coefs <- NULL
    }
    if (control$q != 0) {
      ma.coefs <- param.vector[(final.gamma.pos + control$p + 1):length(param.vector)]
    } else {
      ma.coefs <- NULL
    }
  }
  return(list(lambda = lambda, alpha = alpha, beta = beta, small.phi = small.phi, gamma.one.v = gamma.one.vector, gamma.two.v = gamma.two.vector, ar.coefs = ar.coefs, ma.coefs = ma.coefs))
}

checkAdmissibility <- function(opt.env, box.cox=NULL, small.phi=NULL, ar.coefs=NULL, ma.coefs=NULL, tau=0, bc.lower=0, bc.upper=1) {
  # Check the range of the Box-Cox parameter
  if (!is.null(box.cox)) {
    if ((box.cox <= bc.lower) | (box.cox >= bc.upper)) {
      return(FALSE)
    }
  }
  # Check the range of small.phi
  if (!is.null(small.phi)) {
    if (((small.phi < .8) | (small.phi > 1))) {
      return(FALSE)
    }
  }
  # Check AR part for stationarity
  if (!is.null(ar.coefs)) {
    arlags <- which(abs(ar.coefs) > 1e-08)
    if (length(arlags) > 0L) {
      p <- max(arlags)
      if (min(Mod(polyroot(c(1, -ar.coefs[1L:p])))) < 1 + 1e-2) {
        return(FALSE)
      }
    }
  }
  # Check MA part for invertibility
  if (!is.null(ma.coefs)) {
    malags <- which(abs(ma.coefs) > 1e-08)
    if (length(malags) > 0L) {
      q <- max(malags)
      if (min(Mod(polyroot(c(1, ma.coefs[1L:q])))) < 1 + 1e-2) {
        return(FALSE)
      }
    }
  }
  # Check the eigen values of the D matrix
  D.eigen.values <- eigen(opt.env$D, symmetric = FALSE, only.values = TRUE)$values
  
  return(all(abs(D.eigen.values) < 1 + 1e-2))
}

calcModel <- function(y, x.nought, F, g, w) { # w is passed as a list
  length.ts <- length(y)
  x <- matrix(0, nrow = length(x.nought), ncol = length.ts)
  y.hat <- matrix(0, nrow = 1, ncol = length.ts)
  e <- matrix(0, nrow = 1, ncol = length.ts)
  y.hat[, 1] <- w$w.transpose %*% x.nought
  e[, 1] <- y[1] - y.hat[, 1]
  x[, 1] <- F %*% x.nought + g %*% e[, 1]
  y <- matrix(y, nrow = 1, ncol = length.ts)
  
  loop <- .Call("calcBATS", ys = y, yHats = y.hat, wTransposes = w$w.transpose, Fs = F, xs = x, gs = g, es = e, PACKAGE = "forecast")
  
  return(list(y.hat = loop$y.hat, e = loop$e, x = loop$x))
}



