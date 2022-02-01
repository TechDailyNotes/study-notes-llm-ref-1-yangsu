## Source

- [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://arxiv.org/pdf/1704.04110.pdf)
- RNN based probabilistic forecasting

## Intro

- challenge: the distribution of the magnitude in time series (sales per item in certain timepoint) in real world is highed skewed
- which makes group-based regularization schemes, input standardization, even BN, less effective
- DeepAR ads compared to other approaches
	- learn seasonal and across dependency, minimal manual feature engineering needed
	- can forecast item with little or no history by learning from similar items
- DeepAR ads in prediction accuracy
	- do probabilistic forecast in the form of Monte Carlo samples, can be used to compute consistent quantile estimates
	- does not assume Gaussian noise, user can choose likelihood functions
- previous
	- individual: ARIMA, exp smoothing
	- real world: non-stationarity, non-Gaussian like, non-consistent variance
	- ones capture the real world: zero-inflated Poisson, neg-bio dist, tailored multi-stage likelihood
	- ones capture across info: matrix factorization, bayesian via priors
	- neural network approaches: independent prediction, RNN generalize to multimodal tasks

## Model

- training
	- usual: use feature info *x* as input to current timepoint
	- autoregressive: use last obs. *z* as input to current timepoint
	- recurrent: use last latent state *h* as input to current timepoint
	- model predicts a distribution *l* given its hyperparameters
	- [prev z & h, cur x -> model -> cur h -> cur l -> cur z]
	- optimize l by comparing predicted z and real z
- testing
	- only difference is that every z, h are now predicted value *z'*, *h'*
	- use l and h' to *sample* z'
	- [prev z' & h', cur x -> model -> cur h' -> cur l -> cur z']
- notice that different from a seq-to-seq model that has different encoder and decoder, here there is only a single network with shared weights in *h*
- h at current timeframe is computed by a function with input prev h & z and cur x, the function is implemented by a multi-layer RNN with LSTM cells

## Method

- should be chosen to match the statistical property of the data
- network directly predicts all parameters (ie. mean, sd) of the prob-dist for next timepoint
- can use Gaussian likelihood for real value data, neg-bio likelihood for pos discrete data, and other like beta likelihood for unit interval, etc.
- e.g. for Gaussian likelihood
	- mean = w_mean * h + b_mean, affine function of h
	- sd = log(1 + exp(w_sd * h + b_sd))
- e.g. for neg-bio likelihood
	- mean & (shape param) alpha = log(1 + exp(w_ * h + b_))
	- var = mean + mean ** 2 * alpha
- goal: maximize the log-likelihood of [log l(z|likelihood)] across all timepoint and items
- scale handling
	- use average scale heuristic
	- weighted sampling for high-velocity item forcasting
- feature info x can be item-dependent or time-dependent, or both
	- but they need to be available in prediction range
	- item-dependent features are repeated along timepoints
- evaluation metrics (on prediction)
	- p-risk (quantile loss), mainly 0.5 and 0.9 risk
	- ND, normalized deviation
	- RMSE