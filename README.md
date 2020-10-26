# seme-ts
## question
We suppose that our dataset is composed of 1D-signal, e.g., by time series. We try to answer the following question:
  *how can one use ML algorithm to have consistent predictions when the dataset consists of a small number of observations?*

## setting
We start with a "complete" database of `N` observations (`N` supposed to be large) and we divide it into a train and a test set.

From the train set, we subsample `n << N` observations (in a way that will depend on the dataset) and define a baseline model. The performance of this model will be computed on the test set previously defined. We compare this performance with some model tested on the whole train set.
  *how can we improve such performance?*

### proposed approach
One possible idea is to do some **data augmentation** to have more samples and so to obtain more "robust" estimate.

### future directions
We also would like to include:
 - threshold analysis with respect to `n`: how many samples do I really need to achieve a good result?
 - try different datasets (is the previous question strongly depending on the dataset?)

 ## things to do
  - Check `tsaug` python library.
  - retrieve example datasets
  - test different models (ARMA, RF, XGB... others?) on the datasets and on the subsample
