# seme-ts
## defining the problem
We suppose that our dataset is composed of 1D-signals and we want to predict a real variable (regression problem). We try to answer the following questions:

  *How can one use ML algorithm to have a consistent prediction when the dataset consists of a small number of observations?*

## attacking the problem
### set-up
We start with a "complete" database of `N` observations: `N` is supposed to be large and define the datetime points for each signal. We divide it into a train and a test set.

From the train set, we subsample `n << N` observations (in a way that will be discussed) and define a baseline model on it. The performance of this model is computed on the test set previously defined. We compare this performance with different models tested on the whole train set (or with a known reference value). We then focus on:

  *how can we improve such performance knowing that a temporal order is defined on the observations?*

  *Is it possible to use the datetime attribute to infer new fake observations (+labels!) and augment the dataset size? How is this changing the algorithm performance?*

#### dataset examples
We focus on these open repositories:
 - [Beijing PM2.5 Data Data Set](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)
 - [Appliances energy prediction Data Set](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)

### proposed approach
One possible idea is to do some **data augmentation** to have more samples and so to obtain more "robust" estimate.

One idea is to exploit the temporal order in the observations and to infer values by using interpolation techniques.

### future directions
We also would like to include:
 - threshold analysis with respect to `n`: how many samples do I really need to achieve a good result?

 ## things to do
  - how to divide train/test set
  - Check `tsaug` python library.
  - which interpolation techniques and thus which assumption on the time-series (continuity?) shall one take?
  - test different models (ARMA, RF, XGB... others?) on the datasets and on the subsample
