# seme-ts

## defining the problem
We are given a regression problem with a dataframe consisting in `d` features, corresponding to `d` 1D-signals `X_1, X_2, ..., X_d` (e.g. time-series), and `n` observations. Each observation is the value at a certain time of the `d` signals and we are interested in predicting the continuous variable `Y`, depending on the values of the previous signals.

Suppose that `n` is small (small-data problem). We try to answer the following questions:

    *Does the time-scale nature of the problem gives us more information on the prediction of the variable `Y`?*

    *Is it possible to use the datetime attribute to infer new fake observations (+labels!) and augment the dataset size? How is this changing the algorithm performance?*

### set-up
For the sake of analysis, we consider a database `data_A` with `N` observations where `N>>n`. It is split in `train_A` and `test_A`.

Through a sampling procedure on `train_A`, we derive a smaller database with `n` observations called `data_B`. The possible sampling procedures are explained in the following subsections.

From `data_B`, we do some **data augmentation** to dispose to a larger number of observations and obtain a bigger dataframe called `data_C`.
The new *fake* features and labels are inferred using different techniques which depend on the previous sampling method (random subsample, random portion, etc.).

A same model is then trained on the different `train_X` splits: we'll refer to such model as `model_X` (for X = A, B or C). We evaluate each model on `test_A` and understand whether the imputation technique is improving the result.

### dataset examples
We focus on these open repositories:
 - [Beijing PM2.5 Data Data Set](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)
 - [Appliances energy prediction Data Set](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)

## Current version

### last updates
 - added `subsample.py` by Max
 - added `energydata_complete.csv` as dataframe example from Appliances energy prediction Data Set

### things to do
 - check mathematical bibliography on Time Series Data Augmentation, see e.g. the link provided by Yiye [Time Series Data Augmentation for Deep Learning: A Survey](https://arxiv.org/pdf/2002.12478.pdf), as well as python libraries as `tsaug`
 - add docstrings and useful comments on functions, e.g. in `subsample.py` provided by Max
 - discuss the train/test split in `data_A` as well as the possible subsampling techniques to obtain `data_B`
 - discuss interpolation techniques on ts and thus the assumption we want to assume on the underlying ts (continuity? quasi-stationarity?)
 - provide a few baseline models for the two datasets (RF, XGB, LR... other?)


## attacking the problem

### performance
The performance of this model is computed on the test set previously defined. We compare this performance with different models tested on the whole train set (or with a known reference value). We then focus on:

*how can we improve such performance knowing that a temporal order is defined on the observations?*


### proposed approach
One possible idea is to do some **data augmentation** to have more samples and so to obtain more "robust" estimate.

One idea is to exploit the temporal order in the observations and to infer values by using interpolation techniques.

### future directions
We also would like to include:
 - threshold analysis with respect to `n`: how many samples do I really need to achieve a good result?
