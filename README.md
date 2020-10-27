# seme-ts

## defining the problem
We are given a regression problem with a dataframe consisting in `d` features `X_1, X_2, ..., X_d`,  each feature corresponding to 1D-signal  (e.g. a time-series), and `n` observations. Each observation is the value at a certain time of the `d` signals, i.e. the dataframe is a `N x d` matrix with coefficient `(i,j)` given by `X_j(t_i)` for `i=1,...,d` and `j=1,...,n`. We are interested in predicting a continuous variable `Y` which depends on the values of the 1D-signals `X_1, X_2, ..., X_d`.

Suppose that `n` is small (**small-data problem**). We try to answer the following questions:

*Does the time-series nature of the problem gives us more information on the prediction of the variable `Y`?*

*Is it possible to use the datetime attribute to infer new fake observations (+labels!) and augment the dataset size? How is this changing the algorithm performance?*

### set-up
For the sake of analysis, we consider a database `data_A` with `N` observations where `N>>n`. It is split in `train_A` and `test_A`, this splitting being performed in different ways. We mainly focus on classical train_test_split (random sampling) and time_series_split (a random block of consecutive observations).

Through a sampling procedure on `train_A`, we derive a smaller database with `n` observations called `data_B`. Again the sampling procedure is not unique and it is discussed in the following subsections.

From `data_B`, we perform **data augmentation** to dispose to a larger number of observations and obtain a bigger dataframe called `data_C`.
The new *fake* features and labels are inferred using different techniques which may depend on the previous sampling method and un the underlying assumption on the time-series.

A same machine-learning model is then trained on the different `train_X` splits, where `X = A, B` or `C`: we'll refer to such model as `model_X`. We evaluate each model on `test_A` and understand whether the imputation technique is improving the result or not.

### dataset examples
We focus on these open repositories:
 - [Beijing PM2.5 Data Data Set](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)
 - [Appliances energy prediction Data Set](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)

## Current version
To work with our functions, just download the `tsmall` directory and launch python in the same root directory of `tsmall`. It then suffices to type `from tsmall import *` to retrieve all the functionalities.

### last updates
 - created `tsmall` directory with `__init__.py` file to initialize the module
 - added `subsample.py` by Max in `tsmall`
 - added `models.py` (empty) in `tsmall`
 - added `energydata_complete.csv` in `\` as dataframe example from Appliances energy prediction Data Set

### things to do
 - [ ] check mathematical bibliography on Time Series Data Augmentation, see e.g. the link provided by Yiye [Time Series Data Augmentation for Deep Learning: A Survey](https://arxiv.org/pdf/2002.12478.pdf), as well as python libraries as `tsaug`
 - [ ] add docstrings and useful comments in all python scripts, e.g. in `subsample.py` provided by Max
 - [ ] discuss the train/test split in `data_A` as well as the possible subsampling techniques to obtain `data_B`
 - [ ] discuss interpolation techniques to obtain `data_C` and thus the assumptions we want to take on the underlying ts (continuity? quasi-stationarity?)
 - [ ] provide a few baseline models for the two datasets (RF, XGB, LR... other?) in `models.py`


### far in the future
 We also would like to include:
  - threshold analysis with respect to `n`: how many samples do I really need to achieve a good result?
  - is it possible to do any consistency analysis or prove the procedure improve robustness?
  - is it always possible to apply the previous procedures or does it strongly depend on *normality assumptions* on the data?

## diving into the problem

### bibliography
Research papers:
 1. [Time Series Data Augmentation for Deep Learning: A Survey](https://arxiv.org/abs/2002.12478)
 2. [Data Augmentation Using Synthetic Data for Time Series Classification with Deep Residual Networks](https://arxiv.org/abs/1808.02455)
 3. [Time Series Data Augmentation for Neural Networks by Time Warping with a Discriminative Teacher](https://arxiv.org/abs/2004.08780)
 4. [An Empirical Survey of Data Augmentation for Time Series Classification with Neural Networks](https://arxiv.org/abs/2007.15951)
 5. [Data Augmentation for Time Series Classification using Convolutional Neural Networks](https://halshs.archives-ouvertes.fr/halshs-01357973)

...related to the GitHub repositories:
 - to 2. https://github.com/hfawaz/aaltd18
 - to 3. and 4. https://github.com/uchidalab/time_series_augmentation

 Python libraries:
 - `tsaug` [documentation](https://tsaug.readthedocs.io/en/stable/index.html)

### train test split on `data_A`
Two possibilities: random subsampling (ignoring the time-series order), block subsampling (keeping consecutive observations).

### subsampling
Two possibilities: random subsampling (ignoring the time-series order), block subsampling (keeping consecutive observations).

### data augmentation
The basic idea of data augmentation is to generate synthetic dataset covering unexplored input space while maintaining correct labels. (from [1](https://arxiv.org/abs/2002.12478))

One idea is to exploit the temporal order in the observations and to infer values by using interpolation techniques.


Under stationarity assumptions we can use classical bootstrap techniques.

Under continuity assumptions of the signals, we can use interpolation techniques.
