# seme-ts

## defining the problem
We are given a regression problem with a dataframe consisting in `d` features `X_1, X_2, ..., X_d`,  each feature corresponding to 1D-signal  (e.g. a time-series), and `n` observations. Each observation is the value at a certain time of the `d` signals, i.e. the dataframe is a `N x d` matrix with coefficient `(i,j)` given by `X_j(t_i)` for `i=1,...,d` and `j=1,...,n`. We are interested in predicting a continuous variable `Y` which depends on the values of the 1D-signals `X_1, X_2, ..., X_d`.

Suppose that `n` is small (**small-data problem**). We try to answer the following questions:

*Does the time-series nature of the problem gives us more information on the prediction of the variable `Y`?*

*Is it possible to use the datetime attribute to infer new fake observations (+labels!) and augment the dataset size? How is this changing the algorithm performance?*

### set-up
For the sake of analysis, we consider a database `data_A` with `N` observations where `N>>n`. It is split in `train_A` and `test_A`, this splitting being performed in different ways. We mainly focus on classical train_test_split (random sampling) and time_series_split (random blocks of consecutive observations).

Through a sampling procedure on `train_A`, we derive a smaller database with `n` observations called `data_B`. Again the sampling procedure is not unique and it is discussed in the following subsections.

From `data_B`, we perform **data augmentation** to dispose to a larger number of observations and obtain a bigger dataframe called `data_C`.
The new *fake* features and labels are inferred using different techniques which may depend on the previous sampling method and on the underlying assumption on the time-series.

A same machine-learning model is then trained on the different `train_X` splits, where `X = A, B` or `C`: we'll refer to such model as `model_X`. We evaluate each model on `test_A` and understand whether the imputation technique is improving the result or not.

### dataset examples
We focus on these open repositories at UCI:
 - [Beijing PM2.5 Data Data Set](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)
 - [Appliances energy prediction Data Set](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)

## Current version
To work with our functions, just download the `tsmall` directory and launch python in the same root directory of `tsmall`. It then suffices to type `from tsmall import *` to retrieve all the functionalities.

A baseline routine is provided in `script.py`.

Architecture:
```
tsmall\
    augmentation.py
    subsample.py
script.py
```

### last updates
 - added `augmentation.py` (empty) in `tsmall`
 - added `subsample.py` by Max in `tsmall`
 - created `tsmall` directory with `__init__.py` file to initialize the module
 - added `energydata_complete.csv` in `\` as dataframe example from Appliances energy prediction Data Set


### things to do
 - [x] check mathematical bibliography on Time Series Data Augmentation, see e.g. the link provided by Yiye [Time Series Data Augmentation for Deep Learning: A Survey](https://arxiv.org/pdf/2002.12478.pdf),
 - [ ] check python libraries as `tsaug`
 - [ ] add docstrings and useful comments in all python scripts, e.g. in `subsample.py` provided by Max
 - [ ] discuss the train/test split in `data_A` as well as the possible subsampling techniques to obtain `data_B`
 - [ ] implement proposed strategies in `subsample.py`
 - [ ] create the basic `script.py` with baseline routine
 - [ ] discuss interpolation techniques to obtain `data_C` and thus the assumptions we want to take on the underlying ts (continuity? quasi-stationarity?)
 - [ ] create `augmentation.py` with different augmentation techniques


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

 Observe that most of the cited bibliography is focused on classification problems for NN algorithms. In such framework, the time-series is seen as an input and the augmentation technique allows to generate *similar* time-series on which the model can be trained. For these reasons, many of the aforementioned strategies are not available within our framework (e.g. Dynamic Time Warping in [3]); however a few ideas (e.g. frequency domain transform, decomposition method in [1]) can be developed and put into practice for standard ML algorithms.

 Python libraries:
 - `tsaug` [documentation](https://tsaug.readthedocs.io/en/stable/index.html)

### train test split on `data_A`
Two possibilities: random subsampling (ignoring the time-series order), block subsampling (keeping consecutive observations).

### subsampling
Two possibilities: random subsampling (ignoring the time-series order), block subsampling (keeping consecutive observations).

### data augmentation
Data augmentation is the process of generating artificial data in order to reduce the variance of the predictor and thus avoiding overfitting.


Within our framework, we aim at exploiting the temporal order in the observations and to infer values by using interpolation techniques:
 - Under stationarity assumptions we can use classical bootstrap techniques.
 - Under continuity assumptions of the signals, we can use interpolation techniques.
