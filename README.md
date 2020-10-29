# seme-tsmall

## Defining the problem
We are given a regression problem with a dataframe consisting in `d` features `X_1, X_2, ..., X_d`,  each feature corresponding to 1D-signal  (e.g. a time-series), and `n` observations. Each observation is the value at a certain time of the `d` signals, i.e. the dataframe is a `n x d` matrix with coefficient `(i,j)` given by `X_j(t_i)` for `i=1,...,d` and `j=1,...,n`. We are interested in predicting a continuous variable `Y` which depends on the values of the 1D-signals `X_1, X_2, ..., X_d`.

Suppose that `n` is small (**small-data problem**). We try to answer the following questions:

*Does the time-series nature of the problem gives us more information on the prediction of the variable `Y`?*

*Is it possible to use the datetime attribute to infer new synthetic observations and augment the dataset size? Is this helping reducing overfitting?*

### set-up
For the sake of analysis, we consider a database `data_A` with `N` observations where `N>>n`. It is split in `train_A` and `test_A`, this splitting being performed in different ways. We mainly focus on classical train_test_split (random sampling) and time_series_split (random block of consecutive observations and window cropping).

Through a **sampling procedure** on `train_A`, we derive a smaller database with `n` observations called `data_B`. Again the sampling procedure is not unique and it is discussed in the following subsections.

From `data_B`, we perform **data augmentation** to dispose of a larger number of observations and obtain a bigger dataframe called `data_C`.
The new *fake* features and labels are inferred using different techniques which may depend on the previous sampling method and on the underlying assumption on the time-series.

A same machine-learning algorithm is then trained on the different `train_X` splits, where `X = A, B` or `C`: this yields the three models `model_A`, `model_B` and `model_C`. We evaluate each model on `test_A` and understand whether the imputation technique is improving the result or not.

### dataset examples
We focus on these open repositories available at UCI:
 - [Beijing PM2.5 Data Data Set](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)
 - [Appliances energy prediction Data Set](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)

## Current module version
To work with our functions, just download the `tsmall` directory and launch python in the same root directory of `tsmall`. It then suffices to type `from tsmall import *` to retrieve all the functionalities.

A baseline routine is provided in the notebook `oracle.ipynb`. The ipynb `signal_distortion.ipynb` contains a few examples using the module `tsmall` and, in particular, the function `signal_distortion()`.

Architecture of `tsmall` module:
```
tsmall/
    augment.py   # containing signal_distortion and dfaug
    preprocessing.py  # containing block_sampling and min_max_normalization
```

#### dependencies
You need to pre-install numpy, matplotlib, scikit-learn and pywt (wavelet) for running the code.

### last updates
 - implement different methods for transformation (fourier and wavelet) in `augment.py`
 - added support to modify the `y` in `dfaug(y_dist=True)`
 - added a very first version of the notebook `signal_distortion.ipynb`
 - added `oracle.ipynb` for baseline tests
 - added `preprocessing.py` with `block_sampling()` and `min_max_normalization()`
 - old scripts went to `old` folder


### overall progress
 - [x] check mathematical bibliography on Time Series Data Augmentation
 - [x] check python libraries as `signal_distortion`
 - [x] discuss the train/test split in `data_A` as well as the possible subsampling techniques to obtain `data_B`
 - [x] implement the sampling strategy in `preprocessing.py`
 - [x] discuss interpolation techniques to obtain `data_C` and thus the assumptions we want to take on the underlying ts (continuity? quasi-stationarity?)
 - [x] implement the augmentation techniques through Fourier in `augment.py`
 - [x] create the basic notebook `oracle.py` with baseline routine
 - [x] implement wavelet technique in `augment.py`
 - [ ] test the augmented dataframe
 - [ ] use different metric_score to asset the overall performance
 - [ ] try xgboost
 - [ ] add docstrings and useful comments in all python scripts



#### far in the future
 We also would like to include:
  - is it possible to do any consistency analysis or prove the procedure improve robustness?
  - is it always possible to apply the previous procedures or does it strongly depend on *normality assumptions* on the data?

## Documentation

### bibliography
Research papers:
 1. [Time Series Data Augmentation for Deep Learning: A Survey](https://arxiv.org/abs/2002.12478)
 2. [Data Augmentation Using Synthetic Data for Time Series Classification with Deep Residual Networks](https://arxiv.org/abs/1808.02455)
 3. [Time Series Data Augmentation for Neural Networks by Time Warping with a Discriminative Teacher](https://arxiv.org/abs/2004.08780)
 4. [An Empirical Survey of Data Augmentation for Time Series Classification with Neural Networks](https://arxiv.org/abs/2007.15951)
 5. [Data Augmentation for Time Series Classification using Convolutional Neural Networks](https://halshs.archives-ouvertes.fr/halshs-01357973)
 6. [Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring using Convolutional Neural Networks](https://arxiv.org/abs/1706.00527)
 7. [Improving the Accuracy of Global Forecasting Models using Time Series Data Augmentation](https://arxiv.org/abs/2008.02663v1)
 8. [Generating Synthetic Time Series to Augment Sparse Datasets](https://ieeexplore.ieee.org/document/8215569)
 9. [Data Augmentation for EEG-Based Emotion Recognition with Deep Convolutional Neural Networks](https://link.springer.com/chapter/10.1007/978-3-319-73600-6_8)
 10. [Time-Series Data Augmentation based on Interpolation](https://www.sciencedirect.com/science/article/pii/S1877050920316914)

...related to the GitHub repositories:
 - [2] https://github.com/hfawaz/aaltd18
 - [3, 4] https://github.com/uchidalab/time_series_augmentation
 - [6] https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data

 Observe that most of the cited bibliography is focused on classification problems for NN algorithms. In such framework, the time-series is seen as an input and the augmentation technique allows to generate *similar* time-series on which the model can be trained. For these reasons, many of the aforementioned strategies are not suitable for our framework (e.g. Dynamic Time Warping in [3]); however a few ideas (e.g. frequency domain transform, decomposition method in [1]) could be developed and put into practice for standard ML algorithms.

 Python libraries:
 - `signal_distortion` is useful to modify time-series in the time space, but not necessarily for our scope | see the [documentation](https://signal_distortion.readthedocs.io/en/stable/index.html)
 - `sigment` data augmentation for audio signals | see the [documentation](https://notes.eonu.net/docs/sigment/0.1.1/index.html)

### train test split
The train test split is obtained via a similar procedure to obtaining `data_B`, we thus refer to the following subsection.

### subsampling
We tackle three possible subsampling techniques in order to obtain `data_B`:
 1. random subsampling: take `n` observations from `train_A` uniformly at random without replacement;
 2. block subsampling: consider `u` a uniform random integer between 1 and `size(train_A)-n`, select `n` consecutive rows starting from row `u`;
 3. window cropping (see `block_sampling()`): consider `k` blocks of total size `n`, apply (2) for every block (without oversampling).

 The sampling procedure is validated by comparing old and new histograms: we want to be sure that the sampling procedure has not altered the statistical properties of the original dataframe.

### data augmentation
Data augmentation is the process of generating artificial data in order to reduce the variance of the predictor and thus avoiding overfitting.

Within our framework, we aim at exploiting the temporal order in the observations and to infer values by using interpolation techniques:
 - Under stationarity assumptions we can use classical bootstrap techniques.
 - Under continuity assumptions of the signals, we can use interpolation techniques as in [10], although with not great performances.

For very general 1D-signals, we can try the following procedure:
 1. select a random time-window of the signal;
 2. apply DFT and perturb certain frequencies;
 3. apply IDFT to reconstruct a portion of the original signal;
 4. assign Y[time-window].

This procedure is tackled by `signal_distortion()` for a single 1D-signal and by `dfaug()` for the whole dataframe. A normalization procedure is usually performed to apply the same distortion to every signal.

Some key points:
 - is some frequency more important than others? Is there an automatic way to decide it?
 - understanding fine tuning in Fourier/Wavelet decomposition
