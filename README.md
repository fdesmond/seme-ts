# seme-tsmall

## Defining the problem
We are given a regression problem with a dataframe consisting in `d` features `X_1, X_2, ..., X_d`,  each feature corresponding to a 1D-signal  (e.g. a time-series), and `n` observations. Each observation is the value at a certain time `t_i` of the `d` signals, i.e. the dataframe is a `n x d` matrix with coefficient `(i,j)` given by `X_j(t_i)` for `i=1,...,d` and `j=1,...,n`. We are interested in predicting the variable `Y` which depends on the values of the 1D-signals `X_1, X_2, ..., X_d`.

Suppose that `n` is small (**small-data problem**). We try to answer the following questions:

*Does the time-series nature of the problem gives us more information on the prediction of the variable `Y`?*

*Is it possible to use the datetime attribute to infer new synthetic observations and augment the dataset size? Is this helping reducing overfitting?*

### method
For the sake of analysis, we consider a dataframe `data_A` with `N` observations where `N>>n`. It is split in `train_A` and `test_A`, this splitting being performed in different ways. We mainly focus on classical (uniform) random sampling and on random block sampling (random blocks of consecutive observations), depending on the dataset.

Through a **sampling procedure** on `train_A`, we derive a smaller dataframe with `n` observations called `data_B`. Again the sampling procedure can be performed in different ways, we refer to the the following subsections.

From `data_B`, we perform **data augmentation** to dispose of a larger number of observations and obtain a bigger dataframe called `data_C`, the dimension of `data_C` is usually 2, 3 or 4 times the one of `data_B`. The **synthetic features** and labels are inferred using different techniques, we refer to the data augmnetation section.

A same machine-learning algorithm is then trained on the different `train_X` splits, where `X = A, B` or `C`: this yields the three models `model_A`, `model_B` and `model_C`. We evaluate each model on (different subsets of) `test_A` and understand whether the imputation technique is improving the stability and/or the score of `model_C` with respect to `model_B`.

### dataset examples
We focus on these open repositories available at UCI:
 - [Beijing PM2.5 Data Data Set](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)
 - [Appliances energy prediction Data Set](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)


## Current module version
To work with our functions, just download the `tsmall` directory and launch python in the same root directory of `tsmall`. It then suffices to type `from tsmall import *` to retrieve all the functionalities.

Here a list of the notebooks present in the repository:
- `aug_test.ipynb` : test the data augmentation for energy dataset, it perform the comparison using xgboost/randomForest
- `aug_test_linear.ipynb` : same as above with linearRegression
- `oracle.ipynb` : first implementation of tsmall with the energy dataset (to be deprecated)
- `signal_distortion.ipynb` : contains information about fourier and wavelet discrete transform, it uses the submodule `tsmall.augment`.


Architecture of `tsmall` module:
```
tsmall/
    augment.py          # contain signal_distortion and dfaug
    preprocessing.py    # contain block_sampling and min_max_normalization
```

#### dependencies
You need to pre-install `numpy`, `pandas` and `pywt` (wavelet pkg) for running `tsmall`. The notebooks require `matplotlib`, `scikit-learn` and `xgboost`.

### last updates
 - added `aug_test.ipynb` with `xgboost` and `np.log(y)`
 - implement different methods for transformation (fourier and wavelet) in `augment.py`
 - added support to modify the `y` in `dfaug(y_dist=True)`
 - added a very first version of the notebook `signal_distortion.ipynb`
 - old scripts moved to `old` folder

### overall progress
 - [x] check mathematical bibliography and python libraries on 1d-signal Data Augmentation
 - [x] discuss train/test split in `data_A` as well as the possible subsampling techniques to obtain `data_B`. Implement the sampling strategies in `preprocessing.py`
 - [x] discuss augmentation techniques to obtain `data_C`, the assumptions on the underlying signal (continuity? quasi-stationarity?). Implement techniques in  `augment.py`
 - [x] create the basic notebook `oracle.py` with baseline routine
 - [x] test the augmented dataframe in `aug_test.ipynb` with different score metrics, models (RF, xgboost, linearRegression)
 - [x] add docstrings useful comments in all python scripts
 - [ ] test PM2.5 dataframe
 - [ ] try energy dataset with classification, is this improving the result?
 - [ ] add assertion errors to python code
 - [ ] implement Sonia's idea with classification
 - [ ] start writing report
 - [ ] start writing presentation

#### far in the future
 We also would like to include:
  - mathematical modelization and precise hypothesis on the model for which the procedure should give good results
  - does it makes more sense to create synthetic features by combinations of the existing ones instead of perturbation? (see proposition by Sonia)


## Documentation

### bibliography
Related research papers:
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

GitHub repositories:
 - [2] https://github.com/hfawaz/aaltd18
 - [3, 4] https://github.com/uchidalab/time_series_augmentation
 - [6] https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data

 Observe that most of the cited bibliography is focused on classification problems for NN algorithms. In such framework, the time-series is seen as an input and the augmentation technique allows to generate *synthetic time-series* on which the model can be trained. For these reasons, many of the aforementioned strategies are not suitable for our framework (e.g. Dynamic Time Warping in [3]); however a few ideas (e.g. frequency domain transform, decomposition method in [1]) could be developed and put into practice for standard ML algorithms.

 Python libraries:
 - `tsaug` is useful to modify time-series in the time space, but not necessarily for our scope | see the [documentation](https://signal_distortion.readthedocs.io/en/stable/index.html)
 - `sigment` data augmentation for audio signals | see the [documentation](https://notes.eonu.net/docs/sigment/0.1.1/index.html)

### train test split
The train test split is obtained via a similar procedure to obtaining `data_B`, we thus refer to the following subsection.

### subsampling
We tackle two possible subsampling techniques in order to obtain `data_B`:
 1. random subsampling: take `n` observations from `train_A` uniformly at random without replacement;
 3. window cropping (see `block_sampling()`): consider `k` distinct blocks of total size `n`, each block consists in a fixed number of consecutive rows. The blocks are not overlapping.

 The sampling procedure is checked by comparing old and new histograms: we want to be sure that the sampling procedure has not altered the statistical properties of the original dataframe.

### data augmentation
Data augmentation is the process of generating artificial data in order to reduce the variance of the predictor and thus avoiding overfitting.

Within our framework, we aim at exploiting the temporal order in the observations and to infer values by using interpolation techniques:
 - Under stationarity assumptions we can use classical bootstrap techniques (Max work?).
 - Under continuity assumptions of the signals, we can use interpolation techniques as in [10], although with not great performances.
 - Fourier/wavelet transform, see next subsection
 - good points aggregation and convex combination

#### Fourier and wavelet discrete transform
For very general 1D-signals, we try the following procedure:
 1. select a random time-window of the signal;
 2. apply discrete transform such as FFT or WDF and perturb certain frequencies;
 3. apply inverse transform to reconstruct a portion of the original signal;
 4. assign Y[time-window].

This procedure is tackled by `signal_distortion()` for a single 1D-signal and by `dfaug()` for the whole dataframe.

Some key points:
 - is some frequency more important than others? Is there an automatic way to decide it?
 - understanding fine tuning in Fourier/Wavelet decomposition
 - is signal normalization useful? (it seems not from energy dataframe)

#### aggregate and generate
To add.
