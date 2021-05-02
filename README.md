# seme-tsmall
Work done in [Institut de Mathématiques de Bordeaux](https://www.math.u-bordeaux.fr/imb/spip.php) organized by [AMIES](https://www.agence-maths-entreprises.fr/public/pages/index.html) and in collaboration with [FieldBox.ai](https://www.fieldbox.ai/).

The report can be found on Hal Archives: https://hal.archives-ouvertes.fr/hal-03211100

Link to the event: http://seme-bordeaux.sciencesconf.org/

## Defining the problem
We are given a regression problem with a dataframe consisting in `d` features `X_1, X_2, ..., X_d` and `n` observations.  Each feature corresponds to a 1D-signal  (e.g. a time-series): the i-th observation is the value at a certain time `t_i` of the `d` signals.

Thus, the dataframe is a `n x d` matrix with coefficient `(i,j)` given by `X_j(t_i)` for `i=1,...,d` and `j=1,...,n`.

We are interested in predicting the variable `y` which depends on the values of the 1D-signals `X_1, X_2, ..., X_d`. Namely, we suppose that `y = f(X_1, ..., X_d)`. Note that we do not suppose `y` to be an explicit function of the time.

Suppose that `n` is small (**small-data problem**). We try to answer the following questions:

*Does the time-signal nature of the features gives us more information than the sole observations `X_j(t_i)`?*

*Is it possible to infer new data and augment the dataset size? Is this helping in predicting `y` (e.g., reducing overfitting)?*



### experimental method
For the sake of analysis, we consider a dataframe `data_A` with `N` observations where `N>>n`. It is split in `train_A` and `test_A`. We mainly focus on classical (uniform) random sampling and on random block sampling (random blocks of consecutive observations), depending on the dataset.

#### sampling procedure
Through a **sampling procedure** on `train_A`, we derive a smaller dataframe called `train_B`; similarly we obtain `test_B` from `test_A`. The two dataframes `train_B` and `test_B` form `data_B`: the small dataframe with `n` observations. We refer to the Documentation section for the sampling procedure.

#### data augmentation
From `train_B`, we perform **data augmentation** to dispose of a larger number of observations and obtain a bigger dataframe called `data_C`, the dimension of `data_C` is up to 8 times the one of `train_B`. The **synthetic features** and labels are inferred using different techniques, we refer to the data augmnetation section.

#### evaluation with RMSE and R2 score
A same machine-learning algorithm is then trained on the different `train_X` splits, where `X = A, B` or `C`: this yields the three models `model_A`, `model_B` and `model_C`. We evaluate each model on `test_A` (and `test_B`) to understand whether the imputation technique is improving the stability and/or the score of `model_C` with respect to `model_B`. The metrics under consideration are given by the (root) mean square error and the R2 score.

### dataset examples
We focus on the following open repository available at UCI: [Appliances energy prediction Data Set](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)


## Current module version
To work with our functions, just download the `tsmall` directory and launch python in the same root directory of `tsmall`. It then suffices to type `from tsmall import *` to retrieve all the functionalities.

Here a list of the relevant notebooks present in the repository:
- `DL_aug.ipynb` : test the data augmentation for energy dataset using knn algorithm
- `TS_aug.ipynb` : test data augmentation using LSTM-VAE
- `signal_distortion.ipynb` : contains information about fourier and wavelet discrete transform, it uses the submodule `tsmall.augment`.


Architecture of `tsmall` module:
```
tsmall/
    augment.py          # contain signal_distortion, dfaug and mdfaug
    dl_method.py        # function for LSTM-VAE
    preprocessing.py    # contain block_sampling and min_max_normalization
utils.py                # useful function to lighten notebooks
```

#### dependencies
You need to pre-install `numpy`, `pandas` and `pywt` (wavelet pkg) for running `tsmall`. The notebooks require `matplotlib` and `scikit-learn`. If you want to test DL methods, you need `keras` and `tensorflow < 2`.

If you have `pipenv` installed, the environment is set in the `Pipfile`. You can load it by simply executing `pipenv install` inside the git repository.

### last updates
 - added `utils.py` with useful functions for notebook
 - pipenv environment, code cleaning and refactoring with hint typing
 - clean a bit + `DL_aug.ipynb` and `TS_aug.ipynb` merged from deep-learning branch
 - added `aug_knn.ipynb` and `aug_knn_sstt.ipynb`
 - added `mdfaug()` in `augment.py`
 - added a very first version of the notebook `signal_distortion.ipynb`
 - old scripts moved to `old` folder

### overall progress
 - [x] check mathematical bibliography and python libraries on 1d-signal Data Augmentation
 - [x] discuss train/test split in `data_A` as well as the possible subsampling techniques to obtain `data_B`. Implement the sampling strategies in `preprocessing.py`
 - [x] discuss augmentation techniques to obtain `data_C`, the assumptions on the underlying signal (continuity? quasi-stationarity?). Implement techniques in  `augment.py`
 - [x] test data augmentation for knn in `aug_knn.ipynb` with different score metrics and plot the results
 - [x] add docstrings useful comments in all python scripts
 - [x] test linearRegression and decision trees (xgboost, adaboost, randomforest)
 - [x] prepare presentation for AMIES
 - [ ] add assertion errors to python code
 - [ ] test PM2.5 dataframe
 - [ ] clean `signal_distortion.py` and test removing high frequencies
 - [ ] to-do list in `todo.txt`
 - [ ] write report for AMIES


#### far in the future
 We also would like to include:
  - Which hypothesis on the model for which the procedure should give good results?
  - Does it makes more sense to create synthetic features by combinations of the existing ones instead of perturbation?


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

Within our framework, we can try to exploit the time-signal nature of the observations and to infer new values. Depending on the hypothesis one takes on `y`, different techniques are available:
 - Under stationarity assumptions one can use classical bootstrap techniques or model-methods (ARIMA, ARCH, etc.).
 - Under continuity assumptions of the signals, one can use interpolation techniques as in [10], (however this seems not to substantially improve the results).
 - Fourier/wavelet transform, see next subsection
 - categorical data augmentation: group observations by `y`-labels + transformation in feature-space

 Since we do not want to assume any additional hypothesis on `y`, we exclude the first two possibilities and focus on the last twos. It turns out that categorical data augmentation works pretty well with k-NN.

#### Fourier and wavelet discrete transform
For very general 1D-signals, we try the following procedure:
 1. select a random time-window of the signal;
 2. apply discrete transform such as FFT or WDF and perturb certain frequencies;
 3. apply inverse transform to reconstruct a portion of the original signal;
 4. assign y[time-window].

This procedure is tackled by `signal_distortion()` for a single 1D-signal and by `dfaug()` for the whole dataframe.

#### aggregate and generate
Instead of grouping by time-windows (consecutive observations), we group by `y`-labels (e.g. select observations for which the corresponding `y`-label is in the same quantile interval). We then apply DFT/DWF and assign the original `y` values.
