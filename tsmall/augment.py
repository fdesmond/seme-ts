import numpy as np
from numpy.fft import rfft, irfft
import pandas as pd
import pywt

DISTORT_COEFFICIENT = 0.2
DISTORT_METHOD = 'fourier2'
FRACTION_OF_DISTORTED_FEATURES = 0.5


# 1D signal distortion through Fourier or Wavelet Transform
def signal_distortion(
    signal: np.ndarray,
    sigma: float = DISTORT_COEFFICIENT,
    method: str = DISTORT_METHOD
) -> np.ndarray:
    '''
    Parameters
    ----------
    signal : 1D-ndarray, input vector to distort
    sigma : positive float64, scale of the perturbation (0 iff no perturbation)
    method : str in *{fourier1, fourier2, wavelet}*, type of transformation/perturbation

    Returns
    -------
    transformed_signal : 1D-ndarray of the same size of input

    Depending on the method, it performs a discrete fourier/wavelet transformation,
    perturb it with IID gaussian variables and invert the transformation. The input is
    perturbed with the perturbation magnitude tuned by the sigma parameter.
    '''

    # FFT to real signals, we force the length to be an even number
    signal_length = len(signal)
    signal_length_is_odd = signal_length % 2
    if signal_length_is_odd:
        signal_last = signal[-1]
        signal = signal[:-1]
        signal_length = len(signal)

    if method not in {'fourier1', 'fourier2', 'wavelet'}:
        raise TypeError("Method must be 'fourier1', 'fourier2' or 'wavelet'.")

    # fourier transform on phase
    if method == 'fourier1':
        ft = rfft(signal)
        perturbation = np.random.normal(0, sigma, size=signal_length//2+1)
        # ftb = ft + perturbation   # noise applied on fourier transform
        # signal2 = irfft(ftb)   # inverse fourier transform
        ft_p = ft.real*np.cos(perturbation) - ft.imag*np.sin(perturbation) +\
            1j * (ft.imag*np.cos(perturbation) + ft.real*np.sin(perturbation))
        transformed_signal = irfft(ft_p)

    # fourier transform on amplitudes
    if method == 'fourier2':
        ft = rfft(signal)
        scale_perturbation = np.append(
            np.ones(signal_length//2+1 - signal_length//4),
            np.exp(np.random.normal(scale=sigma, size=signal_length//4))
        )
        ft_p = ft*scale_perturbation
        transformed_signal = irfft(ft_p)

    # wavelet transform on the first component only
    if method == 'wavelet':
        cA, cD = pywt.dwt(signal, 'db2')     # decomposition
        perturbed_cA = cA + np.random.normal(0, sigma, size=len(cA))
        perturbed_cD = cD + np.random.normal(0, 0, size=len(cD))
        transformed_signal = pywt.idwt(perturbed_cA, perturbed_cD, 'db2')

    # restore the original size of the vector
    if signal_length_is_odd:
        transformed_signal = np.append(transformed_signal, signal_last)

    return transformed_signal


# dataframe augmentation
def dfaug(
    X: pd.DataFrame,
    sigma: float = DISTORT_COEFFICIENT,
    frac_features: float = FRACTION_OF_DISTORTED_FEATURES,
    method: str = DISTORT_METHOD,
    y_dist: bool = False
) -> pd.DataFrame:
    '''
    Parameters
    ----------
    X : pd.DataFrame of size (n, d), the first d-1 columns represent the 1d-signal to be disturbed
        (see y_dist for last column)
    sigma : positive float64, it tunes the distortion in signal_distortion() (0 iff no distortion)
    frac_features : float64 in [0,1], fraction of features to perturb for each time window
    method : str in {*fourier1* for phase, *fourier2* for amplitude, *wavelet* for wavelet},
             see signal_distortion()
    y_dist : boolean value, True for changing the last column in X

    Returns
    -------
    X_dist : pandas df consisting of a distorted version of X

    The output is a shuffled copy of the input where random time windows of some (random) features are
    perturbed by means of signal_distortion(). To concatenate the output to the original DataFrame
    just run: pd.concat([X, X_dist], axis=0)
    '''
    n, d = X.shape
    col = X.columns     # save column names for later
    t_int = n // 5      # length of time intervals
    t_points = sorted(np.random.randint(0, n - t_int, 5))       # we select 5 time intervals

    # coping the five blocks into a new 2d-ndarray that will be distorted
    X_dist = np.concatenate((X[t_points[0]:t_points[0]+t_int], X[t_points[1]:t_points[1]+t_int],\
                             X[t_points[2]:t_points[2]+t_int], X[t_points[3]:t_points[3]+t_int],\
                             X[t_points[4]:t_points[4]+t_int]), axis=0)

    # applying distortion to each time-block and for some random features
    for i in [t_int*t for t in range(5)]:
        for j in np.random.choice(range(d-1), int(d*frac_features), replace=False):     # select sqrt(d) different features at random
            X_dist[i:i+t_int, j] = signal_distortion(X_dist[i:i+t_int, j], sigma=sigma, method=method)
            if y_dist: X_dist[i:i+t_int, -1] = signal_distortion(X_dist[i:i+t_int, -1], sigma=sigma*0.5, method=method)

    # back to dataframe
    X_dist = pd.DataFrame(X_dist, columns=col)
    return X_dist


# multiple augmentation using dfaug and the three different methods
def mdfaug(X, n_true, n_f1, n_f2, n_w, \
           sigma_list_f1 = None, ff_list_f1 = None, \
           sigma_list_f2 = None, ff_list_f2 = None, \
           sigma_list_w = None, ff_list_w = None, \
           y_dist=False):
    ''' Mixed Distortion:
        Parameters
        ----------
        X : true dataframe
        n_true : copies of X to include in the output
        n_f1 : numbers of distorted copies from fourier1 to include in the output
        n_f2 : numbers of distorted copies from fourier2 to include in the output
        n_w : numbers of distorted copies from wavelet to include in the output

        sigma_list_f1, ff_list_f1 : list (length n_f1), parameter for fourier distortion, if not given randomly select from [0, 1]
        sigma_list_f2, ff_list_f2 : list (length n_f2), parameter for fourier2 distortion, if not given randomly select from [0, 1]
        sigma_list_w, ff_list_w : list (length n_w), parameter for wavelet distortion, if not given randomly select from [0, 1]

        y_dist : boolean value, True for changing the last column in X. Refer to dfaug()

        Return
        ----------
        X_aug : dataframe composed of n_true copies of X and n_f1 + n_f2 + n_w distorted copies with dfaug.
    '''

    # check length of lists with n_f1, n_f2 and n_w
    #
    if sigma_list_f1 is not None: assert len(sigma_list_f1) == n_f1
    else: sigma_list_f1 = np.random.uniform(size = n_f1)
    if ff_list_f1 is not None: assert len(ff_list_f1) == n_f1
    else: ff_list_f1 = np.random.uniform(size = n_f1)

    #
    if sigma_list_f2 is not None: assert len(sigma_list_f2) == n_f2
    else: sigma_list_f2 = np.random.uniform(size = n_f2)
    if ff_list_f2 is not None: assert len(ff_list_f2) == n_f2
    else: ff_list_f2 = np.random.uniform(size = n_f2)

    #
    if sigma_list_w is not None: assert len(sigma_list_w) == n_w
    else: sigma_list_w = np.random.uniform(size = n_w)
    if ff_list_w is not None: assert len(ff_list_w) == n_w
    else: ff_list_w = np.random.uniform(size = n_w)


    # initialize output with n_true copies of X
    X_aug = pd.concat([X]*n_true)


    # append distorted copies of X through dfaug()
    for k in range(n_f1):
        X_aug = X_aug.append(dfaug(X, sigma=sigma_list_f1[k], frac_features=ff_list_f1[k], method='fourier1', y_dist=y_dist))
    for k in range(n_f2):
        X_aug = X_aug.append(dfaug(X, sigma=sigma_list_f2[k], frac_features=ff_list_f2[k], method='fourier2', y_dist=y_dist))
    for k in range(n_w):
        X_aug = X_aug.append(dfaug(X, sigma=sigma_list_w[k], frac_features=ff_list_w[k], method='wavelet', y_dist=y_dist))


    return X_aug
