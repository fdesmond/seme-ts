import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import TransformedTargetRegressor

from tsmall import mdfaug


def quantile_based_augmentation(
    xtrain_B: np.ndarray,
    ytrain_B: np.ndarray,
    n_bins: int = 9,
) -> pd.DataFrame:
    train_C = pd.DataFrame()

    # avoid concentration on unique values
    ytrain_B = ytrain_B + np.random.normal(0, 0.01, size=len(ytrain_B))
    ytrain_B_bin = pd.qcut(ytrain_B, n_bins).value_counts().sort_index()

    # run quantile-based data augmentation
    for i in range(n_bins):
        row_id = ytrain_B.apply(
            lambda x: np.round(x, 3) in ytrain_B_bin.index.values[i]
        )
        train_B_bin_sample = pd.concat(
            [xtrain_B.loc[row_id], ytrain_B.loc[row_id]],
            axis=1,
        )

        # augmentation with combined techniques: fourier1, fourier2 and wavelet
        train_C_bin_sample = mdfaug(
            train_B_bin_sample,
            n_true=3, n_f1=2, n_f2=2, n_w=2,
            sigma_list_f1=[0.3, 0.3], ff_list_f1=[0.5, 0.4],
            sigma_list_f2=[0.5, 0.2], ff_list_f2=[0.5, 0.6],
            sigma_list_w=[0.7, 0.5], ff_list_w=[0.5, 0.7],
            is_target_distorted=False
        )
        train_C = train_C.append(train_C_bin_sample)
        del train_B_bin_sample, train_C_bin_sample
    return train_C


def run_knn(X_train, Y_train):
    '''Run KNN with GridSearch(cv=3) and apply log transformation
    to target variable.

    Returns:
        - trained model
    '''
    params_knn = {'n_neighbors': list(np.arange(10, 30, 2))}
    knn = KNeighborsRegressor()
    gs = GridSearchCV(
        knn,
        param_grid=params_knn,
        cv=3,
        scoring='neg_mean_squared_error'
    )
    tt = TransformedTargetRegressor(
        regressor=gs,
        func=np.log1p,
        inverse_func=np.expm1
    )
    tt.fit(X_train, Y_train)

    return tt
