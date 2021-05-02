import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, r2_score

from tsmall import mdfaug


def repeat_augmentation_and_get_statistics(
    xtrain_A_t,
    ytrain_A,
    xtest_A_t,
    ytest_A,
    sxtest_A_t,
    p: float = 0.05,
    n_tries: int = 50
):
    # initialize list of results
    R2_B = np.zeros(shape=(n_tries, 2))
    R2_C = np.zeros(shape=(n_tries, 2))
    rmse_B = np.zeros(shape=(n_tries, 2))
    rmse_C = np.zeros(shape=(n_tries, 2))

    for k in range(n_tries):
        # sample data_B
        xtrain_B = xtrain_A_t.sample(frac=p, random_state=k)
        ytrain_B = ytrain_A.loc[xtrain_B.index]
        xtest_B = xtest_A_t.sample(frac=p, random_state=k)
        ytest_B = ytest_A.loc[xtest_B.index]

        train_C: pd.DataFrame = quantile_based_augmentation(
            xtrain_B,
            ytrain_B,
            n_bins=9
        )
        ytrain_C = train_C[train_C.columns[-1]]
        xtrain_C = train_C[train_C.columns[:-1]]

        # scaling
        ss_B = StandardScaler()
        sxtrain_B = ss_B.fit_transform(xtrain_B)
        sxtest_B = ss_B.transform(xtest_B)
        ss_C = StandardScaler()
        sxtrain_C = ss_C.fit_transform(xtrain_C)

        # model fitting
        model_B = run_knn(X_train=sxtrain_B, Y_train=ytrain_B)
        model_C = run_knn(X_train=sxtrain_C, Y_train=ytrain_C)

        # save performances
        R2_B[k, 0] = r2_score(ytest_B, model_B.predict(sxtest_B))
        rmse_B[k, 0] = mean_squared_error(ytest_B, model_B.predict(sxtest_B), squared=False)
        R2_B[k, 1] = r2_score(ytest_A, model_B.predict(sxtest_A_t))
        rmse_B[k, 1] = mean_squared_error(ytest_A, model_B.predict(sxtest_A_t), squared=False)
        R2_C[k, 0] = r2_score(ytest_B, model_C.predict(sxtest_B))
        rmse_C[k, 0] = mean_squared_error(ytest_B, model_C.predict(sxtest_B), squared=False)
        R2_C[k, 1] = r2_score(ytest_A, model_C.predict(sxtest_A_t))
        rmse_C[k, 1] = mean_squared_error(ytest_A, model_C.predict(sxtest_A_t), squared=False)

        return R2_B, R2_C, rmse_B, rmse_C


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


# KNN with GridSearch and cv=3 + applying log transformation to target variable
def run_knn(X_train, Y_train):
    '''run KNN wit CV.

    return: trained model.
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
