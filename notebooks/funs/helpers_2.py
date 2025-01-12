import logging
import pandas as pd
from scipy.stats import f_oneway, kruskal
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from category_encoders import cat_boost

logger = logging.getLogger(__name__)

def _outliers_zscore(array):
    """
    Calculate the number of outliers in an array using the Z-score method.
    """
    z = np.abs((array - np.mean(array)) / np.std(array))
    return np.sum(z > 3)

def _outliers_mean(array):
    """
    Calculate the mean of the outliers relative to overall mean.
    """
    q1 = np.quantile(array, 0.25)
    q3 = np.quantile(array, 0.75)
    iqr = q3 - q1
    q_lower = q1 - 1.5 * iqr
    q_upper = q3 + 1.5 * iqr
    return array[array < q_lower].mean()/array.mean(),array[array > q_upper].mean()/array.mean()

def _outliers_iqr(array):
    """
    Calculate the number of outliers in an array using the Interquartile Range (IQR) method.
    """
    q1 = np.quantile(array, 0.25)
    q3 = np.quantile(array, 0.75)
    iqr = q3 - q1
    q_lower = q1 - 1.5 * iqr
    q_upper = q3 + 1.5 * iqr
    return np.sum((array < q_lower) | (array > q_upper))

def check_outliers(df):
    """
    Check the outliers in the DataFrame.
    """
    for column in df.columns:
        if df[column].dtype in ['float','int']:
            z_outliers = _outliers_zscore(df[column])
            iqr_outliers = _outliers_iqr(df[column])
            if z_outliers > 0 or iqr_outliers > 0:
                logger.info("Column: %s, Z-score outliers: %d, IQR outliers: %d",
                            column, z_outliers, iqr_outliers)
                mean_outliers = _outliers_mean(df[column])
                logger.info("Outliers mean relative to overall mean: %s, %s",
                            round(mean_outliers[0],3),round(mean_outliers[1],3))
    return None

def _winsorize(array, q_lower, q_upper):
    array = np.where(array < q_lower, q_lower, array)
    array = np.where(array > q_upper, q_upper, array)
    return array

def _normalize(array, _mean, _std):
    return (array - _mean) / _std

def _standardize(array, _min, _max):
    return (array - _min) / (_max - _min)

def _onehot_encode(train_df,test_df,col):
    enc = OneHotEncoder(dtype=int)

    train_d = enc.fit_transform(train_df[[col]]).toarray()
    oh_train = pd.DataFrame(train_d, columns=enc.get_feature_names_out())
    train_df.drop(columns = col, inplace=True)
    train_df = pd.concat([train_df, oh_train], axis=1)

    if test_df:
        test_d = enc.transform(test_df[[col]]).toarray()
        oh_test = pd.DataFrame(test_d, columns=enc.get_feature_names_out())
        test_df.drop(columns = col, inplace=True)
        test_df = pd.concat([test_df, oh_test], axis=1)
        return train_df, test_df
    else:
        return train_df

def _cb_encode(train_df,test_df,col,params):
    enc = cat_boost.CatBoostEncoder()
    train_df[col] = enc.fit_transform(train_df[col], train_df[params['target'][0]].cat.codes)
    if test_df:
        test_df[col] = enc.transform(test_df[col])
        return train_df, test_df
    return train_df

def _ordinal_encode(train_df,test_df,col,map_dict):
    mapping = {x:i for i, x in enumerate(map_dict[col])}
    missing_val = [x for x in train_df[col].unique() if x not in mapping.keys()]
    if missing_val:
        logger.warning("Missing ordinal mapping in %s", col)
        logger.warning("Missing order for value %s", missing_val)
        mapping.update({x:np.nan for x in missing_val})

    if test_df:
        test_df[col] = test_df[col].map(mapping).astype('int')
        return train_df, test_df
    else:
        train_df[col] = train_df[col].map(mapping).astype('int')
        return train_df

def process_cat(train_df, test_df, params, params_ordinal):
    """
    Processes categorical features in the test and train dataframes 
    based on specified parameters.
    """
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    for column in [x for x in train_df.columns if train_df[x].dtype in ['category']]:
        if column in params['onehot']:
            train_df, test_df = _onehot_encode(train_df, test_df, column)
        elif column in params['catboost']:
            train_df, test_df = _cb_encode(train_df, test_df, column, params)
        elif column in params['ordinal']:
            train_df, test_df = _ordinal_encode(train_df, test_df, column, params_ordinal)
        else:
            if column not in params['target']:
                logger.warning("Categorical olumn %s not in encoding list", column)
    return train_df, test_df

def process_num(train_df, test_df, params):
    """
    Processes numerical features in the test and train dataframes 
    based on specified parameters.
    """
    for column in [x for x in train_df.columns if train_df[x].dtype in ['float','int']]:
        if column in params['winsorize']:
            q_lower = np.quantile(train_df[column], 0.05)
            q_upper = np.quantile(train_df[column], 0.95)
            train_df[column] = _winsorize(train_df[column], q_lower, q_upper)
            test_df[column] = _winsorize(train_df[column], q_lower, q_upper)
        elif column in params['normalize']:
            _mean = train_df[column].mean()
            _std = train_df[column].std()
            train_df[column] = _normalize(train_df[column], _mean, _std)
            test_df[column] = _normalize(test_df[column], _mean, _std)
        elif column in params['standardize']:
            _min = train_df[column].min()
            _max = train_df[column].max()
            train_df[column] = _standardize(train_df[column], _min, _max)
            test_df[column] = _standardize(test_df[column], _min, _max)
        elif column in params['log']:
            train_df[column] = np.log(train_df[column])
            test_df[column] = np.log(test_df[column])
        elif column in params['none']:
            1==1
        else:
            logger.warning("Numerical column %s not in encoding list", column)

    return train_df, test_df

def drop_columns(test_df, train_df, columns):
    """
    Drops the specified columns from the test and train data.
    """
    if train_df:
        return test_df.drop(columns=columns), train_df.drop(columns=columns)
    else:
        return test_df.drop(columns=columns)

def get_feat_rank_stat(train, params):
    """
    Calculate feature ranking statistics using ANOVA and Kruskal-Wallis tests.
    """

    p_an_tests = []
    p_kru_tests = []
    target_col = params['target']
    columns = [x for x in train.columns if x != target_col]

    for col in columns:
        unique_classes = train[target_col].unique()
        group_data = [train[train[target_col] == cls][col] for cls in unique_classes]
        p_kru = kruskal(*group_data).pvalue
        p_anova = f_oneway(*group_data).pvalue
        p_kru_tests.append(float(p_kru))
        p_an_tests.append(float(p_anova))

    return pd.DataFrame({
        "variable": columns,
        "p_kruskal": p_kru_tests,
        "p_anova": p_an_tests,
        'kruskal_rank': pd.Series(p_kru_tests).rank().astype(int).to_list(),
        "anova_rank":  pd.Series(p_an_tests).rank().astype(int).to_list(),
    }).sort_values('kruskal_rank')

def get_feat_rank_corr(train, params):
    """
    Calculate the correlation of each feature with the target variable and rank them.
    """

    target_col = params['target']
    if train[target_col].dtypes=='category':
        train[target_col] = train[target_col].cat.codes
    else:
        train[target_col] = train[target_col]
    columns = [x for x in train.columns if x != target_col]

    corr = train[columns].corrwith(train[target_col])

    return pd.DataFrame({
        "variable": columns,
        "corr": corr,
        "corr_rank": corr.abs().rank(ascending=False).astype(int).to_list()
    }).sort_values('corr_rank')

def _get_feat_sfs(train, params):
    """
    Perform forward feature selection using multiple classifiers.
    """

    target_col = params['target']
    columns = [x for x in train.columns if x != target_col]

    if params['n_vars'] == 'all':
        n_vars = len(5)
    else:
        n_vars = params['n_vars']
    results = {}
    models = {
        'SVM':SVR(),
        'GBM':GradientBoostingRegressor(),
        'LinReg':LinearRegression()
    }

    sfs = SequentialFeatureSelector(models.get(params['kwarg']),
                                    n_features_to_select=n_vars,
                                    direction=params['method'])
    sfs.fit(train[columns], train[target_col])
    results[params['kwarg']] = sfs.get_feature_names_out()
    return pd.DataFrame(results)

def _get_selected_features(train,params,stat_res=None,corr_res=None):
    """
    Select the features based on the feature selection method.
    """

    method = params['method']
    n_vars = params['n_vars']
    kwarg = params['kwarg']
    _vars = None

    if method == 'rank_stat' and stat_res is not None:
        _vars =  stat_res.sort_values(f'{kwarg}_rank')['variable'].values[:n_vars].tolist()
    elif method == 'rank_corr' and corr_res is not None:
        _vars = corr_res.sort_values('corr_rank')['variable'].values[:n_vars].tolist()
    elif method in ['forward','backward']:
        _vars = _get_feat_sfs(train, params)[kwarg].values.tolist()
    if not _vars:
        logger.error("Invalid method name")
    else:
        logger.info("Selecting %d features based on %s%s method", n_vars, method, kwarg)
        return _vars


def select_features(train, tests, params,stat_res=None,corr_res=None):
    """
    Select the features in the train dataframe.
    """
    n_vars = params['n_vars']
    target_col = params['target']
    if n_vars =='all':
        selected_features = [x for x in train.columns if x != target_col]
        criteria = 'all'
    else:
        selected_features = _get_selected_features(train,params,stat_res,corr_res)
        criteria = {
        'method':params['method'],
        'n_vars':params['n_vars'],
        'kwarg':params['kwarg']
        }
    metrics = {'columns':selected_features, 'criteria':criteria}

    return train[selected_features+[target_col]], tests[selected_features+[target_col]], metrics
