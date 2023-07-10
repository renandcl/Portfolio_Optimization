import pandas as pd
import numpy as np
from typing import Tuple, Literal

from scipy import optimize
from scipy.optimize import OptimizeResult
import scipy.stats as stats
import pmdarima
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM


def interval_group_by_convertion(dataset, interval_type):
    """ Function to group the interval by the type of interval

        Parameters:
        -----------
            `dataset`: DataFrame
                DataFrame with datetime index to be grouped

            `interval_type`: str
                Type of interval to be grouped.
                #Valid intervals: [daily, week_start, week_end, month_start, month_end, quarter_start, quarter_end, half-year_start, half-year_end, year_start, year_end]"

        Returns:
        -----------
            `grouped_dataset`: DataFrame
                DataFrame with the grouped data by interval with a valid value on the interval

        Example:
        -----------
        >>> df = pd.DataFrame({'Value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}, index=pd.date_range('2022-01-01', periods=15, freq='D'))
                    Value
        DATE
        2022-01-01  1
        2022-01-02  2
        2022-01-03  3
        2022-01-04  4
        2022-01-05  5
        2022-01-06  6
        2022-01-07  7
        2022-01-08  8
        2022-01-09  9
        2022-01-10  10
        2022-01-11  11
        2022-01-12  12
        2022-01-13  13
        2022-01-14  14
        2022-01-15  15
        
        >>> interval_group_by_convertion(df, 'week_start')
                    Value
        DATE
        2022-01-01  1
        2022-01-02  2
        2022-01-09  9
    """

    # Copy values to avoid changing the DataFrame
    dataset_ = dataset.copy()

    # Set a Date record to track dates as a new column
    dataset_['Date_Record'] = dataset_.index

    # Define if last valid business day is on start on end of the interval
    def df_agg(df,interval_type_):
        if 'start' in interval_type_:
            return df.first()
        else:
            return df.last()

    # Define the interval and group by interval and year
    if interval_type == 'daily':
        grouped_dataset = dataset_
    elif 'week' in interval_type:
        grouped_dataset = df_agg(dataset_.groupby([dataset_.index.year,pd.DatetimeIndex(dataset_.index).isocalendar().week]),interval_type)
    elif 'month' in interval_type:
        grouped_dataset = df_agg(dataset_.groupby([dataset_.index.year,dataset_.index.month]),interval_type)
    elif 'quarter' in interval_type:
        grouped_dataset = df_agg(dataset_.groupby([dataset_.index.year,dataset_.index.quarter]),interval_type)
    elif 'half-year' in interval_type:
        grouped_dataset = df_agg(dataset_.groupby([dataset_.index.year,dataset_.index.quarter]),interval_type)[::2]
    elif 'year' in interval_type:
        grouped_dataset = df_agg(dataset_.groupby([dataset_.index.year]),interval_type)
    else:
        raise ValueError('Invalid interval type')

    # Retrieve the dates for each value
    grouped_dataset.rename(columns={'Date_Record':'DATE'},inplace=True)
    grouped_dataset.set_index('DATE', inplace= True)

    return grouped_dataset



def assets_calculations(Assets_Values: pd.DataFrame, calculation_method = 'mean', interval_type = 'daily', interval_range = 36, moving_average_type: Literal['SMA', 'EMA'] = None, moving_average_window: int = None, filtered = False, display_out_of_range_assets = False, **kwargs):
    """ Calculates the Assets Returns and Volatility, returning DataFrames of assets values on Range, Last Stock Close values, Assets Returns, Stock Risk and Assets covariance matrix.

        Parameters:
        -----------
            `Assets_Values`: DataFrame
                Dataframe with the Assets values

            `calculation_method`: str, optional
                Type of return to calculate. 
                #Valid types: ['mean', 'log_mean', 'CAPM','log_CAPM','auto_ARIMA', 'CC-ARMA-GARCH', 'log_CC-ARMA-GARCH', 'DCC-ARMA-GARCH', 'log_DCC-ARMA-GARCH']

            `interval_type`: str, optional
                Last valid business day on period. The default is 'daily'.
                #Valid intervals: ['daily', 'week_start', 'week_end', 'month_start', 'month_end', 'quarter_start', 'quarter_end', 'year_start', 'year_end']

            `interval_range`: int, optional 
                Time range to calculate the returns. The default is 24.

            `moving_average_type`: str, optional
                Type of moving average to calculate the returns. The default is None.
                #Valid types: ['SMA', 'EMA']

            `moving_average_window`: int, optional
                Window size to calculate the moving average. The default is inteval_range//4.

            `filtered`: bool, optional
                Interval_range already set in the Assets Data. The default is False.

            `display_out_of_range_Assets`: bool, optional
                Boolean to display the Assets that are out of range. The default is True.
        
        Displays:
        -----------
            `Removed_Assets`: list
                List with the Assets that are out of range

        Returns:
        -----------
            `Assets_Range_`: DataFrame
                Dataframe with the Assets values on range

            `Assets_Return`: DataFrame
                Dataframe with the Assets returns, mean value on time range

            `Assets_Risk`: DataFrame
                Dataframe with the Assets risk, standard deviation on time range
            
            `Assets_Covariance`: DataFrame
                Dataframe with the Assets covariance matrix considering the time range

        Example:
        -----------
            >>> Assets_Values
                        ABEV3  ALPA4  AMER3  ASAI3  ...  WEGE3  YDUQ3       IBOV      Selic
            DATE                                    ...                                     
            1996-06-26    NaN    NaN    NaN    NaN  ...    NaN    NaN    6227.15   1.000637 
            1996-06-27    NaN    NaN    NaN    NaN  ...    NaN    NaN    6181.59   1.001266 
            1996-06-28    NaN    NaN    NaN    NaN  ...    NaN    NaN    6043.89   1.001865 
            1996-07-01    NaN    NaN    NaN    NaN  ...    NaN    NaN    6156.19   1.002457 
            ...           ...    ...    ...    ...  ...    ...    ...        ...        ... 
            2022-03-04  13.93  24.71  30.09  12.71  ...  31.51  19.54  114473.78  10.010280 
            2022-03-07  13.43  22.27  27.01  12.28  ...  31.08  18.01  111593.46  10.013056 
            2022-03-08  13.44  22.96  26.93  12.30  ...  31.57  18.08  111203.45  10.015833 
            2022-03-09  13.93  24.45  27.51  13.40  ...  32.67  19.34  113900.34  10.018610 
            >>> Assets_Range_, Assets_Return, Assets_Risk, Assets_Covar = Assets_Calculations(Assets_Values, calculation_method = 'CAPM',interval_type='week_end',interval_range=20)
            >>> Assets_Range_
                        ABEV3  ALPA4  AMER3  ASAI3  ...  WEGE3  YDUQ3       IBOV      Selic
            DATE                                    ...                                     
            2021-10-29  16.99  38.63  29.70  15.29  ...  37.00  20.86  103500.71   9.808227 
            2021-11-05  17.90  40.81  34.95  15.24  ...  37.68  22.48  104824.23   9.816153 
            2021-11-12  17.51  41.79  37.40  15.53  ...  35.65  23.81  106334.54   9.826071 
            2021-11-19  17.39  43.05  33.57  14.08  ...  35.79  22.40  103035.02   9.834012 
            ...           ...    ...    ...    ...  ...    ...    ...        ...        ... 
            2022-02-18  14.77  26.40  33.72  12.70  ...  29.81  22.18  112879.85   9.988101 
            2022-02-25  15.20  25.73  30.50  13.46  ...  29.40  21.25  113141.94  10.001957 
            2022-03-04  13.93  24.71  30.09  12.71  ...  31.51  19.54  114473.78  10.010280 
            2022-03-09  13.93  24.45  27.51  13.40  ...  32.67  19.34  113900.34  10.018610 
            >>> Assets_Return
            ABEV3    0.003313
            ALPA4    0.003994
            AMER3    0.008736
            ASAI3    0.008587
                    ...   
            WEGE3    0.006806
            YDUQ3    0.010095
            IBOV     0.005218
            Selic    0.001121
            >>> Assets_Risk
            ABEV3    0.034962
            ALPA4    0.050574
            AMER3    0.091581
            ASAI3    0.058812
                    ...   
            WEGE3    0.050082
            YDUQ3    0.065758
            IBOV     0.018757
            Selic    0.000210
            >>> Assets_Covar
                    ABEV3     ALPA4         AMER3         ASAI3  ...     WEGE3         YDUQ3          IBOV         Selic    
            ABEV3  0.001222  0.001052  1.090290e-03  9.104891e-04  ...  0.000393  5.300665e-04  1.883792e-04  1.028199e-06     
            ALPA4  0.001052  0.002558  1.615984e-03  6.144936e-04  ...  0.001118  9.781883e-04  2.468074e-04 -3.631255e-06     
            AMER3  0.001090  0.001616  8.387118e-03  1.354543e-03  ...  0.000940  1.958638e-03  6.536455e-04 -3.829735e-07     
            ASAI3  0.000910  0.000614  1.354543e-03  3.458803e-03  ...  0.001621  1.497035e-03  6.408717e-04  8.060351e-07     
            ...         ...       ...           ...           ...  ...       ...           ...           ...           ...     
            WEGE3  0.000393  0.001118  9.395024e-04  1.620583e-03  ...  0.002508  9.270504e-04  4.880634e-04 -3.806338e-06     
            YDUQ3  0.000530  0.000978  1.958638e-03  1.497035e-03  ...  0.000927  4.324083e-03  7.702726e-04 -6.021454e-07     
            IBOV   0.000188  0.000247  6.536455e-04  6.408717e-04  ...  0.000488  7.702726e-04  3.518096e-04  3.045734e-07     
            Selic  0.000001 -0.000004 -3.829735e-07  8.060351e-07  ... -0.000004 -6.021454e-07  3.045734e-07  4.421833e-08     
        """

    # Copy values to avoid changing the original DataFrame
    Assets_Values_ = Assets_Values.copy()

    # Group the Assets values by interval
    Assets_Values_ = interval_group_by_convertion(Assets_Values_, interval_type)

    # if the dataframe with assets are already filterd with the interval_range, then the interval_range is the length of the dataframe
    if filtered:
        interval_range = len(Assets_Values_)

    # Remove Assets with range of values less than interval_range
    lower_than_range = ((len(Assets_Values_)-Assets_Values_.isna().sum())<interval_range)[((len(Assets_Values_)-Assets_Values_.isna().sum())<interval_range)].index.tolist()
    Assets_Values_ = Assets_Values_.drop(columns=lower_than_range)

    # Display out of range Assets
    if lower_than_range and display_out_of_range_assets:
        sharpe_logger.debug('Removed Assets:')
        sharpe_logger.debug(lower_than_range)
        print('Removed Assets:',lower_than_range)

    # Calculate the returns
    if moving_average_window == None:
        moving_average_window = interval_range//4

    if moving_average_type == 'SMA':
        # Calculate the simple moving average
        Assets_Values_ = Assets_Values_.rolling(moving_average_window).mean()
    elif moving_average_type == 'EMA':   
        # Calculate the exponential moving average
        Assets_Values_ = Assets_Values_.ewm(span=moving_average_window, adjust= False).mean()
    elif moving_average_type == None:
        pass
    else:
        raise ValueError('Invalid moving average type')

    # Calculate the Assets DataFrames
    Assets_Range_ = Assets_Values_.iloc[-interval_range:,:]
    
    # Calculate the Assets Returns
    if calculation_method == 'mean':
        # Calculate the mean value on time range
        Assets_Return_ = Assets_Range_.pct_change().mean()
        Assets_Risk_ = Assets_Range_.pct_change().std()
        Assets_Covar_ = Assets_Range_.pct_change().cov()
    
    elif calculation_method == 'log_mean':
        # Calculate the mean value on time range
        Assets_Return_ = np.log(Assets_Range_).diff().iloc[1:,:].mean()
        Assets_Risk_ = np.log(Assets_Range_).diff().iloc[1:,:].std()
        Assets_Covar_ = np.log(Assets_Range_).diff().iloc[1:,:].cov()

    elif calculation_method == 'CAPM':
        # Calculate the CAPM -> return = (Rf + Beta*(Rm - Rf))
        Assets_Covar_ = Assets_Range_.pct_change().cov()
        Risk_Free = Assets_Range_['Selic'].pct_change().mean()
        Market_Return = Assets_Range_['IBOV'].pct_change().mean()
        beta = Assets_Covar_['IBOV']/(Assets_Range_['IBOV'].pct_change().var())
        Assets_Return_ = Risk_Free + beta*(Market_Return-Risk_Free)

    elif calculation_method == 'log_CAPM':
        # Calculate the CAPM -> return = (Rf + Beta*(Rm - Rf))
        Assets_Covar_ = np.log(Assets_Range_).diff().iloc[1:,:].cov()
        Risk_Free = np.log(Assets_Range_['Selic']).diff()[1:].mean()
        Market_Return = np.log(Assets_Range_['IBOV']).diff()[1:].mean()
        beta = Assets_Covar_['IBOV']/(Assets_Range_['IBOV'].diff()[1:].var())
        Assets_Return_ = Risk_Free + beta*(Market_Return-Risk_Free)

    elif calculation_method == 'auto_ARIMA':
        # Calculate the ARIMA with ponderation of seasonality, trend and residuals
        try:
            Assets_Arima = Assets_Range_.apply(lambda x: pmdarima.auto_arima(x, start_p=1, start_q=1,
                        test='adf',       # use adftest to find optimal 'd'
                        max_p=20, max_q=20, # maximum p and q
                        max_d = 4,   # frequency of series
                        max_order= 60,
                        max_iter=100,       # maximum number of iterations
                        d=None,           # let model determine 'd'
                        seasonal=False,   # No Seasonality
                        start_P=0,
                        trace=False,
                        error_action='ignore',   # don't want to know if an order does not work
                        suppress_warnings=True,  # don't want convergence warnings
                        stepwise=True).predict(n_periods=1), axis=0) #predict 1 period in the future
            Assets_Return_ = ((Assets_Arima - Assets_Range_[-1:].reset_index(drop=True))/Assets_Range_[-1:].reset_index(drop=True)).loc[0]

        except Exception as err:
            try:
                sharpe_logger.error('ARIMA Error: {}'.format(err))
                Assets_Arima = Assets_Range_.apply(lambda x: pmdarima.ARIMA(order=(1,0,1),suppress_warnings=True, max_inter = 100).fit(x).predict(n_periods=1), axis=0)
                Assets_Return_ = ((Assets_Arima - Assets_Range_[-1:].reset_index(drop=True))/Assets_Range_[-1:].reset_index(drop=True)).loc[0]

            except Exception as err2:
                sharpe_logger.error('ARIMA Error: {}'.format(err2))
                Assets_Return_ = ((Assets_Range_[-2:-1].reset_index(drop=True) - Assets_Range_[-1:].reset_index(drop=True))/Assets_Range_[-1:].reset_index(drop=True)).loc[0]
        
        Assets_Risk_ = np.log(Assets_Range_).diff().iloc[1:,:].std()
        Assets_Covar_ = np.log(Assets_Range_).diff().iloc[1:,:].cov()

    elif calculation_method == 'naive-bayes':
        # Calculate the naive bayes
        raise NotImplementedError('Naïve Bayes not implemented yet')

    elif calculation_method == 'LSTM':
        # Calculate the LSTM with keras
        # train set with 80% of the data
        train_set = Assets_Range_.iloc[:int(len(Assets_Range_)*0.8),:]
        # test set with 20% of the data
        test_set = Assets_Range_.iloc[int(len(Assets_Range_)*0.8):,:]

        # train the model of lstm
        model = Sequential()
        model.add(LSTM(units=32, return_sequences=True, input_shape=(train_set.shape[1],1)))
        model.add(LSTM(units=32))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(train_set, test_set, epochs=100, batch_size=5, verbose=2)

        # predict the future values
        Assets_Return_ = model.predict(Assets_Range_.iloc[-1:,:])
        # raise NotImplementedError('LSTM method not implemented yet')
        Assets_Risk_ = np.log(Assets_Range_).diff().iloc[1:,:].std()
        Assets_Covar_ = np.log(Assets_Range_).diff().iloc[1:,:].cov()

    elif calculation_method == 'VaR':
        # Calculate the VaR (Value-at-Risk)
        # Assets_Return_ = Assets_Range_.pct_change().mean()        
        # Assets_VaR_ = Assets_Range_.pct_change().quantile(q=VaR_percentage)
        raise NotImplementedError('VaR method not implemented yet')

    elif calculation_method == 'CVaR':
        # Calculate the CVaR (Conditional Value-at-Risk)
        # Assets_Return_ = Assets_Range_.pct_change().mean()
        # Assets_VaR_ = Assets_Range_.pct_change().quantile(q=VaR_percentage)
        raise NotImplementedError('CVaR method not implemented yet')
    
    elif 'DCC-ARMA-GARCH' in calculation_method:
        # Calculate the GARCH (GARCH-(1,1))
        Assets_Range_, Assets_Return_, Assets_Risk_, Assets_Covar_ = dcc_garch(Assets_Range_, calculation_method)

    elif 'CC-ARMA-GARCH' in calculation_method:
        # Calculate the GARCH (GARCH-(1,1))
        Assets_Range_, Assets_Return_, Assets_Risk_, Assets_Covar_ = cc_garch(Assets_Range_, calculation_method)

    else:
        raise ValueError('Invalid calculation_method')

    return Assets_Range_, Assets_Return_, Assets_Risk_, Assets_Covar_


from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri, vectors
pandas2ri.activate()

base = importr('base')
rugarch = importr('rugarch')
rmgarch = importr('rmgarch')
rlist = r['list']
c = r['c']


def dcc_garch(assets_range_,calculation_method):
    """ DCC-GARCH model(1,1) """

    if 'log' in calculation_method:
        returns = np.log(assets_range_).diff().iloc[1:,:]
    else:
        returns = assets_range_.pct_change().iloc[1:,:]

    r_df = pandas2ri.py2rpy_pandasdataframe(returns)

    garch_spec = rugarch.ugarchspec(
        mean_model = rlist(armaOrder=c(1,1)),
        variance_model = rlist(garchOrder=c(1,1),model='sGARCH'),
        distribution_model = 'norm'
    )

    params_conf = garch_spec.do_slot('model').rx('modelinc')[0]
    u_fit = {}
    u_params = {}
    all_params = ['mu','ar1','ma1','arfima','archm','mxreg','omega','alpha1','beta1','gamma1','eta1','eta2','delta','lambda','vxreg','skew','shape','ghlambda','xi','aux1','aux2','aux3']

    notfitted = []

    for component in returns.columns:
        included_params = []
        try:
            u_fit[component] = rugarch.ugarchfit(
                spec = garch_spec,
                data = r_df.rx(component)
                )
            
            spec_params = dict(zip(all_params,params_conf))

            for key, value in spec_params.items():
                if 'aux' not in key:
                    if value:
                        included_params.append(key)

            params_values = r('coef')(u_fit[component])
            tmp_params = dict(zip(included_params, params_values))
            u_params[component] = tmp_params
            
        except:
            # print('removed: ' + component)
            notfitted.append(component)
            pass

    if notfitted:
        print('removed: ')
        print(notfitted)
        returns.drop(notfitted, axis=1, inplace=True)
        r_df = pandas2ri.py2rpy_pandasdataframe(returns)

    u_spec_start = {}

    for component in returns.columns:
        
        u_spec_start[component] = rugarch.ugarchspec(
            mean_model = rlist(armaOrder=c(1,1)),
            variance_model = rlist(garchOrder=c(1,1),model='sGARCH'),
            distribution_model = 'norm',
            start_pars = vectors.ListVector(u_params[component].items())
        )

    u_multispec_w_start_pars = rugarch.multispec(
        speclist = c([u_spec_start[component] for component in u_spec_start.keys()])
    )

    u_multifit_w_start_pars = rugarch.multifit(
        multispec = u_multispec_w_start_pars,
        data = r_df
    )

    multispec = rugarch.multispec(
        base.replicate(len(returns.columns),
        garch_spec)
    )

    dccmodelspec = rmgarch.dccspec(
        uspec=multispec,
        dccOrder=c(1,1),
        model="DCC",
        distribution="mvnorm"
        )

    dccmodelfit = rmgarch.dccfit(
        spec = dccmodelspec,
        data = r_df,
        fit = u_multifit_w_start_pars
        )

    dcc_forecast = rmgarch.dccforecast(dccmodelfit,n_ahead=1)

    covariance_array = rmgarch.rcov(dcc_forecast)[0].reshape(len(returns.columns),len(returns.columns))
    mean_array = dcc_forecast.slots['mforecast'].rx['mu'][0][0].reshape(len(returns.columns))
    risk_array = r('sigma')(dcc_forecast)[0]

    cov = pd.DataFrame(index = returns.columns, columns = returns.columns, data = covariance_array)
    mean = pd.DataFrame(index = returns.columns, columns = ['mean'], data = mean_array)
    risk = pd.DataFrame(index = returns.columns, columns = ['risk'], data = risk_array)

    return assets_range_.loc[:,~assets_range_.columns.isin(notfitted)], mean.squeeze(), risk.squeeze(), cov


def cc_garch(assets_range_, calculation_method):
    """ Constant Correlation with ARMA(1,1)-GARCH model(1,1) """

    if 'log' in calculation_method:
        returns = np.log(assets_range_).diff().iloc[1:,:]
    else:
        returns = assets_range_.pct_change().iloc[1:,:]

    r_df = pandas2ri.py2rpy_pandasdataframe(returns)

    garch_spec = rugarch.ugarchspec(
        mean_model = rlist(armaOrder=c(1,1)),
        variance_model = rlist(garchOrder=c(1,1),model='sGARCH'),
        distribution_model = 'norm'
    )

    params_conf = garch_spec.do_slot('model').rx('modelinc')[0]
    u_fit = {}
    u_params = {}
    all_params = ['mu','ar1','ma1','arfima','archm','mxreg','omega','alpha1','beta1','gamma1','eta1','eta2','delta','lambda','vxreg','skew','shape','ghlambda','xi','aux1','aux2','aux3']

    notfitted = []

    for component in returns.columns:
        included_params = []
        try:
            u_fit[component] = rugarch.ugarchfit(
                spec = garch_spec,
                data = r_df.rx(component)
                )
            
            spec_params = dict(zip(all_params,params_conf))

            for key, value in spec_params.items():
                if 'aux' not in key:
                    if value:
                        included_params.append(key)

            params_values = r('coef')(u_fit[component])
            tmp_params = dict(zip(included_params, params_values))
            u_params[component] = tmp_params
            
        except:
            # print('removed: ' + component)
            notfitted.append(component)
            pass

    if notfitted:
        print('removed: ')
        print(notfitted)
        returns.drop(notfitted, axis=1, inplace=True)
        r_df = pandas2ri.py2rpy_pandasdataframe(returns)

    u_spec_start = {}

    for component in returns.columns:
        
        u_spec_start[component] = rugarch.ugarchspec(
            mean_model = rlist(armaOrder=c(1,1)),
            variance_model = rlist(garchOrder=c(1,1),model='sGARCH'),
            distribution_model = 'norm',
            start_pars = vectors.ListVector(u_params[component].items())
        )

    u_multispec_w_start_pars = rugarch.multispec(
        speclist = c([u_spec_start[component] for component in u_spec_start.keys()])
    )

    u_multifit_w_start_pars = rugarch.multifit(
        multispec = u_multispec_w_start_pars,
        data = r_df
    )

    cc_forecast = rugarch.multiforecast(u_multifit_w_start_pars, n_ahead=1)

    mean_array = r('fitted')(cc_forecast)[0].reshape(len(returns.columns))
    risk_array = r('sigma')(cc_forecast)[0].reshape(len(returns.columns))

    mean = pd.Series(index = returns.columns, name = 'mean', data = mean_array)
    risk = pd.Series(index = returns.columns, name = 'risk', data = risk_array)
    corr = returns.corr()
    cov = corr.mul(risk, axis=0).mul(risk.T, axis=1)

    return assets_range_.loc[:,~assets_range_.columns.isin(notfitted)], mean, risk, cov


    import scipy.stats as st


