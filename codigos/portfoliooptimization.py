import pandas as pd
import numpy as np
from typing import Tuple, Literal

from scipy import optimize
from scipy.optimize import OptimizeResult
import scipy.stats as stats
from gekko import GEKKO
from amplpy import AMPL


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


# import jpylogger

# sharpe_logger = jpylogger.JupyterLogger('Sharpe')



def sharpe_optimization(Initial_Weigths: np.ndarray, Assets_Return_: np.ndarray, Assets_Covar_: np.ndarray, Risk_Free_Return_: float, display_message = False) -> OptimizeResult:
    """ Function to optmize the Sharpe Ratio and return the optimal weights of the Assets.

        Parameters:
        -----------
            `Initial_Weigths`: ndarray
                Initial Weights of the Assets percentage on the portfolio

            `Assets_Return_`: ndarray
                Dataframe with the Assets Return values

            `Assets_Covar_`: ndarray
                Dataframe with the Assets Covariance matrix

            `Risk_Free_Return_`: float
                Risk free return rate

            `display_message`: bool, optional 
                Boolean to display optmization result. The default is False.

        Displays:
        -----------
            `Optimization Details`: 
                Display maximum Sharpe Ratio and the optimization iteration details

        Returns:
        -----------
            `Optimization_Result`: optimize.OptimizeResult
                Result obtainded from the optimization of Sharpe Ratio

        Example:
        -----------
            >>> Assets_Return_
            ABEV3    0.003313
            ALPA4    0.003994
            AMER3    0.008736
            ASAI3    0.008587
                    ...   
            WEGE3    0.006806
            YDUQ3    0.010095
            IBOV     0.005218
            Selic    0.001121
            >>> Assets_Covar_
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
            >>> Risk_Free_Interest_Rate
            0.0002773038676628925
            >>> Minmize_Result_dict = sharpe_optimization(np.array([]),Assets_Return_,Assets_Covar_,Risk_Free_Interest_Rate, display_message = True)
            Optimization terminated successfully    (Exit mode 0)
                        Current function value: -1.132918791438277
                        Iterations: 23
                        Function evaluations: 2135
                        Gradient evaluations: 23
            >>> Optimize_Result_dict.keys()
            dict_keys(['x', 'fun', 'jac', 'nit', 'nfev', 'njev', 'status', 'message', 'success'])
            >>> Optimize_Result_dict['nit'] # Number of iterations
            25
            >>> Optimize_Result_dict['message'] # Optmization Message
            Optimization terminated successfully
            >>> Optimize_Result_dict['success']
            True
                
        """
    
    # assert variable type
    assert isinstance(Initial_Weigths, np.ndarray), "Initial Weights must be a numpy array"
    assert isinstance(Assets_Return_, np.ndarray), "Assets Return must be a numpy array"
    assert isinstance(Assets_Covar_, np.ndarray), "Assets Covariance must be a numpy array"
    assert isinstance(Risk_Free_Return_, float), "Risk Free Return must be a float"
    assert isinstance(display_message, bool), "Display Message must be a boolean"
    

    # Defining the Sharpe Ratio function to iterate with independed variables and return the negative value
    def fun(x):
        Assets_Percentages = x
        Portfolio_Return = Assets_Percentages.dot(Assets_Return_)
        Portfolio_Risk = np.sqrt(Assets_Percentages.dot(Assets_Covar_.dot(Assets_Percentages.transpose())))
        Sharpe = (Portfolio_Return - Risk_Free_Return_)/Portfolio_Risk
        return -Sharpe
    
    # Defining the independent variables, boundaries, and constraints
    x0 = Initial_Weigths
    if len(x0) == 0:
        x0 = np.random.dirichlet(np.ones(len(Assets_Return_)))
    bnds = tuple([(0,1) for _ in range(len(Assets_Return_))])
    cons = {'type': 'eq', 'fun': lambda x: -np.sum(x)+1}

    # Optimizing the Sharpe Ratio with Sequential Least Squares Programming
    Optimization_Result = optimize.minimize(fun, x0, method='SLSQP', bounds=bnds, constraints=cons, options={'disp':display_message})

    return Optimization_Result


def sharpe_optimization_full(Initial_Weigths, Assets_Return_, Assets_Covar_, Risk_Free_Return_, display_message = False) -> OptimizeResult:
    """ Function to optmize the Sharpe Ratio and return the optimal weights of the Assets.

        Parameters:
        -----------
            `Initial_Weigths`: ndarray
                Initial Weights of the Assets percentage on the portfolio

            `Assets_Return_`: ndarray
                Dataframe with the Assets Return values

            `Assets_Covar_`: ndarray
                Dataframe with the Assets Covariance matrix

            `Risk_Free_Return_`: float
                Risk free return rate

            `display_message`: bool, optional 
                Boolean to display optmization result. The default is False.

        Displays:
        -----------
            `Optimization Details`: 
                Display maximum Sharpe Ratio and the optimization iteration details

        Returns:
        -----------
            `Assets_list`: Series
                Series with Assets and the weights from the optimization of Sharpe Ratio

            `Sharpe_Ratio`: float
                Maximum Sharpe Ratio

            `Sharpe optimization plotting data`: tuple
                Tuple with the plotting data for the optimization of Sharpe Ratio with X's, Y's values and the optimal X and Y 

            `Minimize_Result`: optimize.OptimizeResult
                Result obtainded from the optimization of Sharpe Ratio

        Example:
        -----------
            >>> Assets_Return_
            ABEV3    0.003313
            ALPA4    0.003994
            AMER3    0.008736
            ASAI3    0.008587
                    ...   
            WEGE3    0.006806
            YDUQ3    0.010095
            IBOV     0.005218
            Selic    0.001121
            >>> Assets_Covar_
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
            >>> Risk_Free_Interest_Rate
            0.0002773038676628925
            >>> Assets_percentage, sharpe_value, plot_data, Minmize_Result_dict = Sharp_minimize(np.array([]),Assets_Return_,Assets_Covar_,Risk_Free_Interest_Rate)
            Optimization terminated successfully    (Exit mode 0)
                        Current function value: -1.132918791438277
                        Iterations: 23
                        Function evaluations: 2135
                        Gradient evaluations: 23
            >>> Assets_percentage
            BRAP4     0.250
            CIEL3     0.071
            CPFE3     0.173
            ENBR3     0.053
                    ...  
            SULA11    0.040
            SUZB3     0.039
            TOTS3     0.020
            VIVT3     0.020
            Length: 15, dtype: float64
            >>> sharpe_value
            1.132918791438277
            >>> plot_data[0]
            >>> plot_data[1]
            >>> plot_data[2]
            >>> plot_data[3]
            [0.01638687091104584, 0.016386871095913257, 0.01638687135333397, ..., 0.004761154792584894, 0.004754723514319205, 0.004759055836073518]
            [-0.00046606206830981044, -0.000466062092138692, -0.00046606210936729663, ..., 0.005671302160525368, 0.005664005551252644, 0.005668927653854581]
            0.004759055836073518
            0.005668927653854581
            >>> Optimize_Result_dict.keys()
            >>> Optimize_Result_dict['nit'] # Number of iterations
            >>> Optimize_Result_dict['message'] # Optmization Message
            >>> Optimize_Result_dict['success']
            dict_keys(['x', 'fun', 'jac', 'nit', 'nfev', 'njev', 'status', 'message', 'success'])
            25
            Optimization terminated successfully
            True
                
        """
    
    # Initialize iteration results list
    iteration_results=[]

    # Defining the Sharpe Ratio function to iterate with independed variables and return the negative value
    def fun(x):
        Assets_Percentages = x
        Portfolio_Return = Assets_Percentages.dot(Assets_Return_)
        Portfolio_Risk = np.sqrt(Assets_Percentages.dot(Assets_Covar_.dot(Assets_Percentages.transpose())))
        Sharpe = (Portfolio_Return - Risk_Free_Return_)/Portfolio_Risk
        iteration_results.append([Sharpe,x,Portfolio_Risk,Portfolio_Return])
        return -Sharpe

    def jac(x):
        Assets_Return__ = Assets_Return_.to_numpy()
        Assets_Covar__ = Assets_Covar_.to_numpy()
        Assets_Percentages_ = x
        Portfolio_Return = Assets_Percentages_.dot(Assets_Return__)
        Portfolio_Risk = np.sqrt(Assets_Percentages_.dot(Assets_Covar__.dot(Assets_Percentages_.transpose())))
        def partial_dev(return_i,covar_i):
            return (return_i*Portfolio_Risk**2-(Portfolio_Return-Risk_Free_Return_)*(Assets_Percentages_.dot(covar_i)))/Portfolio_Risk**(3)
        Partial_Jac = lambda i: np.array([partial_dev(Assets_Return__[i],Assets_Covar__[i,:])])
        jac_array = Partial_Jac(np.arange(len(Assets_Percentages_)))[0]
        return jac_array
    
    # Defining the independent variables, boundaries, and constraints
    x0 = Initial_Weigths
    if len(x0) == 0:
        x0 = np.random.dirichlet(np.ones(len(Assets_Return_)))
        # x0 = np.array([1/len(Assets_Return_) for _ in range(len(Assets_Return_))])
    bnds = tuple([(0,1) for _ in range(len(Assets_Return_))])
    # cons = {'type': 'eq', 'fun': lambda x: -np.sum(x)+1, 'jac': lambda x: np.power(x,0)}
    cons = {'type': 'eq', 'fun': lambda x: -np.sum(x)+1}

    rel_step = lambda x: (np.finfo(x.dtype).eps ** (1/2)) * ((x >= 0).astype(float) * 2 - 1 ) * np.abs(x) #np.maximum(1,np.abs(x))
    rel_step_array = lambda i: np.array(rel_step(i))

    options_args = {
        'disp': display_message, 
        'maxiter': 1000, 
        'ftol': 1e-6, 
        'eps': np.finfo(float).eps ** (1/2), 
        'iprint': 1, 
        'finite_diff_rel_step': rel_step_array
        }

    # Optimizing the Sharpe Ratio with Sequential Least Squares Programming
    # Optimization_Result = optimize.minimize(fun, x0, method='SLSQP', bounds=bnds, constraints=cons, jac = jac, options = options_args)
    Optimization_Result = optimize.minimize(fun, x0, method='SLSQP', bounds=bnds, constraints=cons, options={'disp':display_message})
    
    # Getting the results for the plotting data
    Portfolios_Sharpe = []
    Portfolios_Risk = []
    Portfolios_Return = []
    final_sharpe_result = -np.inf
    Final_Weights = np.around(Optimization_Result['x'],decimals=3)

    for iteration_result in iteration_results:
        sharpe = iteration_result[0]
        Portfolios_Sharpe.append(sharpe)
        Portfolios_Risk.append(iteration_result[2])
        Portfolios_Return.append(iteration_result[3])
        if sharpe < final_sharpe_result:
            continue
        else:
            px = iteration_result[2]
            py = iteration_result[3]
            pz = iteration_result[0]

    # Displaying the Assets and weights
    list_assets = {}
    assets_result = dict(zip(Assets_Return_.index.to_list(),Final_Weights))
    for assets,value in assets_result.items():
        if value > 0:
            list_assets[assets] = value
            # print(f'{stock} - {value}')
    
    return pd.Series(data=list_assets), (Portfolios_Risk,Portfolios_Return,Portfolios_Sharpe,px,py,pz), Optimization_Result


def Sharp_GEKKO(Stock_Close,Stocks_Return,Stocks_Covar,Risk_Free_Return,Total_Available_Money):
    """	Calculate Sharpe Ratio with GEKKO for discrete optimization

        Parameters:
        ----------
            `Stock_Close`: DataFrame
                DataFrame with stock close values

            `Stocks_Return`: DataFrame
                DataFrame with stocks return values

            `Stocks_Covar`: DataFrame
                DataFrame with stocks covariance values

            `Risk_Free_Return`: float
                Risk free return value

            `Total_Available_Money`: float 
                Total available money to invest

        Returns:
        ----------
            `Gekko_Model`: dict
                GEKKO model optimization result
        
        Example:
        ----------
            >>> Assets_Return_
            ABEV3    0.003313
            ALPA4    0.003994
            AMER3    0.008736
            ASAI3    0.008587
                    ...   
            WEGE3    0.006806
            YDUQ3    0.010095
            IBOV     0.005218
            Selic    0.001121
            >>> Assets_Covar_
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
            >>> Risk_Free_Interest_Rate
            0.0002773038676628925

    """

    # Initialize GEKKO model
    model = GEKKO(remote=False)
    
    # Define variables with upper and lower bounds based on total available money
    x=model.Array(model.Var,(len(Stock_Close)),integer=True)
    for stock, close_Value,i in zip(Stock_Close.index.tolist(),Stock_Close,range(len(Stock_Close))):
        x[i]=model.Var(1,lb=0,ub=(Total_Available_Money//close_Value),integer=True,name=stock)
    
    # Intermediate calculation for total invested money
    total_invested = model.Intermediate(model.sum(x*Stock_Close.to_numpy()))

    # Intermediate calculation for portfolio return
    Portfolio_Return = model.Intermediate((x*Stock_Close.to_numpy()/total_invested).dot(Stocks_Return.to_numpy()))

    # Intermediate calculation for portfolio risk with breakdown of risk calculations
    risk_part2ij = 0
    risk_part2 = np.empty((len(Stock_Close),len(Stock_Close)),dtype=object)
    for i in range(len(Stock_Close)):
        for j in range(len(Stock_Close)):
            risk_part2[i,j] = model.Intermediate((x[i]*Stock_Close.iloc[i]/total_invested)*(x[j]*Stock_Close.iloc[j]/total_invested)*Stocks_Covar.iloc[i,j])

            risk_part2ij = risk_part2ij + risk_part2[i,j]
    Portfolio_Risk = model.Intermediate(model.sqrt(risk_part2ij))
    
    # Intermediate calculation for portfolio risk
    # Portfolio_Risk = model.Intermediate(model.sqrt((x*Stock_Close.to_numpy()/total_invested).dot(Stocks_Covar.to_numpy().dot((x*Stock_Close.to_numpy()/total_invested).transpose()))))

    # Define constraints for total invested money
    model.Equation(total_invested<=Total_Available_Money)
    
    # Define objective function
    sharp = model.Intermediate((Portfolio_Return - Risk_Free_Return)/Portfolio_Risk)

    # Set maximization objective
    model.Maximize(sharp)
    
    # Solve model APOPT (Interior Point and Active Set Methods)
    model.options.SOLVER=1
    model.solve(disp=False,debug=False)
    
    return model


def utility_sharpe_GEKKO(Stock_Close,Stocks_Return,Stocks_Covar,Risk_Free_Return,Total_Available_Money, VaR, confidence_level=.95, infos=None):
    """	Calculate Sharpe Ratio with GEKKO for discrete optimization

        Parameters:
        ----------
            `Stock_Close`: DataFrame
                DataFrame with stock close values

            `Stocks_Return`: DataFrame
                DataFrame with stocks return values

            `Stocks_Covar`: DataFrame
                DataFrame with stocks covariance values

            `Risk_Free_Return`: float
                Risk free return value

            `Total_Available_Money`: float 
                Total available money to invest

            `VaR`: float
                Money Value at Risk

            `confidence_level`: float
                Confidence level

            `upper_lot_boundary`: Series
                Upper boundary for lots

        Returns:
        ----------
            `Gekko_Model`: dict
                GEKKO model optimization result
        
        Example:
        ----------
            >>> Assets_Return_
            ABEV3    0.003313
            ALPA4    0.003994
            AMER3    0.008736
            ASAI3    0.008587
                    ...   
            WEGE3    0.006806
            YDUQ3    0.010095
            IBOV     0.005218
            Selic    0.001121
            >>> Assets_Covar_
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
            >>> Risk_Free_Interest_Rate
            0.0002773038676628925

    """

    # Initialize GEKKO model
    model = GEKKO(remote=False)

    # define upper lot boundary when None, based on total available money
    if infos is None:
        infos = pd.DataFrame(index=Stock_Close.columns, columns=['lot_size'])
        infos['lower_lot_boundary'] = 0
        infos['upper_lot_boundary'] = Total_Available_Money//Stock_Close
        infos['fees'] = 0
        infos['b3_taxes'] = 0
        infos['income_taxes'] = 0

    bond_tax = .225
    infos['lower_lot_boundary'] = 0
    lower_lot_boundary = infos.lower_lot_boundary[Stock_Close.index]
    upper_lot_boundary = np.minimum(Total_Available_Money//Stock_Close,infos.upper_lot_boundary.loc[Stock_Close.index])
    fees = infos.brokerage_fee.loc[Stock_Close.index]
    b3_taxes = infos.stock_exchange_taxes.loc[Stock_Close.index]
    income_taxes = infos.income_taxes.loc[Stock_Close.index]
    
    # Safety factor for initial lot size
    ss = 1

    # Random starting point for lot sizes
    start = pd.Series(data = (np.random.dirichlet(np.ones(len(Stock_Close)))*(Total_Available_Money*ss)).astype(int), index = Stock_Close.index)
        
    # Define integer variables with upper and lower bounds 
    x=model.Array(model.Var,(len(Stock_Close)),integer=True)
    for i,close_Value in enumerate(Stock_Close):
        x[i]=model.Var(start[i] ,lb=0,ub=np.min([Total_Available_Money//close_Value,upper_lot_boundary[i]]),integer=True,name=Stock_Close.index[i])
        
    # Intermediate calculation for total invested money
    total_variable_invested = model.Intermediate(model.sum(x*Stock_Close.to_numpy()))

    # Intermediate calculation for portfolio return
    Portfolio_Return = model.Intermediate((x*Stock_Close.to_numpy()/total_variable_invested).dot(Stocks_Return.to_numpy()))

    # Intermediate calculation for portfolio risk with breakdown of risk calculations
    partial_risk = np.empty((len(Stock_Close)),dtype=object)
    proportional_covar = np.empty((len(Stock_Close),len(Stock_Close)),dtype=object)
    covar_sum = 0
    for i in range(len(Stock_Close)):
        for j in range(len(Stock_Close)):
            proportional_covar[i,j] = model.Intermediate((x[i]*Stock_Close.iloc[i]/total_variable_invested)*(x[j]*Stock_Close.iloc[j]/total_variable_invested)*Stocks_Covar.iloc[i,j])
            covar_sum += proportional_covar[i,j]
        partial_risk[i] = model.Intermediate(covar_sum)
        covar_sum = 0
    Portfolio_Risk = model.Intermediate(model.sqrt(model.sum(partial_risk)))

    # Intermediate calculation for portfolio with risk free, based on utility function
    sigma_inv = model.Intermediate(total_variable_invested*Portfolio_Risk/Total_Available_Money)
    r_inv = model.Intermediate((Total_Available_Money-total_variable_invested)*Risk_Free_Return/Total_Available_Money+Portfolio_Return*total_variable_invested/Total_Available_Money)

    # Define constraints for total invested money
    model.Equation(total_variable_invested<=Total_Available_Money)

    # Define constraints for utility
    model.Equation(r_inv - sigma_inv*stats.norm.ppf(confidence_level) >= VaR/Total_Available_Money)
    
    # Define objective function
    sharpe = model.Intermediate((Portfolio_Return - Risk_Free_Return)/Portfolio_Risk)

    # Set maximization objective
    model.Maximize(sharpe)
    
    # Solve model APOPT (Interior Point and Active Set Methods)
    model.options.SOLVER=1
    model.solve(disp=False,debug=True)
    
    return model


def utility_sharpe_GEKKO_with_costs(Stock_Close,Stocks_Return,Stocks_Covar,Risk_Free_Return,Total_Available_Money, VaR, confidence_level=.95, infos=None,relaxed=False):
    """	Calculate Sharpe Ratio with GEKKO for discrete optimization

        Parameters:
        ----------
            `Stock_Close`: DataFrame
                DataFrame with stock close values

            `Stocks_Return`: DataFrame
                DataFrame with stocks return values

            `Stocks_Covar`: DataFrame
                DataFrame with stocks covariance values

            `Risk_Free_Return`: float
                Risk free return value

            `Total_Available_Money`: float 
                Total available money to invest

            `VaR`: float
                Money Value at Risk

            `confidence_level`: float
                Confidence level

            `upper_lot_boundary`: Series
                Upper boundary for lots

        Returns:
        ----------
            `Gekko_Model`: dict
                GEKKO model optimization result
        
        Example:
        ----------
            >>> Assets_Return_
            ABEV3    0.003313
            ALPA4    0.003994
            AMER3    0.008736
            ASAI3    0.008587
                    ...   
            WEGE3    0.006806
            YDUQ3    0.010095
            IBOV     0.005218
            Selic    0.001121
            >>> Assets_Covar_
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
            >>> Risk_Free_Interest_Rate
            0.0002773038676628925

    """

    # Initialize GEKKO model
    model = GEKKO(remote=False)

    # define upper lot boundary when None, based on total available money
    if infos is None:
        infos = pd.DataFrame(index=Stock_Close.columns, columns=['lot_size'])
        infos['lower_lot_boundary'] = 0
        infos['upper_lot_boundary'] = Total_Available_Money//Stock_Close
        infos['fees'] = 0
        infos['b3_taxes'] = 0
        infos['income_taxes'] = 0

    bond_tax = .225
    infos['lower_lot_boundary'] = 0
    lower_lot_boundary = infos.lower_lot_boundary[Stock_Close.index]
    upper_lot_boundary = np.minimum(Total_Available_Money//Stock_Close,infos.upper_lot_boundary.loc[Stock_Close.index])
    fees = infos.brokerage_fee.loc[Stock_Close.index]
    b3_taxes = infos.stock_exchange_taxes.loc[Stock_Close.index]
    income_taxes = infos.income_taxes.loc[Stock_Close.index]
    
    # Safety factor for initial lot size
    ss = 1

    # Random starting point for lot sizes
    start = pd.Series(data = (np.random.dirichlet(np.ones(len(Stock_Close)))*(Total_Available_Money*ss)).astype(int), index = Stock_Close.index)
        
    # Define integer variables with upper and lower bounds 
    x=np.empty((len(Stock_Close)),dtype=object)
    # y=np.empty((len(Stock_Close)),dtype=object)
    for i,close_Value in enumerate(Stock_Close):
        x[i]=model.Var(start[i],lb=lower_lot_boundary[i],ub=upper_lot_boundary[i], integer=True,name=Stock_Close.index[i])
        # y[i]=model.if2(-x[i],1,0)
        
    # Intermediate calculation for fees, taxes and artificial return
    # fees_calc = model.Intermediate(model.sum(y*fees.to_numpy()))
    fees_calc=0
    op_costs_calc = model.Intermediate((x*Stock_Close.to_numpy()).dot(b3_taxes.to_numpy()))
    taxes_calc = model.Intermediate(((x*Stock_Close.to_numpy()).dot(Stocks_Return.to_numpy()*income_taxes.to_numpy())))
    stocks_invested = model.Intermediate(model.sum(x*Stock_Close.to_numpy()))
    portfolio_artificial_return = model.Intermediate((x*Stock_Close.to_numpy()).dot(Stocks_Return.to_numpy()))

    # Intermediate calculation for total invested money
    total_variable_invested = model.Intermediate(stocks_invested+op_costs_calc+taxes_calc+fees_calc)

    # Intermediate calculation for portfolio return
    Portfolio_Return = model.Intermediate((portfolio_artificial_return-op_costs_calc-taxes_calc-fees_calc))

    # Intermediate calculation for portfolio risk with breakdown of risk calculations
    partial_risk = np.empty((len(Stock_Close)),dtype=object)
    proportional_covar = np.empty((len(Stock_Close),len(Stock_Close)),dtype=object)
    covar_sum = 0
    for i in range(len(Stock_Close)):
        for j in range(len(Stock_Close)):
            proportional_covar[i,j] = model.Intermediate((x[i]*Stock_Close.iloc[i])*(x[j]*Stock_Close.iloc[j])*Stocks_Covar.iloc[i,j])
            covar_sum += proportional_covar[i,j]
        partial_risk[i] = model.Intermediate(covar_sum)
        covar_sum = 0
    Portfolio_Risk = model.Intermediate(model.sqrt(model.sum(partial_risk)))

    # Intermediate calculation for portfolio with risk free, based on utility function
    sigma_inv = model.Intermediate(Portfolio_Risk*total_variable_invested/Total_Available_Money)
    r_inv = model.Intermediate(((Total_Available_Money-total_variable_invested)/Total_Available_Money)*(Risk_Free_Return/(1+bond_tax))+Portfolio_Return*total_variable_invested/Total_Available_Money)

    # Define constraints for total invested money
    model.Equation(total_variable_invested<=Total_Available_Money)

    # Define constraints for utility
    model.Equation(r_inv - sigma_inv*stats.norm.ppf(confidence_level) >= VaR)
    
    # Define objective function
    sharpe = model.Intermediate((Portfolio_Return - (Risk_Free_Return/(1+bond_tax)*total_variable_invested))/Portfolio_Risk)

    # Set maximization objective
    model.Maximize(sharpe)
    
    # Solve model APOPT (Interior Point and Active Set Methods)
    model.options.SOLVER=1
    if relaxed:
        model.solver_options = ['minlp_as_nlp 1']
    model.solve(disp=False,debug=True)
    
    return model


def sharpe_heuristc(assets_return_,assets_risk_,assets_cov_,risk_free_value,n_assets,clean_factor):
    """ Greedy algorithm to find the best portfolio based on sharpe ratio"""
    
    heuristic_df = pd.DataFrame()
    heuristic_df['mean'] = assets_return_
    heuristic_df['std'] = assets_risk_
    heuristic_df['local_sharpe'] = (heuristic_df['mean']-risk_free_value)/heuristic_df['std']
    heuristic_df['cov_mean'] = assets_cov_.mean().abs()
    heuristic_df['cov_std'] = assets_cov_.std()
    heuristic_df['sharpe_cov'] = heuristic_df.local_sharpe/(heuristic_df.cov_mean/heuristic_df.cov_std)
    heuristic_df.sort_values(by='sharpe_cov',ascending=False,inplace=True)
    heuristic_df['count_neg_corr'] = (assets_cov_<0).sum()
    heuristic_df = heuristic_df.loc[heuristic_df['count_neg_corr']<len(assets_cov_)/clean_factor,:]
    # heuristic_df[:n_assets].style.background_gradient(cmap='summer', axis=0)
    ind = heuristic_df[:n_assets].sort_index().index

    return ind


def utility_sharpe_optimization(Initial_Weigths, Assets_Return_, Assets_Covar_, Risk_Free_Return_, VaR, confidence_level, display_message = False) -> OptimizeResult:
    
    # Initialize iteration results list
    iteration_results=[]

    rf = Risk_Free_Return_
    zscore = stats.norm.ppf(confidence_level)

    r_p = lambda x: x.dot(Assets_Return_)
    sigma_p = lambda x: np.sqrt(x.dot(Assets_Covar_.dot(x.transpose())))

    sigma_inv = lambda x: (VaR-rf)*sigma_p(x)/(r_p(x)-rf-sigma_p(x)*zscore)
    r_inv = lambda x: (1-sigma_inv(x)/sigma_p(x))*rf+(sigma_inv(x)/sigma_p(x))*r_p(x)

    pct_p = lambda x: sigma_inv(x)/sigma_p(x)
    pct_rf = lambda x: (r_inv(x)-r_p(x)*pct_p(x))/rf

    sharpe_func = lambda x: (r_p(x)-rf)/sigma_p(x)

    # Defining the Sharpe Ratio function to iterate with independed variables and return the negative value
    def fun(x):
        Assets_Percentages = x
        Sharpe = sharpe_func(Assets_Percentages)
        iteration_results.append([Sharpe,x,sigma_p(Assets_Percentages),r_p(Assets_Percentages)])
        return -Sharpe

    
    # Defining the independent variables, boundaries, and constraints
    x0 = Initial_Weigths
    if len(x0) == 0:
        x0 = np.random.dirichlet(np.ones(len(Assets_Return_)))
    bnds = tuple([(0,1) for _ in range(len(Assets_Return_))])
    cons = [
        {'type': 'eq', 'fun': lambda x: -np.sum(x)+1},
        {'type': 'eq', 'fun': lambda x: 1-pct_p(x)-pct_rf(x)}
    ]

    # Optimizing the Sharpe Ratio with Sequential Least Squares Programming
    Optimization_Result = optimize.minimize(fun, x0, method='SLSQP', bounds=bnds, constraints=cons, options={'disp':display_message})
    
    # Getting the results for the plotting data
    Portfolios_Sharpe = []
    Portfolios_Risk = []
    Portfolios_Return = []
    final_sharpe_result = -np.inf
    Final_Weights = np.around(Optimization_Result['x'],decimals=3)

    for iteration_result in iteration_results:
        sharpe = iteration_result[0]
        Portfolios_Sharpe.append(sharpe)
        Portfolios_Risk.append(iteration_result[2])
        Portfolios_Return.append(iteration_result[3])
        if sharpe < final_sharpe_result:
            continue
        else:
            px = iteration_result[2]
            py = iteration_result[3]
            pz = iteration_result[0]

    # Displaying the Assets and weights
    list_assets = {}
    assets_result = dict(zip(Assets_Return_.index.to_list(),Final_Weights))
    for assets,value in assets_result.items():
        if value > 0:
            list_assets[assets] = value
            # print(f'{stock} - {value}')
    
    utility_portfolio = {
        'risk_free': rf,
        'portfolio_return': r_p(Optimization_Result['x']),
        'utility_return': r_inv(Optimization_Result['x']),
        'portfolio_risk': sigma_p(Optimization_Result['x']),
        'utility_risk': sigma_inv(Optimization_Result['x']),
        'portfolio_sharpe': sharpe_func(Optimization_Result['x']),
        'pct_p': pct_p(Optimization_Result['x']),
        'pct_rf': pct_rf(Optimization_Result['x'])
    }

    inv_portfolio = pd.Series(data=list_assets)*pct_p(Optimization_Result['x'])
    inv_portfolio['Selic'] = pct_rf(Optimization_Result['x'])

    return pd.Series(data=list_assets), (Portfolios_Risk,Portfolios_Return,Portfolios_Sharpe,px,py,pz), Optimization_Result, pd.DataFrame().from_dict(utility_portfolio,orient='index',columns=['values']), inv_portfolio


def MINLP_sharpe_w_costs_rebalance_n_utility(Stock_Close,Stocks_Return,Stocks_Covar,Risk_Free_Return,Total_Available_Money, VaR, current_lot_allocation=None, confidence_level=.95, infos=None,relaxed=False):
    """	Calculate Sharpe Ratio with GEKKO for discrete optimization

        Parameters:
        ----------
            `Stock_Close`: DataFrame
                DataFrame with stock close values

            `Stocks_Return`: DataFrame
                DataFrame with stocks return values

            `Stocks_Covar`: DataFrame
                DataFrame with stocks covariance values

            `Risk_Free_Return`: float
                Risk free return value

            `Total_Available_Money`: float 
                Total available money to invest

            `VaR`: float
                Money Value at Risk

            `confidence_level`: float
                Confidence level

            `upper_lot_boundary`: Series
                Upper boundary for lots

        Returns:
        ----------
            `Gekko_Model`: dict
                GEKKO model optimization result
        
        Example:
        ----------
            >>> Assets_Return_
            ABEV3    0.003313
            ALPA4    0.003994
            AMER3    0.008736
            ASAI3    0.008587
                    ...   
            WEGE3    0.006806
            YDUQ3    0.010095
            IBOV     0.005218
            Selic    0.001121
            >>> Assets_Covar_
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
            >>> Risk_Free_Interest_Rate
            0.0002773038676628925

    """

    # Initialize GEKKO model
    model = GEKKO(remote=False)

    # define upper lot boundary when None, based on total available money
    tickers = Stock_Close.index
    if infos is None:
        infos = pd.DataFrame(index=tickers, columns=['lot_size'], data=1)
        infos['lower_lot_boundary'] = 0
        infos['upper_lot_boundary'] = Total_Available_Money//Stock_Close
        infos['fees'] = 0
        infos['stock_exchange_taxes'] = 0
        infos['income_taxes'] = 0

    bond_tax = .225
    infos['lower_lot_boundary'] = 0
    lower_lot_boundary = infos.lower_lot_boundary[tickers]
    upper_lot_boundary = np.minimum(Total_Available_Money//Stock_Close,infos.upper_lot_boundary.loc[tickers])
    fees = infos.brokerage_fee.loc[tickers]
    stock_exchange_taxes = infos.stock_exchange_taxes.loc[tickers]
    income_taxes = infos.income_taxes.loc[tickers]
    
    # Safety factor for initial lot size
    ss = 1

    # Random starting point for lot sizes
    if current_lot_allocation is not None and (current_lot_allocation>0).any():
        start = pd.Series(data = current_lot_allocation, index = tickers)
    else:
        # start = pd.Series(data = (np.random.dirichlet(np.ones(len(Stock_Close)))*(Total_Available_Money*ss)/Stock_Close).astype(int), index = tickers)
        start =     pd.Series(
                        np.round( sharpe_optimization(np.array([]),Stocks_Return.to_numpy(),Stocks_Covar.to_numpy(),Risk_Free_Return).x), 
                        index = Stocks_Return.index
                        )*Total_Available_Money/Stock_Close/2
        current_lot_allocation = pd.Series(data = np.zeros(len(tickers)), index = tickers)
    
    print(start.loc[start>0])

    # Define integer variables with upper and lower bounds 
    x=np.empty((len(Stock_Close)),dtype=object)
    y=np.empty((len(Stock_Close)),dtype=object)
    delta_x=np.empty((len(Stock_Close)),dtype=object)
    for i,close_Value in enumerate(Stock_Close):
        x[i]=model.Var(start[i],lb=lower_lot_boundary[i],ub=upper_lot_boundary[i], integer=True,name=tickers[i])
        delta_x[i]=model.abs3(x[i]-current_lot_allocation[i])
        y[i]=model.if2(-delta_x[i],1,0)
        
    # Intermediate calculation for fees, taxes and artificial return
    # fees_calc=0
    fees_calc = model.Intermediate(model.sum(y*fees.to_numpy()))
    op_costs_calc = model.Intermediate((delta_x*Stock_Close.to_numpy()).dot(stock_exchange_taxes.to_numpy()))
    taxes_calc = model.Intermediate(((delta_x*Stock_Close.to_numpy()).dot(Stocks_Return.to_numpy()*income_taxes.to_numpy())))
    stocks_invested = model.Intermediate(model.sum(x*Stock_Close.to_numpy()))
    portfolio_artificial_return = model.Intermediate((x*Stock_Close.to_numpy()).dot(Stocks_Return.to_numpy()))

    # Intermediate calculation for total invested money
    total_variable_invested = model.Intermediate(stocks_invested+op_costs_calc+taxes_calc+fees_calc)

    # Intermediate calculation for portfolio return
    Portfolio_Return = model.Intermediate((portfolio_artificial_return-op_costs_calc-taxes_calc-fees_calc))

    # Intermediate calculation for portfolio risk with breakdown of risk calculations
    partial_risk = np.empty((len(Stock_Close)),dtype=object)
    proportional_covar = np.empty((len(Stock_Close),len(Stock_Close)),dtype=object)
    covar_sum = 0
    for i in range(len(Stock_Close)):
        for j in range(len(Stock_Close)):
            proportional_covar[i,j] = model.Intermediate((x[i]*Stock_Close.iloc[i])*(x[j]*Stock_Close.iloc[j])*Stocks_Covar.iloc[i,j])
            covar_sum += proportional_covar[i,j]
        partial_risk[i] = model.Intermediate(covar_sum)
        covar_sum = 0
    Portfolio_Risk = model.Intermediate(model.sqrt(model.sum(partial_risk)))

    # Intermediate calculation for portfolio with risk free, based on utility function
    sigma_inv = model.Intermediate(Portfolio_Risk*total_variable_invested/Total_Available_Money + 1e-10)
    r_inv = model.Intermediate(((Total_Available_Money-total_variable_invested)/Total_Available_Money)*(Risk_Free_Return/(1+bond_tax))+Portfolio_Return*total_variable_invested/Total_Available_Money)

    # Define constraints for total invested money
    model.Equation(total_variable_invested<=Total_Available_Money)

    # Define constraints for utility
    model.Equation(r_inv - sigma_inv*stats.norm.ppf(confidence_level) >= VaR)
    
    # Define objective function
    sharpe = model.Intermediate((Portfolio_Return - (Risk_Free_Return/(1+bond_tax)*total_variable_invested))/Portfolio_Risk)

    # Set maximization objective
    model.Maximize(sharpe)
    
    # Solve model APOPT (Interior Point and Active Set Methods)
    model.options.SOLVER=1
    if relaxed:
        model.solver_options = ['minlp_as_nlp 1']
    model.solve(disp=True,debug=False)
    
    return model

