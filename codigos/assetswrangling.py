import pandas as pd
import numpy as np



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


def adjust_close_values_with_split(close_values_df, split_values_df):
    """ Function to adjust the close values according to the splits.	
    
        Parameters:
        -----------
            `close_values_df`: DataFrame
                Dataframe with the close values

            `split_values_df`: DataFrame
                Dataframe with the splits
        
        Returns:
        -----------
            `close_values_df_adjusted`: DataFrame
                Dataframe with the adjusted close values multiplied by the splits

        Example:
        -----------
            >>> close_values_df
            CODNEG      ABEV3  ALPA4  AMER3  ASAI3  AZUL4  B3SA3  BBAS3  ...  VIVT3  WEGE3  YDUQ3 
            DATE                                                         ...                         
            1998-03-16    NaN  45.80    NaN    NaN    NaN    NaN   8.00  ...    NaN    NaN    NaN    
            1998-03-17    NaN  45.50    NaN    NaN    NaN    NaN   8.35  ...    NaN    NaN    NaN    
            1998-03-18    NaN  45.40    NaN    NaN    NaN    NaN   8.30  ...    NaN    NaN    NaN    
            1998-03-19    NaN  47.50    NaN    NaN    NaN    NaN   8.39  ...    NaN    NaN    NaN    
            1998-03-20    NaN  50.02    NaN    NaN    NaN    NaN   8.45  ...    NaN    NaN    NaN    
            ...           ...    ...    ...    ...    ...    ...    ...  ...    ...    ...    ...    
            2022-03-03  14.18  25.44  31.10  13.15  23.67  14.77  34.77  ...  48.79  31.25  20.32    
            2022-03-04  13.93  24.71  30.09  12.71  21.83  14.44  33.90  ...  48.64  31.51  19.54    
            2022-03-07  13.43  22.27  27.01  12.28  17.90  13.90  32.45  ...  48.64  31.08  18.01    
            2022-03-08  13.44  22.96  26.93  12.30  19.17  13.62  32.70  ...  48.80  31.57  18.08    
            2022-03-09  13.93  24.45  27.51  13.40  20.86  13.80  34.55  ...  49.65  32.67  19.34   

            >>> split_values_df
                        ALPA4  ABEV3  ASAI3  B3SA3  BIDI11  ...  USIM5  VALE3  VIIA3  WEGE3  YDUQ3  
            2000-05-02    NaN    NaN    NaN    NaN     NaN  ...    NaN    NaN    NaN    NaN    NaN   
            2000-06-21    NaN    NaN    NaN    NaN     NaN  ...    NaN    NaN    NaN    NaN    NaN   
            2000-10-20    NaN    0.3    NaN    NaN     NaN  ...    NaN    NaN    NaN    NaN    NaN   
            2002-03-01    NaN    NaN    NaN    NaN     NaN  ...    NaN    NaN    4.0    NaN    NaN   
            2002-03-04    NaN    NaN    NaN    NaN     NaN  ...    NaN    NaN    NaN    NaN    NaN   
            ...           ...    ...    ...    ...     ...  ...    ...    ...    ...    ...    ...   
            2021-10-22    NaN    NaN    NaN    NaN     NaN  ...    NaN    NaN    NaN    NaN    NaN   
            2021-11-05    NaN    NaN    NaN    NaN     NaN  ...    NaN    NaN    NaN    NaN    NaN   
            2021-12-15    NaN    NaN    NaN    NaN     NaN  ...    NaN    NaN    NaN    NaN    NaN   
            2021-12-21    NaN    NaN    NaN    NaN     NaN  ...    NaN    NaN    NaN    NaN    NaN   
            2021-12-27    NaN    NaN    NaN    NaN     NaN  ...    NaN    NaN    NaN    NaN    NaN   

            >>> CVM_Close_Adjust_Split(CVM_Close, split_values_df)
            CODNEG      ABEV3      ALPA4  AMER3  ASAI3  AZUL4  ...  VBBR3  VIIA3  VIVT3  WEGE3  YDUQ3
            DATE                                               ...                                      
            1998-03-16    NaN   1.376409    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN    NaN   
            1998-03-17    NaN   1.367393    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN    NaN   
            1998-03-18    NaN   1.364388    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN    NaN   
            1998-03-19    NaN   1.427498    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN    NaN   
            1998-03-20    NaN   1.503231    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN    NaN   
            ...           ...        ...    ...    ...    ...  ...    ...    ...    ...    ...    ...   
            2022-03-03  14.18  25.440000  31.10  13.15  23.67  ...  23.35   3.74  48.79  31.25  20.32   
            2022-03-04  13.93  24.710000  30.09  12.71  21.83  ...  22.68   3.62  48.64  31.51  19.54   
            2022-03-07  13.43  22.270000  27.01  12.28  17.90  ...  21.63   3.40  48.64  31.08  18.01   
            2022-03-08  13.44  22.960000  26.93  12.30  19.17  ...  21.78   3.40  48.80  31.57  18.08   
            2022-03-09  13.93  24.450000  27.51  13.40  20.86  ...  22.23   3.41  49.65  32.67  19.34   

        """
    
    # Create a dataframe with Close values columns and Index, and add the non nan values from split_values_df. Additionaly, backfill the nan values with the previous value and forward fill the nan values with 1
    split_values_df_df = pd.DataFrame(index=close_values_df.index,columns=close_values_df.columns).combine_first(split_values_df).fillna(method='bfill').fillna(1).astype(float)

    # Multiply Close values with the split ratio
    split_values_df_adj = (close_values_df/(split_values_df_df.shift(-1).fillna(method='ffill')))

    return split_values_df_adj


def interpolate_nan_close_values(close_values_df):
    """ Function to interpolate NaN values from the close values and return a dataframe

        Parameters:
        -----------
            `close_values_df`: DataFrame 
                Dataframe with close values

        Returns:
        -----------
            `close_values_df_interpolated`: DataFrame
                Dataframe with linear interpolated values for the nan values

        Example:
        -----------
            >>> close_values_df
            CODNEG      ABEV3      ALPA4  AMER3  ASAI3  AZUL4  ...  VBBR3  VIIA3  VIVT3  WEGE3  YDUQ3
            DATE                                               ...                                      
            1998-03-16    NaN   1.376409    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN    NaN   
            1998-03-17    NaN   1.367393    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN    NaN   
            1998-03-18    NaN   1.364388    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN    NaN   
            1998-03-19    NaN   1.427498    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN    NaN   
            1998-03-20    NaN   1.503231    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN    NaN   
            ...           ...        ...    ...    ...    ...  ...    ...    ...    ...    ...    ...   
            2022-03-03  14.18  25.440000  31.10  13.15  23.67  ...  23.35   3.74  48.79  31.25  20.32   
            2022-03-04  13.93  24.710000  30.09  12.71  21.83  ...  22.68   3.62  48.64  31.51  19.54   
            2022-03-07  13.43  22.270000  27.01  12.28  17.90  ...  21.63   3.40  48.64  31.08  18.01   
            2022-03-08  13.44  22.960000  26.93  12.30  19.17  ...  21.78   3.40  48.80  31.57  18.08   
            2022-03-09  13.93  24.450000  27.51  13.40  20.86  ...  22.23   3.41  49.65  32.67  19.34   

            >>> CVM_Close_Adjust_Interpolate(close_values_df)
            CODNEG      ABEV3      ALPA4  AMER3  ASAI3  AZUL4  ...  VBBR3  VIIA3  VIVT3  WEGE3  YDUQ3
            DATE                                               ...
            1998-03-16    NaN   1.376409    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN    NaN
            1998-03-17    NaN   1.367393    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN    NaN
            1998-03-18    NaN   1.364388    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN    NaN
            1998-03-19    NaN   1.427498    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN    NaN
            1998-03-20    NaN   1.503231    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN    NaN
            ...           ...        ...    ...    ...    ...  ...    ...    ...    ...    ...    ...   
            2022-03-03  14.18  25.440000  31.10  13.15  23.67  ...  23.35   3.74  48.79  31.25  20.32
            2022-03-04  13.93  24.710000  30.09  12.71  21.83  ...  22.68   3.62  48.64  31.51  19.54
            2022-03-07  13.43  22.270000  27.01  12.28  17.90  ...  21.63   3.40  48.64  31.08  18.01
            2022-03-08  13.44  22.960000  26.93  12.30  19.17  ...  21.78   3.40  48.80  31.57  18.08
            2022-03-09  13.93  24.450000  27.51  13.40  20.86  ...  22.23   3.41  49.65  32.67  19.34
        """
    
    # Interpolate the adjusted close values
    close_values_df_interpolated = close_values_df.interpolate(method='linear',limit_direction='forward')

    return close_values_df_interpolated


def convert_risk_free(Risk_Free_Values, convert_from = 'annual', convert_to = 'daily', interval_type = 'daily'):
    """ Convert the Risk Free Interest Rate.
    
        Parameters:
        -----------
            `Risk_Free_Values`: DataFrame
                Annualized interest rate as float

            `convert_from`: str, optional
                Last valid business day on period. The default is 'annual'.
                #Valid intervals: [daily, weekly, monthly, quarterly, annual]"

            `convert_to`: str, optional
                Last valid business day on period. The default is 'daily'.
                #Valid intervals: [daily, weekly, monthly, quarterly, annual]"

            `interval_type`: str, optional
                Arrange the Risk Free convertion rate on interval. It should be the same as convert_to. The default is 'daily'. 
                #Valid intervals: [daily, week_start, week_end, month_start, month_end, quarter_start, quarter_end, year_start, year_end]"
        
        Returns:
        -----------
            `Risk_Free_Values`: DataFrame
                Risk Free converted rate dataframe

        Example:
        -----------
            >>> Risk_Free_Values
                            Selic
            DATE
            2022-02-15      0.1065
            2022-02-16      0.1065
            2022-02-17      0.1065
            2022-02-18      0.1065
            2022-02-21      0.1065
            2022-02-22      0.1065
            2022-02-23      0.1065
            2022-02-24      0.1065
            2022-02-25      0.1065
            2022-03-02      0.1065
            2022-03-03      0.1065
            2022-03-04      0.1065
            2022-03-07      0.1065
            2022-03-08      0.1065
            2022-03-09      0.1065

            >>> Convert_Risk_Free(Risk_Free_Values, convert_from = 'annual', convert_to = 'daily', interval_type = 'week_end')
                            Selic
            DATE
            2022-02-18      0.001948
            2022-02-25      0.001948
            2022-03-04      0.001948
            2022-03-09      0.001948
        """
    
    # Copy values to avoid changing the original DataFrame
    Risk_Free_Values_ = Risk_Free_Values.copy()

    # Group the Risk Free values by interval
    Risk_Free_Values_ = interval_group_by_convertion(Risk_Free_Values_, interval_type)

    # Define original convertion rate period
    if convert_from == 'daily':
        convertion_rate_initial = 365
    elif convert_from == 'weekly':
        convertion_rate_initial = 52
    elif convert_from == 'monthly':
        convertion_rate_initial = 12
    elif convert_from == 'quarterly':
        convertion_rate_initial = 4
    elif convert_from == 'annual':
        convertion_rate_initial = 1

    # Define final convertion rate period
    if convert_to == 'daily':
        convertion_rate_final = 365
    elif convert_to == 'weekly':
        convertion_rate_final = 52
    elif convert_to == 'monthly':
        convertion_rate_final = 12
    elif convert_to == 'quarterly':
        convertion_rate_final = 4
    elif convert_to == 'annual':
        convertion_rate_final = 1

    # Calculate the Risk Free convertion and historic values
    return np.power(1 + Risk_Free_Values_, convertion_rate_initial/convertion_rate_final) - 1


def assets_concat(CVM_Values, Ibov_Values, Selic_Values):
    """ Function to concat the CVM, IBOV and Selic values.

        Parameters:
        -----------
            `CVM_Values`: DataFrame
                Dataframe with the CVM values

            `Ibov_Values`: DataFrame
                Dataframe with the IBOV values

            `Selic_Values`: Series
                Series with the Selic daily interest rate values

        Returns:
        -----------
            `Assets_Values`: DataFrame
                Dataframe with the concat values, outer join with Ibov and inner join with Selic

        Example:
        -----------
            >>> Assets_Concat(CVM_Values, Ibov_Values, Selic_Values)
                        ABEV3  ALPA4  AMER3  ASAI3  AZUL4  ...  VIVT3  WEGE3  YDUQ3       IBOV      Selic  
            DATE                                           ...                                              
            1996-06-26    NaN    NaN    NaN    NaN    NaN  ...    NaN    NaN    NaN    6227.15   1.000637   
            1996-06-27    NaN    NaN    NaN    NaN    NaN  ...    NaN    NaN    NaN    6181.59   1.001266   
            1996-06-28    NaN    NaN    NaN    NaN    NaN  ...    NaN    NaN    NaN    6043.89   1.001865   
            1996-07-01    NaN    NaN    NaN    NaN    NaN  ...    NaN    NaN    NaN    6156.19   1.002457   
            1996-07-02    NaN    NaN    NaN    NaN    NaN  ...    NaN    NaN    NaN    6245.50   1.003036   
            ...           ...    ...    ...    ...    ...  ...    ...    ...    ...        ...        ...   
            2022-03-03  14.18  25.44  31.10  13.15  23.67  ...  48.79  31.25  20.32  115165.55  10.007505   
            2022-03-04  13.93  24.71  30.09  12.71  21.83  ...  48.64  31.51  19.54  114473.78  10.010280   
            2022-03-07  13.43  22.27  27.01  12.28  17.90  ...  48.64  31.08  18.01  111593.46  10.013056   
            2022-03-08  13.44  22.96  26.93  12.30  19.17  ...  48.80  31.57  18.08  111203.45  10.015833   
            2022-03-09  13.93  24.45  27.51  13.40  20.86  ...  49.65  32.67  19.34  113900.34  10.018610   

    """
    
    CVM_IBOV_concat = pd.concat([CVM_Values,Ibov_Values],axis=1)
    Selic_interst_to_growth = (Selic_Values + 1).cumprod()
    CVM_IBOV_Selic = pd.concat([CVM_IBOV_concat,Selic_interst_to_growth],axis=1,join='inner')

    return CVM_IBOV_Selic

