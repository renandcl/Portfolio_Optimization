import pandas as pd
import numpy as np

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

