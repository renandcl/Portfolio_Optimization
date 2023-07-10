from typing import Literal
import pandas as pd
from datetime import datetime
import json
import numpy as np
import urllib3
import os
import asyncio
import aiohttp
from typing import Literal
import nest_asyncio
nest_asyncio.apply()


from os import environ, path

environ['data_path'] = path.join(path.expanduser('~'), 'data','b3')

environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

CVM_COTAHIST_ARGS = {
    'wd':[2,8,2,12,3,12,10,3,4,13,13,13,13,13,13,13,5,18,18,13,1,8,7,13,12,3],
    'columns_b3': [
        'TIPREG',
        'DATA',
        'CODBDI',
        'CODNEG',
        'TPMERC',
        'NOMRES',
        'ESPECI',
        'PRAZOT',
        'MODREF',
        'PREABE',
        'PREMAX',
        'PREMIN',
        'PREMED',
        'PREULT',
        'PREOFC',
        'PREOFV',
        'TOTNEG',
        'QUATOT',
        'VOLTOT',
        'PREEXE',
        'INDOPC',
        'DATVEN',
        'FATCOT',
        'POTEXE',
        'CODISI',
        'DISMES'
    ],
    'dtypes_b3': {
        'TIPREG':'int64',
        'DATA':'int64',
        'CODBDI':'category',
        'CODNEG':'object',
        'TPMERC':'int64',
        'NOMRES':'object',
        'ESPECI':'category',
        'PRAZOT':'object',
        'MODREF':'object',
        'PREABE':'float64',
        'PREMAX':'float64',
        'PREMIN':'float64',
        'PREMED':'float64',
        'PREULT':'float64',
        'PREOFC':'float64',
        'PREOFV':'float64',
        'TOTNEG':'int64',
        'QUATOT':'int64',
        'VOLTOT':'float64',
        'PREEXE':'float64',
        'INDOPC':'int64',
        'DATVEN':'int64',
        'FATCOT':'float64',
        'POTEXE':'float64',
        'CODISI':'object',
        'DISMES':'int64'
    }
}


def IBOV_components() -> list:
    """ Function to get actual IBOV components.

        Returns:
        -----------          
            `IBOV_components`: list
                List with the IBOV components
        
        Example:
        ----------
            >>> IBOV_components()
            ['ALPA4','ABEV3','AMER3','ASAI3',...,'VIIA3','VBBR3','WEGE3','YDUQ3']
        """

    # Defining the URL to collect data
    url = "https://sistemaswebb3-listados.b3.com.br/indexProxy/indexCall/GetPortfolioDay/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwiaW5kZXgiOiJJQk9WIiwic2VnbWVudCI6IjEifQ=="

    # GET request to collect data
    req = urllib3.PoolManager().request('GET', url)
    content = json.loads(req.data.decode('utf-8'))

    # Appending components to a list
    IBOV_components = []
    for i in content['results']:
        IBOV_components.append(i['cod'])
    
    return IBOV_components


async def get_stocks_splits_scrap(components) -> pd.DataFrame:
    """ Funtion to return the stocks splits between the range of dates in a dataframe with index of dates and columns of stocks symbols.

        Parameters:
        -----------
            `components`: list
                List of stocks symbols
        
        Returns:
        -----------
            `stocks_splits_in_range`: dataframe
                dataframe with the stocks splits between the range of dates in a dataframe with index of dates and columns of stocks symbols.

        Example:
        -----------
            >>> components = ['ABEV3','AMER3','ASAI3','AZUL4','B3SA3','BBAS3',...,'VIIA3','WEGE3','YDUQ3']
            >>> Stocks_Splits_Scrap(components)
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

        """
    # check if split h5 file is up to date, else collect splits for components from yahoo finance
    try:
        date_check = datetime.fromtimestamp(os.stat(os.path.join(os.getenv('data_path'),'splits.h5')).st_mtime).date() == datetime.today().date()

        if not date_check:
            raise FileNotFoundError

        Stocks_Splits = pd.read_hdf(os.path.join(os.getenv('data_path'),'splits.h5'), 'splits')
        
        components_check = components == Stocks_Splits.columns.tolist()

        if not components_check:
            raise FileNotFoundError

    except:
        async def main():
            # create aiohttp session and each task for stock symbol and run parallel tasks
            async with aiohttp.ClientSession(cookie_jar = aiohttp.DummyCookieJar()) as session:
                tasks = []
                for symbol in components:
                    tasks.append(asyncio.create_task(get_data(session, symbol)))
                splits_list_df = await asyncio.gather(*tasks)

                # create dataframe with splits
                splits_df = pd.concat(splits_list_df, axis=1)

                # Correct the hour of the splits, because yahoo finance considers the milisecond which the data has been added, causing divergence with stocks close timestamp
                splits_df.index = splits_df.index.map(lambda x: datetime.fromtimestamp(x).replace(hour=0,minute=0,second=0,microsecond=0))
            return splits_df

        async def get_data(session, symbol):
            # function to get data from yahoo finance for the symbol
            interval = '1d' #Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]"
            yeardate = str(int(datetime.timestamp(datetime.now())))
            url =   f'https://query2.finance.yahoo.com/v8/finance/chart/{symbol}.SA?formatted=true&crumb=9IJOhpp5VXs&lang=en-US&region=BR&includeAdjustedClose=true&interval={interval}&period1=-2208988800&period2={yeardate}&events=div%7Csplit&useYfid=true&corsDomain=finance.yahoo.com'
            headers = { 
                'Cookie' : """A1=d=AQABBNhP4WMCEGIDyLYyvWt6_qqTL7Jf70YFEgEBAQGh4mPrYwAAAAAA_eMAAA&S=AQAAAl2VHXBoq4ObdzMJMd4kgkE; A3=d=AQABBNhP4WMCEGIDyLYyvWt6_qqTL7Jf70YFEgEBAQGh4mPrYwAAAAAA_eMAAA&S=AQAAAl2VHXBoq4ObdzMJMd4kgkE; A1S=d=AQABBNhP4WMCEGIDyLYyvWt6_qqTL7Jf70YFEgEBAQGh4mPrYwAAAAAA_eMAAA&S=AQAAAl2VHXBoq4ObdzMJMd4kgkE&j=WORLD""" ,
                'User-Agent': """Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/109.0"""
                }

            counter = 0
            while counter < 5:
                counter += 1
                try:
                    async with session.get(url,headers = headers, timeout=15) as response:

                        # start the session request
                        if response.status == 200:
                            data = await response.json()

                            # Check if the symbol contains split key on json data
                            if 'events' in data['chart']['result'][0]:
                                if 'splits' in data['chart']['result'][0]['events']:
                                    splits = data['chart']['result'][0]['events']['splits']
                                    # Creating a dataframe with the data
                                    split_date = []
                                    split_ratio = []
                                    for split in splits:
                                        split_date.append(splits[split]['date'])
                                        split_ratio.append(splits[split]['numerator']/splits[split]['denominator'])

                                    split_data_dict = {
                                        'DATE': split_date,
                                        symbol : split_ratio
                                    }

                                    splits_data = pd.DataFrame(data=split_data_dict)
                                    splits_data = splits_data.set_index('DATE').sort_index()

                                    # Ratio correction according previews splits
                                    mList = splits_data[symbol].to_list()
                                    split_corr = [] 
                                    for i in range(len(mList)):
                                        split_corr.append(np.prod(mList[i:]))
                                    splits_data[symbol] = split_corr
                                    symbol_split = splits_data.sort_index()
                                    # Index date correction from int values to datetime
                                    # symbol_split.index.map(lambda x: datetime.fromtimestamp(x))

                                    return symbol_split
                                else:
                                    return None

                        else:
                            return None
                except:
                    if counter < 5:
                        sharpe_logger.error(f'Error getting split for symbol: {symbol} - Retrying...')
                    else:
                        sharpe_logger.error(f'Error getting split for symbol: {symbol} - Max retries exceeded')
                    pass

        try:
            loop = asyncio.get_running_loop()
        except Exception:  # 'RuntimeError: There is no current event loop...'
            loop = None
        
        # Run the async loop
        if loop and loop.is_running():
            task = []
            task.append(asyncio.create_task(main()))
            tasks_result = await asyncio.gather(*task)
            Stocks_Splits = tasks_result[0]
        else:
            Stocks_Splits = asyncio.run(main())

        # Save the dataframe to h5 file
        Stocks_Splits.to_hdf(os.path.join(os.getenv('data_path'),'splits.h5'), 'splits', mode='w')

    else:
        return Stocks_Splits
    
    return Stocks_Splits


def get_IBOV_historical_quote() -> pd.Series:
    """ Function to get the IBOV Historical Quote.

        Returns:
        -----------
            `IBOV_Hist_Quote`: Series 
                Series with the IBOV Historical Quote from 1994 to today

        Example:
        -----------
            >>> IBOV_Hist_Quote()
                            IBOV
            DATE                  
            1994-01-03     380.090
            1994-01-04     400.549
            1994-01-05     421.045
            1994-01-06     444.174
            1994-01-07     476.395
            ...                ...
            2022-03-03  115165.550
            2022-03-04  114473.780
            2022-03-07  111593.460
            2022-03-08  111203.450
            2022-03-09  113900.340
            
        """
    
    # Defining the path to collect data
    dir_path = os.getenv('data_path')
    current_year = datetime.now().year

    # Collecting data from range of dates
    for i in reversed(range(1994,current_year+1)):
        # check the year due to file differences
        if i > 1997:
            ibov_data = pd.read_csv(os.path.join(dir_path,'IBOV_'+str(i)+'.csv'), sep = ";", engine = 'python', encoding = "ISO-8859-1", skiprows = 1, skipfooter = 2)
            ibov_data.set_index('Dia',inplace=True)
            for column in ibov_data.columns:
                if ibov_data[column].isna().all():
                    pass
                else:
                    ibov_data[column] = ibov_data[column].str.replace('.','',regex=True).str.replace(',','.',regex=True).astype(float)
        else:
            ibov_xls = pd.ExcelFile(os.path.join(dir_path,'IBOVDIA.XLS'))
            ibov_data = pd.read_excel(ibov_xls, str(i), skiprows=1,skipfooter=4)
            ibov_data.rename(columns={ 'PREGÃO' : 'Dia'}, inplace = True)
            ibov_data.set_index('Dia',inplace=True)
        
        # Ibov wrangling
        # Rename months
        columns = dict(zip(ibov_data.columns,list(range(1, 12 + 1))))
        ibov_data.rename(columns=columns, inplace=True)

        # add the year
        ibov_data.columns = pd.MultiIndex.from_product([ibov_data.columns, [i]])

        # rearange the columns order, stack data, and reorder index
        ibov_data = ibov_data.swaplevel(axis=1).stack([-1,-2]).swaplevel(-3,-1).sort_index(ascending=False)

        # Rename the inde
        ibov_data.index.names = ['Year','Month','Day']
        ibov_data.rename_axis = ['Close']

        if i == current_year:
            ibov_data_final = ibov_data
        else:
            ibov_data_final = pd.concat([ibov_data_final,ibov_data])
        

    # Converting the data to a Series and transform index to datetime
    ibov_data_final = ibov_data_final.reset_index()
    ibov_data_final['Month'] = ibov_data_final['Month'].apply(lambda x: '0'+str(x) if (x < 10) else str(x))
    ibov_data_final['DATE'] = pd.to_datetime(ibov_data_final['Year'].astype(str) + ibov_data_final['Month'].astype(str) + ibov_data_final['Day'].astype(str) ,format='%Y%m%d')
    ibov_data_final.set_index('DATE',inplace=True)
    ibov_data_final.drop(['Year','Month','Day'],axis=1,inplace=True)
    ibov_data_final.sort_index(ascending=True,inplace=True)
    ibov_data_final.columns = ['IBOV']
    
    return ibov_data_final


def get_Selic_historical_interest_rate() -> pd.DataFrame:
    """ Function to get the Historical Selic.

        Returns:
        -----------
            `Historical_Selic`: DataFrame 
                DataFrame with the Historical Selic

        Example:
        -----------
            >>> Historical_Selic()
                            Selic
            DATA
            1996-06-26      0.2618
            1996-06-27      0.2576
            1996-06-28      0.2440
            1996-07-01      0.2408
            1996-07-02      0.2346
            ...             ...
            2022-03-03      0.1065
            2022-03-04      0.1065
            2022-03-07      0.1065
            2022-03-08      0.1065
            2022-03-09      0.1065

        """
    
    # Defining the path to collect data
    dir_path = os.getenv('data_path')

    # Collecting data
    Selic_Data = pd.read_csv(os.path.join(dir_path,'SELIC'+'.csv'), sep=";", engine= 'python', encoding="ISO-8859-1",skipfooter=1)
    Selic_Data.columns = ['DATA','Selic']
    Selic_Data['DATA'] = pd.to_datetime(Selic_Data['DATA'],format='%d/%m/%Y')
    Selic_Data.set_index('DATA',inplace=True)

    # Correct percentage value
    Selic_Data = Selic_Data/100

    return Selic_Data


def get_n_storing_B3_historical_data_before_current_year() -> pd.DataFrame:
    """ Function to get the historical CVM data until previous end of the year an store it in a h5 file.
    
            Returns:
            -----------
                `CVM_Hist_Data`: DataFrame 
                    DataFrame with the historical CVM data
    
            Example:
            -----------
                >>> get_storing_CVM_historical_data()
                        TIPREG      DATA CODBDI    CODNEG  ...  FATCOT POTEXE        CODISI  DISMES  
                0             1  20200102     02     AALR3  ...     1.0    0.0  BRAALRACNOR6     101  
                1             1  20200102     02    AAPL34  ...     1.0    0.0  BRAAPLBDR004     131  
                2             1  20200102     02     ABCB4  ...     1.0    0.0  BRABCBACNPR4     133  
                3             1  20200102     02     ABEV3  ...     1.0    0.0  BRABEVACNOR1     122  
                ...         ...       ...    ...       ...  ...     ...    ...           ...     ...  
                3083504       1  20211112     78  BBASA340  ...     1.0    0.0  BRBBASACNOR3     301  
                3083505       1  20211118     78  BBASA340  ...     1.0    0.0  BRBBASACNOR3     301  
                3083506       1  20211119     78  BBASA340  ...     1.0    0.0  BRBBASACNOR3     301  
                3083507       1  20211122     78  BBASA340  ...     1.0    0.0  BRBBASACNOR3     301  
            """

    # Set initial and final year
    initial_year = 1994
    final_year = datetime.today().year - 1

    # Defining paramenters of CSV file from: https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/historico/mercado-a-vista/series-historicas/
    dir_path = os.getenv('data_path')
    
    # Reading all CSV files with B3 components transactions from initial year to last year
    for year in range(initial_year,final_year+1):
        file_name = 'COTAHIST_A'+str(year)+'.TXT' #https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/historico/mercado-a-vista/series-historicas/

        b3_historical_raw = pd.read_fwf(
                                        os.path.join(dir_path,file_name),
                                        header=None,
                                        names=CVM_COTAHIST_ARGS['columns_b3'],
                                        dtype=CVM_COTAHIST_ARGS['dtypes_b3'],
                                        widths=CVM_COTAHIST_ARGS['wd'],
                                        skiprows=1,
                                        skipfooter=1,
                                        encoding="ISO-8859-1",
                                        engine='pyarrow'
                                    )

        if year == initial_year:
            b3_historical_data = b3_historical_raw
        else:
            b3_historical_data = pd.concat([b3_historical_data,b3_historical_raw],ignore_index=True)

        # Correct values due to interpretation dificulty on pandas for NaN values
        b3_historical_data[['CODBDI', 'CODNEG', 'NOMRES', 'ESPECI', 'PRAZOT', 'MODREF', 'CODISI']] = b3_historical_data[['CODBDI', 'CODNEG', 'NOMRES', 'ESPECI', 'PRAZOT', 'MODREF', 'CODISI']].applymap(str)
    
    # Save as h5 file
    b3_historical_data.to_hdf(os.path.join(dir_path,'b3_historical_data.h5'),key='b3_historical_data', mode='w')

    return b3_historical_data


def get_n_storing_B3_historical_data_for_current_year() -> pd.DataFrame:
    """ Function to get the historical CVM data for 2022 up to date and store it in a h5 file.
    
            Returns:
            -----------
                `CVM_Hist_Data`: DataFrame 
                    DataFrame with the historical CVM data
    
            Example:
            -----------
                >>> get_storing_CVM_historical_data()
                        TIPREG      DATA CODBDI    CODNEG  ...  FATCOT POTEXE        CODISI  DISMES  
                0             1  20200102     02     AALR3  ...     1.0    0.0  BRAALRACNOR6     101  
                1             1  20200102     02    AAPL34  ...     1.0    0.0  BRAAPLBDR004     131  
                2             1  20200102     02     ABCB4  ...     1.0    0.0  BRABCBACNPR4     133  
                3             1  20200102     02     ABEV3  ...     1.0    0.0  BRABEVACNOR1     122  
                ...         ...       ...    ...       ...  ...     ...    ...           ...     ...  
                3083504       1  20211112     78  BBASA340  ...     1.0    0.0  BRBBASACNOR3     301  
                3083505       1  20211118     78  BBASA340  ...     1.0    0.0  BRBBASACNOR3     301  
                3083506       1  20211119     78  BBASA340  ...     1.0    0.0  BRBBASACNOR3     301  
                3083507       1  20211122     78  BBASA340  ...     1.0    0.0  BRBBASACNOR3     301  
            """
    # Get current year
    current_year = datetime.today().year

    # Get the data from the h5 file
    dir_path = os.getenv('data_path')

    # Read current_year data
    file_name = f'COTAHIST_A{str(current_year)}.TXT' #https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/historico/mercado-a-vista/series-historicas/

    b3_historical_current_year = pd.read_fwf(
                                    os.path.join(dir_path,file_name),
                                    header=None,
                                    names=CVM_COTAHIST_ARGS['columns_b3'],
                                    dtype=CVM_COTAHIST_ARGS['dtypes_b3'],
                                    widths=CVM_COTAHIST_ARGS['wd'],
                                    skiprows=1,
                                    skipfooter=1,
                                    encoding="ISO-8859-1",
                                    engine='pyarrow'
                                )
    
    b3_historical_current_year[['CODBDI', 'CODNEG', 'NOMRES', 'ESPECI', 'PRAZOT', 'MODREF', 'CODISI']] = b3_historical_current_year[['CODBDI', 'CODNEG', 'NOMRES', 'ESPECI', 'PRAZOT', 'MODREF', 'CODISI']].applymap(str)
    
    # Save as h5 file
    key_name = f'b3_historical_{str(current_year)}'
    b3_historical_current_year.to_hdf(os.path.join(dir_path,key_name+'.h5'),key=key_name, mode='w')

    return b3_historical_current_year


def get_B3_historical_data(filter_type = 'IBOV', components_list: list = None, total_transactions_on_period = 10000, period: Literal['month','week','day'] = 'month'):
    """ Function to get the historical CVM data end of the year an store it in a h5 file.

            Parameters:
            -----------
                `components_list`: list, optional
                    List of components to be filtered.

                `filter_type`: str, optional
                    Type of filter to be applied to the data. The default is 'IBOV'.
                    #Valid values: 'IBOV', 'transactions_per_period', 'components_list'
                
                `total_transactions_on_period`: int
                    Total transactions to be retrieved from the data. The default is 10000.
                
                `period`: str
                    Period of the data to be retrieved. The default is 'month'.
                    #Valid values: 'month', 'week', 'day'
            
            Returns:
            -----------
                `CVM_Hist_Data`: DataFrame 
                    DataFrame with the historical CVM data
    
            Example:
            -----------
                >>> B3_historical_data()
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
            """

    # Get current year
    current_year = datetime.today().year

    # Define the data path
    dir_path = os.getenv('data_path')

    # Check if the h5 file is up to date, if not, update it
    key_name = f'b3_historical_{str(current_year)}'
    if datetime.fromtimestamp(os.stat(os.path.join(os.getenv('data_path'),key_name+'.h5')).st_mtime).date() == datetime.today().date():
        b3_historical_current_year = pd.read_hdf(os.path.join(dir_path,key_name+'.h5'),key = key_name, usecols=['DATA','PREULT','CODNEG','TOTNEG'])
    else:
        b3_historical_current_year = get_n_storing_B3_historical_data_for_current_year()
        b3_historical_current_year = b3_historical_current_year[['DATA','PREULT','CODNEG','TOTNEG']]
    
    # Read file upto current_year
    b3_historical_before_current_year = pd.read_hdf(os.path.join(dir_path,'b3_historical_data.h5'),key='b3_historical_data',usecols=['DATA','PREULT','CODNEG','TOTNEG'])

    # Concatenate the data
    b3_historical_data = pd.concat([b3_historical_before_current_year,b3_historical_current_year],ignore_index=True)

    # Correction on columns values to decimal values 
    b3_historical_data['PREULT'] = b3_historical_data['PREULT'].values/100
    
    # Correction on columns date data types
    b3_historical_data['DATA'] = pd.to_datetime(b3_historical_data['DATA'],format='%Y%m%d')   

    # Filter the components by filter type
    if filter_type == 'IBOV':
        components = IBOV_components()
    elif filter_type == 'components_list':
        components = components_list
    elif filter_type == 'transactions_per_period':
        if period == 'month':
            initial_date = datetime.strftime(datetime.today() - pd.DateOffset(months=1), '%Y-%m-%d')
            final_date = datetime.strftime(datetime.now(),format='%Y-%m-%d')
        elif period == 'day':
            initial_date = datetime.strftime(datetime.today() - pd.DateOffset(days=1), '%Y-%m-%d')
            final_date = datetime.strftime(datetime.now(),format='%Y-%m-%d')
        elif period == 'week':
            initial_date = datetime.strftime(datetime.today() - pd.DateOffset(weeks=1), '%Y-%m-%d')
            final_date = datetime.strftime(datetime.now(),format='%Y-%m-%d')
        else:
            raise ValueError('Period must be "month", "day" or "week"') 
        
        # Filtering the dataframe with the initial and final dates
        between_dates = b3_historical_data.loc[b3_historical_data['DATA'].isin(pd.date_range(initial_date,final_date))]

        # Filtering the dataframe with the total transactions per month
        total_transactions = between_dates.groupby(['CODNEG'])['TOTNEG'].sum()
        filter_transactions = total_transactions[total_transactions > total_transactions_on_period]

        # Defining the filtered transaction components as index
        list_stocks_count = pd.DataFrame()
        list_stocks_count['TckrSymb'] = filter_transactions.index.values

        # Reading CSV file wit the list of B3 components and its paramenters details    
        file_name = 'InstrumentsConsolidatedFile_'+datetime.today().strftime('%Y%m%d')+'_1.csv' #https://arquivos.b3.com.br/Web/Consolidated
        B3_instruments = pd.read_csv(os.path.join(dir_path,file_name),sep=';', encoding = "ISO-8859-1", low_memory=False)

        odd_lot_shares = B3_instruments.loc[(B3_instruments.SctyCtgyNm == 'SHARES') & (B3_instruments.SgmtNm == 'ODD LOT')].TckrSymb.unique()
        std_lot_shares = B3_instruments.loc[(B3_instruments.SctyCtgyNm == 'SHARES') & (B3_instruments.SgmtNm == 'CASH')].TckrSymb.unique()
        funds = B3_instruments.loc[(B3_instruments.SctyCtgyNm == 'SHARES') & (B3_instruments.SgmtNm == 'FUNDS') & ~(B3_instruments.CrpnNm.str.contains('IMOB', na=True))].TckrSymb.unique()
        real_state_funds = B3_instruments.loc[(B3_instruments.SctyCtgyNm == 'SHARES') & (B3_instruments.SgmtNm == 'FUNDS') & (B3_instruments.CrpnNm.str.contains('IMOB', na=False))].TckrSymb.unique()
        bdr_shares = B3_instruments.loc[(B3_instruments.SctyCtgyNm == 'BDR')].TckrSymb.unique()
        etf_shares = B3_instruments.loc[(B3_instruments.SctyCtgyNm == 'ETF FOREIGN INDEX') | (B3_instruments.SctyCtgyNm == 'ETF EQUITIES')].TckrSymb.unique()
        unit_shares = B3_instruments.loc[(B3_instruments.SctyCtgyNm == 'UNIT')].TckrSymb.unique()

        # Filtering the dataframe with stock types
        B3_instruments_filter = B3_instruments.loc[
            (B3_instruments.TckrSymb.isin(odd_lot_shares)) |
            (B3_instruments.TckrSymb.isin(std_lot_shares)) |
            (B3_instruments.TckrSymb.isin(funds)) |
            (B3_instruments.TckrSymb.isin(real_state_funds)) |
            (B3_instruments.TckrSymb.isin(bdr_shares)) |
            (B3_instruments.TckrSymb.isin(etf_shares)) |
            (B3_instruments.TckrSymb.isin(unit_shares))
            ,['TckrSymb']
            ]

        # Merging the filtered transactions stocks with the filtered components details
        stocks_list = pd.merge(list_stocks_count,B3_instruments_filter,on='TckrSymb')

        components = stocks_list['TckrSymb'].to_list()
    else:
        raise ValueError('Filter type must be "IBOV", "transactions_per_period" or "components_list"')

    # Selecting only the selected components
    b3_historical_data = b3_historical_data[b3_historical_data['CODNEG'].isin(components)]

    # Selecting only close values, sorting by date and symbols
    b3_historical_data[['DATA','CODNEG','PREULT']].sort_values(by=['DATA','CODNEG'])

    b3_historical_data.rename(columns={'DATA':'DATE'},inplace=True)

    # Pivot table with symbols on columns, date on index and close values on values
    b3_historical_data = pd.pivot_table(b3_historical_data, values = 'PREULT', index='DATE',columns='CODNEG')

    return b3_historical_data


def get_stocks_splits_scrap_wrapper(*args, **kwargs) -> pd.DataFrame:
    """ Function to return the stocks splits between the range of dates in a dataframe with index of dates and columns of stocks symbols.

        Parameters:
        -----------
            `components`: list
                List of stocks symbols
        
        Returns:
        -----------
            `stocks_splits_in_range`: dataframe
                dataframe with the stocks splits between the range of dates in a dataframe with index of dates and columns of stocks symbols.

        Example:
        -----------
            >>> components = ['ABEV3','AMER3','ASAI3','AZUL4','B3SA3','BBAS3',...,'VIIA3','WEGE3','YDUQ3']
            >>> Stocks_Splits_Scrap(components)
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

        """

    # function to allow running within jupyter notebook
    return asyncio.run(get_stocks_splits_scrap(*args, **kwargs))


def get_assets_infos(components=None) -> pd.DataFrame:
    """ Function to return the assets infos in a dataframe with index stocks symbols.

        Parameters:
        -----------
            `components`: list
                List of stocks symbols

        Returns:
        -----------
            `assets_infos`: dataframe 
                Datafreame with the assets infos in a dataframe with index stocks symbols.

        Example:
        -----------
            >>> components = ['ABEV3','AMER3','ASAI3','AZUL4','B3SA3','BBAS3',...,'VIIA3','WEGE3','YDUQ3']
            >>> Assets_Infos(components)
                                            type  lot_size  income_taxes  stock_exchange_taxes   brokerage_fee  upper_lot_boundary  
            003H11                None       NaN          0.00              0.000325               0              999999     
            2WAV3       std_lot_shares       1.0          0.15              0.000325               0              999999     
            5GTK11          etf_shares       1.0          0.15              0.000325               0              999999     
            5GTK11T               None       NaN          0.00              0.000325               0              999999     
            A1AP34          bdr_shares       1.0          0.15              0.000325               0              999999     
            ...                    ...       ...           ...                   ...             ...                 ...     
            IMABETF11H            None       NaN          0.00              0.000325               0              999999     
            IMBB11                None       NaN          0.00              0.000325               0              999999     
            IMBBETF11H            None       NaN          0.00              0.000325               0              999999     
            IRFM11                None       NaN          0.00              0.000325               0              999999     
            IRFMETF11H            None       NaN          0.00              0.000325               0              999999     
    """

    file_name = 'InstrumentsConsolidatedFile_'+datetime.today().strftime('%Y%m%d')+'_1.csv' #https://arquivos.b3.com.br/Web/Consolidated
    dir_path = os.environ['data_path']
    instruments = pd.read_csv(os.path.join(dir_path,file_name),sep=';', encoding = "ISO-8859-1", low_memory=False)
    assets_info = pd.DataFrame(index=instruments.TckrSymb.unique())

    instruments.set_index('TckrSymb',drop=False,inplace=True)
    instruments['odd_lot'] = (instruments.TckrSymb+'F'==instruments.TckrSymb.shift(-1)) # instruments.TckrSymb.apply(lambda x: np.where(x+'F' in instruments.TckrSymb.values, True, False))
    instruments.loc[instruments.odd_lot,'AllcnRndLot'] = 100

    odd_lot_shares = instruments.loc[(instruments.SctyCtgyNm == 'SHARES') & (instruments.SgmtNm == 'ODD LOT')].TckrSymb.unique()
    std_lot_shares = instruments.loc[(instruments.SctyCtgyNm == 'SHARES') & (instruments.SgmtNm == 'CASH')].TckrSymb.unique()
    funds = instruments.loc[(instruments.SctyCtgyNm == 'SHARES') & (instruments.SgmtNm == 'FUNDS') & ~(instruments.CrpnNm.str.contains('IMOB', na=True))].TckrSymb.unique()
    real_state_funds = instruments.loc[(instruments.SctyCtgyNm == 'SHARES') & (instruments.SgmtNm == 'FUNDS') & (instruments.CrpnNm.str.contains('IMOB', na=False))].TckrSymb.unique()
    bdr_shares = instruments.loc[(instruments.SctyCtgyNm == 'BDR')].TckrSymb.unique()
    etf_shares = instruments.loc[(instruments.SctyCtgyNm == 'ETF FOREIGN INDEX') | (instruments.SctyCtgyNm == 'ETF EQUITIES')].TckrSymb.unique()
    unit_shares = instruments.loc[(instruments.SctyCtgyNm == 'UNIT')].TckrSymb.unique()
    index = instruments.loc[(instruments.SctyCtgyNm == 'INDEX')].TckrSymb.unique()

    assets_info['type'] = None
    assets_info['lot_size'] = np.NaN
    assets_info['income_taxes'] = 0
    assets_info['stock_exchange_taxes'] = 0.0325/100
    assets_info['brokerage_fee'] = 0
    assets_info['upper_lot_boundary'] = 999999

    assets_info.loc[assets_info.index.isin(odd_lot_shares),'type'] = 'odd_lot_shares'
    assets_info.loc[assets_info.index.isin(std_lot_shares),'type'] = 'std_lot_shares'
    assets_info.loc[assets_info.index.isin(funds),'type'] = 'funds'
    assets_info.loc[assets_info.index.isin(real_state_funds),'type'] = 'real_state_funds'
    assets_info.loc[assets_info.index.isin(bdr_shares),'type'] = 'bdr_shares'
    assets_info.loc[assets_info.index.isin(etf_shares),'type'] = 'etf_shares'
    assets_info.loc[assets_info.index.isin(unit_shares),'type'] = 'unit_shares'
    assets_info.loc[assets_info.index.isin(index),'type'] = 'index'

    assets_info.loc[assets_info.index.isin(odd_lot_shares),'lot_size'] = instruments.loc[instruments.index.isin(odd_lot_shares)].AllcnRndLot
    assets_info.loc[assets_info.index.isin(std_lot_shares),'lot_size'] = instruments.loc[instruments.index.isin(std_lot_shares)].AllcnRndLot
    assets_info.loc[assets_info.index.isin(funds),'lot_size'] = instruments.loc[instruments.index.isin(funds)].AllcnRndLot
    assets_info.loc[assets_info.index.isin(real_state_funds),'lot_size'] = instruments.loc[instruments.index.isin(real_state_funds)].AllcnRndLot
    assets_info.loc[assets_info.index.isin(bdr_shares),'lot_size'] = instruments.loc[instruments.index.isin(bdr_shares)].AllcnRndLot
    assets_info.loc[assets_info.index.isin(etf_shares),'lot_size'] = instruments.loc[instruments.index.isin(etf_shares)].AllcnRndLot
    assets_info.loc[assets_info.index.isin(unit_shares),'lot_size'] = instruments.loc[instruments.index.isin(unit_shares)].AllcnRndLot
    assets_info.loc[assets_info.index.isin(index),'lot_size'] = instruments.loc[instruments.index.isin(index)].AllcnRndLot


    assets_info.loc[assets_info.index.isin(odd_lot_shares),'income_taxes'] = 0.15
    assets_info.loc[assets_info.index.isin(std_lot_shares),'income_taxes'] = 0.15
    assets_info.loc[assets_info.index.isin(funds),'income_taxes'] = 0.15
    assets_info.loc[assets_info.index.isin(real_state_funds),'income_taxes'] = 0.2
    assets_info.loc[assets_info.index.isin(bdr_shares),'income_taxes'] = 0.15
    assets_info.loc[assets_info.index.isin(etf_shares),'income_taxes'] = 0.15
    assets_info.loc[assets_info.index.isin(unit_shares),'income_taxes'] = 0.15
    assets_info.loc[assets_info.index.isin(index),'income_taxes'] = 0.15

    assets_info.loc[(instruments.SgmtNm == 'ODD LOT'),'upper_lot_boundary'] = 99

    assets_info = pd.concat([assets_info, pd.DataFrame([['index', 1, 0.225, 0.000734, 0,999999]], columns=assets_info.columns, index=['IBOV'])], axis=0)
    assets_info = pd.concat([assets_info, pd.DataFrame([['treasury_bonds', 1, 0.225, 0.000734, 0,999999]], columns=assets_info.columns, index=['Selic'])], axis=0)

    if components:
        return assets_info.loc[components]
    else:
        return assets_info