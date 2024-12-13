import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def get_rolling_ratings_ret_df(df):
    """
    Input: pandas dataframe -> glassd dataframe (make sure index is datetime)
    
    Output: a dataframe like the rolling_period_ratings_stats.csv
    """
    ratings_cols = df.filter(like='rating').columns
    numeric_ratings_col = ratings_cols.values[:-3]

    ratings_avg = pd.DataFrame()
    for col in numeric_ratings_col:
        col_avg = df[['ticker',col]].groupby('ticker').resample('M').agg(
                        total=(col, 'sum'),
                        count=(col, lambda x: x[x!=0].count())
        )
        col_avg = col_avg.reset_index()
        col_avg.rename(columns={'total': col+'_total', 'count': col+'_count'},inplace=True)
        ratings_avg = pd.concat([ratings_avg, col_avg[[col+'_total', col+'_count']]],axis=1).reset_index(drop=True)

    ratings_stat = pd.concat([col_avg[['ticker','reviewDateTime']], ratings_avg],axis=1)

    ratings_stat['reviewDateTime']=pd.to_datetime(ratings_stat['reviewDateTime'])
    ratings_stat.set_index('reviewDateTime',drop=True,inplace=True)

    rolling_df = pd.DataFrame()
    rolling_periods = [1,3,6,9,12,24,36]
    for i in rolling_periods:
        rolling = ratings_stat.groupby('ticker').rolling(window=i).sum()
        rolling = rolling.add_prefix(f"{i}_")
        rolling = rolling.reset_index()
        rolling_df = pd.concat([rolling_df,rolling.drop(['ticker','reviewDateTime'],axis=1)],axis=1)

    rolling_df = pd.concat([rolling[['ticker','reviewDateTime']],rolling_df],axis=1)

    for i in rolling_periods:
        for col in numeric_ratings_col:
            rolling_df[f"{i}_{col}_mean"] = rolling_df[f"{i}_{col}_total"]/rolling_df[f"{i}_{col}_count"]
    
    ret_rolling = pd.read_csv('/home/group3/group3/Finalized/data/rolling_future_ret.csv')
    ret_rolling['reviewDateTime'] = pd.to_datetime(ret_rolling['reviewDateTime'])
    ret_rolling_df = pd.merge(rolling_df, ret_rolling, left_on=['ticker','reviewDateTime'], right_on=['ticker','reviewDateTime'])

    return ret_rolling_df




def monthly_ret_concat(ret_df):
    rolling_ret = pd.read_csv('/home/group3/group3/Finalized/data/rolling_future_ret.csv')
    rolling_ret['date'] = pd.to_datetime(rolling_ret['date'])
    rolling_ret_rating = pd.merge(ret_df,rolling_ret, left_on=['ticker','reviewDateTime'], right_on=['ticker','date'])
    return rolling_ret_rating

    
