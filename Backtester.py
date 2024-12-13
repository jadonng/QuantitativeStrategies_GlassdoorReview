
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class BackTest:
    """
    Backtester for backtesting alpha ideas

    - returns are preprocessed: columns being ticker, row being EOM date (2020-01-31) means return from 2020-02-01 - 2020-02-28

    """
    def __init__(self, features, grouping=None, start_date='2014-01-01', end_date='2022-12-30', universe=None, market_neutral=True, decay=3):
        self.company = pd.read_parquet('data/company_overview_cleaned.parquet')

        self.ret = pd.read_csv('data/universe_ret.csv',index_col=0)
        self.ret = self.ret.loc[start_date:end_date,:]

        self.features = features
        self.market_neutral=market_neutral
        self.start_date = start_date
        self.end_date = end_date
        self.decay=decay

        # result dataframes
        self.weights = None
        self.portfolio_returns = pd.DataFrame()
        self.metrics = pd.DataFrame()

        # grouping
        if isinstance(grouping, pd.DataFrame):
            # filter stocks first (universe check)
            universe = self.ret.columns.unique().values
            self.ret = self.ret.loc[:,universe]
            
            self.features = self.features.loc[:, self.features.columns.get_level_values('ticker').isin(universe)]

            # 1. group ret (assume equal weight portfolio)
            ret_merge = pd.merge(self.ret.T, grouping, left_on=self.ret.T.index, right_on='ticker')
            ret_merge.set_index('ticker',inplace=True)
            self.ret = ret_merge.groupby('group').mean().T

            # 2. group features (take mean)
            columns = self.features.columns.get_level_values(0).unique()
            df = pd.DataFrame()
            new_columns = []
            for feature in columns:
                merge = pd.merge(self.features[feature].T, grouping, left_on=self.features[feature].T.index, right_on='ticker')
                merge.set_index('ticker',inplace=True)
                merge = merge.groupby('group').mean().T
                df = pd.concat([df, merge],axis=1)
                new_columns += [(feature,col) for col in merge.columns]

            new_columns = pd.MultiIndex.from_tuples(new_columns)
            df.columns = new_columns
            self.features = df
            self.features.columns = self.features.columns.set_names(['feature', 'ticker'])


        # define universe, we do not need this if grouping is defined
        if not isinstance(grouping, pd.DataFrame):
            if universe is not None:
                universe = list(set(universe) & set(self.ret.columns.unique()))
                self.ret = self.ret.loc[:,universe]

        if universe is not None:
            self.universe = list(set(universe) & set(self.ret.columns.unique().values) & set(self.features.columns.get_level_values('ticker').unique().values))
        else:
            self.universe = list(set(self.ret.columns.unique().values) & set(self.features.columns.get_level_values('ticker').unique().values))
        
        self.features = self.features.loc[:, self.features.columns.get_level_values('ticker').isin(self.universe)]
        self.ret = self.ret.loc[:, self.universe]

    def create_alpha(self, weight):
        weight = weight.loc[self.start_date:self.end_date,:]
        self.weights = weight
    
    def run(self, display_plot=True):
        assert self.weights is not None, "Please assign weights before running with self.weight = {your defined weights}"
        
        self.weights = self.weights.loc[:,self.universe]
        if self.market_neutral:
            self.weights = self.make_market_neutral(self.weights)

        if self.decay>0:
            self.weights = self.weight_decay(self.weights,window=self.decay)

        self.portfolio_returns = self.weights.mul(self.ret).sum(axis=1)
        self.metrics = self.get_metrics()
        if display_plot:
            self.plot_pnl()
            display(self.metrics)
        return self.metrics

    # portfolio transformation functions
    
    def make_market_neutral(self, weights):
        row_means = weights.mean(axis=1, skipna=True)
        centered_weights = weights.sub(row_means, axis=0)
        abs_sum = centered_weights.abs().sum(axis=1, skipna=True)
        abs_sum = abs_sum.replace(0, np.nan)
        normalized_weights = centered_weights.div(abs_sum, axis=0)

        return normalized_weights

    def weight_decay(self, weight, window=6):
        weight = weight.rolling(window=window,min_periods=1).mean()
        return weight


    def plot_pnl(self,display_plot=True):
        fig  = plt.figure(figsize=(10, 6))
        plt.plot(self.portfolio_returns.index, (1+self.portfolio_returns).cumprod(), label='PnL')
        plt.xlabel('date')
        plt.ylabel('return')
        plt.title('PnL')
        plt.legend()
        # unique_years = pd.to_datetime(self.portfolio_returns.index).year
        # labels = [(f"{year}-01-31") for year in unique_years]
        plt.xticks(range(0, len(self.portfolio_returns.index), int(len(self.portfolio_returns.index)/5)))
        if display_plot:
            plt.show()
        else:
            plt.close(fig)

        return self.portfolio_returns.index, (1+self.portfolio_returns).cumprod()

    def get_metrics(self):
        metrics = {}
        # portfolio return
        metrics['ret'] = (1+self.portfolio_returns).cumprod()[-1]

        # std
        metrics['std'] = (self.portfolio_returns).std()

        # sharpe 
        metrics['sharpe'] = (self.portfolio_returns.mean()*12-0.0086)/((self.portfolio_returns).std()*np.sqrt(12))

        # max drawdown
        uni = self.weights.count(axis=1)
        uni=uni[uni!=0]
        metrics['universe'] = uni.mean()

        metrics_df = pd.DataFrame(metrics,index=[0])
        return metrics_df
    
    # =========================================
    # some helper function for constructing weights/alphas
    # =========================================
    def get(self, col):
        return self.features[col]
    
    # 4 arithmetic operations
    def add(self, *args):
        df = self.features[args[0]]
        for col in args[1:]:
            df+= self.features[col]
        return df
    def minus(self, A, B):
        df = self.features[A] - self.features[B]
        return df
    def mul(self, *args):
        df = self.features[args[0]]
        for col in args[1:]:
            df = df.mul(self.features[col])
        return df
    def div(self, A, B):
        df = self.features[A].div(self.features[B])
        return df

    # cross-section operations
    def rank(self, A):
        if isinstance(A, str):
            rank_df = self.features[A].rank(axis=1)
        elif isinstance(A, pd.DataFrame):
            rank_df = A.rank(axis=1)
        return rank_df

    # time-series operations
    def diff(self, col, window):
        df = self.features[col]  - self.features[col].shift(window)
        return df
    
    def SMA(self, col, window, min_periods=None):
        df = self.features[col].rolling(window=window, min_periods=min_periods).mean()
        return df



