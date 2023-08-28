
# Part 1. Task Discription
# We train a DRL agent for stock trading. This task is modeled as a Markov Decision Process (MDP), and the objective function is maximizing (expected) cumulative return.
# 
# We specify the state-action-reward as follows:
# 
# * **State s**: The state space represents an agent's perception of the market environment. Just like a human trader analyzing various information, here our agent passively observes many features and learns by interacting with the market environment (usually by replaying historical data).
# 
# * **Action a**: The action space includes allowed actions that an agent can take at each state. For example, a ∈ {−1, 0, 1}, where −1, 0, 1 represent
# selling, holding, and buying. When an action operates multiple shares, a ∈{−k, ..., −1, 0, 1, ..., k}, e.g.. "Buy
# 10 shares of AAPL" or "Sell 10 shares of AAPL" are 10 or −10, respectively
# 
# * **Reward function r(s, a, s′)**: Reward is an incentive for an agent to learn a better policy. For example, it can be the change of the portfolio value when taking a at state s and arriving at new state s',  i.e., r(s, a, s′) = v′ − v, where v′ and v represent the portfolio values at state s′ and s, respectively
# 
# 
# **Market environment**: 30 consituent stocks of Dow Jones Industrial Average (DJIA) index. Accessed at the starting date of the testing period.
# 
# 
# The data for this case study is obtained from Yahoo Finance API. The data contains Open-High-Low-Close price and volume.
# 


# # Part 2. Install Python Packages
# ## 2.1. Install packages

# install required packages
# !pip install swig
# !pip install wrds
# !pip install pyportfolioopt
# install finrl library
# !pip install -q condacolab
# import condacolab
# condacolab.install()
# !apt-get update - y - qq & & apt-get install - y - qq cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx swig
# !pip install git+https://github.com/AI4Finance-Foundation/FinRL.git


# ## 2.2. A list of Python packages 
# * Yahoo Finance API
# * pandas
# * numpy
# * matplotlib
# * stockstats
# * OpenAI gym
# * stable-baselines
# * tensorflow
# * pyfolio

# ## 2.3. Import Packages
import itertools
import sys
from pprint import pprint
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from finrl.meta.data_processor import DataProcessor
from stable_baselines3.common.logger import configure
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
# from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.data_processor_crypto import DataProcessor
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
sys.path.append("../FinRL")


# ## 2.4. Create Folders
from finrl import config
from finrl import config_tickers
import os
from finrl.main import check_and_make_directories
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)
check_and_make_directories(
    [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

# # Part 3. Download Data
# Yahoo Finance provides stock data, financial news, financial reports, etc. Yahoo Finance is free.
# * FinRL uses a class **YahooDownloader** in FinRL-Meta to fetch data via Yahoo Finance API
# * Call Limit: Using the Public API (without authentication), you are limited to 2,000 requests per hour per IP (or up to a total of 48,000 requests a day).
# 
# 
# -----
# class YahooDownloader:
#     Retrieving daily stock data from
#     Yahoo Finance API
# 
#     Attributes
#     ----------
#         start_date : str
#             start date of the data (modified from config.py)
#         end_date : str
#             end date of the data (modified from config.py)
#         ticker_list : list
#             a list of stock tickers (modified from config.py)
# 
#     Methods
#     -------
#     fetch_data()
# 


TRAIN_START_DATE = '2020-09-01'
TRAIN_END_DATE = '2023-01-01'
TRADE_START_DATE = '2023-01-02'
TRADE_END_DATE = '2023-08-24'
TIME_INTERVAL='1d'
INDICATORS = ['macd', 'rsi', 'cci', 'dx']
TICKER_LIST = ['BTCUSDT','ETHUSDT']
DATA_SOURCE = 'binance'

DP = DataProcessor(data_source=DATA_SOURCE, start_date=TRAIN_START_DATE,
                       end_date=TRAIN_END_DATE, time_interval=TIME_INTERVAL)

price_array, tech_array, turbulence_array = DP.run(ticker_list=TICKER_LIST,
                                                       technical_indicator_list=INDICATORS,
                                                       if_vix=False, cache=True)


# df = YahooDownloader(start_date=TRAIN_START_DATE,
#                      end_date=TRADE_END_DATE,
#                      ticker_list=config_tickers.DOW_30_TICKER).fetch_data()

# import pandas as pd

# Specify the path to your dataset
file_path = "/home/wyq/FinRL/data/dataset.csv"

# rename time as date
# Load the newly provided dataset
new_dataset_2 = pd.read_csv(file_path)

# Check if the column is named 'time' or 'date'
column_name = 'time' if 'time' in new_dataset_2.columns else 'date'

# Rename the column to 'date' for uniformity
new_dataset_2.rename(columns={column_name: 'date'}, inplace=True)

# Convert the 'date' column to datetime format and add the 'day' column
new_dataset_2['date'] = pd.to_datetime(new_dataset_2['date'])
date_mapping_2 = {date: i for i, date in enumerate(sorted(new_dataset_2['date'].unique()))}
new_dataset_2['day'] = new_dataset_2['date'].map(date_mapping_2)

# Save the updated dataset to a CSV file
file_path = "/home/wyq/FinRL/data/dataset.csv"
new_dataset_2.to_csv(file_path, index=False)

# Load the dataset into a pandas DataFrame
df = pd.read_csv(file_path)




df.shape



df['tic']



df.sort_values(['date', 'tic'], ignore_index=True).head()


# # Part 4: Preprocess Data
# We need to check for missing data and do feature engineering to convert the data point into a state.
# * **Adding technical indicators**. In practical trading, various information needs to be taken into account, such as historical prices, current holding shares, technical indicators, etc. Here, we demonstrate two trend-following technical indicators: MACD and RSI.
# * **Adding turbulence index**. Risk-aversion reflects whether an investor prefers to protect the capital. It also influences one's trading strategy when facing different market volatility level. To control the risk in a worst-case scenario, such as financial crisis of 2007–2008, FinRL employs the turbulence index that measures extreme fluctuation of asset price.


fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_vix=True,
    use_turbulence=False,
    user_defined_feature=False)

processed = fe.preprocess_data(df)
processed



list_ticker = processed["tic"].unique().tolist()
list_ticker



list_date = list(pd.date_range(
    processed['date'].min(), processed['date'].max()).astype(str))

list_date



combination = list(itertools.product(list_date, list_ticker))
combination



processed['date'] = processed['date'].astype(str)
processed['date']
processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
    processed, on=["date", "tic"], how="left")
processed['date']
# processed_full



list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(
    processed['date'].min(), processed['date'].max()).astype(str))
combination = list(itertools.product(list_date, list_ticker))

processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
    processed, on=["date", "tic"], how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date', 'tic'])

processed_full = processed_full.fillna(0)



processed_full.sort_values(['date', 'tic'], ignore_index=True).head(10)



mvo_df = processed_full.sort_values(['date', 'tic'], ignore_index=True)[
    ['date', 'tic', 'close']]


# # Part 5. Build A Market Environment in OpenAI Gym-style
# The training process involves observing stock price change, taking an action and reward's calculation. By interacting with the market environment, the agent will eventually derive a trading strategy that may maximize (expected) rewards.
# 
# Our market environment, based on OpenAI Gym, simulates stock markets with historical market data.


# ## Data Split
# We split the data into training set and testing set as follows:
# 
# Training data period: 2009-01-01 to 2020-07-01
# 
# Trading data period: 2020-07-01 to 2021-10-31
# 


train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
print(len(train))
print(len(trade))



stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")



buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}


e_train_gym = StockTradingEnv(df=train, **env_kwargs)



# ## Environment for Training

env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))



# # Part 6: Train DRL Agents
# * The DRL algorithms are from **Stable Baselines 3**. Users are also encouraged to try **ElegantRL** and **Ray RLlib**.
# * FinRL includes fine-tuned standard DRL algorithms, such as DQN, DDPG, Multi-Agent DDPG, PPO, SAC, A2C and TD3. We also allow users to
# design their own DRL algorithms by adapting these DRL algorithms.

agent = DRLAgent(env=env_train)

if_using_a2c = True
if_using_ddpg = True
if_using_ppo = True
if_using_td3 = True
if_using_sac = True



# ### Agent Training: 5 algorithms (A2C, DDPG, PPO, TD3, SAC)
# ### Agent 1: A2C
agent = DRLAgent(env=env_train)
model_a2c = agent.get_model("a2c")

if if_using_a2c:
    # set up logger
    tmp_path = RESULTS_DIR + '/a2c'
    new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_a2c.set_logger(new_logger_a2c)



trained_a2c = agent.train_model(model=model_a2c,
                                tb_log_name='a2c',
                                total_timesteps=50000) if if_using_a2c else None



# ### Agent 2: DDPG
agent = DRLAgent(env=env_train)
model_ddpg = agent.get_model("ddpg")

if if_using_ddpg:
    # set up logger
    tmp_path = RESULTS_DIR + '/ddpg'
    new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_ddpg.set_logger(new_logger_ddpg)



trained_ddpg = agent.train_model(model=model_ddpg,
                                 tb_log_name='ddpg',
                                 total_timesteps=50000) if if_using_ddpg else None



# ### Agent 3: PPO
agent = DRLAgent(env=env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
}
model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)

if if_using_ppo:
    # set up logger
    tmp_path = RESULTS_DIR + '/ppo'
    new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_ppo.set_logger(new_logger_ppo)



trained_ppo = agent.train_model(model=model_ppo,
                                tb_log_name='ppo',
                                total_timesteps=50000) if if_using_ppo else None



# ### Agent 4: TD3
agent = DRLAgent(env=env_train)
TD3_PARAMS = {"batch_size": 100,
              "buffer_size": 1000000,
              "learning_rate": 0.001}

model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)

if if_using_td3:
    # set up logger
    tmp_path = RESULTS_DIR + '/td3'
    new_logger_td3 = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_td3.set_logger(new_logger_td3)



trained_td3 = agent.train_model(model=model_td3,
                                tb_log_name='td3',
                                total_timesteps=50000) if if_using_td3 else None



# ### Agent 5: SAC
agent = DRLAgent(env=env_train)
SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}

model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)

if if_using_sac:
    # set up logger
    tmp_path = RESULTS_DIR + '/sac'
    new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_sac.set_logger(new_logger_sac)



trained_sac = agent.train_model(model=model_sac,
                                tb_log_name='sac',
                                total_timesteps=50000) if if_using_sac else None



# ## In-sample Performance
# Assume that the initial capital is $1,000,000.


# ### Set turbulence threshold
# Set the turbulence threshold to be greater than the maximum of insample turbulence data. If current turbulence index is greater than the threshold, then we assume that the current market is volatile
data_risk_indicator = processed_full[(processed_full.date < TRAIN_END_DATE) & (
    processed_full.date >= TRAIN_START_DATE)]
insample_risk_indicator = data_risk_indicator.drop_duplicates(subset=['date'])



# insample_risk_indicator.vix.describe()



# insample_risk_indicator.vix.quantile(0.996)



# insample_risk_indicator.turbulence.describe()



# insample_risk_indicator.turbulence.quantile(0.996)



# ### Trading (Out-of-sample Performance)
# 
# We update periodically in order to take full advantage of the data, e.g., retrain quarterly, monthly or weekly. We also tune the parameters along the way, in this notebook we use the in-sample data from 2009-01 to 2020-07 to tune the parameters once, so there is some alpha decay here as the length of trade date extends. 
# 
# Numerous hyperparameters – e.g. the learning rate, the total number of samples to train on – influence the learning process and are usually determined by testing some variations.


e_trade_gym = StockTradingEnv(
    df=trade, **env_kwargs)
# env_trade, obs_trade = e_trade_gym.get_sb_env()

trained_moedl = trained_a2c
df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
    model=trained_moedl,
    environment=e_trade_gym)


trained_moedl = trained_ddpg
df_account_value_ddpg, df_actions_ddpg = DRLAgent.DRL_prediction(
    model=trained_moedl,
    environment=e_trade_gym)


trained_moedl = trained_ppo
df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
    model=trained_moedl,
    environment=e_trade_gym)


trained_moedl = trained_td3
df_account_value_td3, df_actions_td3 = DRLAgent.DRL_prediction(
    model=trained_moedl,
    environment=e_trade_gym)


trained_moedl = trained_sac
df_account_value_sac, df_actions_sac = DRLAgent.DRL_prediction(
    model=trained_moedl,
    environment=e_trade_gym)


# # Part 6.5: Mean Variance Optimization
fst = mvo_df
fst = fst.iloc[0*29:0*29+29, :]
tic = fst['tic'].tolist()

mvo = pd.DataFrame(columns=tic)

for k in range(len(tic)):
    mvo[tic[k]] = 0

# for i in range(mvo_df.shape[0]//29):
#     n = mvo_df
#     n = n.iloc[i*29:i*29+29, :]
#     date = n['date'][i*29]
#     mvo.loc[date] = n['close'].tolist()

for i in range(mvo_df.shape[0]//29):
    n = mvo_df
    n = n.iloc[i*29:i*29+29, :]
    date = n['date'].iloc[0]
    mvo.loc[date] = pd.Series(n['close'].values, index=tic)



# ### Helper functions
from scipy import optimize
from scipy.optimize import linprog

# function obtains maximal return portfolio using linear programming


def MaximizeReturns(MeanReturns, PortfolioSize):

    # dependencies

    c = (np.multiply(-1, MeanReturns))
    A = np.ones([PortfolioSize, 1]).T
    b = [1]
    res = linprog(c, A_ub=A, b_ub=b, bounds=(0, 1), method='simplex')

    return res


def MinimizeRisk(CovarReturns, PortfolioSize):

    def f(x, CovarReturns):
        func = np.matmul(np.matmul(x, CovarReturns), x.T)
        return func

    def constraintEq(x):
        A = np.ones(x.shape)
        b = 1
        constraintVal = np.matmul(A, x.T)-b
        return constraintVal

    xinit = np.repeat(0.1, PortfolioSize)
    cons = ({'type': 'eq', 'fun': constraintEq})
    lb = 0
    ub = 1
    bnds = tuple([(lb, ub) for x in xinit])

    opt = optimize.minimize(f, x0=xinit, args=(CovarReturns),  bounds=bnds,
                            constraints=cons, tol=10**-3)

    return opt


def MinimizeRiskConstr(MeanReturns, CovarReturns, PortfolioSize, R):

    def f(x, CovarReturns):

        func = np.matmul(np.matmul(x, CovarReturns), x.T)
        return func

    def constraintEq(x):
        AEq = np.ones(x.shape)
        bEq = 1
        EqconstraintVal = np.matmul(AEq, x.T)-bEq
        return EqconstraintVal

    def constraintIneq(x, MeanReturns, R):
        AIneq = np.array(MeanReturns)
        bIneq = R
        IneqconstraintVal = np.matmul(AIneq, x.T) - bIneq
        return IneqconstraintVal

    xinit = np.repeat(0.1, PortfolioSize)
    cons = ({'type': 'eq', 'fun': constraintEq},
            {'type': 'ineq', 'fun': constraintIneq, 'args': (MeanReturns, R)})
    lb = 0
    ub = 1
    bnds = tuple([(lb, ub) for x in xinit])

    opt = optimize.minimize(f, args=(CovarReturns), method='trust-constr',
                            x0=xinit,   bounds=bnds, constraints=cons, tol=10**-3)

    return opt



def StockReturnsComputing(StockPrice, Rows, Columns):
    import numpy as np

    if Rows <= 1:
        raise ValueError("The input data must have more than one row.")

    StockReturn = np.zeros([Rows-1, Columns])
    for j in range(Columns):        # j: Assets
        for i in range(Rows-1):     # i: Daily Prices
            StockReturn[i, j] = (
                (StockPrice[i+1, j]-StockPrice[i, j])/StockPrice[i, j]) * 100

    return StockReturn



# ### Calculate mean returns and variance-covariance matrix


# Obtain optimal portfolio sets that maximize return and minimize risk

# Dependencies
import numpy as np
import pandas as pd

# input k-portfolio 1 dataset comprising 15 stocks
# StockFileName = './DJIA_Apr112014_Apr112019_kpf1.csv'

Rows = 1259  # excluding header
Columns = 15  # excluding date
portfolioSize = 29  # set portfolio size

# read stock prices in a dataframe
# df = pd.read_csv(StockFileName,  nrows= Rows)

# extract asset labels
# assetLabels = df.columns[1:Columns+1].tolist()
# print(assetLabels)

# extract asset prices
# StockData = df.iloc[0:, 1:]
# StockData = mvo.head(mvo.shape[0]-336)
StockData = mvo.head(mvo.shape[0])
TradeData = mvo.tail(336)
# df.head()
TradeData.to_numpy()



arStockPrices = np.asarray(StockData)
[Rows, Cols] = arStockPrices.shape
Rows, Cols


# compute asset returns
arStockPrices = np.asarray(StockData)
[Rows, Cols] = arStockPrices.shape
arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)


# compute mean returns and variance covariance matrix of returns
meanReturns = np.mean(arReturns, axis=0)
covReturns = np.cov(arReturns, rowvar=False)

# set precision for printing results
np.set_printoptions(precision=3, suppress=True)

# display mean returns and variance-covariance matrix of returns
print('Mean returns of assets in k-portfolio 1\n', meanReturns)
print('Variance-Covariance matrix of returns\n', covReturns)



from pypfopt.efficient_frontier import EfficientFrontier

ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
raw_weights_mean = ef_mean.max_sharpe()
cleaned_weights_mean = ef_mean.clean_weights()
mvo_weights = np.array([1000000 * cleaned_weights_mean[i] for i in range(29)])
mvo_weights



StockData.tail(1)



LastPrice = np.array([1/p for p in StockData.tail(1).to_numpy()[0]])
Initial_Portfolio = np.multiply(mvo_weights, LastPrice)
Initial_Portfolio



Portfolio_Assets = TradeData @ Initial_Portfolio
MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])
MVO_result



# # Part 7: Backtesting Results
# Backtesting plays a key role in evaluating the performance of a trading strategy. Automated backtesting tool is preferred because it reduces the human error. We usually use the Quantopian pyfolio package to backtest our trading strategies. It is easy to use and consists of various individual plots that provide a comprehensive image of the performance of a trading strategy.


df_result_a2c = df_account_value_a2c.set_index(df_account_value_a2c.columns[0])
df_result_ddpg = df_account_value_ddpg.set_index(
    df_account_value_ddpg.columns[0])
df_result_td3 = df_account_value_td3.set_index(df_account_value_td3.columns[0])
df_result_ppo = df_account_value_ppo.set_index(df_account_value_ppo.columns[0])
df_result_sac = df_account_value_sac.set_index(df_account_value_sac.columns[0])
df_account_value_a2c.to_csv("df_account_value_a2c.csv")
# baseline stats
print("==============Get Baseline Stats===========")
df_dji_ = get_baseline(
    ticker="^DJI",
    start=TRADE_START_DATE,
    end=TRADE_END_DATE)
stats = backtest_stats(df_dji_, value_col_name='close')
df_dji = pd.DataFrame()
df_dji['date'] = df_account_value_a2c['date']
df_dji['account_value'] = df_dji_['close'] / \
    df_dji_['close'][0] * env_kwargs["initial_amount"]
df_dji.to_csv("df_dji.csv")
df_dji = df_dji.set_index(df_dji.columns[0])
df_dji.to_csv("df_dji+.csv")

result = pd.merge(df_result_a2c, df_result_ddpg, left_index=True,
                  right_index=True, suffixes=('_a2c', '_ddpg'))
result = pd.merge(result, df_result_td3, left_index=True,
                  right_index=True, suffixes=('', '_td3'))
result = pd.merge(result, df_result_ppo, left_index=True,
                  right_index=True, suffixes=('', '_ppo'))
result = pd.merge(result, df_result_sac, left_index=True,
                  right_index=True, suffixes=('', '_sac'))
# result = pd.merge(result, MVO_result, left_index=True, right_index=True)
result = pd.merge(result, df_dji, left_index=True, right_index=True)
result.columns = ['a2c', 'ddpg', 'td3', 'ppo', 'sac',  'dji']
# result.columns = ['a2c', 'ddpg', 'td3', 'ppo','dji']
print("result: ", result)
result.to_csv("result.csv")

# %matplotlib inline
plt.rcParams["figure.figsize"] = (15, 5)
plt.figure()
result.plot()

# Save the plot in the current directory
plt.savefig("result_plot.png")
