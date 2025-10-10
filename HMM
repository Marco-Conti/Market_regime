import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import os
import yfinance as yf
from datetime import datetime

np.random.seed(100)

class HiddenMarkovModel:
    def __init__(self, ticker, symbol_name):
        self.assets = None
        self.data = None
        self.test_data = None
        self.test_score = None
        self.best_params_daily = None
        self.start_date = "1900-01-01"
        self.end_date = datetime.today().strftime('%Y-%m-%d')
        self.ticker = ticker
        self.symbol_name = symbol_name
        self.sma = None
        self.test_data = None

    def load_data(self):
        self.assets = yf.download(self.ticker, start=self.start_date, end=self.end_date)

        if self.assets.empty:
            raise ValueError(f"Nessun dato scaricato per {self.ticker}. Verifica il ticker e la connessione.")

        # Correzione dei nomi delle colonne
        self.assets.rename(columns={
            'Close': f'{self.symbol_name}_Adj_Close', #rinomino colonna
        }, inplace=True)


    def calculate_daily_returns_log(self):
        global simple_returns

        self.data = pd.DataFrame()
        self.data[f'{self.symbol_name}_Log_Return'] = (np.log(
            self.assets[f'{self.symbol_name}_Adj_Close'] / self.assets[f'{self.symbol_name}_Adj_Close'].shift(
                1))*100).iloc[1:] #calcolo rendimento logaritmici da dare in input per il modello

        simple_returns = self.assets[f'{self.symbol_name}_Adj_Close'].pct_change().iloc[1:]  #calcolo rendimenti semplici

    def fit_hmm_model(self):
        train_size = int(0.8 * len(self.data))
        train_data = self.data[:train_size]
        self.test_data = self.data[train_size:]

        parameters = {'n_components': [2],
                      'covariance_type': ["diag"],
                      'n_iter': [100, 500, 1000, 2500, 5000, 10000, 50000],
                      'init_params': ['smct'],
                      'params': ["mct", 'smct'], }

        model = hmm.GaussianHMM()

        grid_search = GridSearchCV(model, parameters, cv=10)
        grid_search.fit(train_data)

        self.best_params_daily = grid_search.best_params_

        self.model = hmm.GaussianHMM(**self.best_params_daily)
        self.model.fit(train_data)

    def predict_states_daily(self):
        global hidden_states_daily
        score_test = np.array(self.model.decode(self.test_data, algorithm='viterbi')[0])
        hidden_states_daily = np.array(self.model.decode(self.test_data, algorithm='viterbi')[1])
        return hidden_states_daily

    def states_assign_value(self):
        global test

        # Build dataframe with predicted regimes and returns
        test = pd.DataFrame()
        test["Regimes"] = hidden_states_daily
        returns = np.array(self.test_data[f'{self.symbol_name}_Log_Return'])
        test['Returns'] = returns

        # Calculate mean returns for each regime
        regime_means = test.groupby('Regimes')['Returns'].mean()

        # Ensure Regime 0 is bearish (lower mean) and Regime 1 is bullish (higher mean)
        regime_means_sorted = regime_means.sort_values()
        bearish_regime = regime_means_sorted.index[0]  # Lowest mean (bearish)
        bullish_regime = regime_means_sorted.index[-1]  # Highest mean (bullish)

        # Assign regimes: 0 for bearish, 1 for bullish
        test['Assigned_Regime'] = np.where(test['Regimes'] == bearish_regime, 0,
                                          np.where(test['Regimes'] == bullish_regime, 1, test['Regimes']))

    def calculate_wma(self):
        # Calculate 100-day and 200-day Weighted Moving Averages
        self.assets[f'{self.symbol_name}_WMA_100'] = self.assets[f'{self.symbol_name}_Adj_Close'].ewm(span=100, adjust=False).mean()
        self.assets[f'{self.symbol_name}_WMA_200'] = self.assets[f'{self.symbol_name}_Adj_Close'].ewm(span=200, adjust=False).mean()

    def plot_daily_close_prices_with_states(self):
        plt.figure(figsize=(15, 8))
        test_data_index = self.data.index[-len(self.test_data):]

        regimes = np.sort(test['Assigned_Regime'].unique())

        # Define custom colors: red for Regime 0 (bearish), green for Regime 1 (bullish)
        colors = {0: '#FF0000', 1: '#00FF00'}  # Red for bearish, green for bullish

        print("Observation (days) per regime:")
        for i in regimes:
            pos = (test['Assigned_Regime'] == i).values
            num_obs = np.sum(pos)
            print(f"Regime {i}: {num_obs} days")

            plt.scatter(test_data_index[pos],
                        self.assets[f'{self.symbol_name}_Adj_Close'][-len(self.test_data):][pos],
                        label=f'Regime {i} {"(Bearish)" if i == 0 else "(Bullish)"}',
                        s=6, alpha=1, color=colors[i])

        plt.legend()
        plt.title(f"{self.symbol_name} Close Price with Decoded States and WMAs (Test Data) - Daily")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.show()

    def get_regime_characteristics(self):
        global regime_df

        state_means = []
        state_Stds = []
        skewness = []
        kurtosis = []
        time = []
        count = []
        cumulative_performance = []

        for regime, group in test.groupby('Assigned_Regime'):
            state_means.append(group['Returns'].mean())
            state_Stds.append(group['Returns'].std())
            skewness.append(group['Returns'].skew())
            kurtosis.append(group['Returns'].kurtosis())
            time.append(round(len(group['Returns']) / len(self.test_data) * 100, 2))
            count.append(group['Returns'].count())
            cumulative_performance.append(group['Returns'].mean() * group['Returns'].count())

        regimes = np.sort(test['Assigned_Regime'].unique())

        regime_df = pd.DataFrame({'Regimes': regimes,
                                  'Mean': state_means,
                                  'StD': state_Stds,
                                  'Skewness': skewness,
                                  'Kurtosis': kurtosis,
                                  'Time %': time,
                                  'Days': count,
                                  'Cumul. Perf': cumulative_performance})

        regime_df = regime_df.sort_values(by='Mean')

        print(f"Printing regime characteristics for {self.symbol_name}")
        print(regime_df.to_string(index=False))


    def plot_probability_regimes(self):
        probabilities = self.model.predict_proba(self.test_data)
        probabilities = pd.DataFrame(probabilities)
        probabilities['Assigned_Regime'] = test['Assigned_Regime']

        cmap = plt.get_cmap('coolwarm', len(probabilities.columns[:-1]))

        for i in probabilities.columns[:-1]:
            plt.figure(figsize=(12, 3))
            plt.plot(self.test_data.index[-len(self.test_data):], probabilities[i],
                     color=cmap(i))
            plt.title(f"Probability Regime {i} for {self.symbol_name}")
            plt.xlabel("Time")
            plt.ylabel("Probability")
            plt.ylim(0, 1)
            plt.show()

        print()
        print("Printing evolution of Regime 0 probabilities in the last 5 days:")
        print(probabilities[0].iloc[-5:])


if __name__ == "__main__":
    tickers = {
        "^GSPC": "S&P 500",  # S&P 500 Index
        "^RUT": "Russell 2000",  # Russell 2000 Index
        "^DJI": "Dow Jones",  # Alternative ticker for Dow Jones
        "^NYA": "NYSE COMPOSITE",  # NYSE Composite Index
        "^GDAXI": "DAX",  # DAX (Germany)
        "^VIX": "VIX"
    }



    models = {}
    for ticker, symbol_name in tickers.items():
        hhmm_model = HiddenMarkovModel(ticker, symbol_name)
        hhmm_model.load_data()
        hhmm_model.calculate_daily_returns_log()
        hhmm_model.fit_hmm_model()
        hidden_states = hhmm_model.predict_states_daily()
        hhmm_model.states_assign_value()
        hhmm_model.get_regime_characteristics()
        hhmm_model.plot_daily_close_prices_with_states()
        hhmm_model.plot_probability_regimes()
