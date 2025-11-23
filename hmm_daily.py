import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import warnings
import os
warnings.filterwarnings("ignore")
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
        self.model = None
        self.regime_map = {}
    def load_data(self):
        self.assets = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        if self.assets.empty:
            raise ValueError(f"Nessun dato scaricato per {self.ticker}. Verifica il ticker e la connessione.")
        if isinstance(self.assets.columns, pd.MultiIndex):
             self.assets.columns = self.assets.columns.get_level_values(0)
        self.assets.rename(columns={
            'Close': f'{self.symbol_name}_Adj_Close', 
            'High': f'{self.symbol_name}_High', 
            'Low': f'{self.symbol_name}_Low' 
        }, inplace=True)
        
        self.first_date = self.assets.index[0].strftime('%Y-%m-%d')
        self.last_date = self.assets.index[-1].strftime('%Y-%m-%d')
        
        print(f"--- Data Range for {self.symbol_name} ---")
        print(f"First trading day: {self.first_date}")
        print(f"Last trading day: {self.last_date}")
        print(f"Total observations: {len(self.assets)}")
    def calculate_daily_returns_log(self):
        self.data = pd.DataFrame()
        adj_close = self.assets[f'{self.symbol_name}_Adj_Close']
        self.data[f'{self.symbol_name}_Log_Return'] = (np.log(
            adj_close / adj_close.shift(1))*100).iloc[1:]
        
        self.simple_returns = adj_close.pct_change().iloc[1:]
    def fit_hmm_model(self):
        print("\n--- HMM Model Results ---")
        train_size = int(0.8 * len(self.data))
        train_data = self.data[:train_size]
        self.test_data = self.data[train_size:]
        parameters = {'n_components': [2],
                      'covariance_type': ["diag"],
                      'n_iter': [100, 500, 1000], 
                      'init_params': ['smct'],
                      'params': ["mct", 'smct'], }
        model = hmm.GaussianHMM()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_search = GridSearchCV(model, parameters, cv=5)
            grid_search.fit(train_data)
        self.best_params_daily = grid_search.best_params_
        self.model = hmm.GaussianHMM(**self.best_params_daily)
        self.model.fit(train_data)
        print("Transition Matrix:")
        rounded_transmat = np.round(self.model.transmat_, decimals=4)
        print(rounded_transmat)
        self.test_score = self.model.score(self.test_data)
        self.aic = self.model.aic(self.test_data)
        self.bic = self.model.bic(self.test_data)
        print(f"AIC: {self.aic:.2f}")
        print(f"BIC: {self.bic:.2f}")
    def predict_states_daily(self):
        log_prob, state_sequence = self.model.decode(self.test_data, algorithm='viterbi')
        self.hidden_states_daily = np.array(state_sequence)
        return self.hidden_states_daily
    def states_assign_value(self):
        print("\n--- Regime Assignment ---")
        print("Codifica: Stato 0 = Regime NEGATIVO | Stato 1 = Regime POSITIVO")
        
        self.test_df = pd.DataFrame()
        self.test_df["Regimes"] = self.hidden_states_daily
        returns = np.array(self.test_data[f'{self.symbol_name}_Log_Return'])
        self.test_df['Returns'] = returns
        regime_means = self.test_df.groupby('Regimes')['Returns'].mean()
        lowest_mean_regime = regime_means.idxmin()
        highest_mean_regime = regime_means.idxmax()
        
        self.regime_map = {}
        if len(regime_means) == 2:
             self.regime_map[lowest_mean_regime] = 0
             self.regime_map[highest_mean_regime] = 1
             self.test_df['Assigned_Regime'] = np.where(self.test_df['Regimes'] == lowest_mean_regime, 0, 1)
        else:
            middle_regimes = [r for r in regime_means.index if r not in [lowest_mean_regime, highest_mean_regime]]
            if middle_regimes:
                middle_mean_regime = middle_regimes[0]
                self.regime_map[middle_mean_regime] = 2
            
            self.regime_map[lowest_mean_regime] = 0
            self.regime_map[highest_mean_regime] = 1
            self.test_df['Assigned_Regime'] = np.where(self.test_df['Regimes'] == lowest_mean_regime, 0,
                                            np.where(self.test_df['Regimes'] == highest_mean_regime, 1, 2))
    def get_regime_characteristics(self):
        print("\n--- Regime Characteristics ---")
        state_means = []
        state_Stds = []
        skewness = []
        kurtosis = []
        time = []
        count = []
        cumulative_performance = []
        for regime, group in self.test_df.groupby('Assigned_Regime'):
            state_means.append(group['Returns'].mean())
            state_Stds.append(group['Returns'].std())
            skewness.append(group['Returns'].skew())
            kurtosis.append(group['Returns'].kurtosis())
            time.append(round(len(group['Returns']) / len(self.test_data) * 100, 2))
            count.append(group['Returns'].count())
            cumulative_performance.append(group['Returns'].mean() * group['Returns'].count())
        regimes = np.sort(self.test_df['Assigned_Regime'].unique())
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
    def create_summary_plot(self):
        """Crea un grafico riassuntivo per Telegram"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        test_data_index = self.data.index[-len(self.test_data):]
        regimes = np.sort(self.test_df['Assigned_Regime'].unique())
        cmap = plt.get_cmap('brg', len(regimes))
        
        # 1. Close Price with States (Last 1 Year)
        ax1 = fig.add_subplot(gs[0, :])
        one_year_ago = datetime.now() - pd.DateOffset(years=1)
        last_year_mask = test_data_index >= one_year_ago
        
        if last_year_mask.any():
            for i in regimes:
                pos = (self.test_df['Assigned_Regime'] == i).values
                combined_mask = last_year_mask & pos
                
                if combined_mask.any():
                    ax1.scatter(test_data_index[combined_mask],
                                self.assets[f'{self.symbol_name}_Adj_Close'][-len(self.test_data):][combined_mask],
                                label=f'State {i}',
                                s=10, alpha=1, color=cmap(i))
        
        ax1.legend()
        ax1.set_title(f"{self.symbol_name} Close Price with States (Last 1 Year)", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Close Price")
        ax1.grid(True, alpha=0.3)
        
        # 2. Probability Regimes (Last 1 Year)
        probabilities_hmm = self.model.predict_proba(self.test_data)
        n_samples = probabilities_hmm.shape[0]
        n_regimes = len(self.regime_map)
        probabilities_remapped = np.zeros((n_samples, n_regimes))
        
        for hmm_regime, assigned_regime in self.regime_map.items():
            probabilities_remapped[:, assigned_regime] = probabilities_hmm[:, hmm_regime]
        
        probabilities = pd.DataFrame(probabilities_remapped)
        dates = self.test_data.index[-len(self.test_data):]
        
        if last_year_mask.any():
            dates_1y = dates[last_year_mask]
            prob_0_1y = probabilities[0][last_year_mask]
            prob_1_1y = probabilities[1][last_year_mask]
            
            # Regime 0 (NEGATIVO)
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.plot(dates_1y, prob_0_1y, color='red', linewidth=1.5)
            ax2.set_title('Regime 0 (NEGATIVO) - Last 1 Year', fontsize=10)
            ax2.set_ylabel('Probability')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            
            # Regime 1 (POSITIVO)
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.plot(dates_1y, prob_1_1y, color='green', linewidth=1.5)
            ax3.set_title('Regime 1 (POSITIVO) - Last 1 Year', fontsize=10)
            ax3.set_ylabel('Probability')
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)
        
        # 3. Regime Statistics
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Statistiche regime
        regime_stats = []
        for regime, group in self.test_df.groupby('Assigned_Regime'):
            mean_ret = group['Returns'].mean()
            std_ret = group['Returns'].std()
            days = len(group)
            pct_time = round(days / len(self.test_df) * 100, 1)
            regime_stats.append([f"Regime {regime}", f"{mean_ret:.4f}%", f"{std_ret:.4f}%", days, f"{pct_time}%"])
        
        table = ax4.table(cellText=regime_stats,
                         colLabels=['Regime', 'Mean Return', 'Std Dev', 'Days', '% Time'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0.1, 0.3, 0.8, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Colora le celle header
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Info aggiuntive
        current_regime = self.test_df['Assigned_Regime'].iloc[-1]
        regime_label = "NEGATIVO" if current_regime == 0 else "POSITIVO"
        
        info_text = f"Current Regime: {current_regime} ({regime_label})\n"
        info_text += f"Last Update: {self.last_date}\n"
        info_text += f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        ax4.text(0.5, 0.1, info_text, ha='center', va='top', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'HMM Daily Report - {self.symbol_name}', fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig('hmm_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n✅ Summary plot saved as 'hmm_plot.png'")
        
        # Salva le date in un file per il workflow
        with open('report_info.txt', 'w') as f:
            f.write(f"LAST_DATA_DATE={self.last_date}\n")
            f.write(f"REPORT_DATE={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
if __name__ == "__main__":
    print("=" * 60)
    print("HMM DAILY REPORT")
    print("=" * 60)
    
    ticker = "^GSPC"
    symbol_name = "S&P 500"
    
    try:
        print(f"\nProcessing {symbol_name}...")
        hhmm_model = HiddenMarkovModel(ticker, symbol_name)
        hhmm_model.load_data()
        hhmm_model.calculate_daily_returns_log()
        hhmm_model.fit_hmm_model()
        hhmm_model.predict_states_daily()
        hhmm_model.states_assign_value()
        hhmm_model.get_regime_characteristics()
        hhmm_model.create_summary_plot()
        
        print("\n" + "=" * 60)
        print("✅ REPORT COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
