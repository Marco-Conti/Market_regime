import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
        
        self.test_df = pd.DataFrame(index=self.test_data.index)
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
        
    def create_summary_plot(self, fig, timeframe='full'):
        """Crea un grafico riassuntivo per una pagina del PDF
        
        Args:
            fig: matplotlib figure
            timeframe: 'full' per tutti i dati, 'last_year' per ultimo anno
        """
        from matplotlib.dates import MonthLocator, DateFormatter
        
        gs = fig.add_gridspec(3, 1, hspace=0.35, wspace=0.3, height_ratios=[1.2, 0.8, 1])
        
        test_data_index = self.data.index[-len(self.test_data):]
        regimes = np.sort(self.test_df['Assigned_Regime'].unique())
        cmap = plt.get_cmap('brg', len(regimes))
        
        # Determina la maschera temporale in base al timeframe
        if timeframe == 'last_year':
            one_year_ago = datetime.now() - pd.DateOffset(years=1)
            time_mask = test_data_index >= one_year_ago
            title_suffix = "(Last 1 Year)"
            period_label = "Last 1 Year"
        else:  # 'full'
            time_mask = np.array([True] * len(test_data_index))
            title_suffix = "(Full History)"
            period_label = "Full History"
        
        # 1. Close Price with States
        ax1 = fig.add_subplot(gs[0])
        
        if time_mask.any():
            for i in regimes:
                pos = (self.test_df['Assigned_Regime'] == i).values
                combined_mask = time_mask & pos
                
                if combined_mask.any():
                    ax1.scatter(test_data_index[combined_mask],
                                self.assets[f'{self.symbol_name}_Adj_Close'][-len(self.test_data):][combined_mask],
                                label=f'State {i}',
                                s=10, alpha=1, color=cmap(i))
        
        ax1.legend()
        ax1.set_title(f"{self.symbol_name} Close Price with States {title_suffix}", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Close Price")
        ax1.grid(True, alpha=0.3)
        
        # Formattazione date mensile solo per last_year
        if timeframe == 'last_year':
            ax1.xaxis.set_major_locator(MonthLocator())
            ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Probability Regime 0 (NEGATIVO) - larghezza piena
        probabilities_hmm = self.model.predict_proba(self.test_data)
        n_samples = probabilities_hmm.shape[0]
        n_regimes = len(self.regime_map)
        probabilities_remapped = np.zeros((n_samples, n_regimes))
        
        for hmm_regime, assigned_regime in self.regime_map.items():
            probabilities_remapped[:, assigned_regime] = probabilities_hmm[:, hmm_regime]
        
        probabilities = pd.DataFrame(probabilities_remapped)
        dates = self.test_data.index[-len(self.test_data):]
        
        if time_mask.any():
            dates_filtered = dates[time_mask]
            prob_0_filtered = probabilities[0][time_mask]
            
            # Regime 0 (NEGATIVO) - occupa tutta la larghezza
            ax2 = fig.add_subplot(gs[1])
            ax2.plot(dates_filtered, prob_0_filtered, color='red', linewidth=1.5)
            ax2.set_title(f'Regime 0 (NEGATIVO) Probability - {period_label}', fontsize=10)
            ax2.set_ylabel('Probability')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            
            # Formattazione date mensile solo per last_year
            if timeframe == 'last_year':
                ax2.xaxis.set_major_locator(MonthLocator())
                ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Regime Statistics - spostata più in alto
        ax3 = fig.add_subplot(gs[2])
        ax3.axis('off')
        
        # Filtra i dati in base al timeframe per le statistiche
        if timeframe == 'last_year':
            # Crea un DataFrame filtrato per l'ultimo anno
            one_year_ago = datetime.now() - pd.DateOffset(years=1)
            test_df_filtered = self.test_df[self.test_df.index >= one_year_ago].copy()
        else:
            test_df_filtered = self.test_df.copy()
        
        # Statistiche regime (basate sul timeframe)
        regime_stats = []
        for regime, group in test_df_filtered.groupby('Assigned_Regime'):
            mean_ret = group['Returns'].mean()
            std_ret = group['Returns'].std()
            days = len(group)
            pct_time = round(days / len(test_df_filtered) * 100, 1)
            regime_stats.append([f"Regime {regime}", f"{mean_ret:.4f}%", f"{std_ret:.4f}%", days, f"{pct_time}%"])
        
        table = ax3.table(cellText=regime_stats,
                         colLabels=['Regime', 'Mean Return', 'Std Dev', 'Days', '% Time'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0.1, 0.45, 0.8, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Colora le celle header
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Info regime corrente - spostato più in alto
        current_regime = self.test_df['Assigned_Regime'].iloc[-1]
        regime_label = "NEGATIVO" if current_regime == 0 else "POSITIVO"
        regime_color = 'red' if current_regime == 0 else 'green'
        
        regime_text = f"Current Regime: {current_regime} ({regime_label})"
        
        ax3.text(0.5, 0.15, regime_text, ha='center', va='center', 
                fontsize=16, fontweight='bold', color=regime_color,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
                         edgecolor=regime_color, linewidth=2.5, alpha=0.9))
        
        plt.suptitle(f'HMM Daily Report - {self.symbol_name} {title_suffix}', fontsize=16, fontweight='bold', y=0.98)
def process_all_tickers():
    """Processa tutti i ticker e crea un PDF con 2 pagine per ticker (full + last year)"""
    
    # Dizionario dei ticker
    tickers = {
        "^GSPC": "S&P 500",
        "SPY": "SPY ETF (S&P 500)",
        "^IXIC": "NASDAQ",
        "^VIX": "VIX",
        "^RUT": "Russell 2000",
        "^DJI": "Dow Jones",
        "^NYA": "NYSE COMPOSITE",
        "^GDAXI": "DAX",
        "^N225": "NIKKEI 225",
        "^HSI": "HANG SENG",
    }
    
    # Crea la cartella di output se non esiste
    output_dir = os.path.expanduser('~/Desktop/Report HMM')
    os.makedirs(output_dir, exist_ok=True)
    
    # Nome del file PDF con data
    today_date = datetime.now().strftime('%Y-%m-%d')
    pdf_filename = f'HMM_Report_{today_date}.pdf'
    pdf_path = os.path.join(output_dir, pdf_filename)
    
    print("=" * 60)
    print("HMM MULTI-TICKER REPORT")
    print("=" * 60)
    print(f"\nProcessing {len(tickers)} tickers...")
    print(f"1 page per ticker (Last Year Only)")
    print(f"Total pages: {len(tickers)}")
    print(f"Output directory: {output_dir}")
    print(f"Output file: {pdf_filename}\n")
    
    # Verifica se il file esiste già
    if os.path.exists(pdf_path):
        print(f"⚠️  File già esistente - verrà sovrascritto: {pdf_filename}\n")
    
    # Crea il PDF
    with PdfPages(pdf_path) as pdf:
        for idx, (ticker, symbol_name) in enumerate(tickers.items(), 1):
            print(f"\n{'=' * 60}")
            print(f"[{idx}/{len(tickers)}] Processing {symbol_name} ({ticker})...")
            print(f"{'=' * 60}")
            
            try:
                # Crea e processa il modello HMM
                hhmm_model = HiddenMarkovModel(ticker, symbol_name)
                hhmm_model.load_data()
                hhmm_model.calculate_daily_returns_log()
                hhmm_model.fit_hmm_model()
                hhmm_model.predict_states_daily()
                hhmm_model.states_assign_value()
                hhmm_model.get_regime_characteristics()
                
                # PAGINA: Last Year
                print(f"  📊 Creating page: Last Year...")
                fig2 = plt.figure(figsize=(16, 10))
                hhmm_model.create_summary_plot(fig2, timeframe='last_year')
                pdf.savefig(fig2, dpi=150, bbox_inches='tight')
                plt.close(fig2)
                
                print(f"✅ {symbol_name} completed (1 page added to PDF)")
                
            except Exception as e:
                print(f"❌ Error processing {symbol_name}: {e}")
                import traceback
                traceback.print_exc()
                
                # Crea una pagina di errore
                fig = plt.figure(figsize=(16, 10))
                ax = fig.add_subplot(111)
                ax.axis('off')
                error_text = f"Error processing {symbol_name} ({ticker})\n\n{str(e)}"
                ax.text(0.5, 0.5, error_text, ha='center', va='center', 
                       fontsize=14, color='red',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                plt.suptitle(f'Error - {symbol_name}', fontsize=16, fontweight='bold', color='red')
                pdf.savefig(fig, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                continue
    
    print("\n" + "=" * 60)
    print("✅ MULTI-TICKER REPORT COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\n📄 PDF report saved:")
    print(f"   Location: {output_dir}")
    print(f"   Filename: {pdf_filename}")
    print(f"   Full path: {pdf_path}")
    print(f"   Total pages: {len(tickers)} (1 per ticker)")
    
    # Salva le informazioni del report
    info_file = os.path.join(output_dir, 'last_report_info.txt')
    with open(info_file, 'w') as f:
        f.write(f"REPORT_DATE={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"TOTAL_TICKERS={len(tickers)}\n")
        f.write(f"PAGES_PER_TICKER=1\n")
        f.write(f"TOTAL_PAGES={len(tickers)}\n")
        f.write(f"PDF_FILENAME={pdf_filename}\n")
        f.write(f"PDF_PATH={pdf_path}\n")
if __name__ == "__main__":
    process_all_tickers()
