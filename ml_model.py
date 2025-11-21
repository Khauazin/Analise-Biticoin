import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import warnings
import joblib
from scipy.stats import mode

# Ignora avisos irrelevantes
warnings.filterwarnings(
    "ignore",
    message="R^2 score is not well-defined with less than two samples.",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Training interrupted by user.",
)


# ======================================================
# SISTEMA DE LOGS
# ======================================================
def iniciar_logger(nome_logger="MLModel"):
    """Configura o sistema de logs (arquivo diário + console)."""
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    log_filename = datetime.now().strftime("ml_log_%Y-%m-%d.txt")
    log_path = os.path.join(logs_dir, log_filename)

    logger = logging.getLogger(nome_logger)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        fmt = "%(asctime)s - %(levelname)s - %(message)s"
        file_formatter = logging.Formatter(fmt)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(fmt)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


# ======================================================
# CLASSE PRINCIPAL
# ======================================================
class MLModel:
    def __init__(self):
        self.linear_model = LinearRegression()
        self.nn_models = []
        self.random_forest_model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='constant', fill_value=0)
        self.trained = False
        self.best_model = None
        self.best_model_name = None
        self.feature_cols = []
        self.logger = iniciar_logger("MLModel")
        # Inicializar logger para backend
        self.backend_logger = iniciar_logger("backend")
        self.last_train_time = None
        self.retrain_interval = 3600  # 1 hour in seconds
        # Force retrain for testing
        self.last_train_time = None

    def prepare_features(self, df, n_lags=5):
        df = df.copy()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        df.drop(columns=datetime_cols, inplace=True, errors='ignore')

        if 'close' not in df.columns:
            raise ValueError("O DataFrame deve conter a coluna 'close'.")

        # Alterar alvo para prever a mudanca percentual de preco futura
        # (regressao)
        df['target'] = df['close'].pct_change().shift(-1)
        df['returns'] = df['close'].pct_change()

        # Obter o preco atual para conversao de predicoes percentuais
        # para precos
        self.current_close = df['close'].iloc[-1] if not df.empty else None
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Momentum e volatilidade multi‑janela
        for w in [3, 5, 10, 20]:
            df[f'returns_{w}'] = df['close'].pct_change(w)
            df[f'vol_{w}'] = df['returns'].rolling(w).std()

        for lag in range(1, n_lags + 1):
            df[f'lag_{lag}'] = df['close'].shift(lag)
            df[f'return_lag_{lag}'] = df['returns'].shift(lag)

        df['ma_5'] = df['close'].rolling(5).mean()
        df['ma_10'] = df['close'].rolling(10).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['ema_gap'] = df['ema_12'] - df['ema_26']

        # Calcular indicadores adicionais se não presentes
        if 'volume' not in df.columns:
            df['volume'] = np.nan
        if 'volatility' not in df.columns:
            df['volatility'] = df['close'].rolling(20).std()
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))
        if 'macd' not in df.columns:
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            df['macd'] = macd_line
            df['macd_signal'] = signal_line
            df['macd_hist'] = macd_line - signal_line

        # ATR aproximado se possível
        if {'high', 'low', 'close'}.issubset(set(df.columns)):
            tr1 = (df['high'] - df['low']).abs()
            tr2 = (df['high'] - df['close'].shift(1)).abs()
            tr3 = (df['low'] - df['close'].shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr_14'] = tr.rolling(14).mean()

        df.dropna(subset=['target'], inplace=True)
        # Definir lista fixa de features para evitar inconsistências
        feature_cols = []
        # retornos/log
        feature_cols += ['returns', 'log_returns']
        # momentum/vol
        feature_cols += [f'returns_{w}' for w in [3, 5, 10, 20]]
        feature_cols += [f'vol_{w}' for w in [3, 5, 10, 20]]
        # lags
        feature_cols += [f'lag_{lag}' for lag in range(1, n_lags + 1)]
        feature_cols += [f'return_lag_{lag}' for lag in range(1, n_lags + 1)]
        # médias e MACD
        feature_cols += [
            'ma_5',
            'ma_10',
            'ema_12',
            'ema_26',
            'ema_gap',
            'macd',
            'macd_signal',
            'macd_hist',
        ]
        # indicadores
        feature_cols += ['volume', 'volatility', 'rsi', 'atr_14']
        # garantir colunas existentes
        for col in feature_cols:
            if col not in df.columns:
                df[col] = np.nan

        X = df[feature_cols].values
        y = df['target'].values

        return X, y, feature_cols

    def train(self, df):
        try:
            self.backend_logger.info("Starting ML model training")
            X, y, self.feature_cols = self.prepare_features(df)
            # Reinstanciar modelos para evitar estado antigo
            # com shapes diferentes
            self.linear_model = LinearRegression()
            self.random_forest_model = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
            )
            self.nn_models = []
            if len(X) < 2:
                self.logger.warning("⚠️ Dados insuficientes para treinamento.")
                return

            test_size = 0.2 if len(X) > 10 else (0.5 if len(X) > 5 else 0.7)
            if len(X) <= 2:
                test_size = 0.001

            if len(X) > 2:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=min(test_size, 0.5), random_state=42
                )
            else:
                X_train, y_train, X_test, y_test = X, y, X, y

            self.imputer.fit(X_train)
            X_train = self.imputer.transform(X_train)
            X_test = self.imputer.transform(X_test)

            self.scaler.fit(X_train)
            X_train = self.scaler.transform(X_train)
            X_test = self.scaler.transform(X_test)

            self.linear_model.fit(X_train, y_train)
            self.random_forest_model.fit(X_train, y_train)

            # Optimized hyperparameter grid for regression
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'lbfgs'],
                'alpha': [0.0001, 0.001],
                'max_iter': [1000, 2000, 5000],
                'early_stopping': [True]
            }

            base_nn = MLPRegressor(random_state=42)
            cv_folds = min(5, len(X_train)) if len(X_train) > 1 else 1
            # Use RandomizedSearchCV for efficiency
            random_search = RandomizedSearchCV(
                base_nn,
                param_grid,
                n_iter=25,
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=1,
                random_state=42,
            )
            random_search.fit(X_train, y_train)
            best_params = random_search.best_params_
            self.logger.info(f"Best NN params: {best_params}")

            self.nn_models = [
                MLPRegressor(**best_params, random_state=42 + i).fit(
                    X_train,
                    y_train,
                )
                for i in range(3)
            ]

            y_pred_lr = self.linear_model.predict(X_test)
            y_pred_rf = self.random_forest_model.predict(X_test)
            nn_predictions = (
                np.array([m.predict(X_test) for m in self.nn_models])
                if self.nn_models
                else np.array([np.zeros_like(y_test)])
            )
            y_pred_nn = (
                np.mean(nn_predictions, axis=0)
                if self.nn_models
                else np.zeros_like(y_test)
            )

            mse_scores = {
                'linear_regression': mean_squared_error(y_test, y_pred_lr),
                'neural_network': mean_squared_error(y_test, y_pred_nn),
                'random_forest': mean_squared_error(y_test, y_pred_rf)
            }

            scores = {k: -v for k, v in mse_scores.items()}
            # Negate MSE for maximization

            self.best_model_name = max(scores, key=scores.get)
            self.best_model = (
                self.nn_models if self.best_model_name == 'neural_network'
                else self.linear_model
                if self.best_model_name == 'linear_regression'
                else self.random_forest_model
            )
            self.trained = True
            self.last_train_time = datetime.now()

            # Initialize last_volatility after first training
            if df is not None and 'close' in df.columns:
                self.last_volatility = df['close'].pct_change().tail(20).std()

            self.backend_logger.info(
                f"ML model training completed. Best model: "
                f"{self.best_model_name}"
            )
            return {'mse': mse_scores, 'best_model': self.best_model_name}
        except Exception as e:
            self.logger.error(f"Erro crítico no treinamento ML: {e}")
            self.backend_logger.error(f"Erro crítico no treinamento ML: {e}")
            raise

    def prepare_single_input(self, data):
        if not isinstance(data, (dict, pd.Series, np.ndarray)):
            raise TypeError(
                "Entrada deve ser um dicionario, pandas.Series "
                "ou numpy.ndarray."
            )

        if isinstance(data, np.ndarray):
            df_row = pd.DataFrame([data], columns=self.feature_cols)
        elif isinstance(data, dict):
            df_row = pd.DataFrame([data])
        else:
            df_row = pd.DataFrame([data.to_dict()])

        for col in self.feature_cols:
            if col not in df_row.columns:
                df_row[col] = np.nan

        df_row = df_row[self.feature_cols]
        return self.imputer.transform(df_row.values).flatten()

    def predict(self, df_row):
        if not self.trained:
            raise Exception("O modelo ainda não foi treinado.")

        features = self.prepare_single_input(df_row)
        X_input = self.scaler.transform(features.reshape(1, -1))

        pred_lr_pct = self.linear_model.predict(X_input)[0]
        pred_rf_pct = self.random_forest_model.predict(X_input)[0]
        nn_predictions_pct = (
            np.array([m.predict(X_input)[0] for m in self.nn_models])
            if self.nn_models
            else np.array([pred_lr_pct])
        )
        pred_nn_pct = (
            np.mean(nn_predictions_pct) if self.nn_models else pred_lr_pct
        )

        pred_best_pct = (
            pred_nn_pct if self.best_model_name == 'neural_network'
            else pred_lr_pct if self.best_model_name == 'linear_regression'
            else pred_rf_pct
        )

        # Converter predições percentuais para preços reais
        current_close = df_row.get('close', self.current_close)
        if current_close is None:
            current_close = 0  # Fallback se não houver preço atual

        pred_lr = current_close * (1 + pred_lr_pct)
        pred_nn = current_close * (1 + pred_nn_pct)
        pred_rf = current_close * (1 + pred_rf_pct)
        pred_best = current_close * (1 + pred_best_pct)

        # Calibrar ml_score com acordo entre modelos
        # e magnitude vs ATR (volatilidade)
        try:
            ups = sum(
                1 for v in (pred_lr, pred_nn, pred_rf) if v > current_close
            )
            downs = 3 - ups
            agreement = max(ups, downs) / 3.0  # 0.33..1.0
            atr_ref = df_row.get('atr_14', None)
            denom = max(
                (
                    atr_ref if atr_ref and atr_ref > 0
                    else current_close * 0.005
                ),
                1e-9,
            )
            magnitude = min(abs(pred_best - current_close) / denom, 1.0)
            ml_score = float(0.6 * agreement + 0.4 * magnitude)
        except Exception:
            ml_score = 0.5

        self.backend_logger.info(f"ML prediction made: {pred_best}")
        return {
            'linear_regression_prediction': pred_lr,
            'neural_network_prediction': pred_nn,
            'random_forest_prediction': pred_rf,
            'best_model_prediction': pred_best,
            'ml_score': ml_score
        }

    def walk_forward_validate(self, df, train_size=500, test_size=100):
        """Walk-forward simples para validar direcao e retorno ajustado.
        Retorna dict com accuracy direcional e um Sharpe-like
        (mean/std de retornos previstos).
        """
        try:
            X, y, feats = self.prepare_features(df)
            n = len(y)
            if n < train_size + test_size + 10:
                return {'error': 'insufficient_data', 'n': n}
            accs = []
            rets = []
            idx = 0
            while idx + train_size + test_size <= n:
                X_tr = X[idx:idx+train_size]
                y_tr = y[idx:idx+train_size]
                X_te = X[idx+train_size:idx+train_size+test_size]
                y_te = y[idx+train_size:idx+train_size+test_size]
                # Fit lightweight models
                lr = LinearRegression().fit(X_tr, y_tr)
                rf = RandomForestRegressor(
                    n_estimators=150,
                    random_state=42,
                ).fit(X_tr, y_tr)
                # Predict percentage change
                p_lr = lr.predict(X_te)
                p_rf = rf.predict(X_te)
                p = 0.5 * p_lr + 0.5 * p_rf
                # Directional accuracy
                acc = float(np.mean(np.sign(p) == np.sign(y_te)))
                accs.append(acc)
                # Return proxy: product of predicted sign and actual returns
                rets.append(float(np.mean(np.sign(p) * y_te)))
                idx += test_size
            accuracy = float(np.mean(accs)) if accs else 0.0
            mean_ret = float(np.mean(rets)) if rets else 0.0
            std_ret = float(np.std(rets) + 1e-9)
            sharpe_like = float(mean_ret / std_ret)
            return {
                'accuracy': accuracy,
                'mean_return': mean_ret,
                'risk_adjusted': sharpe_like,
            }
        except Exception as e:
            return {'error': str(e)}

    def predict_multi_step(self, initial_features, n_steps=1):
        if not self.trained:
            raise Exception("O modelo ainda não foi treinado.")

        predictions = []
        current_features = self.prepare_single_input(initial_features)
        current_close = (
            initial_features.get('close', self.current_close)
            if isinstance(initial_features, dict)
            else self.current_close
        )
        if current_close is None:
            current_close = 0

        for _ in range(n_steps):
            X_input = self.scaler.transform(current_features.reshape(1, -1))
            if self.best_model_name == 'neural_network' and self.nn_models:
                nn_predictions_pct = np.array(
                    [m.predict(X_input)[0] for m in self.nn_models]
                )
                pred_pct = mode(nn_predictions_pct, axis=0)[0][0]
            else:
                pred_pct = self.best_model.predict(X_input)[0]

            # Converter predição percentual para preço
            pred_price = current_close * (1 + pred_pct)
            predictions.append(pred_price)

            # Atualizar features para próximo passo
            new_features = np.zeros_like(current_features)
            new_features[0] = pred_pct  # mudança percentual
            new_features[1] = current_features[0]  # lag anterior
            current_features = new_features
            current_close = pred_price  # atualizar preço para próximo cálculo

        self.backend_logger.info(
            f"ML multi-step prediction made: {len(predictions)} steps"
        )
        return np.array(predictions).flatten()

    def save_model(self, filepath=None):
        """Salva o modelo treinado em um arquivo."""
        if not self.trained:
            raise Exception("O modelo não está treinado para salvar.")
        if filepath is None:
            filepath = os.path.join(
                os.path.dirname(__file__),
                "models",
                "ml_model.pkl",
            )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'linear_model': self.linear_model,
            'nn_models': self.nn_models,
            'random_forest_model': self.random_forest_model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'trained': self.trained,
            'best_model_name': self.best_model_name,
            'feature_cols': self.feature_cols,
            'last_train_time': self.last_train_time,
            'retrain_interval': self.retrain_interval,
            'last_volatility': getattr(self, 'last_volatility', None)
        }
        joblib.dump(model_data, filepath)
        self.logger.info(f"Modelo salvo em {filepath}")

    def load_model(self, filepath=None):
        """Carrega o modelo treinado de um arquivo."""
        if filepath is None:
            filepath = os.path.join(
                os.path.dirname(__file__),
                "models",
                "ml_model.pkl",
            )
        if not os.path.exists(filepath):
            self.logger.warning(
                f"Arquivo de modelo não encontrado: {filepath}"
            )
            return False
        try:
            model_data = joblib.load(filepath)
            self.linear_model = model_data['linear_model']
            self.nn_models = model_data['nn_models']
            self.random_forest_model = model_data['random_forest_model']
            self.scaler = model_data['scaler']
            self.imputer = model_data['imputer']
            self.trained = model_data['trained']
            self.best_model_name = model_data['best_model_name']
            self.feature_cols = model_data['feature_cols']
            self.last_train_time = model_data.get('last_train_time', None)
            self.retrain_interval = model_data.get('retrain_interval', 3600)
            self.last_volatility = model_data.get('last_volatility', None)
            self.best_model = (
                self.nn_models if self.best_model_name == 'neural_network'
                else self.linear_model
                if self.best_model_name == 'linear_regression'
                else self.random_forest_model
            )
            self.logger.info(f"Modelo carregado de {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {e}")
            return False

    def should_retrain(self, df=None):
        """Verifica se o modelo deve ser retreinado.
        Decorrido ou mudancas no regime de mercado.
        """
        if self.last_train_time is None:
            return True

        elapsed = (datetime.now() - self.last_train_time).total_seconds()
        time_based_retrain = elapsed >= self.retrain_interval

        # Verificar mudanca no regime de mercado baseado na volatilidade
        if df is not None and len(df) > 20:
            current_volatility = df['close'].tail(20).pct_change().std()
            if hasattr(self, 'last_volatility'):
                volatility_change = (
                    abs(current_volatility - self.last_volatility)
                    / self.last_volatility
                )
                regime_change = volatility_change > 0.5  # 50% change
                if regime_change:
                    self.logger.info(
                        "Regime change detected. "
                        f"Volatility change: {volatility_change:.2%}"
                    )
                    self.last_volatility = current_volatility
                    return True
            else:
                self.last_volatility = current_volatility

        return time_based_retrain
