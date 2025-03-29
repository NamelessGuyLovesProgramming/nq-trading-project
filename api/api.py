from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import sys
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

# Füge Root-Verzeichnis zum Pfad hinzu für richtige Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importiere die benötigten Module aus dem bestehenden Projekt
from src.data.fetcher import DataFetcher
from src.data.processor import DataProcessor
from src.models.lstm import LSTMModel
from src.backtesting.engine import BacktestEngine
from src.backtesting.metrics import calculate_performance_metrics

# Strategien
from src.strategies.ml_strategy import MLStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.combined import CombinedStrategy
from src.strategies.volume_profile import VolumeProfileStrategy
from src.strategies.market_regime import MarketRegimeStrategy
from src.strategies.ensemble import EnsembleStrategy
from src.strategies.risk_managed import RiskManagedStrategy

# Konstanten aus config
from config import (
    DEFAULT_SYMBOL, DEFAULT_INTERVAL, DEFAULT_PERIOD,
    RSI_OVERBOUGHT, RSI_OVERSOLD, VOLUME_THRESHOLD,
    INITIAL_CAPITAL, COMMISSION, WINDOW_SIZE, EPOCHS, BATCH_SIZE
)

# Verzeichnisse für Daten und Ergebnisse
TEMP_DIR = "temp"
OUTPUT_DIR = "output"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")

# Erstelle Verzeichnisse, falls sie nicht existieren
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# Speicher für laufende Aufgaben
tasks = {}

# FastAPI App erstellen
app = FastAPI(
    title="NQ-Trading-Backtest-API",
    description="API for NQ-Trading-Backtest-Tool",
    version="1.0.0"
)

# CORS einrichten (für Frontend-Zugriff)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Für Produktionsumgebungen einschränken
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- Modelle für Request/Response -----

class DataRequest(BaseModel):
    symbol: str = DEFAULT_SYMBOL
    period: str = DEFAULT_PERIOD
    interval: str = DEFAULT_INTERVAL
    force_download: bool = False
    custom_file: Optional[str] = None
    combine_all_years: bool = False


class TrainModelRequest(BaseModel):
    data_id: str
    window_size: int = WINDOW_SIZE
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    test_size: float = 0.2


class MeanReversionParams(BaseModel):
    rsi_overbought: int = RSI_OVERBOUGHT
    rsi_oversold: int = RSI_OVERSOLD
    bb_trigger: float = 0.8


class VolumeProfileParams(BaseModel):
    volume_threshold: float = VOLUME_THRESHOLD
    lookback: int = 20


class CombinedParams(BaseModel):
    trend_weight: float = 0.4
    momentum_weight: float = 0.3
    volatility_weight: float = 0.3
    threshold: float = 0.15


class EnsembleParams(BaseModel):
    voting_method: str = "majority"


class MLParams(BaseModel):
    model_id: str
    threshold: float = 0.005
    window_size: int = WINDOW_SIZE


class RiskManagementParams(BaseModel):
    enabled: bool = False
    risk_per_trade: float = 0.01
    position_size_method: str = "fixed"
    atr_risk_multiplier: float = 1.5
    max_drawdown: float = -0.05


class BacktestRequest(BaseModel):
    data_id: str
    strategy: str
    strategy_params: Dict[str, Any] = {}
    risk_management: RiskManagementParams = RiskManagementParams()
    initial_capital: float = INITIAL_CAPITAL
    commission: float = COMMISSION


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ----- Hilfsfunktionen -----

# In-Memory Speicher für Daten und Modelle
data_cache = {}
model_cache = {}
backtest_results_cache = {}


def get_strategy(strategy_type, strategy_params, risk_management_params):
    """
    Erstellt eine Strategie basierend auf dem Typ und den Parametern.
    """
    # Erstelle die Basis-Strategie
    if strategy_type == "mean_reversion":
        params = MeanReversionParams(**strategy_params)
        strategy = MeanReversionStrategy(
            rsi_overbought=params.rsi_overbought,
            rsi_oversold=params.rsi_oversold,
            bb_trigger=params.bb_trigger
        )

    elif strategy_type == "volume":
        params = VolumeProfileParams(**strategy_params)
        strategy = VolumeProfileStrategy(
            volume_threshold=params.volume_threshold,
            lookback=params.lookback
        )

    elif strategy_type == "combined":
        params = CombinedParams(**strategy_params)
        weights = {
            'trend': params.trend_weight,
            'momentum': params.momentum_weight,
            'volatility': params.volatility_weight
        }
        strategy = CombinedStrategy(
            weights=weights,
            threshold=params.threshold
        )

    elif strategy_type == "regime":
        strategy = MarketRegimeStrategy()

    elif strategy_type == "ensemble":
        params = EnsembleParams(**strategy_params)
        # Erstelle Standard-Strategien für das Ensemble
        mr_strategy = MeanReversionStrategy()
        vol_strategy = VolumeProfileStrategy()
        combined_strategy = CombinedStrategy()

        strategy = EnsembleStrategy(
            strategies=[mr_strategy, vol_strategy, combined_strategy],
            voting_method=params.voting_method
        )

    elif strategy_type == "ml":
        params = MLParams(**strategy_params)
        if params.model_id not in model_cache:
            raise HTTPException(status_code=404, detail=f"Model with ID {params.model_id} not found")

        model = model_cache[params.model_id]

        # Bereite Daten für ML vor
        processor = DataProcessor()
        data_id = strategy_params.get("data_id")

        if data_id and data_id in data_cache:
            _, _, _, _, scaler = processor.prepare_data_for_ml(
                data_cache[data_id], window_size=params.window_size
            )
        else:
            # Wenn kein spezifischer Datensatz angegeben, verwende den letzten
            for d_id in data_cache:
                _, _, _, _, scaler = processor.prepare_data_for_ml(
                    data_cache[d_id], window_size=params.window_size
                )
                break

        strategy = MLStrategy(
            model.model, scaler, window_size=params.window_size, threshold=params.threshold
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy type: {strategy_type}"
        )

    # Wenn Risikomanagement aktiviert, umhülle die Strategie
    if risk_management_params.enabled:
        strategy = RiskManagedStrategy(
            strategy,
            max_drawdown=risk_management_params.max_drawdown,
            volatility_filter=True,
            risk_per_trade=risk_management_params.risk_per_trade,
            position_size_method=risk_management_params.position_size_method,
            atr_risk_multiplier=risk_management_params.atr_risk_multiplier
        )

    return strategy


# Background Tasks
def load_data_task(task_id, request: DataRequest):
    """Background Task zum Laden von Daten"""
    try:
        tasks[task_id] = {"status": "running", "progress": 0.1}

        # Erstelle DataFetcher-Instanz
        fetcher = DataFetcher(symbol=request.symbol)

        # Lade Daten basierend auf den Parametern
        if request.combine_all_years:
            tasks[task_id]["progress"] = 0.2
            data = fetcher.load_custom_file("nq-1m*.csv")
            tasks[task_id]["progress"] = 0.7
        elif request.custom_file:
            tasks[task_id]["progress"] = 0.2
            data = fetcher.fetch_data(
                period=request.period,
                interval=request.interval,
                force_download=request.force_download,
                custom_file=request.custom_file
            )
            tasks[task_id]["progress"] = 0.7
        elif request.period.startswith("nq-1m"):
            tasks[task_id]["progress"] = 0.2
            data = fetcher.load_nq_minute_data(request.period)
            tasks[task_id]["progress"] = 0.7
        else:
            tasks[task_id]["progress"] = 0.2
            data = fetcher.fetch_data(
                period=request.period,
                interval=request.interval,
                force_download=request.force_download
            )
            tasks[task_id]["progress"] = 0.7

        # Fehlerprüfung
        if data is None or data.empty:
            tasks[task_id] = {
                "status": "error",
                "error": "No data could be loaded. Check parameters."
            }
            return

        # Füge technische Indikatoren hinzu
        tasks[task_id]["progress"] = 0.8
        processor = DataProcessor()
        data_with_indicators = processor.add_technical_indicators(data)

        # Speichere im Cache
        data_id = str(uuid.uuid4())
        data_cache[data_id] = data_with_indicators

        # Bereite Ergebnis vor
        result = {
            "data_id": data_id,
            "symbol": request.symbol,
            "period": request.period,
            "interval": request.interval,
            "rows": len(data_with_indicators),
            "date_range": {
                "start": data_with_indicators.index[0].isoformat(),
                "end": data_with_indicators.index[-1].isoformat()
            },
            "columns": list(data_with_indicators.columns)
        }

        tasks[task_id] = {
            "status": "completed",
            "progress": 1.0,
            "result": result
        }

    except Exception as e:
        tasks[task_id] = {
            "status": "error",
            "error": str(e)
        }


def train_model_task(task_id, request: TrainModelRequest):
    """Background Task zum Trainieren eines ML-Modells"""
    try:
        tasks[task_id] = {"status": "running", "progress": 0.1}

        # Prüfe, ob Daten-ID gültig ist
        if request.data_id not in data_cache:
            tasks[task_id] = {
                "status": "error",
                "error": f"Data with ID {request.data_id} not found"
            }
            return

        data = data_cache[request.data_id]
        tasks[task_id]["progress"] = 0.2

        # Daten vorbereiten
        processor = DataProcessor()
        X_train, y_train, X_test, y_test, scaler = processor.prepare_data_for_ml(
            data, window_size=request.window_size, test_size=request.test_size
        )

        tasks[task_id]["progress"] = 0.3

        # Prüfe, ob genug Daten vorhanden sind
        if X_train.shape[0] < 10 or X_test.shape[0] < 5:
            if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                tasks[task_id] = {
                    "status": "error",
                    "error": "No training or test data available."
                }
                return

        # Modell erstellen und trainieren
        input_shape = (request.window_size, X_train.shape[2])
        model = LSTMModel(input_shape, output_dir=MODELS_DIR)
        model.build_model()

        # Callback für den Fortschritt
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress_base = 0.3  # Start bei 30%
                progress_per_epoch = 0.6 / request.epochs  # 60% für Training
                tasks[task_id]["progress"] = progress_base + (epoch + 1) * progress_per_epoch

        # Training
        history = model.train(
            X_train, y_train,
            X_test, y_test,
            epochs=request.epochs,
            batch_size=request.batch_size,
            callbacks=[ProgressCallback()]
        )

        tasks[task_id]["progress"] = 0.9

        # Vorhersagen auf Testdaten
        predictions = model.predict(X_test)

        # Leistungsmetriken berechnen
        mse = np.mean((predictions - y_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_test))
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

        # Speichere Modell in Cache
        model_id = str(uuid.uuid4())
        model_cache[model_id] = model

        # Erstelle Ergebnis
        result = {
            "model_id": model_id,
            "data_id": request.data_id,
            "window_size": request.window_size,
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "metrics": {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape)
            },
            "training_shape": list(X_train.shape),
            "testing_shape": list(X_test.shape)
        }

        tasks[task_id] = {
            "status": "completed",
            "progress": 1.0,
            "result": result
        }

    except Exception as e:
        tasks[task_id] = {
            "status": "error",
            "error": str(e)
        }


def run_backtest_task(task_id, request: BacktestRequest):
    """Background Task zum Durchführen eines Backtests"""
    try:
        tasks[task_id] = {"status": "running", "progress": 0.1}

        # Prüfe, ob Daten-ID gültig ist
        if request.data_id not in data_cache:
            tasks[task_id] = {
                "status": "error",
                "error": f"Data with ID {request.data_id} not found"
            }
            return

        data = data_cache[request.data_id]
        tasks[task_id]["progress"] = 0.2

        # Erstelle die Strategie
        strategy = get_strategy(
            request.strategy,
            request.strategy_params,
            request.risk_management
        )

        tasks[task_id]["progress"] = 0.3

        # Backtest durchführen
        backtest_engine = BacktestEngine(
            data,
            strategy,
            initial_capital=request.initial_capital,
            commission=request.commission
        )

        tasks[task_id]["progress"] = 0.4
        results = backtest_engine.run()
        tasks[task_id]["progress"] = 0.7

        # Berechne Benchmark-Renditen (Buy & Hold)
        benchmark_returns = data['Close'].pct_change()

        # Berechne erweiterte Metriken mit Benchmark
        additional_metrics = calculate_performance_metrics(
            results['returns'],
            benchmark_returns=benchmark_returns
        )

        tasks[task_id]["progress"] = 0.9

        # Formatiere die Ergebnisse für JSON-Response
        formatted_results = {
            "equity_curve": [
                {"date": date.isoformat(), "value": float(value)}
                for date, value in zip(results['portfolio_value'].index, results['portfolio_value'].values)
            ],
            "drawdown": [
                {"date": date.isoformat(), "value": float(value)}
                for date, value in zip(results['drawdown'].index, results['drawdown'].values)
            ],
            "trades": int(results['trades']),
            "win_rate": float(results['win_rate']) if results['win_rate'] is not None else None,
            "sharpe_ratio": float(results['sharpe_ratio']) if results['sharpe_ratio'] is not None else None,
            "max_drawdown": float(results['max_drawdown']),
            "total_return": float(results['portfolio_value'].iloc[-1] / request.initial_capital - 1),
            "additional_metrics": {
                k: float(v) if isinstance(v, (float, np.float32, np.float64)) else v
                for k, v in additional_metrics.items() if v is not None
            }
        }

        # Speichere Ergebnisse im Cache
        result_id = str(uuid.uuid4())
        backtest_results_cache[result_id] = {
            "results": results,
            "strategy": strategy,
            "data_id": request.data_id,
            "formatted_results": formatted_results
        }

        # Erstelle Ergebnis
        result = {
            "result_id": result_id,
            "data_id": request.data_id,
            "strategy": request.strategy,
            "results": formatted_results
        }

        tasks[task_id] = {
            "status": "completed",
            "progress": 1.0,
            "result": result
        }

    except Exception as e:
        tasks[task_id] = {
            "status": "error",
            "error": str(e)
        }


# ----- API Endpunkte -----

@app.get("/", tags=["Info"])
async def root():
    """API-Root Endpunkt"""
    return {
        "name": "NQ-Trading-Backtest-API",
        "version": "1.0.0",
        "endpoints": [
            "/api/data",
            "/api/model",
            "/api/backtest",
            "/api/task/{task_id}"
        ]
    }


@app.post("/api/data", tags=["Data"], response_model=Dict[str, Any])
async def load_data(request: DataRequest, background_tasks: BackgroundTasks):
    """Lade Daten asynchron"""
    task_id = str(uuid.uuid4())
    background_tasks.add_task(load_data_task, task_id, request)
    return {"task_id": task_id, "status": "started"}


@app.post("/api/model", tags=["Model"], response_model=Dict[str, Any])
async def train_model(request: TrainModelRequest, background_tasks: BackgroundTasks):
    """Trainiere ML-Modell asynchron"""
    if request.data_id not in data_cache:
        raise HTTPException(status_code=404, detail=f"Data with ID {request.data_id} not found")

    task_id = str(uuid.uuid4())
    background_tasks.add_task(train_model_task, task_id, request)
    return {"task_id": task_id, "status": "started"}


@app.post("/api/backtest", tags=["Backtest"], response_model=Dict[str, Any])
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Führe Backtest asynchron durch"""
    if request.data_id not in data_cache:
        raise HTTPException(status_code=404, detail=f"Data with ID {request.data_id} not found")

    # Überprüfe, ob bei ML-Strategie ein Modell angegeben ist
    if request.strategy == "ml":
        if "model_id" not in request.strategy_params:
            raise HTTPException(status_code=400, detail="model_id is required for ML strategy")

        model_id = request.strategy_params["model_id"]
        if model_id not in model_cache:
            raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")

    task_id = str(uuid.uuid4())
    background_tasks.add_task(run_backtest_task, task_id, request)
    return {"task_id": task_id, "status": "started"}


@app.get("/api/task/{task_id}", tags=["Tasks"], response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Rufe den Status einer Aufgabe ab"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found")

    return {
        "task_id": task_id,
        **tasks[task_id]
    }


@app.get("/api/data/{data_id}/preview", tags=["Data"], response_model=Dict[str, Any])
async def get_data_preview(data_id: str, rows: int = 10):
    """Rufe eine Vorschau der Daten ab"""
    if data_id not in data_cache:
        raise HTTPException(status_code=404, detail=f"Data with ID {data_id} not found")

    data = data_cache[data_id]
    preview = data.head(rows).reset_index()

    # Konvertiere DatetimeIndex zu String
    preview["Date"] = preview["Date"].astype(str)

    return {
        "data_id": data_id,
        "total_rows": len(data),
        "preview": preview.to_dict(orient="records")
    }


@app.get("/api/models", tags=["Model"], response_model=List[Dict[str, str]])
async def list_models():
    """Liste alle verfügbaren Modelle auf"""
    return [{"model_id": model_id} for model_id in model_cache.keys()]


@app.get("/api/backtest/{result_id}", tags=["Backtest"], response_model=Dict[str, Any])
async def get_backtest_result(result_id: str):
    """Rufe ein Backtest-Ergebnis ab"""
    if result_id not in backtest_results_cache:
        raise HTTPException(status_code=404, detail=f"Backtest result with ID {result_id} not found")

    return backtest_results_cache[result_id]["formatted_results"]


# Start Server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)