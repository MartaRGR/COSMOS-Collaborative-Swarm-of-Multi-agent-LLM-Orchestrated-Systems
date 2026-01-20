import os
import pandas as pd

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from src.utils.required_inputs_catalog import REQUIRED_INPUTS_CATALOG
from src.utils.models_catalog import MODELS_CATALOG
from src.utils.setup_logger import get_agent_logger
from src.utils.base_agent import BaseAgent


AGENT_METADATA = {
    "function": "Forecast occupancy based on detected people and historical data",
    "required_inputs": [
        REQUIRED_INPUTS_CATALOG["task_definition"],
        REQUIRED_INPUTS_CATALOG["forecast_horizon"]
    ],
    "output": "occupancy_forecast time series",
    "class": "OccupancyForecastAgent",
    "models": [
        MODELS_CATALOG["prophet"]
    ]
}


class OccupancyForecastAgent(BaseAgent):
    def _setup_agent(self, model_name: str, crew_id: int):
        self.logger = get_agent_logger(f"Occupancy Forecast Agent - Crew {crew_id} - Model {model_name} - Hyperparameters {self.hyperparameters}")
        self.model_name = model_name

        self.df = self._read_data()

        self.model = self._initialize()

    @staticmethod
    def _read_data(
            csv_path=r"C:\Users\rt01306\OneDrive - Telefonica\Desktop\Doc\Doctorado\TESIS\Python_Code\prueba_langraph\src\agents\test2 - Time Series Forecasting\data\historical_occupancy.csv"
    ):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Historical CSV not found at {csv_path}")
        df = pd.read_csv(csv_path, sep=",", parse_dates=["ds"])
        if "y" not in df.columns:
            raise ValueError("CSV must have columns: ds (datetime), y (count)")
        return df

    def _initialize(self):
        from prophet import Prophet
        return Prophet(
            weekly_seasonality=self.hyperparameters["weekly_seasonality"],
            seasonality_prior_scale=self.hyperparameters["seasonality_prior_scale"],
            changepoint_prior_scale=self.hyperparameters["changepoint_prior_scale"],
            seasonality_mode=self.hyperparameters["seasonality_mode"]
        )


    def run(self, input_data):
        try:
            result = self._get_dispatch_entry()["run"](input_data)
            self.logger.info(f">>> Task result:\n{result}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to run {self.model_name}: {e}")
            raise


if __name__ == "__main__":
    config = {
        "model": "prophet",
        "hyperparameters": {
            "weekly_seasonality": 'auto',
            "seasonality_prior_scale": 5,
            "changepoint_prior_scale": 0.1,
            "seasonality_mode": "multiplicative"
        }
    }
    agent = OccupancyForecastAgent("crew_1", config)

    result = agent.run({
        "occupancy": {"ds": "2025-08-01 22:00:00", "y": 20},
        "periods": 5
    })
    print(result)