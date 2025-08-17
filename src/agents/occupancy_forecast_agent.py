import os
import pandas as pd

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from src.utils.required_inputs_catalog import REQUIRED_INPUTS_CATALOG
from src.utils.models_catalog import MODELS_CATALOG
from src.utils.setup_logger import get_agent_logger
from src.utils.base_agent import BaseAgent


AGENT_METADATA = {
    "function": "Forecast occupancy from historical time series",
    "required_inputs": [
        REQUIRED_INPUTS_CATALOG["task_definition"],
        REQUIRED_INPUTS_CATALOG["forecast_horizon_min"]
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

        self.model = self._initialize()

        self.csv_path = self.hyperparameters.get(
            "historical_csv_path", "data/historical_occupancy.csv"
        )
        self.prophet_params = self.hyperparameters.get("prophet_params", {
            "daily_seasonality": True,
            "seasonality_mode": "additive",
            "changepoint_prior_scale": 0.05,
            "yearly_seasonality": True,
            "weekly_seasonality": True
        })

    def _initialize(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Historical CSV not found at {self.csv_path}")
        self.df = pd.read_csv(self.csv_path, parse_dates=["ds"])
        if "y" not in self.df.columns:
            raise ValueError("CSV must have columns: ds (datetime), y (count)")
        return self._get_dispatch_entry()["init"]()

    def run(self, input_data):
        try:

            horizon_min = int(input_data.get("forecast_horizon_min", 60))
            freq_min = int(input_data.get("freq_min", 5))
            periods = int(horizon_min / freq_min)

            self.logger.info(f"Forecasting {horizon_min} min ahead with step {freq_min} min")

            result = self._get_dispatch_entry()["run"](
                input_data={
                    "historical_data": self.df,
                    "periods": periods,
                    "freq": f"{freq_min}min",
                    "prophet_params": self.prophet_params
                }
            )

            self.logger.info(f">>> Task result:\n{result}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to run {self.model_name}: {e}")
            raise


if __name__ == "__main__":
    config = {
        "model": "prophet",
    }
    agent = OccupancyForecastAgent("crew_1", config)

    result = agent.run(config)
    print(result)