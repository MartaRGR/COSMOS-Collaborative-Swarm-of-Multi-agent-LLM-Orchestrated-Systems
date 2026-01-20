import os
import pandas as pd

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from src.utils.required_inputs_catalog import REQUIRED_INPUTS_CATALOG
from src.utils.models_catalog import MODELS_CATALOG
from src.utils.setup_logger import get_agent_logger
from src.utils.base_agent import BaseAgent


AGENT_METADATA = {
    "function": "Predict the temperature evolution of a room based on the room's occupancy forecast",
    "required_inputs": [
        REQUIRED_INPUTS_CATALOG["task_definition"],
        REQUIRED_INPUTS_CATALOG["forecast_horizon"],
        REQUIRED_INPUTS_CATALOG["initial_temperature"],
        REQUIRED_INPUTS_CATALOG["outside_temperature"],
        REQUIRED_INPUTS_CATALOG["room_volume"]
    ],
    "output": "temperature evolution",
    "class": "ThermodynamicPredictionAgent",
    "models": [
        MODELS_CATALOG["thermodynamic_prediction"]
    ]
}


class ThermodynamicPredictionAgent(BaseAgent):
    def _setup_agent(self, model_name: str, crew_id: int):
        self.logger = get_agent_logger(f"Thermodynamic Prediction Agent - Crew {crew_id} - Model {model_name} - Hyperparameters {self.hyperparameters}")
        self.model_name = model_name


    def run(self, input_data, dt_min=60, max_substep=5):
        try:
            from io import StringIO
            from datetime import datetime, timedelta
            import math

            # Parse occupancy forecast string input into a DataFrame
            occupancy_forecast_str = input_data["occupancy_forecast"]
            forecast_df = pd.read_csv(StringIO(occupancy_forecast_str), delim_whitespace=True).reset_index()
            if 'index' in forecast_df.columns:
                forecast_df['ds'] = forecast_df['index'].astype(str) + " " + forecast_df['ds']
                forecast_df.drop(columns=['index'], inplace=True)

            # Parse initial timestamp from the first row of the DataFrame
            initial_time = datetime.strptime(forecast_df.loc[0, 'ds'], "%Y-%m-%d %H:%M:%S")

            # Calculate number of steps per time block based on minute steps
            num_substeps = max(1, math.ceil(dt_min / max_substep))
            dt_sub = dt_min / num_substeps

            # Calculate mass of the air in the room and surface area
            room_volume = float(input_data["room_volume"])
            air_mass = room_volume * self.hyperparameters["air_density"]
            surface_area = 6 * (room_volume ** (2 / 3))  # assuming cube shape for simplicity

            outside_temperature = float(input_data["outside_temperature"])

            results = {}
            timestamps = {}
            # Identify scenario columns (all except 'ds')
            occ_scenarios = [col for col in forecast_df.columns if col != "ds"]
            temp_scenarios = {occ_scenario: occ_scenario.split("_")[0] + "_temperature_evolution" for occ_scenario in
                              occ_scenarios}

            for scenario in occ_scenarios:
                temperature_evolution = []
                timestamp_list = []
                temperature = float(input_data["initial_temperature"])
                current_time = initial_time

                for _, row in forecast_df.iterrows():
                    occupants = row[scenario]

                    # Instead of 1 step of dt_min, do num_substeps of dt_sub each
                    for _ in range(num_substeps):
                        heat_generated = occupants * self.hyperparameters["person_heat"]
                        dQ_persons = heat_generated * 60 * dt_sub
                        ventilation_volume = self.hyperparameters["ACH"] * room_volume / 60 * dt_sub
                        mass_outside_air = ventilation_volume * self.hyperparameters["air_density"]
                        dQ_vent = mass_outside_air * self.hyperparameters["c_p"] * (
                                outside_temperature - temperature
                        )
                        Q_walls = self.hyperparameters["U"] * surface_area * (
                                temperature - outside_temperature) * 60 * dt_sub
                        dT = (dQ_persons + dQ_vent - Q_walls) / (air_mass * self.hyperparameters["c_p"])
                        temperature += dT

                    temperature_evolution.append(round(temperature, 2))
                    timestamp_list.append(current_time)
                    current_time += timedelta(minutes=dt_min)

                results[temp_scenarios[scenario]] = temperature_evolution
                timestamps[temp_scenarios[scenario]] = timestamp_list

            df_timestamps = pd.DataFrame({'timestamp': timestamps[temp_scenarios[occ_scenarios[0]]]})

            # Adding columns for each scenario with the temperature evolution
            for scenario in occ_scenarios:
                df_timestamps[temp_scenarios[scenario]] = results[temp_scenarios[scenario]]

            self.logger.info(f">>> Task result:\\n{df_timestamps}")
            return {"temperature_evolution": df_timestamps.to_string(index=False)}

        except Exception as e:
            self.logger.error(f"Failed to run {self.model_name}: {e}")
            raise


if __name__ == "__main__":
    config = {
        "model": "thermodynamic_prediction",
        "hyperparameters": {
            "c_p": 1005,  # Specific heat capacity of air in J/(kg·K)
            "air_density": 1.225,  # Air density in kg/m³ at sea level and 15°C
            "person_heat": 100,  # Power generated per person in watts
            "ACH": 1.0,  # Air Changes per Hour,
            "U": 0.5  # W/(m²·K), typical heat transfer coefficient
        }
    }
    agent = ThermodynamicPredictionAgent("crew_1", config)

    result = agent.run({
        "occupancy_forecast": "                 ds  base_occupancy_prediction  pessimistic_occupancy_prediction  optimistic_occupancy_prediction\n2025-09-01 13:00:00                        7.0                               4.0                             10.0\n2025-09-01 14:00:00                        5.0                               2.0                              8.0\n2025-09-01 15:00:00                        4.0                               2.0                              7.0",
        "initial_temperature": 22.0,
        "outside_temperature": 30.0,
        "room_volume": 50.0,
        "forecast_horizon": 3
    })
    print(result)