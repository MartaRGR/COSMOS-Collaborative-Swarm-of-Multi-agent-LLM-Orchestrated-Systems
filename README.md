# COSMOS Framework

COSMOS is a modular framework designed to integrate and evaluate Multi-agent Systems with Swarm Intelligence. This project is part of a research effort for a scientific paper and a doctoral thesis.

## Abstract

COSMOS is a comprehensive framework that integrates Multi-agent Systems and Swarm Intelligence to tackle complex tasks in machine learning. By combining these two paradigms, COSMOS aims to enhance the performance and adaptability of models across various applications. The framework supports a range of tasks such as occupancy forecasting, object recognition, and depth analysis, showcasing its potential for real-world scenarios.
Besides, COSMOS incorporates swarm intelligence techniques and answer aggregation methods to optimize decision-making processes in multi-agent environments and improve traceability and interpretability of results.
The framework emphasizes flexibility and scalability, allowing researchers to customize hyperparameters, preprocess data dynamically, and incorporate new models with minimal effort. Its predictive capabilities are demonstrated through applications  COSMOS is designed to support reproducible research, making it a valuable tool for advancing the state of the art in machine learning and its applications.

## Features
- **Multi-agent Systems**: Implement and evaluate various multi-agent strategies.
- **Swarm Intelligence**: Leverage swarm algorithms for enhanced problem-solving.
- **Dynamic Preprocessing**: Adapt data preprocessing steps based on task requirements.
- **Hyperparameter Tuning**: Easily adjust and optimize model parameters.
- **Task Flexibility**: Support for diverse tasks including classification, regression, and forecasting.
- **Modularity**: Easily integrate new models and customize workflows.
- **Reproducibility**: Ensure consistent results across experiments.

## Requirements

- **Python**: 3.10 or higher
- **Dependencies**: Listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/MartaRGR/cosmos-framework.git
   cd cosmos-framework
   
2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4. Set up environment variables for API keys if using OpenAI or Azure AI Foundry models.

5. Configure the framework by editing the `config.yaml` file to specify system settings.

6. Run registration script to register agents:
    ```bash
    python registry_creator_agent.py
    ```

7. Run the framework:
    ```bash
    python coordinator_agent.py
    ```

## Project Structure
- src/: Contains the project's source code.
- tools/: Phoenix tracing in case you want to use it.
- requirements.txt: Project dependencies.

## LICENSE: Project license.
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details. 

## Contributions
Contributions are welcome! Please open an issue or submit a pull request to suggest improvements.  

## Author
Marta Romero García-Rubio