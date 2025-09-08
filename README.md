# Collaborative Swarm of Multi-agent LLM-Orchestrated-Systems (COSMOS)
# COSMOS Framework

COSMOS is a modular framework designed to integrate and evaluate Multi-Agent Systems with Swarm Intelligence. This project is part of a research effort for a scientific paper and a doctoral thesis.

## Abstract

This repository provides the official implementation of COSMOS (Collaborative Swarm of Multi-agent LLM-Orchestrated Systems), 
a flexible framework that brings together Swarm Intelligence and coordinated Multi-Agent Systems powered by Large Language Models. 

The COSMOS architecture includes a modular Coordinator that dynamically decomposes tasks, forms agent crews, and manages 
inter-agent communication. Aggregation mechanisms inspired by Social Choice Theory—such as plurality voting, 
weighted averages, and cognitive aggregation—enable the system to generate robust and explainable outcomes.
COSMOS is easily extensible: users can modify agents, adjust hyperparameters, and run end-to-end experiments for diverse 
tasks, including embedding and contextual answer generation, occupancy detection and forecasting, or multi-modal thermodynamic simulation. 
All scenarios, code, and configurations from the associated publication can be reproduced or extended in this repository, 
providing a hands-on platform to build or benchmark new agent-based solutions.

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
   git clone https://github.com/MartaRGR/COSMOS-Collaborative-Swarm-of-Multi-agent-LLM-Orchestrated-Systems
   cd COSMOS-Collaborative-Swarm-of-Multi-agent-LLM-Orchestrated-Systems
   
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

## Project license
This project is released under the MIT License - see the [LICENSE](./LICENSE) file for full details.

## Contributions
Contributions are welcome! If you have suggestions or improvements, please feel free to open an issue or submit a pull request.  

## Author
Marta Romero García-Rubio
