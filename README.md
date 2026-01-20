# Collaborative Swarm of Multi-agent LLM-Orchestrated-Systems (COSMOS)
# COSMOS Framework

COSMOS is a modular framework designed to integrate and evaluate Multi-Agent Systems with Swarm Intelligence. 
This project is part of a research effort for a scientific paper and a doctoral thesis.

## Abstract

This repository contains the official implementation of COSMOS (Collaborative Swarm of Multi-agent LLM-Orchestrated Systems), 
a modular framework for orchestrating heterogeneous Multi-Agent Systems powered by Large Language Models and inspired by 
principles of Swarm Intelligence.

The COSMOS architecture features a centralized coordinator that dynamically decomposes tasks, assembles agent crews, and 
manages their execution. Aggregation mechanisms inspired by Social Choice Theory -such as plurality voting, weighted averaging, 
and cognitive aggregation- are used to synthesize agent outputs into robust and explainable decisions.

The framework is designed to be easily extensible: users can integrate new agents, adjust configurations and hyperparameters, 
and run end-to-end experiments across a range of tasks, including embedding-based retrieval and contextual answer generation, 
and multimodal thermodynamic simulation. All experimental scenarios, code, and configurations described in the associated 
publication can be reproduced and extended using this repository, providing a practical platform for developing and benchmarking 
agent-based solutions.

## Features
- **Multi-Agent Systems**: Implement and evaluate various multi-agent strategies.
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
