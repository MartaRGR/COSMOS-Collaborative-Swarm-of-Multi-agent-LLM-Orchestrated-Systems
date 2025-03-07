import numpy as np
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict, List, Dict, Annotated
import random
from datetime import datetime
import operator
import logging
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.graph import MermaidDrawMethod

# Pydantic models
class Particle(TypedDict):
    position: Dict[str, float]
    velocity: Dict[str, float]
    best_position: Dict[str, float]
    best_score: float

class CrewState(TypedDict):
    particles: List[Particle]
    best_position: Dict[str, float]
    best_score: float
    last_updated: datetime

class SwarmState(TypedDict):
    crews: Annotated[List[CrewState], operator.add]
    # TODO: poner para que se escriban a la vez y entonces hacer el máximo
    global_best: Annotated[Dict[str, float], operator.add]
    global_best_score: float
    iteration: int
    convergence: float
    end: bool


HYPERPARAM_SPACE = {
    'learning_rate': (0.001, 0.1),
    'dropout_rate': (0.1, 0.5),
    'batch_size': (16, 128)
}

class PSOManager:
    def __init__(self, w=0.8, c1=1.4, c2=1.4):
        self.w = w
        self.c1 = c1
        self.c2 = c2

    @staticmethod
    def initialize_particle() -> Particle:
        """Initialize a particle with random position and velocity"""
        position = {k: random.uniform(v[0], v[1]) for k, v in HYPERPARAM_SPACE.items()}
        # TODO: Dividimos entre 10 para reducir velocidad por valor de hiperparámetros ---> ver cómo normalizar cuando hay gran variación
        velocity = {k: random.uniform(-(v[1] - v[0]) / 10, (v[1] - v[0]) / 10) for k, v in HYPERPARAM_SPACE.items()}
        return {
            'position': position,
            'velocity': velocity,
            'best_position': position.copy(),
            'best_score': float('inf')
        }

    def update_velocity(self, particle: Particle, global_best: Dict[str, float]) -> Dict[str, float]:
        """Update velocity according to PSO equations"""
        new_velocity = {}
        for param in HYPERPARAM_SPACE:
            r1, r2 = random.random(), random.random()
            cognitive = self.c1 * r1 * (particle['best_position'][param] - particle['position'][param])
            social = self.c2 * r2 * (global_best[param] - particle['position'][param])
            new_velocity[param] = self.w * particle['velocity'][param] + cognitive + social
        return new_velocity

    @staticmethod
    def update_position(position: Dict[str, float], velocity: Dict[str, float]) -> Dict[str, float]:
        """Update the position by applying constraints"""
        new_pos = {}
        for param, value in position.items():
            new_val = value + velocity[param]
            min_val, max_val = HYPERPARAM_SPACE[param]
            new_pos[param] = np.clip(new_val, min_val, max_val)
        return new_pos


class ImageRecognitionEvaluator:
    def __init__(self, max_latency=10.0):
        self.max_acceptable_latency = max_latency

    def evaluate(self, particle_position, accuracy: float, latency: float) -> float:
        """Función de evaluación simulada (implementar con modelos reales)"""
        # Simula las métricas de un modelo real
        accuracy = random.uniform(0.7, 0.95)  # Precisión simulada
        latency = random.uniform(0.1, 2.0)  # Latencia simulada
        accuracy_penalty = (1 - accuracy) ** 2  # cuadrática
        latency_norm = np.clip(latency / self.max_acceptable_latency, 0, 1)  # latencia normalizada
        return (0.8 * accuracy_penalty) + (0.2 * latency_norm)


class SwarmOptimizer:
    def __init__(self, num_crews=3, particles_per_crew=5, max_iterations=100, convergence_threshold=0.01):
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger("GraphCreatorAgent")

        self.num_crews = num_crews
        self.particles_per_crew = particles_per_crew
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        self.pso = PSOManager()
        self.evaluator = ImageRecognitionEvaluator()

        self.memory = MemorySaver()
        self.graph = self.create_graph()
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)

    def initialize_crews(self, state: SwarmState) -> SwarmState:
        """Initializes the crews and assigns heterogeneous agents"""
        state['crews'] = [{
            'particles': [self.pso.initialize_particle() for _ in range(self.particles_per_crew)],
            'best_position': {},
            'best_score': float('inf'),
            'last_updated': datetime.now()
        } for _ in range(self.num_crews)]

        state['global_best'] = {}
        state['global_best_score'] = float('inf')
        state['iteration'] = 0
        state['convergence'] = 1.0 # max convergencia según la función de evaluación definida por accuracy y latency
        return state

    def evaluate_crew_for_id(self, crew_id):
        def evaluate(state: SwarmState):
            """Evaluates particles within a specific crew based on its ID."""
            crew = state['crews'][crew_id]
            for particle in crew['particles']:
                # TODO: chequar la función de evaluación
                score = self.evaluator.evaluate(particle['position'], None, None)

                # Actualiza mejores puntuaciones y posiciones del particle
                if score < particle.get('best_score', float('inf')):
                    particle['best_score'] = score
                    particle['best_position'] = particle['position'].copy()

                # Actualiza mejor posición del crew
                if score < crew['best_score']:
                    crew['best_score'] = score
                    crew['best_position'] = particle['position'].copy()
                    crew['last_updated'] = datetime.now()

                # Actualiza el mejor global
                if score < state['global_best_score']:
                    state['global_best_score'] = score
                    state['global_best'] = particle['position'].copy()
            return state

        return evaluate

    def update_crew_for_id(self, crew_id):
        def update(state: SwarmState):
            """Updates the velocities and positions of particles within a specific crew."""
            crew = state['crews'][crew_id]
            for particle in crew['particles']:
                particle['velocity'] = self.pso.update_velocity(
                    particle,
                    crew['best_position'],
                    state['global_best']
                )
                particle['position'] = self.pso.update_position(
                    particle['position'],
                    particle['velocity']
                )
            return state

        return update

    def check_convergence_for_id(self, crew_id):
        def check(state: SwarmState):
            """Calculates convergence for a specific crew."""
            crew = state['crews'][crew_id]
            positions = [list(particle['position'].values()) for particle in crew['particles']]

            # Calcula la divergencia promedio (desviación estándar)
            crew['convergence'] = np.mean(np.std(positions, axis=0))
            return state

        return check

    def finish_crew_for_id(self, crew_id):
        def finish(state: SwarmState):
            """Checks if a crew has converged and stops optimization."""
            crew = state['crews'][crew_id]
            if crew['convergence'] < self.convergence_threshold or state['iteration'] >= self.max_iterations:
                state['end'] = True
            return state
        return finish

    def create_graph(self):
        self.logger.info("Starting the dynamic creation of the graph...")
        graph = StateGraph(SwarmState)
        # Initial configuration for the start node
        graph.add_node("initialize_crews", self.initialize_crews)
        graph.set_entry_point("initialize_crews")

        # Create nodes and edges for each crew
        for crew_id in range(self.num_crews):
            evaluate_node = f"evaluate_crew_{crew_id}"
            update_node = f"update_crew_{crew_id}"
            check_node = f"check_convergence_crew_{crew_id}"
            finish_node = f"finish_crew_{crew_id}"

            graph.add_node(evaluate_node, self.evaluate_crew_for_id(crew_id))
            graph.add_node(update_node, self.update_crew_for_id(crew_id))
            graph.add_node(check_node, self.check_convergence_for_id(crew_id))

            # Unión de nodos específicos del crew
            graph.add_edge("initialize_crews", evaluate_node)
            graph.add_edge(evaluate_node, update_node)
            graph.add_edge(update_node, check_node)

            graph.add_conditional_edges(
                check_node,
                lambda state: "continue" if not state["crews"][crew_id]["finished_detection"] else "stop",
                {"continue": evaluate_node, "stop": END}
            )

        return graph


if __name__ == "__main__":
    # Create a SwarmOptimizer instance
    optimizer = SwarmOptimizer(num_crews=3, particles_per_crew=5, max_iterations=100, convergence_threshold=0.01)

    # Run the optimizer
    file_path = "graph_SI.png"
    graph = optimizer.compiled_graph
    graph_png_data = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    with open(file_path, "wb") as file:
        file.write(graph_png_data)
    print(f"Graph successfully saved in: {file_path}")

    # Run the agent
    state = SwarmState({"end": False})
    # Thread
    thread = {"configurable": {"thread_id": "1"}}
    while not state["end"]:
        for event in graph.stream(state, thread, stream_mode="values"):
            print(event)
            state = event

    # Print the results for testing
    print("Optimization finished!")


    #
    #
    # def initialize_crews(self, state: SwarmState) -> SwarmState:
    #     """Initializes the crews and assigns heterogeneous agents"""
    #     state['crews'] = [{
    #         'particles': [self.pso.initialize_particle() for _ in range(self.particles_per_crew)],
    #         'best_position': {},
    #         'best_score': float('inf'),
    #         'last_updated': datetime.now()
    #     } for _ in range(self.num_crews)]
    #
    #     state['global_best'] = {}
    #     state['global_best_score'] = float('inf')
    #     state['iteration'] = 0
    #     return state
    #
    # # def create_crew_subgraph(self, crew_id: str):
    # #     """Creates a subgraph to execute optimization for each crew"""
    # #     # PSO Subgraph for the current crew
    # #     crew_graph = StateGraph(SwarmState)
    # #
    # #     # Subgraph-specific nodes
    # #     crew_graph.add_node("evaluate_particles", lambda state: self.evaluate_crew(state, crew_id))
    # #     crew_graph.add_node("update_particles", lambda state: self.update_crew(state, crew_id))
    # #     crew_graph.add_node("check_convergence", lambda state: self.check_crew_convergence(state, crew_id))
    # #
    # #     # Connections within the subgraph
    # #     crew_graph.set_entry_point("evaluate_particles")
    # #     crew_graph.add_edge("evaluate_particles", "update_particles")
    # #     crew_graph.add_edge("update_particles", "check_convergence")
    # #
    # #     crew_graph.add_conditional_edges(
    # #         "check_convergence",
    # #         lambda state: self.decide_crew_convergence(state, crew_id),
    # #         {"continue": "evaluate_particles", "stop": END}
    # #     )
    # #
    # #     return crew_graph.compile()
    #
    # def evaluate_crew(self, state: SwarmState, crew_id: int) -> SwarmState:
    #     """Evaluates particles within a specific crew"""
    #     crew = state['crews'][crew_id]
    #     for particle in crew['particles']:
    #         score = self.evaluator.evaluate(particle['position'])
    #
    #         # Update personal best position
    #         if score < particle['best_score']:
    #             particle['best_score'] = score
    #             particle['best_position'] = particle['position'].copy()
    #
    #         # Update crew's best position
    #         if score < crew['best_score']:
    #             crew['best_score'] = score
    #             crew['best_position'] = particle['position'].copy()
    #             crew['last_updated'] = datetime.now()
    #     return state
    #
    # def update_crew(self, state: SwarmState, crew_id: int) -> SwarmState:
    #     """Updates velocities and positions for particles in a crew"""
    #     crew = state['crews'][crew_id]
    #     for particle in crew['particles']:
    #         particle['velocity'] = self.pso.update_velocity(
    #             particle,
    #             crew['best_position'],
    #             state['global_best']
    #         )
    #         particle['position'] = self.pso.update_position(
    #             particle['position'],
    #             particle['velocity']
    #         )
    #     return state
    #
    # def check_crew_convergence(self, state: SwarmState, crew_id: int) -> SwarmState:
    #     """Calculates the convergence metric for a specific crew"""
    #     crew = state['crews'][crew_id]
    #     all_positions = [particle['position'].values() for particle in crew['particles']]
    #     crew['convergence'] = np.mean(np.std(all_positions, axis=0))
    #     return state
    #
    # def decide_crew_convergence(self, state: SwarmState, crew_id: int) -> str:
    #     """Decides whether to continue or stop optimization for a crew"""
    #     crew = state['crews'][crew_id]
    #     if crew['convergence'] < 0.01 or state['iteration'] >= 100:
    #         return "stop"
    #     return "continue"
    #
    # def run_crew_subgraph(self, crew_id: int):
    #     """Runs the subgraph for a specific crew"""
    #     subgraph = self.create_crew_subgraph
    #     response = subgraph.invoke({"crew_id": crew_id})
    #     return response
