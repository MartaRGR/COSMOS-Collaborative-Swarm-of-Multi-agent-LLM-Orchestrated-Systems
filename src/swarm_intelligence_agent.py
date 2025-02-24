from typing import TypedDict, List, Dict, Callable, Optional
import asyncio
import json
import random
import numpy as np
from pydantic import BaseModel
from langgraph.graph import StateGraph
import time
from datetime import datetime
from agents.object_detection_agent import ObjectDetection


# Pydantic models
class PSOState(TypedDict):
    iteration: int
    best_position: list
    best_score: float
    particles: list

class CrewState(BaseModel):
    models: List[str]
    hyperparameters: Dict[str, float]
    velocity: Dict[str, float]
    pbest: Dict[str, float]
    pbest_score: float = float('inf')
    last_used: datetime = datetime.now()

class SwarmState(BaseModel):
    crews: List[CrewState]
    gbest: Dict[str, float]
    gbest_score: float = float('inf')
    q_table: Dict[str, List[float]] = {}
    iteration: int = 0
    diversity: float = 1.0
    performance_logs: List[Dict] = []

# Reinforce Learning
class QLearningController:
    def __init__(self, state_size: int, action_size: int):
        self.alpha = 0.1
        self.gamma = 0.6
        self.epsilon = 0.1
        self.q_table = np.zeros((state_size, action_size))

    def get_action(self, state: int) -> int:
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.q_table.shape[1] - 1)
        return np.argmax(self.q_table[state])

    def update_q(self, state: int, action: int, reward: float, next_state: int):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        self.q_table[state, action] = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)

# Crews' Subgraph
class CrewSubgraph(CrewState):
    def __init__(self, crew_id: int, models: List[str], action_space: List[str]):
        super().__init__(name=f"crew_{crew_id}")
        self.models = models
        self.action_space = action_space
        self.rl_controller = QLearningController(state_size=10, action_size=len(action_space))

        # Añadir nodos específicos
        self.add_node("select_action", self.select_action)
        self.add_node("execute_models", self.execute_models)
        self.add_node("update_policy", self.update_policy)

        # Definir flujo
        self.add_edge("select_action", "execute_models")
        self.add_edge("execute_models", "update_policy")

    async def select_action(self, state: SwarmState) -> SwarmState:
        current_state = self._get_state_representation(state)
        action = self.rl_controller.get_action(current_state)
        state.current_action = self.action_space[action]
        return state

    async def execute_models(self, state: SwarmState) -> SwarmState:
        crew = next(c for c in state.crews if c.id == self.name)
        image_processor = MetaImageProcessor(crew.models)
        results = await image_processor.process(state.current_image)
        crew.last_score = self._calculate_merit(results)
        return state

    async def update_policy(self, state: SwarmState) -> SwarmState:
        current_state = self._get_state_representation(state)
        next_state = (current_state + 1) % 10  # Simplified state progression
        reward = self._calculate_reward(state)
        self.rl_controller.update_q(current_state, state.current_action, reward, next_state)
        return state

    def _calculate_merit(self, results: List[Dict]) -> float:
        """Merit function that combines precision, latency and diversity."""
        avg_acc = np.mean([r['accuracy'] for r in results])
        avg_lat = np.mean([r['latency'] for r in results])
        diversity = len({r['prediction'] for r in results}) / len(results)
        return (0.5 * avg_acc) + (0.3 * (1 - avg_lat)) + (0.2 * diversity)


class MetaImageProcessor:
    def __init__(self, model_configs: List[str]):
        self.models = [ObjectDetection(config) for config in model_configs]

    async def process(self, image_path: str) -> List[Dict]:
        results = []
        for model in self.models:
            start = time.time()
            detections = model.predict(image_path)
            latency = time.time() - start
            results.append({
                'model': str(type(model)),
                'accuracy': np.mean([d['confidence'] for d in detections]),
                'latency': latency,
                'predictions': detections
            })
        return results


class SwarmOptimizer:
    def __init__(self, num_crews: int = 5):
        self.workflow = StateGraph(SwarmState)
        self.num_crews = num_crews
        self._setup_infrastructure()

    def _setup_infrastructure(self):
        # Crear subgrafos para cada crew
        for i in range(self.num_crews):
            models = random.sample(AGENT_METADATA['models'], k=2)
            crew_graph = CrewSubgraph(i, models, ["explore", "exploit", "diversify"])
            self.workflow.add_subgraph(crew_graph)

        # Nodos globales
        self.workflow.add_node("initialize", self.initialize_swarm)
        self.workflow.add_node("evaluate_diversity", self.calculate_diversity)
        self.workflow.add_node("update_gbest", self.update_global_best)

        # Conexiones
        self.workflow.set_entry_point("initialize")
        self.workflow.add_edge("initialize", "evaluate_diversity")
        self.workflow.add_edge("evaluate_diversity", "update_gbest")

    def initialize_swarm(self, state: SwarmState) -> SwarmState:
        state.crews = [CrewState(
            models=random.sample(AGENT_METADATA['models'], 2),
            hyperparams=self._random_hyperparams(),
            velocity={k: random.uniform(-0.1, 0.1) for k in hyperparams},
            pbest=hyperparams.copy()
        ) for _ in range(self.num_crews)]
        return state

    async def calculate_diversity(self, state: SwarmState) -> SwarmState:
        positions = np.array([list(c.hyperparams.values()) for c in state.crews])
        state.diversity = np.mean(np.std(positions, axis=0))
        return state

    async def update_global_best(self, state: SwarmState) -> SwarmState:
        best_crew = min(state.crews, key=lambda x: x.pbest_score)
        if best_crew.pbest_score < state.gbest_score:
            state.gbest = best_crew.pbest.copy()
            state.gbest_score = best_crew.pbest_score
        return state

# Optimization Function based on PSO
class PSOMechanics:
    def __init__(self, w=0.8, c1=2, c2=2):
        self.w = w  # Inertia --> posibilidad de ir reduciendo linealmente la inercia en cada iteración, i.e., w - 0.1 (explotación -0 vs exploración -1)
        self.c1 = c1  # Cognitive factor (own)
        self.c2 = c2  # Social factor (group)

        # c1 = 0 y c2 = 1 --> convergencia
        # c1 = 1 y c2 = 0 ---> no convergencia
        # VER ARTÍCULO Application of a particle swarm optimization for shape optimization in hydraulic machinery ---> para valores recomendados y adaptative inertia

    def update_velocity(self, current: Dict, pbest: Dict, gbest: Dict) -> Dict:
        """Updates speed according to adapted PSO equation"""
        new_velocity = {}
        for key in current.keys():
            r1, r2 = random.random(), random.random()
            cognitive = self.c1 * r1 * (pbest[key] - current[key])
            social = self.c2 * r2 * (gbest[key] - current[key])
            new_velocity[key] = self.w * current[key] + cognitive + social
        return new_velocity

    @staticmethod
    def update_position(position: Dict, velocity: Dict) -> Dict:
        """Update the position of the hyperparameters"""
        return {k: max(0.01, position[k] + velocity[k]) for k in position.keys()}


class EnhancedSwarmAgent:
    def __init__(self, num_crews=10, num_agents=3, agents_file="agents_registry.json"):
        self.num_crews = num_crews
        self.num_agents = num_agents
        self.agents_file = agents_file
        self.swarm_state = SwarmState(crews=[], gbest={})
        self.pso = PSOMechanics()
        self._setup_workflow()
        self.max_acceptable_latency = 10.0  # DEFINIR SEGUNDOS DE LATENCIA
        self.performance_logs = []
        # Considerar meter reinforcement learning con sistemas de recompensas

    def _setup_workflow(self):
        """Configura el grafo de estado con LangGraph"""
        workflow = StateGraph(SwarmState)
        workflow.add_node("initialize", self.initialize_swarm)
        workflow.add_node("evaluate", self.evaluate_crews)
        workflow.add_node("update", self.update_swarm)
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "evaluate")
        workflow.add_edge("evaluate", "update")
        workflow.add_edge("update", "evaluate")
        self.workflow = workflow

    def _create_heterogeneous_crew(self) -> CrewState:
        """Creates a crew with a heterogeneous configuration"""
        with open(self.agents_file) as f:
            agents = json.load(f)

        selected_models = random.sample(agents['models'], k=self.num_agents)  # self.num_agents models combination
        hyperparams = {
            'learning_rate': random.uniform(0.001, 0.1),
            'batch_size': random.choice([16, 32, 64]),
            'attention_dropout': random.uniform(0.1, 0.5)
        }

        return CrewState(
            models=selected_models,
            hyperparameters=hyperparams,
            velocity={k: random.uniform(-0.1, 0.1) for k in hyperparams},
            pbest=hyperparams.copy()
        )

    def initialize_swarm(self, state: SwarmState) -> SwarmState:
        """Initialize the crew swarm"""
        state.crews = [self._create_heterogeneous_crew() for _ in range(self.num_crews)]
        return state

    async def evaluate_crews(self, state: SwarmState) -> SwarmState:
        """Evaluate each crew using combined metrics"""
        evaluation_tasks = []
        for crew in state.crews:
            task = self._evaluate_single_crew(crew)
            evaluation_tasks.append(task)

        scores = await asyncio.gather(*evaluation_tasks)

        # Update best scores
        for crew, score in zip(state.crews, scores):
            if score < crew.pbest_score:
                crew.pbest = crew.hyperparameters.copy()
                crew.pbest_score = score

            if score < state.gbest_score:
                state.gbest = crew.hyperparameters.copy()
                state.gbest_score = score

        state.performance_logs.append({
            'iteration': state.iteration,
            'best_score': state.gbest_score,
            'avg_score': np.mean(scores)
        })

        return state

    async def _evaluate_single_crew(self, crew: CrewConfiguration) -> float:
        """Evaluation function combining multiple metrics"""
        # Simular ejecución con métricas reales
        accuracy = random.uniform(0.7, 0.95)  # Simular precisión
        latency = random.uniform(0.1, 2.0)  # Simular latencia
        return self._combined_metric(accuracy, latency)
    # async def _evaluate_single_crew(self, crew: CrewConfiguration) -> float:
    #     # implementación parecida con modelos reales
    #     results = []
    #     for model in crew.models:
    #         start = time.time()
    #         accuracy = await model.predict(image)
    #         latency = time.time() - start
    #         results.append((accuracy, latency))
    #
    #     avg_accuracy = np.mean([r[0] for r in results])
    #     avg_latency = np.mean([r[1] for r in results])
    #     return self._combined_metric(avg_accuracy, avg_latency)

    def _combined_metric(self, accuracy: float, latency: float) -> float:
        """Metric combined with adaptive weights"""
        # return (0.8 * (1 - accuracy)) + (0.2 * latency) --> lineal
        accuracy_penalty = (1 - accuracy) ** 2  # cuadrática
        latency_norm = np.clip(latency / self.max_acceptable_latency, 0, 1) # latencia normalizada
        return (0.6 * accuracy_penalty) + (0.4 * latency_norm)

    def update_swarm(self, state: SwarmState) -> SwarmState:
        """Update positions and speeds using PSO"""
        for crew in state.crews:
            # Update speed
            crew.velocity = self.pso.update_velocity(
                crew.velocity,
                crew.pbest,
                state.gbest
            )

            # Update Position
            new_position = self.pso.update_position(
                crew.hyperparameters,
                crew.velocity
            )

            # Apply restriction
            crew.hyperparameters = {
                k: min(max(v, 0.001), 1.0) for k, v in new_position.items()
            }

        state.iteration += 1
        return state

    async def optimize(self, max_iterations=100):
        """Run the optimization process"""
        state = SwarmState(crews=[], gbest={})
        state = await self.workflow.arun(state, iterations=max_iterations)
        return state

async def main():
    swarm = EnhancedSwarmAgent(num_crews=10)
    final_state = await swarm.optimize(max_iterations=50)

    print(f"Mejor configuración encontrada:")
    print(final_state.gbest)
    print(f"Puntuación: {final_state.gbest_score:.4f}")

if __name__ == "__main__":
    asyncio.run(main())

# class SwarmAgent:
#     """Agent designed to handle subtasks using swarm intelligence (crew-based task resolution)."""
#
#     def __init__(
#             self, num_crews=10, agents_file="agents_registry.json",
#             run_in_parallel=True, custom_executor: Callable = None
#     ):
#         """
#         Initializes the SwarmAgent.
#
#         Args:
#             num_crews (int): Number of crews to create to resolve subtasks.
#             run_in_parallel (bool): Whether crews should execute tasks in parallel or sequentially.
#             custom_executor (Callable): A user-defined function to customize task execution logic.
#         """
#         self.num_crews = num_crews
#         self.agents_file = agents_file
#         self.run_in_parallel = run_in_parallel
#         self.executor = custom_executor or self.default_executor
#         self.logger = self._setup_logger()
#         self.performance_logs = []
#         # Velocity function
#         # w = inertia factor
#         # c1 = cognitive term weight (own success)
#         # c2 = social term weight (global success)
#         # bt = local best position (found by the considered bird up to time t)
#         # gt = global best position (found by the whole flock up to time t)
#         # xt = position of the considered bird at time t
#         """"
#         ----- ARTICLE: Parallelization of Swarm Intelligence Algorithms: Literature Review -----
#         Velocity function:
#         ------- vt+1 = w * vt + c1 * r1 * (bt - xt) + c2 * r2 (gt * xt) -------
#         After updating the speed, each particle will update its position according to the following formula:
#         -------  xt+1 = xt + vt+1 -------
#         where xt and xt+1 are the position of the particle at time t and t+1, respectively, and vt+1 is the
#         speed of the particle at time t+1.
#
#         After performing the speed and position updates, the fitness of each particle at its current
#         position is calculated. The current local best fitness is then compared to the new value and
#         possibly updated. Moreover, the current global best position will be updated, if a particle
#         has obtained a better fitness than the best known one. After these updates, the mentioned
#         steps are repeated, until a termination criterion is fulfilled. A fixed number of iterations is
#         often used as termination criterion. In some cases, the stagnation of the fitness is used
#         instead, which means that the iterations will stop when the swarm stops improving after a
#         given number of unsuccessful attempts.
#
#         Some authors propose to transmit just the location of the best position found so far by
#         the sending swarm. Other approaches perform an exchange of particles among swarms.
#         Both approaches increase the capability of finding better positions in the search space. But
#         there are no comparisons that could help determine which approach leads to better
#         solutions.
#
#         Another approach is to run different SI algorithms in parallel. Such an approach has
#         been investigated in [63]. The algorithms need not belong to the same family. As each
#         optimization algorithm has different operators and communication patters, such an
#         approach can benefit from the strengths of each algorithm. Using this concept, more
#         complex structures can be built. For example, in [63], the authors propose the idea of
#         having multiple archipelagos. Each archipelago runs one algorithm; in this case PSO, a
#         Genetic Algorithm (GA) and Simulated Annealing (SA). On an archipelago, each island
#         runs an instance of the algorithm with different parameters. Islands communicate with each
#         other using a given topology in an asynchronous way, and so do archipelagos.
#         """
#
#     async def handle_subtasks(self, tasks: List[list]) -> None:
#         """Handles the subtasks provided by the CoordinatorAgent."""
#         self.logger.info(f"Received {len(tasks)} tasks. Creating {self.num_crews} crews.")
#
#         # Step 1: Create crews and configure agents heterogeneously
#         crews = self._create_crews(tasks)
#
#         # Persist crew configurations
#         self._persist_configurations(crews)
#
#         # Step 2: Assign an execution order to the subtasks
#         self._assign_order_to_subtasks(crews)
#
#         # Step 3: Execute each crew in parallel (or sequentially)
#         if self.run_in_parallel:
#             self.logger.info("Executing crews in parallel.")
#             await asyncio.gather(*(self.executor(crew) for crew in crews))
#         else:
#             self.logger.info("Executing crews sequentially.")
#             for crew in crews:
#                 await self.executor(crew)
#
#         # Step 4: Collect performance metrics
#         self._evaluate_crews(crews)
#
#         self.logger.info("All subtasks have been resolved.")
#
#     def _create_crews(self, tasks: List[list]) -> List[dict]:
#         """Step 1: Create 'n' crews with heterogeneous agent configurations."""
#         crews = []
#         subtasks = [item for task in tasks for item in task.subtasks]
#         for i in range(self.num_crews):
#             # Select a random subset of agents/models for the crew (heterogeneous selection)
#             agent_configurations = self._select_heterogeneous_agents(subtasks=subtasks)
#             crews.append({
#                 "name": f"crew_{i + 1}",
#                 "agents": agent_configurations,
#                 "subtasks": subtasks
#             })
#         return crews
#
#     def _select_heterogeneous_agents(self, subtasks: List) -> List[dict]:
#         """Selects heterogeneous agents and models for a crew."""
#         with open(self.agents_file, "r") as file:
#             agents_registry = json.load(file)
#         return [
#             {
#                 "subtask": subtask.subtask,
#                 "agent": subtask.agent,
#                 "model": random.choice(agents_registry[subtask.agent]["models"]) if subtask.agent in agents_registry else None,
#             }
#             for subtask in subtasks
#         ]
#
#     def _persist_configurations(self, crews: List[dict]) -> None:
#         """Persists crew configurations for reproducibility."""
#         with open("crews_configurations.json", "w") as file:
#             json.dump(crews, file, indent=4)
#
#     def _assign_order_to_subtasks(self, crews: List[dict]) -> None:
#         """Step 2: Assign an execution order to subtasks within each crew."""
#         for crew in crews:
#             random.shuffle(crew["subtasks"])
#
#     @staticmethod
#     async def default_executor(crew: dict) -> None:
#         """Step 4: Execute subtasks within each crew."""
#         print(f"Executing tasks for {crew['name']} with agents: {crew['agents']}")
#         for agent in crew["agents"]:
#             for subtask in crew["subtasks"]:
#                 # Simulate agent execution (e.g., calling object_detection_agent)
#                 print(f"Agent {agent['agent']} using {agent['model']} is executing subtask: {subtask}")
#                 await asyncio.sleep(1)
#
#     def _evaluate_crews(self, crews: List[dict]) -> None:
#         """Step 5: Save the performance (or scores) of each crew."""
#         for crew in crews:
#             # Simulating performance metrics (e.g., average execution time, accuracy, etc.)
#             performance_score = random.uniform(0, 1)
#             self.performance_logs.append({
#                 "crew": crew["name"],
#                 "score": performance_score
#             })
#
#         # Persist performance logs
#         with open("performance_logs.json", "w") as file:
#             json.dump(self.performance_logs, file, indent=4)
#
#     @staticmethod
#     def _setup_logger():
#         """Sets up the logger for the SwarmAgent."""
#         import logging
#         logging.basicConfig(
#             level=logging.INFO,
#             format='[%(asctime)s] [%(levelname)s] %(message)s',
#             datefmt='%Y-%m-%d %H:%M:%S'
#         )
#         return logging.getLogger("SwarmAgent")

# if __name__ == "__main__":
#     async def main():
#         from typing import List
#         from pydantic import BaseModel, Field
#
#         class Subtask(BaseModel):
#             subtask: str = Field(description="Description of the subtask")
#             agent: str = Field(description="Name of the assigned agent")
#             dependencies: List[str] = Field(default_factory=list, description="List of dependencies")
#
#         class Task(BaseModel):
#             task: str = Field(description="Name of the main task")
#             subtasks: List[Subtask] = Field(description="List of subtasks")
#
#         tasks = [
#             Task(
#                 task='Analyze Image',
#                 subtasks=[
#                     Subtask(subtask='Detect objects in the image', agent='object_detection_agent', dependencies=[]),
#                     Subtask(subtask='Summarize the content of the image based on detected objects', agent='LLM',
#                             dependencies=['Detect objects in the image'])
#                 ]
#             )
#         ]
#
#         sw_agent = SwarmAgent(num_crews=2)
#         await sw_agent.handle_subtasks(tasks)
#
#     asyncio.run(main())
