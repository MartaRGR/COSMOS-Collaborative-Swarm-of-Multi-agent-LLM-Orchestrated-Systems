import asyncio
from typing import List, Callable
import json
import random


class SwarmAgent:
    """Agent designed to handle subtasks using swarm intelligence (crew-based task resolution)."""

    def __init__(
            self, num_crews=10, agents_file="agents_registry.json",
            run_in_parallel=True, custom_executor: Callable = None
    ):
        """
        Initializes the SwarmAgent.

        Args:
            num_crews (int): Number of crews to create to resolve subtasks.
            run_in_parallel (bool): Whether crews should execute tasks in parallel or sequentially.
            custom_executor (Callable): A user-defined function to customize task execution logic.
        """
        self.num_crews = num_crews
        self.agents_file = agents_file
        self.run_in_parallel = run_in_parallel
        self.executor = custom_executor or self.default_executor
        self.logger = self._setup_logger()
        self.performance_logs = []
        # Velocity function
        # w = inertia factor
        # c1 = cognitive term weight (own success)
        # c2 = social term weight (global success)
        # bt = local best position (found by the considered bird up to time t)
        # gt = global best position (found by the whole flock up to time t)
        # xt = position of the considered bird at time t
        """"
        ----- ARTICLE: Parallelization of Swarm Intelligence Algorithms: Literature Review -----
        Velocity function:
        ------- vt+1 = w * vt + c1 * r1 * (bt - xt) + c2 * r2 (gt * xt) -------
        After updating the speed, each particle will update its position according to the following formula:
        -------  xt+1 = xt + vt+1 ------- 
        where xt and xt+1 are the position of the particle at time t and t+1, respectively, and vt+1 is the 
        speed of the particle at time t+1.
        
        After performing the speed and position updates, the fitness of each particle at its current
        position is calculated. The current local best fitness is then compared to the new value and
        possibly updated. Moreover, the current global best position will be updated, if a particle
        has obtained a better fitness than the best known one. After these updates, the mentioned
        steps are repeated, until a termination criterion is fulfilled. A fixed number of iterations is
        often used as termination criterion. In some cases, the stagnation of the fitness is used
        instead, which means that the iterations will stop when the swarm stops improving after a
        given number of unsuccessful attempts.
        
        Some authors propose to transmit just the location of the best position found so far by
        the sending swarm. Other approaches perform an exchange of particles among swarms.
        Both approaches increase the capability of finding better positions in the search space. But
        there are no comparisons that could help determine which approach leads to better
        solutions.
        
        Another approach is to run different SI algorithms in parallel. Such an approach has
        been investigated in [63]. The algorithms need not belong to the same family. As each
        optimization algorithm has different operators and communication patters, such an
        approach can benefit from the strengths of each algorithm. Using this concept, more
        complex structures can be built. For example, in [63], the authors propose the idea of
        having multiple archipelagos. Each archipelago runs one algorithm; in this case PSO, a
        Genetic Algorithm (GA) and Simulated Annealing (SA). On an archipelago, each island
        runs an instance of the algorithm with different parameters. Islands communicate with each
        other using a given topology in an asynchronous way, and so do archipelagos.
        """

    async def handle_subtasks(self, tasks: List[list]) -> None:
        """Handles the subtasks provided by the CoordinatorAgent."""
        self.logger.info(f"Received {len(tasks)} tasks. Creating {self.num_crews} crews.")

        # Step 1: Create crews and configure agents heterogeneously
        crews = self._create_crews(tasks)

        # Persist crew configurations
        self._persist_configurations(crews)

        # Step 2: Assign an execution order to the subtasks
        self._assign_order_to_subtasks(crews)

        # Step 3: Execute each crew in parallel (or sequentially)
        if self.run_in_parallel:
            self.logger.info("Executing crews in parallel.")
            await asyncio.gather(*(self.executor(crew) for crew in crews))
        else:
            self.logger.info("Executing crews sequentially.")
            for crew in crews:
                await self.executor(crew)

        # Step 4: Collect performance metrics
        self._evaluate_crews(crews)

        self.logger.info("All subtasks have been resolved.")

    def _create_crews(self, tasks: List[list]) -> List[dict]:
        """Step 1: Create 'n' crews with heterogeneous agent configurations."""
        crews = []
        subtasks = [item for task in tasks for item in task.subtasks]
        for i in range(self.num_crews):
            # Select a random subset of agents/models for the crew (heterogeneous selection)
            agent_configurations = self._select_heterogeneous_agents(subtasks=subtasks)
            crews.append({
                "name": f"crew_{i + 1}",
                "agents": agent_configurations,
                "subtasks": subtasks
            })
        return crews

    def _select_heterogeneous_agents(self, subtasks: List) -> List[dict]:
        """Selects heterogeneous agents and models for a crew."""
        with open(self.agents_file, "r") as file:
            agents_registry = json.load(file)
        return [
            {
                "subtask": subtask.subtask,
                "agent": subtask.agent,
                "model": random.choice(agents_registry[subtask.agent]["models"]) if subtask.agent in agents_registry else None,
            }
            for subtask in subtasks
        ]

    def _persist_configurations(self, crews: List[dict]) -> None:
        """Persists crew configurations for reproducibility."""
        with open("crews_configurations.json", "w") as file:
            json.dump(crews, file, indent=4)

    def _assign_order_to_subtasks(self, crews: List[dict]) -> None:
        """Step 2: Assign an execution order to subtasks within each crew."""
        for crew in crews:
            random.shuffle(crew["subtasks"])

    @staticmethod
    async def default_executor(crew: dict) -> None:
        """Step 4: Execute subtasks within each crew."""
        print(f"Executing tasks for {crew['name']} with agents: {crew['agents']}")
        for agent in crew["agents"]:
            for subtask in crew["subtasks"]:
                # Simulate agent execution (e.g., calling object_detection_agent)
                print(f"Agent {agent['agent']} using {agent['model']} is executing subtask: {subtask}")
                await asyncio.sleep(1)

    def _evaluate_crews(self, crews: List[dict]) -> None:
        """Step 5: Save the performance (or scores) of each crew."""
        for crew in crews:
            # Simulating performance metrics (e.g., average execution time, accuracy, etc.)
            performance_score = random.uniform(0, 1)
            self.performance_logs.append({
                "crew": crew["name"],
                "score": performance_score
            })

        # Persist performance logs
        with open("performance_logs.json", "w") as file:
            json.dump(self.performance_logs, file, indent=4)

    @staticmethod
    def _setup_logger():
        """Sets up the logger for the SwarmAgent."""
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger("SwarmAgent")

if __name__ == "__main__":
    async def main():
        from typing import List
        from pydantic import BaseModel, Field

        class Subtask(BaseModel):
            subtask: str = Field(description="Description of the subtask")
            agent: str = Field(description="Name of the assigned agent")
            dependencies: List[str] = Field(default_factory=list, description="List of dependencies")

        class Task(BaseModel):
            task: str = Field(description="Name of the main task")
            subtasks: List[Subtask] = Field(description="List of subtasks")

        tasks = [
            Task(
                task='Analyze Image',
                subtasks=[
                    Subtask(subtask='Detect objects in the image', agent='object_detection_agent', dependencies=[]),
                    Subtask(subtask='Summarize the content of the image based on detected objects', agent='LLM',
                            dependencies=['Detect objects in the image'])
                ]
            )
        ]

        sw_agent = SwarmAgent(num_crews=2)
        await sw_agent.handle_subtasks(tasks)

    asyncio.run(main())
