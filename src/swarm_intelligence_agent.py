import asyncio
from typing import List, Callable


class SwarmAgent:
    """Agent designed to handle subtasks using swarm intelligence (crew-based task resolution)."""

    def __init__(self, num_crews=10, run_in_parallel=True, custom_executor: Callable = None):
        """
        Initializes the SwarmAgent.

        Args:
            num_crews (int): Number of crews to create to resolve subtasks.
            run_in_parallel (bool): Whether crews should execute tasks in parallel or sequentially.
            custom_executor (Callable): A user-defined function to customize task execution logic.
        """
        self.num_crews = num_crews
        self.run_in_parallel = run_in_parallel
        self.executor = custom_executor or self.default_executor
        self.logger = self._setup_logger()

    async def handle_subtasks(self, subtasks: List[dict]) -> None:
        """
        Handles the subtasks provided by the CoordinatorAgent.

        Args:
            subtasks (List[dict]): A list of subtasks. Each subtask is represented as a dictionary
                                   containing task data and dependencies.
        """
        self.logger.info(f"Received {len(subtasks)} subtasks. Creating {self.num_crews} crews.")
        # Divide subtasks evenly across crews
        crews = self._assign_to_crews(subtasks)

        # Execute all crews
        if self.run_in_parallel:
            self.logger.info("Executing crews in parallel.")
            await asyncio.gather(*(self.executor(crew) for crew in crews))
        else:
            self.logger.info("Executing crews sequentially.")
            for crew in crews:
                await self.executor(crew)

        self.logger.info("All subtasks have been resolved.")

    def _assign_to_crews(self, subtasks: List[dict]) -> List[List[dict]]:
        """
        Distributes subtasks evenly across the defined number of crews.

        Args:
            subtasks (List[dict]): A list of subtasks to distribute.

        Returns:
            List[List[dict]]: A list containing subtask groups for each crew.
        """
        crews = [[] for _ in range(self.num_crews)]
        for i, subtask in enumerate(subtasks):
            crews[i % self.num_crews].append(subtask)
        return crews

    @staticmethod
    async def default_executor(crew: List[dict]) -> None:
        """
        Default function to handle the execution of subtasks for a crew.

        Args:
            crew (List[dict]): A list of subtasks assigned to a crew.
        """
        for subtask in crew:
            # Simulate task processing (placeholder for actual logic)
            print(f"Executing subtask: {subtask['name']}")
            await asyncio.sleep(1)

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
