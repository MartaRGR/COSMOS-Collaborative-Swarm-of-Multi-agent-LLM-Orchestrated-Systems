import os
import glob
import time
import asyncio
import aiofiles
from utils.setup_logger import get_agent_logger


class AgentRegistry:
    """Automatically scan and register available agents."""

    def __init__(self, agents_folder="agents", registry_file="agents_registry.json"):
        """
        Initializes agent registration.
        Args:
            agents_folder: Folder where the agents are defined.
            registry_file: File where the agent registry will be saved.
        """
        self.logger = get_agent_logger("AgentRegistry")
        self.logger.info("Initializing...")

        self.agents_folder = agents_folder
        self.registry_file = registry_file
        self.agents = {}

    async def discover_agents(self):
        """Discover agents asynchronously in the specified folder."""
        if not os.path.exists(self.agents_folder):
            error_message = f"Folder {self.agents_folder} does not exist in path {os.path.abspath(self.agents_folder)}"
            self.logger.error(error_message)
            raise ValueError(error_message)
        agents_files = [
            file for file in glob.glob(
                os.path.join(os.path.abspath(self.agents_folder), "*.py")
            ) if os.path.basename(file) != "__init__.py"
        ]
        if not agents_files:
            self.logger.warning(f"No Python files found in directory: {self.agents_folder}")
            return {}

        self.logger.info(f"Found {len(agents_files)} agent files: {agents_files}")
        module_names = [os.path.basename(file) for file in agents_files]
        # Async parallel tasks
        tasks = [self.load_agent_metadata(module_name) for module_name in module_names]
        results = await asyncio.gather(*tasks)
        for item in results:
            if item is not None:
                module_name, metadata = item
                self.agents[module_name] = metadata

    async def load_agent_metadata(self, module_name):
        """Load metadata of an agent by reading the file asynchronously."""
        import re
        import json

        def extract_agent_metadata(text):
            # IMPORTANT TO NOTE: every agent MUST have AGENT_METADATA variable to be processed
            # Searching the beginning of AGENT_METADATA. If literal not found, then return None
            match = re.search(r'AGENT_METADATA\s*=\s*\{', text)
            if not match:
                return None

            start = match.end() - 1  # Beginning from first '{'
            stack = 0  # Counter of stacked keys
            end = start

            # Searching the end of the stacked keys
            for i in range(start, len(text)):
                if text[i] == '{':
                    stack += 1
                elif text[i] == '}':
                    stack -= 1
                    if stack == 0:  # block completely closed
                        end = i + 1
                        break
            return json.loads(text[start:end])

        try:
            agent_file = os.path.join(os.path.abspath(self.agents_folder), module_name)
            if not os.path.exists(agent_file):
                self.logger.warning(f"File {module_name} not found.")
                return None

            async with aiofiles.open(agent_file, mode="r", encoding="utf-8") as file:
                contents = await file.read()

            # IMPORTANT TO NOTE: every agent MUST have AGENT_METADATA variable to be processed
            metadata = extract_agent_metadata(contents)
            if metadata:
                self.logger.debug(f"Loaded metadata for module {module_name}: {metadata}")
                return module_name, metadata
            else:
                self.logger.warning(f"AGENT_METADATA not found in module {module_name}.")
                return None

        except Exception as e:
            print(f"Error loading {module_name}: {e}")

    async def save_registry(self):
        """Save agent list to JSON file asynchronously."""
        import json
        
        if not self.registry_file:
            error_message = "Registry file path not set."
            self.logger.error(error_message)
            raise ValueError(error_message)
        
        async with aiofiles.open(self.registry_file, mode="w", encoding="utf-8") as f:
            await f.write(json.dumps(self.agents, indent=4))
        self.logger.info(f"Wrote registry file: {self.registry_file}")

    async def run(self):
        """Run the discovery and registration process."""
        self.logger.info("REGISTRY CREATOR AGENT - Looking for agents...")
        start_time = time.time()

        await self.discover_agents()
        self.logger.info(f"Discovery finished in {time.time() - start_time:.4f}s.")

        start_time = time.time()
        await self.save_registry()
        self.logger.info(f"Registry saved in {time.time() - start_time:.4f}s")
        self.logger.info("END OF REGISTRY CREATOR AGENT")


if __name__ == "__main__":
    async def main():
        register = AgentRegistry(agents_folder="agents")
        await register.run()
    asyncio.run(main())



