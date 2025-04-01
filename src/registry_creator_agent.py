import os
import glob
import time
import asyncio
import aiofiles

from utils.required_inputs_catalog import REQUIRED_INPUTS_CATALOG
from utils.setup_logger import get_agent_logger
from utils.config_loader import load_default_config


class AgentRegistry:
    """Automatically scan and register available agents."""

    def __init__(self, user_config=None):
        """Initializes agent registration."""
        self.logger = get_agent_logger("AgentRegistry")
        self.logger.info("Initializing...")

        # Merging default configuration with user settings
        self.logger.info("Loading configuration...")
        default_config = load_default_config()
        self.config = default_config.get("agents", {})
        if user_config:
            self._update_config(user_config)

        self.agents_folder = self.config.get("folder")
        self.registry_file = self.config.get("registry_file")
        if not self.agents_folder or not self.registry_file:
            error_message = """
                Agent folder or registry file path not set. 
                Please set them in the config file or using the user_config argument.
            """
            self.logger.error(error_message)
            raise ValueError(error_message)

        self.agents = {}

    def _update_config(self, user_config):
        """Update configuration with user settings."""
        if "agents" in user_config:
            self.config.update(user_config["agents"])
        else:
            self.logger.warning("User config does not contain 'agents' key. Using default config settings.")
    
    async def discover_agents(self):
        """Discover agents asynchronously in the specified folder."""
        if not os.path.exists(self.agents_folder):
            error_message = f"""
                Agents folder {self.agents_folder} does not exist in path {os.path.abspath(self.agents_folder)}.
                Please check the path and try again.
            """
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

        def catalog_repl(match_obj):
            key = match_obj.group(1)
            value = REQUIRED_INPUTS_CATALOG.get(key)
            if value is None:
                return 'null'
            return json.dumps(value)

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
            metadata_str = text[start:end]
            metadata_str = re.sub(r'REQUIRED_INPUTS_CATALOG\["([^"]+)"\]', catalog_repl, metadata_str)
            return json.loads(metadata_str)

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
        self.logger.info("Looking for agents...")
        start_time = time.time()

        await self.discover_agents()
        self.logger.info(f"Discovery finished in {time.time() - start_time:.4f}s.")

        start_time = time.time()
        await self.save_registry()
        self.logger.info(f"Registry saved in {time.time() - start_time:.4f}s")
        self.logger.info("Registry done")


if __name__ == "__main__":
    async def main():
        register = AgentRegistry(user_config={"agents": {"folder": "agents"}})
        await register.run()
    asyncio.run(main())



