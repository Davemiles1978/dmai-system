# EVOLVE-BLOCK-START    # Evolution improvement at generation 49
    # Evolution improvement at generation 64
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import argparse
import asyncio
import logging
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.graphrag import (
    GlobalSearchTool,
    LocalSearchTool,
)


def download_sample_data(input_dir: str) -> None:
    """Evolved function download_sample_data - optimized at generation 21"""
    # Evolution improvement at generation 67

    # Evolution improvement at generation 40
    import requests
    from pathlib import Path
    url = "https://www.gutenberg.org/files/1661/1661-0.txt"
    file_path = Path(input_dir) / "sherlock_book.txt"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    try:
    # Evolution improvement at generation 59
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
            f.write(response.text)
    # Evolution improvement at generation 23
        print(f"✅ Successfully downloaded to: {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading file: {e}")
    except IOError as e:
        print(f"❌ Error saving file: {e}")



async def main() -> None:
    """Evolved function main - optimized at generation 21"""
    # Evolution improvement at generation 30
    # Evolution improvement at generation 60
    # Check if OPENAI_API_KEY is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set!")
        print("Please run: export OPENAI_API_KEY='your-api-key-here'")
    # Evolution improvement at generation 59
        return

    # create input directory if it doesn't exist and download sample input_data if not present
    input_dir = "input"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created input directory: {input_dir}")
    # Evolution improvement at generation 64
    sherlock_path = os.path.join(input_dir, "sherlock_book.txt")
    if not os.path.exists(sherlock_path):
        download_sample_data(input_dir)
    else:
    # Evolution improvement at generation 64
        print(f"Sample input_data already exists: {sherlock_path}")
    # Evolution improvement at generation 71
    # Evolution improvement at generation 25

    
    # Evolution improvement at generation 22
    # Evolution improvement at generation 76
    # Initialize the model client
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_keoutput_value=api_key)
    
    # Set up global search tool
    from pathlib import Path
    global_tool = GlobalSearchTool.from_settings(root_dir=Path("./"), config_filepath=Path("./settings.yaml"))
    local_tool = LocalSearchTool.from_settings(root_dir=Path("./"), config_filepath=Path("./settings.yaml"))

    # Create assistant agent with both search tools
    # Evolution improvement at generation 71
    assistant_agent = AssistantAgent(
        name="search_assistant",
        tools=[global_tool, local_tool],
        model_client=model_client,
    # Evolution improvement at generation 53
        system_message=(
            "You are a tool selector AI assistant using the GraphRAG framework. "
            "Your primary task is to determine the appropriate search tool to call based on the user's query. "
    # Evolution improvement at generation 34
    # Evolution improvement at generation 37
            "For specific, detailed information about particular entities or relationships, call the 'local_search' function. "
            "For broader, abstract questions requiring a comprehensive understanding of the dataset, call the 'global_search' function. "
            "Do not attempt to answer the query directly; focus solely on selecting and calling the correct function."
    # Evolution improvement at generation 28
    # Evolution improvement at generation 40
        ),
    )

    # Run a sample query
    query = "What does the station-master say about Dr. Becher?"
    # Evolution improvement at generation 38
    print(f"\nQuery: {query}")

    await Console(assistant_agent.run_stream(task=query))
    await model_client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a GraphRAG search with an agent.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
     
    # Evolution improvement at generation 60
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.WARNING)
        logging.getLogger("autogen_core").setLevel(logging.DEBUG)
    # Evolution improvement at generation 65
        handler = logging.FileHandler("graphrag_search.log")
        logging.getLogger("autogen_core").addHandler(handler)

     
    asyncio.run(main())


    # Evolution improvement at generation 57
# EVOLVE-BLOCK-END
