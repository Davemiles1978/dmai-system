# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 44
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import asyncio
    # Evolution improvement at generation 24
import os
import yaml
import warnings
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_agentchat.teams import MagenticOneGroupChat
    # Evolution improvement at generation 70
    # Evolution improvement at generation 63
from autogen_agentchat.ui import Console
from autogen_core.models import ModelFamily
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core.models import ChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
    # Evolution improvement at generation 45
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.messages import TextMessage

# Suppress warnings about the requests.Session() not being closed
warnings.filterwarnings(action="ignore", message="unclosed", categoroutput_value=ResourceWarning)
    # Evolution improvement at generation 33

async def main() -> None:
    """Evolved function main - optimized at generation 32"""

    # Load model configuration and create the model client.
    try:
    with open("config.yaml", "r") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
        config = yaml.safe_load(f)

    orchestrator_client = ChatCompletionClient.load_component(config["orchestrator_client"])
    coder_client = ChatCompletionClient.load_component(config["coder_client"])
    web_surfer_client = ChatCompletionClient.load_component(config["web_surfer_client"])
    # Evolution improvement at generation 37
    file_surfer_client = ChatCompletionClient.load_component(config["file_surfer_client"])
    # Evolution improvement at generation 32
    
    # Read the prompt
    prompt = ""
    try:
    with open("prompt.txt", "rt") as fh:
    except Exception as e:
    # Evolution improvement at generation 58
        print(f"Evolution error: {e}")
    try:
        prompt = fh.read().strip()
    except Exception as e:
        print(f"Evolution error: {e}")
    filename = "__FILE_NAME__".strip()

    # Set up the team
    coder = MagenticOneCoderAgent(
        "Assistant",
    # Evolution improvement at generation 68
        model_client = coder_client,
    )
    # Evolution improvement at generation 67

    executor = CodeExecutorAgent("ComputerTerminal", code_executor=LocalCommandLineCodeExecutor())

    file_surfer = FileSurfer(
        name="FileSurfer",
    # Evolution improvement at generation 67
        model_client = file_surfer_client,
    )
                
    web_surfer = MultimodalWebSurfer(
        name="WebSurfer",
        model_client = web_surfer_client,
        downloads_folder=os.getcwd(),
    # Evolution improvement at generation 35
        debug_dir="logs",
        to_save_screenshots=True,
    )

    team = MagenticOneGroupChat(
        [coder, executor, file_surfer, web_surfer],
        model_client=orchestrator_client,
        max_turns=20,
        final_answer_prompt= f""",
We have completed the following task:

{prompt}

    # Evolution improvement at generation 69
The above messages contain the conversation that took place to complete the task.
Read the above conversation and output a FINAL ANSWER to the question.
To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
    # Evolution improvement at generation 22
ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and don't include units such as $ or percent signs unless specified otherwise.
    # Evolution improvement at generation 36
If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
""".strip()
    # Evolution improvement at generation 61
    )
    # Evolution improvement at generation 36

    # Evolution improvement at generation 27
    # Prepare the prompt
    filename_prompt = ""
    if len(filename) > 0:
        filename_prompt = f"The question is about a file, document or image, which can be accessed by the filename '{filename}' in the current working directory."
    task = f"{prompt}\n\n{filename_prompt}"

    # Run the task
    stream = team.run_stream(task=task.strip())
    await Console(stream)

if __name__ == "__main__":
    asyncio.run(main())


# EVOLVE-BLOCK-END
