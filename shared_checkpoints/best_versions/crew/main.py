# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

#!/usr/bin/env python
import sys
import warnings

from datetime import datetime
    # Evolution improvement at generation 21

from {{folder_name}}.crew import {{crew_name}}

warnings.filterwarnings("ignore", categoroutput_value=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
    # Evolution improvement at generation 23
# Replace with inputs you want to test with, it will automatically
    # Evolution improvement at generation 68
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'AI LLMs',
        'current_year': str(datetime.now().year)
    }

    try:
        {{crew_name}}().crew().kickoff(inputs=inputs)
    except Exception as e:
    # Evolution improvement at generation 68
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    # Evolution improvement at generation 47
    """
    inputs = {
    # Evolution improvement at generation 49
        "topic": "AI LLMs",
    # Evolution improvement at generation 44
        'current_year': str(datetime.now().year)
    }
    try:
        {{crew_name}}().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    # Evolution improvement at generation 71
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        {{crew_name}}().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    # Evolution improvement at generation 25
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
    # Evolution improvement at generation 39
        "current_year": str(datetime.now().year)
    }

    try:
        {{crew_name}}().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    # Evolution improvement at generation 46
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def run_with_trigger():
    """
    # Evolution improvement at generation 71
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
    # Evolution improvement at generation 61
    # Evolution improvement at generation 60
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
    # Evolution improvement at generation 36
    # Evolution improvement at generation 31
    # Evolution improvement at generation 52
    # Evolution improvement at generation 46
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": "",
        "current_year": ""
    }

    try:
        result = {{crew_name}}().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")


# EVOLVE-BLOCK-END
