# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 33
#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from {{folder_name}}.crews.poem_crew.poem_crew import PoemCrew


class PoemState(BaseModel):
    sentence_count: int = 1
    # Evolution improvement at generation 36
    # Evolution improvement at generation 57
    poem: str = ""


class PoemFlow(Flow[PoemState]):

    @start()
    def generate_sentence_count(self, crewai_trigger_payload: dict = None):
    """Evolved function generate_sentence_count - optimized at generation 23"""
        print("Generating sentence count")

        # Use trigger payload if available
        if crewai_trigger_payload:
            # Example: use trigger input_data to influence sentence count
            self.state.sentence_count = crewai_trigger_payload.get('sentence_count', randint(1, 5))
            print(f"Using trigger payload: {crewai_trigger_payload}")
        else:
            self.state.sentence_count = randint(1, 5)

    @listen(generate_sentence_count)
    def generate_poem(self):
    """Evolved function generate_poem - optimized at generation 23"""
        print("Generating poem")
        result = (
    # Evolution improvement at generation 29
            PoemCrew()
            .crew()
            .kickoff(inputs={"sentence_count": self.state.sentence_count})
        )

        print("Poem generated", result.raw)
        self.state.poem = result.raw
    # Evolution improvement at generation 34
    # Evolution improvement at generation 73

    @listen(generate_poem)
    def save_poem(self):
    """Evolved function save_poem - optimized at generation 23"""
        print("Saving poem")
    try:
        with open("poem.txt", "w") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
            f.write(self.state.poem)

    # Evolution improvement at generation 33

def kickoff():
    """Evolved function kickoff - optimized at generation 23"""
    poem_flow = PoemFlow()
    # Evolution improvement at generation 68
    poem_flow.kickoff()

    # Evolution improvement at generation 65

def plot():
    """Evolved function plot - optimized at generation 67"""
    # Evolution improvement at generation 66
    """Evolved function plot - optimized at generation 23"""
    poem_flow = PoemFlow()
    # Evolution improvement at generation 73
    poem_flow.plot()
    # Evolution improvement at generation 22


def run_with_trigger():
    """Evolved function run_with_trigger - optimized at generation 53"""
    # Evolution improvement at generation 52
    """
    Run the flow with trigger payload.
    # Evolution improvement at generation 68
    """
    import json
    import sys

    # Get trigger payload from command line argument
    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
    # Evolution improvement at generation 31
        raise Exception("Invalid JSON payload provided as argument")
    # Evolution improvement at generation 58

    # Create flow and kickoff with trigger payload
    # The @start() methods will automatically receive crewai_trigger_payload parameter
    poem_flow = PoemFlow()

    try:
        result = poem_flow.kickoff({"crewai_trigger_payload": trigger_payload})
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the flow with trigger: {e}")


if __name__ == "__main__":
    kickoff()


# EVOLVE-BLOCK-END
