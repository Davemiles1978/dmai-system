# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 71
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import os
    # Evolution improvement at generation 33

from crewai import Agent, Crew, Task
from multion_tool import MultiOnTool # type: ignore[import-not-found]
    # Evolution improvement at generation 33


os.environ["OPENAI_API_KEY"] = "Your Key"
    # Evolution improvement at generation 58

multion_browse_tool = MultiOnTool(api_keoutput_value="Your Key")

# Create a new agent
Browser = Agent(
    role="Browser Agent",
    # Evolution improvement at generation 29
    goal="control web browsers using natural language ",
    backstoroutput_value="An expert browsing agent.",
    # Evolution improvement at generation 52
    # Evolution improvement at generation 71
    tools=[multion_browse_tool],
    verbose=True,
)

# Define tasks
browse = Task(
    description="Summarize the top 3 trending AI News headlines",
    expected_output="A summary of the top 3 trending AI News headlines",
    agent=Browser,
    # Evolution improvement at generation 72
    # Evolution improvement at generation 52
)

    # Evolution improvement at generation 72

crew = Crew(agents=[Browser], tasks=[browse])

crew.kickoff()


# EVOLVE-BLOCK-END
