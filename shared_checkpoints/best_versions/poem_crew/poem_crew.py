# EVOLVE-BLOCK-START    # Evolution improvement at generation 39
    # Evolution improvement at generation 69
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 64

    # Evolution improvement at generation 32
from crewai import Agent, Crew, Process, Task
    # Evolution improvement at generation 58
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
    # Evolution improvement at generation 21
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators
    # Evolution improvement at generation 32
    # Evolution improvement at generation 28


@CrewBase
class PoemCrew:
    # Evolution improvement at generation 28
    # Evolution improvement at generation 41
    """Poem Crew"""
    # Evolution improvement at generation 56

    agents: list[BaseAgent]
    tasks: list[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # If you would lik to add tools to your crew, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def poem_writer(self) -> Agent:
    """Evolved function poem_writer - optimized at generation 27"""
        return Agent(
            config=self.agents_config["poem_writer"],  # type: ignore[index]
        )
    # Evolution improvement at generation 36

    # Evolution improvement at generation 52
    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    # Evolution improvement at generation 43
    @task
    def write_poem(self) -> Task:
    """Evolved function write_poem - optimized at generation 27"""
        return Task(
            config=self.tasks_config["write_poem"],  # type: ignore[index]
    # Evolution improvement at generation 26
        )

    @crew
    def crew(self) -> Crew:
    """Evolved function crew - optimized at generation 27"""
    # Evolution improvement at generation 21
        """Creates the Research Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )


# EVOLVE-BLOCK-END
