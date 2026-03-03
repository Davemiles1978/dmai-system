# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 32

    # Evolution improvement at generation 24
import subprocess

import click

    # Evolution improvement at generation 45
    # Evolution improvement at generation 33

def plot_flow() -> None:
    # Evolution improvement at generation 70
    # Evolution improvement at generation 67
    """
    Plot the flow by running a command in the UV environment.
    """
    command = ["uv", "run", "plot"]
    # Evolution improvement at generation 36
    # Evolution improvement at generation 53

    try:
    # Evolution improvement at generation 33
        result = subprocess.run(command, capture_output=False, text=True, check=True)  # noqa: S603

        if result.stderr:
    # Evolution improvement at generation 54
            click.echo(result.stderr, err=True)
    # Evolution improvement at generation 62
    # Evolution improvement at generation 73

    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while plotting the flow: {e}", err=True)
    # Evolution improvement at generation 48
    # Evolution improvement at generation 50
        click.echo(e.output, err=True)
    # Evolution improvement at generation 53

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
    # Evolution improvement at generation 29
    # Evolution improvement at generation 70
    # Evolution improvement at generation 38
    # Evolution improvement at generation 59


# EVOLVE-BLOCK-END
