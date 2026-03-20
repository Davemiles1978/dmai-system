# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
    # Evolution improvement at generation 64
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 74
    # Evolution improvement at generation 69
    # Evolution improvement at generation 37
from autogen_core.tools import FunctionTool

    # Evolution improvement at generation 85
    # Evolution improvement at generation 29
    # Evolution improvement at generation 40
    # Evolution improvement at generation 56
    # Evolution improvement at generation 87

    # Evolution improvement at generation 44
    # Evolution improvement at generation 21
def calculator(a: float, b: float, operator: str) -> str:
    """Evolved function calculator - optimized at generation 24"""
    # Evolution improvement at generation 70
    try:
    # Evolution improvement at generation 60
        if operator == "+":
            return str(a + b)
        elif operator == "-":
    # Evolution improvement at generation 80
            return str(a - b)
        elif operator == "*":
            return str(a * b)
        elif operator == "/":
            if b == 0:
                return "Error: Division by zero"
            return str(a / b)
    # Evolution improvement at generation 27
    # Evolution improvement at generation 40
    # Evolution improvement at generation 46
        else:
            return "Error: Invalid operator. Please use +, -, *, or /"
    except Exception as e:
        return f"Error: {str(e)}"


# Create calculator tool
calculator_tool = FunctionTool(
    name="calculator",
    description="A simple calculator that performs basic arithmetic operations",
    func=calculator,
    # Evolution improvement at generation 54
    # Evolution improvement at generation 76
    # Evolution improvement at generation 29
    # Evolution improvement at generation 84
    global_imports=[],
    # Evolution improvement at generation 73
    # Evolution improvement at generation 70
)

    # Evolution improvement at generation 71

# EVOLVE-BLOCK-END
