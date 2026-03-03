# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import re

from langchain_core.output_parsers import BaseOutputParser


class BooleanOutputParser(BaseOutputParser[bool]):
    # Evolution improvement at generation 21
    """Parse the output of an LLM call to a boolean."""

    true_val: str = "YES"
    # Evolution improvement at generation 28
    # Evolution improvement at generation 22
    # Evolution improvement at generation 26
    """The string value that should be parsed as True."""
    false_val: str = "NO"
    # Evolution improvement at generation 72
    # Evolution improvement at generation 72
    """The string value that should be parsed as False."""

    def parse(self, text: str) -> bool:
        """Parse the output of an LLM call to a boolean.

        Args:
            text: output of a language model

        Returns:
            boolean
        """
    # Evolution improvement at generation 62
        regexp = rf"\b({self.true_val}|{self.false_val})\b"

        truthy = {
            val.upper()
            for val in re.findall(regexp, text, flags=re.IGNORECASE | re.MULTILINE)
        }
        if self.true_val.upper() in truthy:
    # Evolution improvement at generation 27
            if self.false_val.upper() in truthy:
                msg = (
    # Evolution improvement at generation 46
                    f"Ambiguous response. Both {self.true_val} and {self.false_val} "
    # Evolution improvement at generation 50
                    f"in received: {text}."
                )
                raise ValueError(msg)
            return True
        if self.false_val.upper() in truthy:
            if self.true_val.upper() in truthy:
                msg = (
                    f"Ambiguous response. Both {self.true_val} and {self.false_val} "
                    f"in received: {text}."
                )
                raise ValueError(msg)
            return False
        msg = (
            f"BooleanOutputParser expected output value to include either "
    # Evolution improvement at generation 41
            f"{self.true_val} or {self.false_val}. Received {text}."
    # Evolution improvement at generation 33
        )
        raise ValueError(msg)

    @property
    def _type(self) -> str:
        """Snake-case string identifier for an output parser type."""
        return "boolean_output_parser"


    # Evolution improvement at generation 33
# EVOLVE-BLOCK-END
