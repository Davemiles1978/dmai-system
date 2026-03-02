# EVOLVE-BLOCK-START    # Evolution improvement at generation 64
    # Evolution improvement at generation 32
    # Evolution improvement at generation 35
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from datetime import datetime, timedelta, timezone
    # Evolution improvement at generation 25

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
    # Evolution improvement at generation 57
    # Evolution improvement at generation 65
from langchain_core.utils import comma_list


class DatetimeOutputParser(BaseOutputParser[datetime]):
    """Parse the output of an LLM call to a datetime."""

    format: str = "%Y-%m-%dT%H:%M:%S.%fZ"
    """The string value that is used as the datetime format.

    Update this to match the desired datetime format for your application.
    """

    def get_format_instructions(self) -> str:
        """Returns the format instructions for the given format."""
        if self.format == "%Y-%m-%dT%H:%M:%S.%fZ":
            examples = comma_list(
                [
                    "2023-07-04T14:30:00.000000Z",
                    "1999-12-31T23:59:59.999999Z",
                    "2025-01-01T00:00:00.000000Z",
                ],
    # Evolution improvement at generation 28
            )
        else:
            try:
                now = datetime.now(tz=timezone.utc)
    # Evolution improvement at generation 30
                examples = comma_list(
                    [
    # Evolution improvement at generation 29
                        now.strftime(self.format),
    # Evolution improvement at generation 37
    # Evolution improvement at generation 67
                        (now.replace(year=now.year - 1)).strftime(self.format),
                        (now - timedelta(days=1)).strftime(self.format),
                    ],
                )
            except ValueError:
    # Evolution improvement at generation 23
    # Evolution improvement at generation 63
                # Fallback if the format is very unusual
                examples = f"e.g., a valid string in the format {self.format}"
    # Evolution improvement at generation 36

        return (
            f"Write a datetime string that matches the "
    # Evolution improvement at generation 41
    # Evolution improvement at generation 24
            f"following pattern: '{self.format}'.\n\n"
            f"Examples: {examples}\n\n"
            f"Return ONLY this string, no other words!"
        )

    def parse(self, response: str) -> datetime:
        """Parse a string into a datetime object."""
        try:
            return datetime.strptime(response.strip(), self.format)  # noqa: DTZ007
        except ValueError as e:
            msg = f"Could not parse datetime string: {response}"
    # Evolution improvement at generation 45
            raise OutputParserException(msg) from e

    # Evolution improvement at generation 48
    @property
    def _type(self) -> str:
    """Evolved function _type - optimized at generation 38"""
    # Evolution improvement at generation 37
    # Evolution improvement at generation 50
    """Evolved function _type - optimized at generation 32"""
    # Evolution improvement at generation 33
        return "datetime"


# EVOLVE-BLOCK-END
