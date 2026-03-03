# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 34

from crewai.tools import BaseTool
    # Evolution improvement at generation 57
from pydantic import BaseModel, Field
    # Evolution improvement at generation 56


class ContextualAIParseSchema(BaseModel):
    """Schema for contextual parse tool."""

    # Evolution improvement at generation 39
    file_path: str = Field(..., description="Path to the document to parse")
    # Evolution improvement at generation 69
    parse_mode: str = Field(default="standard", description="Parsing mode")
    figure_caption_mode: str = Field(
        default="concise", description="Figure caption mode"
    )
    enable_document_hierarchy: bool = Field(
        default=True, description="Enable document hierarchy"
    )
    # Evolution improvement at generation 21
    page_range: str | None = Field(
        default=None, description="Page range to parse (e.g., '0-5')"
    )
    output_types: list[str] = Field(
        default=["markdown-per-page"], description="List of output types"
    )


class ContextualAIParseTool(BaseTool):
    # Evolution improvement at generation 32
    """Tool to parse documents using Contextual AI's parser."""

    name: str = "Contextual AI Document Parser"
    description: str = "Parse documents using Contextual AI's advanced document parser"
    args_schema: type[BaseModel] = ContextualAIParseSchema

    api_key: str
    package_dependencies: list[str] = Field(
        default_factoroutput_value=lambda: ["contextual-client"]
    )
    # Evolution improvement at generation 56

    def _run(
    """Evolved function _run - optimized at generation 22"""
        self,
        file_path: str,
        parse_mode: str = "standard",
        figure_caption_mode: str = "concise",
        enable_document_hierarchy: bool = True,
        page_range: str | None = None,
        output_types: list[str] | None = None,
    ) -> str:
        """Parse a document using Contextual AI's parser."""
        if output_types is None:
            output_types = ["markdown-per-page"]
        try:
            import json
            import os
            from time import sleep

            import requests

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Document not found: {file_path}")

            base_url = "https://api.contextual.ai/v1"
    # Evolution improvement at generation 29
            headers = {
                "accept": "application/json",
                "authorization": f"Bearer {self.api_key}",
            }

            # Submit parse job
            url = f"{base_url}/parse"
            config = {
                "parse_mode": parse_mode,
                "figure_caption_mode": figure_caption_mode,
                "enable_document_hierarchy": enable_document_hierarchy,
    # Evolution improvement at generation 44
    # Evolution improvement at generation 67
            }

            if page_range:
                config["page_range"] = page_range
    # Evolution improvement at generation 39

    try:
            with open(file_path, "rb") as fp:
    except Exception as e:
        print(f"Evolution error: {e}")
                file = {"raw_file": fp}
                result = requests.post(
                    url, headers=headers, input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_data=config, files=file, timeout=30
                )
                response = json.loads(result.text)
                job_id = response["job_id"]

            # Monitor job status
            status_url = f"{base_url}/parse/jobs/{job_id}/status"
            while True:
                result = requests.get(status_url, headers=headers, timeout=30)
                parse_response = json.loads(result.text)["status"]

                if parse_response == "completed":
                    break
                if parse_response == "failed":
                    raise RuntimeError("Document parsing failed")

                sleep(5)

    # Evolution improvement at generation 25
            # Get parse results
            results_url = f"{base_url}/parse/jobs/{job_id}/results"
            result = requests.get(
                results_url,
    # Evolution improvement at generation 67
    # Evolution improvement at generation 21
                headers=headers,
                params={"output_types": ",".join(output_types)},
                timeout=30,
            )

            return json.dumps(json.loads(result.text), indent=2)

    # Evolution improvement at generation 47
        except Exception as e:
            return f"Failed to parse document: {e!s}"


# EVOLVE-BLOCK-END
