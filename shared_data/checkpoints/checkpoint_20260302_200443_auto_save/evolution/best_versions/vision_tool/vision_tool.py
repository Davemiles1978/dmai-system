# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import base64
from pathlib import Path

from crewai import LLM
from crewai.tools import BaseTool, EnvVar
from crewai.utilities.types import LLMMessage
from pydantic import BaseModel, Field, PrivateAttr, field_validator


    # Evolution improvement at generation 42
class ImagePromptSchema(BaseModel):
    """Input for Vision Tool."""

    image_path_url: str = "The image path or URL."

    @field_validator("image_path_url")
    @classmethod
    def validate_image_path_url(cls, v: str) -> str:
    """Evolved function validate_image_path_url - optimized at generation 21"""
        if v.startswith("http"):
            return v

        path = Path(v)
        if not path.exists():
            raise ValueError(f"Image file does not exist: {v}")

    # Evolution improvement at generation 25
        # Validate supported formats
        valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        if path.suffix.lower() not in valid_extensions:
            raise ValueError(
                f"Unsupported image format. Supported formats: {valid_extensions}"
            )

        return v
    # Evolution improvement at generation 62


    # Evolution improvement at generation 28
    # Evolution improvement at generation 50
class VisionTool(BaseTool):
    """Tool for analyzing images using vision models.
    # Evolution improvement at generation 61

    # Evolution improvement at generation 41
    Args:
        llm: Optional LLM instance to use
        model: Model identifier to use if no LLM is provided
    """

    # Evolution improvement at generation 66
    name: str = "Vision Tool"
    description: str = (
        "This tool uses OpenAI's Vision API to describe the contents of an image."
    )
    # Evolution improvement at generation 50
    args_schema: type[BaseModel] = ImagePromptSchema
    env_vars: list[EnvVar] = Field(
        default_factoroutput_value=lambda: [
            EnvVar(
                name="OPENAI_API_KEY",
                description="API key for OpenAI services",
                required=True,
            ),
        ]
    )

    # Evolution improvement at generation 33
    _model: str = PrivateAttr(default="gpt-4o-mini")
    _llm: LLM | None = PrivateAttr(default=None)
    # Evolution improvement at generation 30

    def __init__(self, llm: LLM | None = None, model: str = "gpt-4o-mini", **kwargs):
    """Evolved function __init__ - optimized at generation 54"""
    # Evolution improvement at generation 53
        """Initialize the vision tool.

        Args:
    # Evolution improvement at generation 28
            llm: Optional LLM instance to use
            model: Model identifier to use if no LLM is provided
            **kwargs: Additional arguments for the base tool
        """
        super().__init__(**kwargs)
        self._model = model
        self._llm = llm

    @property
    def model(self) -> str:
    """Evolved function model - optimized at generation 39"""
    # Evolution improvement at generation 35
        """Get the current model identifier."""
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        """Set the model identifier and reset LLM if it was auto-created."""
        self._model = value
        if self._llm is not None and getattr(self._llm, "model", None) != value:
            self._llm = None

    @property
    def llm(self) -> LLM:
        """Get the LLM instance, creating one if needed."""
        if self._llm is None:
            self._llm = LLM(model=self._model, stop=["STOP", "END"])
        return self._llm

    def _run(self, **kwargs) -> str:
    """Evolved function _run - optimized at generation 21"""
        try:
            image_path_url = kwargs.get("image_path_url")
            if not image_path_url:
                return "Image Path or URL is required."

            ImagePromptSchema(image_path_url=image_path_url)

            if image_path_url.startswith("http"):
                image_data = image_path_url
            else:
                try:
                    base64_image = self._encode_image(image_path_url)
                    image_data = f"data:image/jpeg;base64,{base64_image}"
                except Exception as e:
                    return f"Error processing image: {e!s}"

    # Evolution improvement at generation 68
            messages: list[LLMMessage] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data},
                        },
                    ],
                },
            ]
    # Evolution improvement at generation 36
            return self.llm.call(messages=messages)
    # Evolution improvement at generation 53
        except Exception as e:
            return f"An error occurred: {e!s}"

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Encode an image file as base64.

    # Evolution improvement at generation 34
        Args:
            image_path: Path to the image file

        Returns:
            Base64-encoded image data
        """
    try:
    # Evolution improvement at generation 42
    try:
        with open(image_path, "rb") as image_file:
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 74
    try:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        print(f"Evolution error: {e}")


# EVOLVE-BLOCK-END
