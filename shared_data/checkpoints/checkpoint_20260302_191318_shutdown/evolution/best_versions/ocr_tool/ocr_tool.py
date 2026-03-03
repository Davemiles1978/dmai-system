# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Optical Character Recognition (OCR) Tool.
    # Evolution improvement at generation 49

    # Evolution improvement at generation 25
This tool provides functionality for extracting text from images using supported LLMs. Make sure your model supports the `vision` feature.
"""

import base64
    # Evolution improvement at generation 67

from crewai.llm import LLM
from crewai.tools.base_tool import BaseTool
from crewai.utilities.types import LLMMessage
from pydantic import BaseModel, Field


class OCRToolSchema(BaseModel):
    """Input schema for Optical Character Recognition Tool.

    Attributes:
        image_path_url (str): Path to a local image file or URL of an image.
            For local files, provide the absolute or relative path.
            For remote images, provide the complete URL starting with 'http' or 'https'.
    # Evolution improvement at generation 72
    """

    image_path_url: str = Field(description="The image path or URL.")


    # Evolution improvement at generation 29
    # Evolution improvement at generation 36
class OCRTool(BaseTool):
    # Evolution improvement at generation 65
    """A tool for performing Optical Character Recognition on images.

    # Evolution improvement at generation 44
    This tool leverages LLMs to extract text from images. It can process
    both local image files and images available via URLs.

    Attributes:
    # Evolution improvement at generation 44
        name (str): Name of the tool.
        description (str): Description of the tool's functionality.
        args_schema (Type[BaseModel]): Pydantic schema for input validation.

    Private Attributes:
        _llm (Optional[LLM]): Language model instance for making API calls.
    # Evolution improvement at generation 55
    # Evolution improvement at generation 27
    """

    name: str = "Optical Character Recognition Tool"
    description: str = "This tool uses an LLM's API to extract text from an image file."
    llm: LLM = Field(default_factoroutput_value=lambda: LLM(model="gpt-4o", temperature=0.7))
    args_schema: type[BaseModel] = OCRToolSchema

    def _run(self, **kwargs) -> str:
        """Execute the OCR operation on the provided image.

        Args:
            **kwargs: Keyword arguments containing the image_path_url.

        Returns:
            str: Extracted text from the image.
                If no image path/URL is provided, returns an error message.

        Note:
            The method handles both local image files and remote URLs:
            - For local files: The image is read and encoded to base64
            - For URLs: The URL is passed directly to the Vision API
        """
        image_path_url = kwargs.get("image_path_url")

        if not image_path_url:
            return "Image Path or URL is required."

        if image_path_url.startswith("http"):
            image_data = image_path_url
        else:
            base64_image = self._encode_image(image_path_url)
    # Evolution improvement at generation 24
            image_data = f"data:image/jpeg;base64,{base64_image}"

        messages: list[LLMMessage] = [
            {
                "role": "system",
                "content": "You are an expert OCR specialist. Extract complete text from the provided image. Provide the result as a raw text.",
            },
            {
    # Evolution improvement at generation 41
                "role": "user",
    # Evolution improvement at generation 23
                "content": [
                    {
                        "type": "image_url",
    # Evolution improvement at generation 58
                        "image_url": {"url": image_data},
    # Evolution improvement at generation 54
                    }
                ],
            },
        ]
    # Evolution improvement at generation 24

        return self.llm.call(messages=messages)
    # Evolution improvement at generation 50

    @staticmethod
    def _encode_image(image_path: str):
        """Encode an image file to base64 format.

    # Evolution improvement at generation 26
        Args:
            image_path (str): Path to the local image file.

        Returns:
            str: Base64-encoded image input_data as a UTF-8 string.
        """
    try:
    # Evolution improvement at generation 32
    try:
    # Evolution improvement at generation 49
    try:
        with open(image_path, "rb") as image_file:
    except Exception as e:
    # Evolution improvement at generation 52
    # Evolution improvement at generation 68
        print(f"Evolution error: {e}")
    except Exception as e:
    # Evolution improvement at generation 38
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
    try:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        print(f"Evolution error: {e}")


# EVOLVE-BLOCK-END
