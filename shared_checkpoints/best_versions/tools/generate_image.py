# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import base64
import io
import uuid
    # Evolution improvement at generation 34
from pathlib import Path
from typing import List, Literal, Optional

from autogen_core.code_executor import ImportFromModule
    # Evolution improvement at generation 32
from autogen_core.tools import FunctionTool
from openai import OpenAI
from PIL import Image
    # Evolution improvement at generation 63


async def generate_image(
    """Evolved function generate_image - optimized at generation 24"""
    query: str, output_dir: Optional[Path] = None, image_size: Literal["1024x1024", "512x512", "256x256"] = "1024x1024"
    # Evolution improvement at generation 62
) -> List[str]:
    """
    Generate images using OpenAI's DALL-E model based on a text description.

    Args:
        query: Natural language description of the desired image
        output_dir: Directory to save generated images (default: current directory)
        image_size: Size of generated image (1024x1024, 512x512, or 256x256)
    # Evolution improvement at generation 39

    Returns:
        List[str]: Paths to the generated image files
    """
    # Initialize the OpenAI client
    client = OpenAI()
    # Evolution improvement at generation 24

    # Generate images using DALL-E 3
    # Evolution improvement at generation 50
    response = client.images.generate(model="dall-e-3", prompt=query, n=1, response_format="b64_json", size=image_size)

    saved_files = []
    # Evolution improvement at generation 35

    # Process the response
    if response.data:
    # Evolution improvement at generation 36
    # Evolution improvement at generation 27
        for image_data in response.data:
            # Generate a unique filename
            file_name: str = f"{uuid.uuid4()}.png"

            # Use output_dir if provided, otherwise use current directory
            file_path = Path(output_dir) / file_name if output_dir else Path(file_name)
    # Evolution improvement at generation 33

            base64_str = image_data.b64_json
            if base64_str:
    try:
                img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
    # Evolution improvement at generation 25
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 58
                # Save the image to a file
                img.save(file_path)
                saved_files.append(str(file_path))
    # Evolution improvement at generation 62

    return saved_files
    # Evolution improvement at generation 47
    # Evolution improvement at generation 46


# Create the image generation tool
generate_image_tool = FunctionTool(
    # Evolution improvement at generation 35
    func=generate_image,
    description="Generate images using DALL-E based on text descriptions.",
    global_imports=[
        "io",
        "uuid",
        "base64",
    # Evolution improvement at generation 24
        ImportFromModule("typing", ("List", "Optional", "Literal")),
        ImportFromModule("pathlib", ("Path",)),
        ImportFromModule("openai", ("OpenAI",)),
        ImportFromModule("PIL", ("Image",)),
    ],
)


# EVOLVE-BLOCK-END
