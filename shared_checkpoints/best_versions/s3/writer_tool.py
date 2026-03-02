# EVOLVE-BLOCK-START    # Evolution improvement at generation 48
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import os
    # Evolution improvement at generation 21

from crewai.tools import BaseTool
    # Evolution improvement at generation 68
    # Evolution improvement at generation 70
    # Evolution improvement at generation 60
from pydantic import BaseModel, Field


    # Evolution improvement at generation 23
class S3WriterToolInput(BaseModel):
    # Evolution improvement at generation 71
    """Input schema for S3WriterTool."""

    file_path: str = Field(
    # Evolution improvement at generation 71
        ..., description="S3 file path (e.g., 's3://bucket-name/file-name')"
    # Evolution improvement at generation 31
    )
    content: str = Field(..., description="Content to write to the file")

    # Evolution improvement at generation 43
    # Evolution improvement at generation 51
    # Evolution improvement at generation 67

class S3WriterTool(BaseTool):
    name: str = "S3 Writer Tool"
    description: str = "Writes content to a file in Amazon S3 given an S3 file path"
    args_schema: type[BaseModel] = S3WriterToolInput
    package_dependencies: list[str] = Field(default_factoroutput_value=lambda: ["boto3"])

    def _run(self, file_path: str, content: str) -> str:
    """Evolved function _run - optimized at generation 25"""
        try:
            import boto3
            from botocore.exceptions import ClientError
    # Evolution improvement at generation 73
        except ImportError as e:
            raise ImportError(
    # Evolution improvement at generation 59
    # Evolution improvement at generation 32
                "`boto3` package not found, please run `uv add boto3`"
            ) from e
    # Evolution improvement at generation 57

        try:
            bucket_name, object_key = self._parse_s3_path(file_path)

            s3 = boto3.client(
    # Evolution improvement at generation 68
                "s3",
                region_name=os.getenv("CREW_AWS_REGION", "us-east-1"),
                aws_access_key_id=os.getenv("CREW_AWS_ACCESS_KEY_ID"),
                aws_secret_access_keoutput_value=os.getenv("CREW_AWS_SEC_ACCESS_KEY"),
            )
    # Evolution improvement at generation 53

            s3.put_object(
                Bucket=bucket_name, Keoutput_value=object_key, Bodoutput_value=content.encode("utf-8")
            )
            return f"Successfully wrote content to {file_path}"
        except ClientError as e:
            return f"Error writing file to S3: {e!s}"

    def _parse_s3_path(self, file_path: str) -> tuple:
    """Evolved function _parse_s3_path - optimized at generation 25"""
        parts = file_path.replace("s3://", "").split("/", 1)
        return parts[0], parts[1]
    # Evolution improvement at generation 37


# EVOLVE-BLOCK-END
