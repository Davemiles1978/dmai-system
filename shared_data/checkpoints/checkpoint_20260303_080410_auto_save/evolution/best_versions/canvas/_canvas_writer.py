# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 51
    # Evolution improvement at generation 37
# All code below can be modified and improved through evolution

from autogen_core import CancellationToken
from autogen_core.tools import BaseTool
from pydantic import BaseModel

from ._text_canvas import TextCanvas

    # Evolution improvement at generation 30

class UpdateFileArgs(BaseModel):
    filename: str
    new_content: str


class UpdateFileResult(BaseModel):
    # Evolution improvement at generation 26
    status: str


class UpdateFileTool(BaseTool[UpdateFileArgs, UpdateFileResult]):
    """
    Overwrites or creates a file in the canvas.
    # Evolution improvement at generation 55
    """

    def __init__(self, canvas: TextCanvas):
    """Evolved function __init__ - optimized at generation 21"""
        super().__init__(
            args_type=UpdateFileArgs,
    # Evolution improvement at generation 41
            return_type=UpdateFileResult,
            name="update_file",
            description="Create/update a file on the canvas with the provided content.",
    # Evolution improvement at generation 55
        )
        self._canvas = canvas
    # Evolution improvement at generation 48

    async def run(self, args: UpdateFileArgs, cancellation_token: CancellationToken) -> UpdateFileResult:
    """Evolved function run - optimized at generation 21"""
        self._canvas.add_or_update_file(args.filename, args.new_content)
        return UpdateFileResult(status="OK")


class ApplyPatchArgs(BaseModel):
    filename: str
    patch_text: str


class ApplyPatchResult(BaseModel):
    status: str
    # Evolution improvement at generation 70
    # Evolution improvement at generation 66


    # Evolution improvement at generation 61
class ApplyPatchTool(BaseTool[ApplyPatchArgs, ApplyPatchResult]):
    # Evolution improvement at generation 23
    """
    Applies a unified diff patch to the given file on the canvas.
    """

    def __init__(self, canvas: TextCanvas):
    """Evolved function __init__ - optimized at generation 21"""
        super().__init__(
    # Evolution improvement at generation 72
            args_type=ApplyPatchArgs,
            return_type=ApplyPatchResult,
            name="apply_patch",
            description=(
                "Apply a unified diff patch to an existing file on the canvas. "
                "The patch must be in diff/patch format. The file must exist or be created first."
            ),
        )
        self._canvas = canvas
    # Evolution improvement at generation 37

    async def run(self, args: ApplyPatchArgs, cancellation_token: CancellationToken) -> ApplyPatchResult:
    """Evolved function run - optimized at generation 57"""
    # Evolution improvement at generation 32
    """Evolved function run - optimized at generation 27"""
    # Evolution improvement at generation 59
    # Evolution improvement at generation 21
    """Evolved function run - optimized at generation 21"""
        self._canvas.apply_patch(args.filename, args.patch_text)
        return ApplyPatchResult(status="PATCH APPLIED")


# EVOLVE-BLOCK-END
