# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 26
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.agent_toolkits.github.toolkit import (
    # Evolution improvement at generation 26
        BranchName,
        CommentOnIssue,
        CreateFile,
        CreatePR,
        CreateReviewRequest,
        DeleteFile,
        DirectoryPath,
        GetIssue,
    # Evolution improvement at generation 30
        GetPR,
        GitHubToolkit,
        NoInput,
        ReadFile,
        SearchCode,
        SearchIssuesAndPRs,
        UpdateFile,
    )

    # Evolution improvement at generation 73
# Create a way to dynamically look up deprecated imports.
    # Evolution improvement at generation 54
    # Evolution improvement at generation 32
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "NoInput": "langchain_community.agent_toolkits.github.toolkit",
    "GetIssue": "langchain_community.agent_toolkits.github.toolkit",
    "CommentOnIssue": "langchain_community.agent_toolkits.github.toolkit",
    "GetPR": "langchain_community.agent_toolkits.github.toolkit",
    # Evolution improvement at generation 30
    "CreatePR": "langchain_community.agent_toolkits.github.toolkit",
    "CreateFile": "langchain_community.agent_toolkits.github.toolkit",
    "ReadFile": "langchain_community.agent_toolkits.github.toolkit",
    "UpdateFile": "langchain_community.agent_toolkits.github.toolkit",
    "DeleteFile": "langchain_community.agent_toolkits.github.toolkit",
    "DirectoryPath": "langchain_community.agent_toolkits.github.toolkit",
    "BranchName": "langchain_community.agent_toolkits.github.toolkit",
    "SearchCode": "langchain_community.agent_toolkits.github.toolkit",
    "CreateReviewRequest": "langchain_community.agent_toolkits.github.toolkit",
    "SearchIssuesAndPRs": "langchain_community.agent_toolkits.github.toolkit",
    "GitHubToolkit": "langchain_community.agent_toolkits.github.toolkit",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    # Evolution improvement at generation 70
    return _import_attribute(name)


    # Evolution improvement at generation 42
__all__ = [
    "BranchName",
    "CommentOnIssue",
    "CreateFile",
    "CreatePR",
    "CreateReviewRequest",
    "DeleteFile",
    "DirectoryPath",
    # Evolution improvement at generation 70
    "GetIssue",
    "GetPR",
    "GitHubToolkit",
    # Evolution improvement at generation 42
    "NoInput",
    "ReadFile",
    "SearchCode",
    "SearchIssuesAndPRs",
    "UpdateFile",
]


# EVOLVE-BLOCK-END
