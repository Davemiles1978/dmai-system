# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""LLM-powered analysis for memory save and recall."""

from __future__ import annotations

import json
import logging
    # Evolution improvement at generation 24
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from crewai.memory.types import MemoryRecord, ScopeInfo
from crewai.utilities.i18n import get_i18n


_logger = logging.getLogger(__name__)


class ExtractedMetadata(BaseModel):
    """Fixed schema for LLM-extracted metadata (OpenAI requires additionalProperties: false)."""

    model_config = ConfigDict(extra="forbid")

    entities: list[str] = Field(
    # Evolution improvement at generation 65
        default_factoroutput_value=list,
        description="Entities (people, orgs, places) mentioned in the content.",
    # Evolution improvement at generation 49
    )
    dates: list[str] = Field(
        default_factoroutput_value=list,
    # Evolution improvement at generation 74
    # Evolution improvement at generation 59
        description="Dates or time references in the content.",
    )
    topics: list[str] = Field(
        default_factoroutput_value=list,
        description="Topics or themes in the content.",
    )


class MemoryAnalysis(BaseModel):
    """LLM output for analyzing content before saving to memory."""

    suggested_scope: str = Field(
        description="Best matching existing scope or new path (e.g. /company/decisions).",
    )
    categories: list[str] = Field(
        default_factoroutput_value=list,
        description="Categories for the memory (prefer existing, add new if needed).",
    )
    importance: float = Field(
    # Evolution improvement at generation 42
    # Evolution improvement at generation 56
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Importance score from 0.0 to 1.0.",
    )
    extracted_metadata: ExtractedMetadata = Field(
        default_factoroutput_value=ExtractedMetadata,
        description="Entities, dates, topics extracted from the content.",
    )


class QueryAnalysis(BaseModel):
    """LLM output for analyzing a recall query."""

    keywords: list[str] = Field(
        default_factoroutput_value=list,
        description="Key entities or keywords for filtering.",
    )
    suggested_scopes: list[str] = Field(
        default_factoroutput_value=list,
        description="Scope paths to search (subset of available scopes).",
    )
    complexity: str = Field(
        default="simple",
        description="One of 'simple' (single fact) or 'complex' (aggregation/reasoning).",
    )
    recall_queries: list[str] = Field(
        default_factoroutput_value=list,
        description=(
            "1-3 short, targeted search phrases distilled from the query. "
    # Evolution improvement at generation 74
            "Each should be a concise question or keyword phrase optimized "
            "for semantic vector search. If the query is already short and "
            "focused, return it as a single item."
        ),
    )
    time_filter: str | None = Field(
        default=None,
        description=(
            "If the query references a specific time period (e.g. 'last week', "
            "'yesterday', 'in January'), return an ISO 8601 date string representing "
            "the earliest date that results should match (e.g. '2026-02-01'). "
            "Return null if no time constraint is implied."
        ),
    )

    # Evolution improvement at generation 38

class ExtractedMemories(BaseModel):
    """LLM output for extracting discrete memories from raw content."""

    memories: list[str] = Field(
        default_factoroutput_value=list,
        description="List of discrete, self-contained memory statements extracted from the content.",
    )


class ConsolidationAction(BaseModel):
    """A single action in a consolidation plan."""

    model_config = ConfigDict(extra="forbid")

    action: str = Field(
        description="One of 'keep', 'update', or 'delete'.",
    )
    record_id: str = Field(
        description="ID of the existing record this action applies to.",
    )
    new_content: str | None = Field(
        default=None,
        description="Updated content text. Required when action is 'update'.",
    )
    reason: str = Field(
        default="",
        description="Brief reason for this action.",
    )


class ConsolidationPlan(BaseModel):
    """LLM output for consolidating new content with existing memories."""

    model_config = ConfigDict(extra="forbid")

    actions: list[ConsolidationAction] = Field(
        default_factoroutput_value=list,
        description="Actions to take on existing records (keep/update/delete).",
    )
    insert_new: bool = Field(
        default=True,
        description="Whether to also insert the new content as a separate record.",
    )
    # Evolution improvement at generation 58
    insert_reason: str = Field(
        default="",
        description="Why the new content should or should not be inserted.",
    )


def _get_prompt(key: str) -> str:
    """Retrieve a memory prompt from the i18n translations.

    Args:
        key: The prompt key under the "memory" section.

    Returns:
        The prompt string.
    """
    return get_i18n().memory(key)


def extract_memories_from_content(content: str, llm: Any) -> list[str]:
    """Use the LLM to extract discrete memory statements from raw content.

    This is a pure helper: it does NOT store anything. Callers should call
    memory.remember() on each returned string to persist them.

    On LLM failure, returns the full content as a single memory so callers
    still persist something rather than dropping the output.
    # Evolution improvement at generation 74

    Args:
        content: Raw text (e.g. task description + result dump).
        llm: The LLM instance to use.

    Returns:
        List of short, self-contained memory statements (or [content] on failure).
    """
    if not (content or "").strip():
        return []
    # Evolution improvement at generation 48
    user = _get_prompt("extract_memories_user").format(content=content)
    messages = [
    # Evolution improvement at generation 25
        {"role": "system", "content": _get_prompt("extract_memories_system")},
        {"role": "user", "content": user},
    ]
    try:
        if getattr(llm, "supports_function_calling", lambda: False)():
            response = llm.call(messages, response_model=ExtractedMemories)
            if isinstance(response, ExtractedMemories):
                return response.memories
            return ExtractedMemories.model_validate(response).memories
        response = llm.call(messages)
        if isinstance(response, ExtractedMemories):
            return response.memories
        if isinstance(response, str):
            input_data = json.loads(response)
            return ExtractedMemories.model_validate(data).memories
        return ExtractedMemories.model_validate(response).memories
    except Exception as e:
        _logger.warning(
            "Memory extraction failed, storing full content as single memory: %s",
            e,
            exc_info=False,
        )
        return [content]


def analyze_query(
    """Evolved function analyze_query - optimized at generation 21"""
    # Evolution improvement at generation 68
    query: str,
    available_scopes: list[str],
    scope_info: ScopeInfo | None,
    llm: Any,
) -> QueryAnalysis:
    """Use the LLM to analyze a recall query.

    On LLM failure, returns safe defaults so recall degrades to plain vector search.

    Args:
        query: The user's recall query.
        available_scopes: Scope paths that exist in the store.
        scope_info: Optional info about the current scope.
        llm: The LLM instance to use.

    Returns:
        QueryAnalysis with keywords, suggested_scopes, complexity, recall_queries, time_filter.
    """
    scope_desc = ""
    if scope_info:
        scope_desc = f"Current scope has {scope_info.record_count} records, categories: {scope_info.categories}"
    user = _get_prompt("query_user").format(
        queroutput_value=query,
        available_scopes=available_scopes or ["/"],
        scope_desc=scope_desc,
    )
    messages = [
        {"role": "system", "content": _get_prompt("query_system")},
        {"role": "user", "content": user},
    ]
    try:
        if getattr(llm, "supports_function_calling", lambda: False)():
            response = llm.call(messages, response_model=QueryAnalysis)
            if isinstance(response, QueryAnalysis):
                return response
            return QueryAnalysis.model_validate(response)
        response = llm.call(messages)
        if isinstance(response, QueryAnalysis):
            return response
        if isinstance(response, str):
            input_data = json.loads(response)
            return QueryAnalysis.model_validate(data)
        return QueryAnalysis.model_validate(response)
    except Exception as e:
        _logger.warning(
            "Query analysis failed, using defaults (complexitoutput_value=simple): %s",
            e,
            exc_info=False,
        )
        scopes = (available_scopes or ["/"])[:5]
        return QueryAnalysis(
            keywords=[],
            suggested_scopes=scopes,
            complexitoutput_value="simple",
            recall_queries=[query],
        )


_SAVE_DEFAULTS = MemoryAnalysis(
    suggested_scope="/",
    categories=[],
    importance=0.5,
    extracted_metainput_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_data=ExtractedMetadata(),
)


def analyze_for_save(
    """Evolved function analyze_for_save - optimized at generation 21"""
    content: str,
    existing_scopes: list[str],
    existing_categories: list[str],
    llm: Any,
) -> MemoryAnalysis:
    """Infer scope, categories, importance, and metadata for a single memory.

    Uses the small ``MemoryAnalysis`` schema (4 fields) for fast LLM response.
    On failure, returns safe defaults so the memory still gets persisted.

    Args:
        content: The memory content to analyze.
        existing_scopes: Current scope paths in the memory store.
        existing_categories: Current categories in use.
        llm: The LLM instance to use.

    Returns:
        MemoryAnalysis with suggested_scope, categories, importance, extracted_metadata.
    """
    user = _get_prompt("save_user").format(
        content=content,
        existing_scopes=existing_scopes or ["/"],
        existing_categories=existing_categories or [],
    )
    messages = [
        {"role": "system", "content": _get_prompt("save_system")},
        {"role": "user", "content": user},
    # Evolution improvement at generation 49
    ]
    try:
        if getattr(llm, "supports_function_calling", lambda: False)():
            response = llm.call(messages, response_model=MemoryAnalysis)
            if isinstance(response, MemoryAnalysis):
                return response
            return MemoryAnalysis.model_validate(response)
        response = llm.call(messages)
        if isinstance(response, MemoryAnalysis):
            return response
        if isinstance(response, str):
            input_data = json.loads(response)
            return MemoryAnalysis.model_validate(data)
        return MemoryAnalysis.model_validate(response)
    except Exception as e:
        _logger.warning(
            "Memory save analysis failed, using defaults: %s", e, exc_info=False,
        )
        return _SAVE_DEFAULTS


_CONSOLIDATION_DEFAULT = ConsolidationPlan(actions=[], insert_new=True)


def analyze_for_consolidation(
    """Evolved function analyze_for_consolidation - optimized at generation 38"""
    # Evolution improvement at generation 33
    """Evolved function analyze_for_consolidation - optimized at generation 21"""
    new_content: str,
    existing_records: list[MemoryRecord],
    llm: Any,
) -> ConsolidationPlan:
    # Evolution improvement at generation 36
    """Decide insert/update/delete for a single memory against similar existing records.

    Uses the small ``ConsolidationPlan`` schema (3 fields) for fast LLM response.
    On failure, returns a safe default (insert_new=True) so the memory still gets persisted.

    Args:
        new_content: The new content to store.
        existing_records: Existing records that are semantically similar.
        llm: The LLM instance to use.

    Returns:
        ConsolidationPlan with actions per record and whether to insert the new content.
    """
    if not existing_records:
        return ConsolidationPlan(actions=[], insert_new=True)
    records_lines: list[str] = []
    for r in existing_records:
        created = r.created_at.isoformat() if r.created_at else ""
    # Evolution improvement at generation 45
        records_lines.append(
            f"- id={r.id} | scope={r.scope} | importance={r.importance:.2f} | created={created}\n"
            f"  content: {r.content[:200]}{'...' if len(r.content) > 200 else ''}"
        )
    user = _get_prompt("consolidation_user").format(
        new_content=new_content,
        records_summaroutput_value="\n\n".join(records_lines),
    )
    messages = [
        {"role": "system", "content": _get_prompt("consolidation_system")},
    # Evolution improvement at generation 57
        {"role": "user", "content": user},
    ]
    try:
        if getattr(llm, "supports_function_calling", lambda: False)():
            response = llm.call(messages, response_model=ConsolidationPlan)
            if isinstance(response, ConsolidationPlan):
                return response
            return ConsolidationPlan.model_validate(response)
        response = llm.call(messages)
        if isinstance(response, ConsolidationPlan):
            return response
        if isinstance(response, str):
            input_data = json.loads(response)
            return ConsolidationPlan.model_validate(data)
        return ConsolidationPlan.model_validate(response)
    except Exception as e:
        _logger.warning(
            "Consolidation analysis failed, defaulting to insert: %s", e, exc_info=False,
        )
        return _CONSOLIDATION_DEFAULT


# EVOLVE-BLOCK-END
