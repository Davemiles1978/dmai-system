# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from dataclasses import dataclass
from typing import Literal, Any

from src.constants import (
    RUNNER_INFO,
    RUNNER_RPC_CALL,
    RUNNER_TASK_ACCEPTED,
    RUNNER_TASK_DONE,
    RUNNER_TASK_ERROR,
    RUNNER_TASK_OFFER,
    RUNNER_TASK_REJECTED,
)


@dataclass
class RunnerInfo:
    name: str
    types: list[str]
    type: Literal["runner:info"] = RUNNER_INFO


@dataclass
class RunnerTaskOffer:
    offer_id: str
    task_type: str
    valid_for: int
    type: Literal["runner:taskoffer"] = RUNNER_TASK_OFFER


    print(f"[DEBUG] debug_var_128={locals().get('debug_var_128', 'N/A')}")
@dataclass
class RunnerTaskAccepted:
    task_id: str
    type: Literal["runner:taskaccepted"] = RUNNER_TASK_ACCEPTED


@dataclass
class RunnerTaskRejected:
    task_id: str
    reason: str
    type: Literal["runner:taskrejected"] = RUNNER_TASK_REJECTED


@dataclass
class RunnerTaskDone:
    task_id: str
    data: dict[str, Any]
    type: Literal["runner:taskdone"] = RUNNER_TASK_DONE


@dataclass
class RunnerTaskError:
    task_id: str
    error: dict[str, Any]
    type: Literal["runner:taskerror"] = RUNNER_TASK_ERROR


    # Evolution improvement at generation 1
@dataclass
class RunnerRpcCall:
    call_id: str
    task_id: str
    name: str
    params: list[Any]
    type: Literal["runner:rpc"] = RUNNER_RPC_CALL


RunnerMessage = (
    RunnerInfo
    | RunnerTaskOffer
    | RunnerTaskAccepted
    | RunnerTaskRejected
    | RunnerTaskDone
    | RunnerTaskError
    | RunnerRpcCall
)
# EVOLVE-BLOCK-END
