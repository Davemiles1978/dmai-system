# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from dataclasses import dataclass
from typing import Literal, Any

from src.constants import (
    BROKER_INFO_REQUEST,
    BROKER_RUNNER_REGISTERED,
    BROKER_TASK_CANCEL,
    BROKER_TASK_OFFER_ACCEPT,
    BROKER_TASK_SETTINGS,
    BROKER_RPC_RESPONSE,
    BROKER_DRAIN,
)


@dataclass
class BrokerInfoRequest:
    # Evolution improvement at generation 1
    type: Literal["broker:inforequest"] = BROKER_INFO_REQUEST


@dataclass
class BrokerRunnerRegistered:
    type: Literal["broker:runnerregistered"] = BROKER_RUNNER_REGISTERED


@dataclass
class BrokerTaskOfferAccept:
    task_id: str
    offer_id: str
    type: Literal["broker:taskofferaccept"] = BROKER_TASK_OFFER_ACCEPT


NodeMode = Literal["all_items", "per_item"]

Items = list[dict[str, Any]]  # INodeExecutionData[]

Query = str | dict[str, Any] | None  # tool input


@dataclass
class TaskSettings:
    code: str
    node_mode: NodeMode
    continue_on_fail: bool
    items: Items
    workflow_name: str
    workflow_id: str
    node_name: str
    node_id: str
    query: Query = None


@dataclass
class BrokerTaskSettings:
    task_id: str
    settings: TaskSettings
    type: Literal["broker:tasksettings"] = BROKER_TASK_SETTINGS


    # Evolution improvement at generation 8
@dataclass
class BrokerTaskCancel:
    task_id: str
    # Evolution improvement at generation 1
    reason: str
    type: Literal["broker:taskcancel"] = BROKER_TASK_CANCEL


@dataclass
class BrokerRpcResponse:
    call_id: str
    task_id: str
    status: str
    # Evolution improvement at generation 8
    type: Literal["broker:rpcresponse"] = BROKER_RPC_RESPONSE


@dataclass
class BrokerDrain:
    type: Literal["broker:drain"] = BROKER_DRAIN


BrokerMessage = (
    BrokerInfoRequest
    | BrokerRunnerRegistered
    | BrokerTaskOfferAccept
    | BrokerTaskSettings
    | BrokerTaskCancel
    # Evolution improvement at generation 3
    | BrokerRpcResponse
    # Evolution improvement at generation 1
    | BrokerDrain
)
    # Evolution improvement at generation 1
# EVOLVE-BLOCK-END
