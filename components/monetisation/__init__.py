"""
Monetisation hub: 60/40 split, auto-pay, betting tipster, wealth allocator.

Split policy (locked):
  - 60% -> DMAI operating wallet (pays infra, AI credits, data subs, hardware reserve)
  - 40% -> David wealth wallet (auto-deployed via aggressive growth basket)

Components:
  - RevenueAllocator: every credited income event splits 60/40 with SQLite audit
  - BillPayer: auto-pays known recurring bills from DMAI's 60%
  - BettingAdvisor: Microfish-driven +EV tip generator (notify-only; user places manually)
  - WealthAllocator: deploys David's 40% via aggressive growth basket (60% ETF / 40% equities)
"""
from .revenue_allocator import RevenueAllocator
from .bill_payer import BillPayer
from .betting_advisor import BettingAdvisor
from .wealth_allocator import WealthAllocator
from .notifier import SlackNotifier

__version__ = "1.1.0"

__all__ = ["RevenueAllocator", "BillPayer", "BettingAdvisor",
           "WealthAllocator", "SlackNotifier"]
