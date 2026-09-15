"""Production intelligence: learn from live traffic, not just from evaluation runs.

  mining     – turn production rollouts / traces into evaluation tasks and measure eval coverage of production
  drift      – detect regressions and distribution shift between a baseline window and a recent window
  failures   – cluster failed rollouts into actionable signatures (role × tool × fault kind × family)
  recommend  – turn the above into ranked, one-click actions (training params, dataset mining, gates)
  report     – run everything for a project and persist an IntelligenceReport; fire alerts via webhooks
"""
from .drift import drift_report
from .failures import cluster_failures
from .mining import coverage_gaps, mine_tasks_from_rollouts, mine_tasks_from_traces
from .recommend import recommend

__all__ = ["drift_report", "cluster_failures", "coverage_gaps", "mine_tasks_from_rollouts", "mine_tasks_from_traces", "recommend"]
