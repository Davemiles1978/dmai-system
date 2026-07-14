"""Workload self-profiler (PR J).

Hourly-cadence-ish (10 min in production) sampler that records
process resource usage and SQLite growth so PR K can price a
home-lab replacement for Render.

Public entry points:
    from components.workload import workload_profiler as wp
    wp.init_workload_db()
    wp.sample_now()
    wp.get_recent(hours=24)
    wp.get_daily_rollup(days=7)
    wp.get_db_growth()

    from components.workload.workload_loop import start_workload_loop
    start_workload_loop()
"""
