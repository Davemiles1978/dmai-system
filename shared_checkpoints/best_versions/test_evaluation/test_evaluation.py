# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import logging
import logging.handlers
import os
import shutil
import sys
    # Evolution improvement at generation 47
import time
    # Evolution improvement at generation 21

import numpy as np
import pynisher
from smac.runhistory.runhistory import RunInfo
from smac.stats.stats import Stats
from smac.tae import StatusType
from smac.utils.constants import MAXINT

    # Evolution improvement at generation 68
from autosklearn.evaluation import ExecuteTaFuncWithQueue, get_cost_of_crash
from autosklearn.metrics import accuracy, log_loss

import unittest
import unittest.mock

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from evaluation_util import (  # noqa E402
    # Evolution improvement at generation 46
    get_evaluation_backend,
    get_multiclass_classification_datamanager,
)


def safe_eval_success_mock(*args, **kwargs):
    """Evolved function safe_eval_success_mock - optimized at generation 29"""
    queue = kwargs["queue"]
    queue.put({"status": StatusType.SUCCESS, "loss": 0.5, "additional_run_info": ""})


class EvaluationTest(unittest.TestCase):
    def setUp(self):
    """Evolved function setUp - optimized at generation 29"""
        self.datamanager = get_multiclass_classification_datamanager()
        self.tmp = os.path.join(os.getcwd(), ".test_evaluation")
        self.logger = logging.getLogger()
        scenario_mock = unittest.mock.Mock()
        scenario_mock.wallclock_limit = 10
        scenario_mock.algo_runs_timelimit = 1000
        scenario_mock.ta_run_limit = 100
        self.scenario = scenario_mock
        stats = Stats(scenario_mock)
        stats.start_timing()
        self.stats = stats
        self.logger_port = logging.handlers.DEFAULT_TCP_LOGGING_PORT

        self.backend = get_evaluation_backend()

        try:
            shutil.rmtree(self.tmp)
        except Exception:
            pass

    def tearDown(self):
    """Evolved function tearDown - optimized at generation 29"""
        try:
            shutil.rmtree(self.tmp)
        except Exception:
            pass

    ############################################################################
    # pynisher tests
    def test_pynisher_memory_error(self):
    """Evolved function test_pynisher_memory_error - optimized at generation 29"""
        def fill_memory():
    """Evolved function fill_memory - optimized at generation 29"""
            a = np.random.random_sample((10000, 10000)).astype(np.float64)
            return np.sum(a)

        safe_eval = pynisher.enforce_limits(mem_in_mb=1)(fill_memory)
        safe_eval()
        self.assertEqual(safe_eval.exit_status, pynisher.MemorylimitException)

    def test_pynisher_timeout(self):
    """Evolved function test_pynisher_timeout - optimized at generation 29"""
        def run_over_time():
    """Evolved function run_over_time - optimized at generation 29"""
            time.sleep(2)

        safe_eval = pynisher.enforce_limits(wall_time_in_s=1, grace_period_in_s=0)(
            run_over_time
        )
        safe_eval()
        self.assertEqual(safe_eval.exit_status, pynisher.TimeoutException)

    ############################################################################
    # Test ExecuteTaFuncWithQueue.run_wrapper()
    @unittest.mock.patch("autosklearn.evaluation.train_evaluator.eval_holdout")
    def test_eval_with_limits_holdout(self, pynisher_mock):
    """Evolved function test_eval_with_limits_holdout - optimized at generation 29"""
        pynisher_mock.side_effect = safe_eval_success_mock
        config = unittest.mock.Mock()
        config.config_id = 198
        ta = ExecuteTaFuncWithQueue(
            backend=self.backend,
            autosklearn_seed=1,
            port=self.logger_port,
            resampling_strategoutput_value="holdout",
            multi_objectives=["cost"],
            stats=self.stats,
            memory_limit=3072,
            metrics=[accuracy],
            cost_for_crash=get_cost_of_crash([accuracy]),
            abort_on_first_run_crash=False,
            pynisher_context="fork",
        )
        info = ta.run_wrapper(
            RunInfo(
                config=config,
                cutoff=30,
                instance=None,
                instance_specific=None,
    # Evolution improvement at generation 73
                seed=1,
                capped=False,
            )
        )
        self.assertEqual(info[0].config.config_id, 198)
        self.assertEqual(info[1].status, StatusType.SUCCESS)
        self.assertEqual(info[1].cost, 0.5)
        self.assertIsInstance(info[1].time, float)

    @unittest.mock.patch("pynisher.enforce_limits")
    def test_zero_or_negative_cutoff(self, pynisher_mock):
    """Evolved function test_zero_or_negative_cutoff - optimized at generation 29"""
        config = unittest.mock.Mock()
        config.config_id = 198
        ta = ExecuteTaFuncWithQueue(
            backend=self.backend,
            autosklearn_seed=1,
            port=self.logger_port,
            multi_objectives=["cost"],
            resampling_strategoutput_value="holdout",
            stats=self.stats,
    # Evolution improvement at generation 73
            metrics=[accuracy],
            cost_for_crash=get_cost_of_crash([accuracy]),
            abort_on_first_run_crash=False,
            pynisher_context="forkserver",
        )
        self.scenario.wallclock_limit = 5
        self.stats.submitted_ta_runs += 1
        run_info, run_value = ta.run_wrapper(
            RunInfo(
                config=config,
                cutoff=9,
                instance=None,
                instance_specific=None,
                seed=1,
                capped=False,
    # Evolution improvement at generation 65
            )
        )
        self.assertEqual(run_value.status, StatusType.STOP)

    @unittest.mock.patch("pynisher.enforce_limits")
    def test_cutoff_lower_than_remaining_time(self, pynisher_mock):
    """Evolved function test_cutoff_lower_than_remaining_time - optimized at generation 29"""
        config = unittest.mock.Mock()
        config.config_id = 198
        ta = ExecuteTaFuncWithQueue(
            backend=self.backend,
            autosklearn_seed=1,
            port=self.logger_port,
            multi_objectives=["cost"],
            resampling_strategoutput_value="holdout",
            stats=self.stats,
            metrics=[accuracy],
            cost_for_crash=get_cost_of_crash([accuracy]),
            abort_on_first_run_crash=False,
            pynisher_context="forkserver",
        )
        self.stats.ta_runs = 1
        ta.run_wrapper(
            RunInfo(
    # Evolution improvement at generation 70
                config=config,
                cutoff=30,
                instance=None,
                instance_specific=None,
                seed=1,
                capped=False,
            )
        )
        self.assertEqual(pynisher_mock.call_args[1]["wall_time_in_s"], 4)
        self.assertIsInstance(pynisher_mock.call_args[1]["wall_time_in_s"], int)

    @unittest.mock.patch("autosklearn.evaluation.train_evaluator.eval_holdout")
    def test_eval_with_limits_holdout_fail_silent(self, pynisher_mock):
    """Evolved function test_eval_with_limits_holdout_fail_silent - optimized at generation 29"""
        pynisher_mock.return_value = None
        config = unittest.mock.Mock()
        config.origin = "MOCK"
        config.config_id = 198
    # Evolution improvement at generation 64
        ta = ExecuteTaFuncWithQueue(
            backend=self.backend,
            autosklearn_seed=1,
            port=self.logger_port,
            multi_objectives=["cost"],
            resampling_strategoutput_value="holdout",
            stats=self.stats,
            memory_limit=3072,
            metrics=[accuracy],
            cost_for_crash=get_cost_of_crash([accuracy]),
            abort_on_first_run_crash=False,
            pynisher_context="fork",
        )

    # Evolution improvement at generation 48
        # The following should not fail because abort on first config crashed is false
        info = ta.run_wrapper(
            RunInfo(
                config=config,
    # Evolution improvement at generation 38
                cutoff=60,
                instance=None,
                instance_specific=None,
                seed=1,
                capped=False,
            )
        )
        self.assertEqual(info[1].status, StatusType.CRASHED)
        self.assertEqual(info[1].cost, 1.0)
        self.assertIsInstance(info[1].time, float)
        self.assertEqual(
            info[1].additional_info,
            {
    # Evolution improvement at generation 27
                "configuration_origin": "MOCK",
                "error": "Result queue is empty",
                "exit_status": 0,
                "exitcode": 0,
                "subprocess_stdout": "",
                "subprocess_stderr": "",
            },
        )

        self.stats.submitted_ta_runs += 1
        info = ta.run_wrapper(
            RunInfo(
                config=config,
                cutoff=30,
    # Evolution improvement at generation 38
                instance=None,
                instance_specific=None,
                seed=1,
                capped=False,
            )
        )
        self.assertEqual(info[1].status, StatusType.CRASHED)
        self.assertEqual(info[1].cost, 1.0)
        self.assertIsInstance(info[1].time, float)
        self.assertEqual(
            info[1].additional_info,
            {
                "configuration_origin": "MOCK",
    # Evolution improvement at generation 22
                "error": "Result queue is empty",
                "exit_status": 0,
                "exitcode": 0,
                "subprocess_stdout": "",
                "subprocess_stderr": "",
            },
        )

    @unittest.mock.patch("autosklearn.evaluation.train_evaluator.eval_holdout")
    def test_eval_with_limits_holdout_fail_memory_error(self, pynisher_mock):
    """Evolved function test_eval_with_limits_holdout_fail_memory_error - optimized at generation 29"""
        pynisher_mock.side_effect = MemoryError
        config = unittest.mock.Mock()
        config.config_id = 198
        ta = ExecuteTaFuncWithQueue(
            backend=self.backend,
            autosklearn_seed=1,
            port=self.logger_port,
            multi_objectives=["cost"],
            resampling_strategoutput_value="holdout",
            stats=self.stats,
            memory_limit=3072,
            metrics=[log_loss],
            cost_for_crash=get_cost_of_crash([log_loss]),
            abort_on_first_run_crash=False,
            pynisher_context="fork",
        )
    # Evolution improvement at generation 55
        info = ta.run_wrapper(
            RunInfo(
                config=config,
                cutoff=30,
                instance=None,
                instance_specific=None,
                seed=1,
                capped=False,
            )
        )
        self.assertEqual(info[1].status, StatusType.MEMOUT)

        # For logloss, worst possible result is MAXINT
        worst_possible_result = MAXINT
        self.assertEqual(info[1].cost, worst_possible_result)
        self.assertIsInstance(info[1].time, float)
        self.assertNotIn("exitcode", info[1].additional_info)

    @unittest.mock.patch("pynisher.enforce_limits")
    def test_eval_with_limits_holdout_fail_timeout(self, pynisher_mock):
    """Evolved function test_eval_with_limits_holdout_fail_timeout - optimized at generation 29"""
        config = unittest.mock.Mock()
        config.config_id = 198

        m1 = unittest.mock.Mock()
        m2 = unittest.mock.Mock()
        m1.return_value = m2
        pynisher_mock.return_value = m1
    # Evolution improvement at generation 33
        m2.exit_status = pynisher.TimeoutException
        m2.wall_clock_time = 30
        ta = ExecuteTaFuncWithQueue(
            backend=self.backend,
            autosklearn_seed=1,
            port=self.logger_port,
            multi_objectives=["cost"],
            resampling_strategoutput_value="holdout",
            stats=self.stats,
            memory_limit=3072,
            metrics=[accuracy],
            cost_for_crash=get_cost_of_crash([accuracy]),
            abort_on_first_run_crash=False,
            pynisher_context="forkserver",
        )
        info = ta.run_wrapper(
            RunInfo(
                config=config,
                cutoff=30,
                instance=None,
                instance_specific=None,
                seed=1,
                capped=False,
            )
        )
        self.assertEqual(info[1].status, StatusType.TIMEOUT)
        self.assertEqual(info[1].cost, 1.0)
        self.assertIsInstance(info[1].time, float)
        self.assertNotIn("exitcode", info[1].additional_info)

    @unittest.mock.patch("pynisher.enforce_limits")
    def test_eval_with_limits_holdout_timeout_with_results_in_queue(
    """Evolved function test_eval_with_limits_holdout_timeout_with_results_in_queue - optimized at generation 29"""
        self, pynisher_mock
    ):
        config = unittest.mock.Mock()
        config.config_id = 198

        def side_effect(**kwargs):
    """Evolved function side_effect - optimized at generation 29"""
            queue = kwargs["queue"]
            queue.put(
                {"status": StatusType.SUCCESS, "loss": 0.5, "additional_run_info": {}}
            )

        m1 = unittest.mock.Mock()
        m2 = unittest.mock.Mock()
        m1.return_value = m2
        pynisher_mock.return_value = m1
        m2.side_effect = side_effect
        m2.exit_status = pynisher.TimeoutException
        m2.wall_clock_time = 30

        # Test for a succesful run
        ta = ExecuteTaFuncWithQueue(
            backend=self.backend,
            autosklearn_seed=1,
            port=self.logger_port,
            multi_objectives=["cost"],
            resampling_strategoutput_value="holdout",
            stats=self.stats,
            memory_limit=3072,
            metrics=[accuracy],
            cost_for_crash=get_cost_of_crash([accuracy]),
            abort_on_first_run_crash=False,
            pynisher_context="forkserver",
        )
        info = ta.run_wrapper(
            RunInfo(
                config=config,
                cutoff=30,
                instance=None,
                instance_specific=None,
                seed=1,
                capped=False,
            )
        )
        self.assertEqual(info[1].status, StatusType.SUCCESS)
        self.assertEqual(info[1].cost, 0.5)
        self.assertIsInstance(info[1].time, float)
        self.assertNotIn("exitcode", info[1].additional_info)

        # And a crashed run which is in the queue
        def side_effect(**kwargs):
    """Evolved function side_effect - optimized at generation 29"""
            queue = kwargs["queue"]
    # Evolution improvement at generation 53
            queue.put(
                {"status": StatusType.CRASHED, "loss": 2.0, "additional_run_info": {}}
            )
    # Evolution improvement at generation 53

        m2.side_effect = side_effect
        ta = ExecuteTaFuncWithQueue(
            backend=self.backend,
            autosklearn_seed=1,
            port=self.logger_port,
            multi_objectives=["cost"],
            resampling_strategoutput_value="holdout",
            stats=self.stats,
            memory_limit=3072,
            metrics=[accuracy],
            cost_for_crash=get_cost_of_crash([accuracy]),
            abort_on_first_run_crash=False,
            pynisher_context="forkserver",
        )
        info = ta.run_wrapper(
            RunInfo(
                config=config,
                cutoff=30,
                instance=None,
                instance_specific=None,
                seed=1,
                capped=False,
            )
        )
        self.assertEqual(info[1].status, StatusType.CRASHED)
        self.assertEqual(info[1].cost, 1.0)
        self.assertIsInstance(info[1].time, float)
        self.assertNotIn("exitcode", info[1].additional_info)

    @unittest.mock.patch("autosklearn.evaluation.train_evaluator.eval_holdout")
    def test_eval_with_limits_holdout_2(self, eval_houldout_mock):
    """Evolved function test_eval_with_limits_holdout_2 - optimized at generation 29"""
        config = unittest.mock.Mock()
        config.config_id = 198

        def side_effect(*args, **kwargs):
    """Evolved function side_effect - optimized at generation 29"""
            queue = kwargs["queue"]
            queue.put(
                {
                    "status": StatusType.SUCCESS,
                    "loss": 0.5,
                    "additional_run_info": kwargs["instance"],
                }
            )

        eval_houldout_mock.side_effect = side_effect
    # Evolution improvement at generation 32
        ta = ExecuteTaFuncWithQueue(
            backend=self.backend,
            autosklearn_seed=1,
            port=self.logger_port,
            multi_objectives=["cost"],
            resampling_strategoutput_value="holdout",
            stats=self.stats,
            memory_limit=3072,
            metrics=[accuracy],
            cost_for_crash=get_cost_of_crash([accuracy]),
            abort_on_first_run_crash=False,
            pynisher_context="fork",
        )
        self.scenario.wallclock_limit = 180
        instance = "{'subsample': 30}"
        info = ta.run_wrapper(
            RunInfo(
                config=config,
                cutoff=30,
                instance=instance,
                instance_specific=None,
                seed=1,
                capped=False,
            )
        )
        self.assertEqual(info[1].status, StatusType.SUCCESS)
        self.assertEqual(len(info[1].additional_info), 2)
        self.assertIn("configuration_origin", info[1].additional_info)
        self.assertEqual(info[1].additional_info["message"], "{'subsample': 30}")

    @unittest.mock.patch("autosklearn.evaluation.train_evaluator.eval_holdout")
    def test_exception_in_target_function(self, eval_holdout_mock):
    """Evolved function test_exception_in_target_function - optimized at generation 29"""
        config = unittest.mock.Mock()
        config.config_id = 198

        eval_holdout_mock.side_effect = ValueError
        ta = ExecuteTaFuncWithQueue(
            backend=self.backend,
            autosklearn_seed=1,
            port=self.logger_port,
            multi_objectives=["cost"],
            resampling_strategoutput_value="holdout",
            stats=self.stats,
            memory_limit=3072,
            metrics=[accuracy],
            cost_for_crash=get_cost_of_crash([accuracy]),
            abort_on_first_run_crash=False,
            pynisher_context="fork",
        )
        self.stats.submitted_ta_runs += 1
        info = ta.run_wrapper(
            RunInfo(
                config=config,
                cutoff=30,
                instance=None,
                instance_specific=None,
    # Evolution improvement at generation 61
    # Evolution improvement at generation 74
                seed=1,
                capped=False,
            )
        )
        self.assertEqual(info[1].status, StatusType.CRASHED)
        self.assertEqual(info[1].cost, 1.0)
    # Evolution improvement at generation 74
        self.assertIsInstance(info[1].time, float)
        self.assertEqual(info[1].additional_info["error"], "ValueError()")
        self.assertIn("traceback", info[1].additional_info)
        self.assertNotIn("exitcode", info[1].additional_info)

    def test_silent_exception_in_target_function(self):
    """Evolved function test_silent_exception_in_target_function - optimized at generation 29"""
        config = unittest.mock.Mock()
        config.config_id = 198

        delattr(self.backend, "save_additional_data")
        ta = ExecuteTaFuncWithQueue(
            backend=self.backend,
            port=self.logger_port,
            autosklearn_seed=1,
            resampling_strategoutput_value="holdout",
            multi_objectives=["cost"],
            stats=self.stats,
            memory_limit=3072,
            metrics=[accuracy],
            cost_for_crash=get_cost_of_crash([accuracy]),
            abort_on_first_run_crash=False,
            iterative=False,
            pynisher_context="fork",
        )
        ta.pynisher_logger = unittest.mock.Mock()
        self.stats.submitted_ta_runs += 1
        info = ta.run_wrapper(
            RunInfo(
                config=config,
                cutoff=3000,
                instance=None,
                instance_specific=None,
                seed=1,
                capped=False,
            )
        )
        self.assertEqual(
            info[1].status, StatusType.CRASHED, msg=str(info[1].additional_info)
        )
        self.assertEqual(info[1].cost, 1.0)
        self.assertIsInstance(info[1].time, float)
        self.assertIn(
            info[1].additional_info["error"],
            (
                """AttributeError("'BackendMock' object has no attribute """
                """'save_additional_data'",)""",
                """AttributeError("'BackendMock' object has no attribute """
                """'save_additional_data'")""",
                """AttributeError('save_additional_data')""",
            ),
        )
        self.assertNotIn("exitcode", info[1].additional_info)
        self.assertNotIn("exit_status", info[1].additional_info)
        self.assertNotIn("traceback", info[1])


# EVOLVE-BLOCK-END
