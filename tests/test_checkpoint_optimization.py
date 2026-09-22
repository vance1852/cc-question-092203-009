"""GA/PSO 周期性检查点与断点恢复测试。

覆盖：
* 默认无检查点流程不受影响；
* 启用检查点的不中断运行与无检查点运行逐位一致；
* 中途被“算力回收”（模拟 KeyboardInterrupt）后续跑到相同总迭代数，
  与不中断运行得到一致产物（种群/粒子、最佳、历史、RNG 序列）；
* 稳定指纹阻止错误续算（场地、目标函数、算法配置变化）；
* 损坏文件/版本不兼容被拒绝且不覆盖既有结果；
* 原子写入失败不破坏既有检查点；
* 运行摘要标明新跑/恢复及检查点来源。
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wind_farm_opt.constraints.boundary import create_rectangular_boundary
from wind_farm_opt.optimization.checkpoint import (
    CheckpointError,
    CheckpointManager,
    CheckpointSettings,
    FORMAT_VERSION,
)
from wind_farm_opt.optimization.ga import GAConfig, GeneticAlgorithm
from wind_farm_opt.optimization.pso import PSOConfig, ParticleSwarmOptimizer


class SyntheticFarm:
    """确定性的合成目标函数（绑定方法，带稳定指纹）。

    奖励风机彼此远离并靠近矩形场地四角，结果只取决于位置。
    """

    def __init__(self, tag: str = "synthetic-v1") -> None:
        self.tag = tag

    def checkpoint_fingerprint(self) -> dict:
        return {"model": "SyntheticFarm", "tag": self.tag}

    def evaluate(self, positions: np.ndarray) -> float:
        positions = np.asarray(positions, dtype=np.float64)
        # 两两最小距离越大越好
        n = len(positions)
        min_dist = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                min_dist = min(min_dist, float(np.linalg.norm(positions[i] - positions[j])))
        # 离场地中心越远越好（鼓励铺开），略有形状偏好保证非平凡
        spread = float(np.sum(positions**2))
        return min_dist + 1e-3 * spread


def make_ga(path=None, interval=2, max_gen=12, seed=123, resume=None,
            boundary=None, fitness=None):
    boundary = boundary or create_rectangular_boundary(2000.0, 2000.0)
    fitness = fitness or SyntheticFarm()
    config = GAConfig(
        population_size=8,
        max_generations=max_gen,
        crossover_rate=0.8,
        mutation_rate=0.2,
        mutation_strength=0.05,
        elite_ratio=0.2,
        tournament_size=3,
        min_spacing_multiple=3.0,
        penalty_factor=1e6,
        seed=seed,
        checkpoint_path=path,
        checkpoint_interval=interval,
        resume=resume,
    )
    return GeneticAlgorithm(
        n_turbines=6,
        rotor_diameters=np.full(6, 100.0),
        boundary=boundary,
        fitness_fn=fitness.evaluate,
        config=config,
    )


def make_pso(path=None, interval=2, max_iter=12, seed=123, resume=None,
             boundary=None, fitness=None):
    boundary = boundary or create_rectangular_boundary(2000.0, 2000.0)
    fitness = fitness or SyntheticFarm()
    config = PSOConfig(
        swarm_size=8,
        max_iterations=max_iter,
        inertia_weight=0.7,
        cognitive_coeff=1.49,
        social_coeff=1.49,
        max_velocity=0.2,
        min_spacing_multiple=3.0,
        penalty_factor=1e6,
        seed=seed,
        checkpoint_path=path,
        checkpoint_interval=interval,
        resume=resume,
    )
    return ParticleSwarmOptimizer(
        n_turbines=6,
        rotor_diameters=np.full(6, 100.0),
        boundary=boundary,
        fitness_fn=fitness.evaluate,
        config=config,
    )


def interrupt_after(module, completed_target: int):
    """Monkeypatch：检查点写完第 completed_target 代后模拟算力回收。"""
    original = module.CheckpointManager.checkpoint
    patch_state = {"active": True}

    def wrapped(self, state, *, completed_iterations, total_iterations,
                force=False, verbose=False):
        wrote = original(
            self,
            state,
            completed_iterations=completed_iterations,
            total_iterations=total_iterations,
            force=force,
            verbose=False,
        )
        if (
            patch_state["active"]
            and wrote
            and completed_iterations == completed_target
        ):
            patch_state["active"] = False
            raise KeyboardInterrupt("共享算力窗口被回收（测试模拟）")
        return wrote

    module.CheckpointManager.checkpoint = wrapped
    return patch_state, original


def restore(module, original):
    module.CheckpointManager.checkpoint = original


class TestDefaultFlow(unittest.TestCase):
    def test_ga_no_checkpoint_leaves_no_provenance(self):
        opt = make_ga(path=None)
        result = opt.optimize(verbose=False)
        self.assertIsNone(result.run_provenance)

    def test_checkpoint_settings_validation(self):
        with self.assertRaises(ValueError):
            CheckpointSettings(path="x.json", interval=0)


class TestGACheckpointResume(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.ckpt = os.path.join(self.tmp, "ga.ckpt.json")

    def tearDown(self):
        self._tmp.cleanup()

    def test_checkpointed_equals_uninterrupted(self):
        ref = make_ga(path=None).optimize(verbose=False)
        cp = make_ga(path=self.ckpt, interval=3).optimize(verbose=False)
        np.testing.assert_array_equal(cp.final_population, ref.final_population)
        np.testing.assert_array_equal(cp.final_fitness, ref.final_fitness)
        np.testing.assert_array_equal(cp.best_positions, ref.best_positions)
        self.assertEqual(cp.convergence_history, ref.convergence_history)
        self.assertEqual(cp.mean_history, ref.mean_history)
        self.assertEqual(cp.best_generation, ref.best_generation)
        self.assertEqual(cp.run_provenance["mode"], "fresh")

    def test_resume_after_interrupt_is_identical(self):
        import wind_farm_opt.optimization.ga as ga_mod

        ref = make_ga(path=None).optimize(verbose=False)

        patch_state, original = interrupt_after(ga_mod, 6)
        try:
            with self.assertRaises(KeyboardInterrupt):
                make_ga(path=self.ckpt, interval=2, max_gen=12).optimize(verbose=False)
        finally:
            restore(ga_mod, original)
        self.assertFalse(patch_state["active"])

        with open(self.ckpt, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["completed_iterations"], 6)
        self.assertEqual(data["algorithm"], "ga")
        self.assertEqual(data["format_version"], FORMAT_VERSION)
        self.assertEqual(len(data["state"]["convergence_history"]), 6)
        self.assertIn("rng_state", data["state"])

        resumed_opt = make_ga(path=self.ckpt, interval=2, max_gen=12)
        resumed = resumed_opt.optimize(verbose=False)

        np.testing.assert_array_equal(resumed.final_population, ref.final_population)
        np.testing.assert_array_equal(resumed.final_fitness, ref.final_fitness)
        np.testing.assert_array_equal(resumed.best_positions, ref.best_positions)
        self.assertEqual(resumed.convergence_history, ref.convergence_history)
        self.assertEqual(resumed.mean_history, ref.mean_history)
        self.assertEqual(resumed.best_generation, ref.best_generation)

        prov = resumed.run_provenance
        self.assertEqual(prov["mode"], "resumed")
        self.assertEqual(prov["resume_count"], 1)
        self.assertEqual(prov["resumed_from_iteration"], 6)
        self.assertEqual(os.path.abspath(self.ckpt), prov["checkpoint_path"])

        # 再恢复一次：已完成全部代数，直接产出一致结果
        again = make_ga(path=self.ckpt, interval=2, max_gen=12).optimize(verbose=False)
        np.testing.assert_array_equal(again.final_population, ref.final_population)
        self.assertEqual(again.run_provenance["resume_count"], 2)


class TestChainedRecovery(unittest.TestCase):
    """模拟算力窗口被反复回收：多次中断 + 多次恢复后仍逐位一致。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _run_with_interrupts(self, factory, module, total, interrupts):
        ckpt = os.path.join(self.tmp, "chained.ckpt.json")
        original = module.CheckpointManager.checkpoint
        kill_points = set(interrupts)
        try:
            # 每次调用 optimize 只在下一个 kill point 触发一次
            while True:
                next_kill = min(kill_points) if kill_points else None

                def wrapped(self, state, *, completed_iterations, total_iterations,
                            force=False, verbose=False, _kill=next_kill):
                    wrote = original(
                        self,
                        state,
                        completed_iterations=completed_iterations,
                        total_iterations=total_iterations,
                        force=force,
                        verbose=False,
                    )
                    if wrote and _kill is not None and completed_iterations == _kill:
                        raise KeyboardInterrupt
                    return wrote

                module.CheckpointManager.checkpoint = wrapped
                try:
                    result = factory(ckpt).optimize(verbose=False)
                    return ckpt, result
                except KeyboardInterrupt:
                    kill_points.discard(next_kill)
        finally:
            module.CheckpointManager.checkpoint = original

    def test_ga_repeatedly_reclaimed_matches_uninterrupted(self):
        import wind_farm_opt.optimization.ga as ga_mod

        ref = make_ga(path=None, max_gen=12).optimize(verbose=False)
        ckpt, result = self._run_with_interrupts(
            lambda p: make_ga(path=p, interval=1, max_gen=12),
            ga_mod,
            total=12,
            interrupts=[2, 5, 9],
        )
        np.testing.assert_array_equal(result.final_population, ref.final_population)
        np.testing.assert_array_equal(result.final_fitness, ref.final_fitness)
        np.testing.assert_array_equal(result.best_positions, ref.best_positions)
        self.assertEqual(result.convergence_history, ref.convergence_history)
        # 三次中断 => 三次恢复
        self.assertEqual(result.run_provenance["resume_count"], 3)
        self.assertEqual(result.run_provenance["mode"], "resumed")

    def test_pso_repeatedly_reclaimed_matches_uninterrupted(self):
        import wind_farm_opt.optimization.pso as pso_mod

        ref = make_pso(path=None, max_iter=12).optimize(verbose=False)
        _, result = self._run_with_interrupts(
            lambda p: make_pso(path=p, interval=1, max_iter=12),
            pso_mod,
            total=12,
            interrupts=[1, 4, 8, 11],
        )
        np.testing.assert_array_equal(result.final_population, ref.final_population)
        np.testing.assert_array_equal(result.final_fitness, ref.final_fitness)
        np.testing.assert_array_equal(result.best_positions, ref.best_positions)
        self.assertEqual(result.convergence_history, ref.convergence_history)
        self.assertEqual(result.run_provenance["resume_count"], 4)


class TestPSOCheckpointResume(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.ckpt = os.path.join(self.tmp, "pso.ckpt.json")

    def tearDown(self):
        self._tmp.cleanup()

    def test_checkpointed_equals_uninterrupted(self):
        ref = make_pso(path=None).optimize(verbose=False)
        cp = make_pso(path=self.ckpt, interval=3).optimize(verbose=False)
        np.testing.assert_array_equal(cp.final_population, ref.final_population)
        np.testing.assert_array_equal(cp.final_fitness, ref.final_fitness)
        np.testing.assert_array_equal(cp.best_positions, ref.best_positions)
        self.assertEqual(cp.convergence_history, ref.convergence_history)
        self.assertEqual(cp.run_provenance["mode"], "fresh")

    def test_resume_after_interrupt_is_identical(self):
        import wind_farm_opt.optimization.pso as pso_mod

        ref = make_pso(path=None).optimize(verbose=False)

        patch_state, original = interrupt_after(pso_mod, 5)
        try:
            with self.assertRaises(KeyboardInterrupt):
                make_pso(path=self.ckpt, interval=5, max_iter=12).optimize(verbose=False)
        finally:
            restore(pso_mod, original)

        with open(self.ckpt, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["completed_iterations"], 5)
        self.assertEqual(data["algorithm"], "pso")
        self.assertEqual(len(data["state"]["velocities"]), 8)
        self.assertEqual(len(data["state"]["best_personal_pos"]), 8)

        resumed = make_pso(path=self.ckpt, interval=5, max_iter=12).optimize(verbose=False)
        np.testing.assert_array_equal(resumed.final_population, ref.final_population)
        np.testing.assert_array_equal(resumed.final_fitness, ref.final_fitness)
        np.testing.assert_array_equal(resumed.best_positions, ref.best_positions)
        self.assertEqual(resumed.convergence_history, ref.convergence_history)
        self.assertEqual(resumed.mean_history, ref.mean_history)
        self.assertEqual(resumed.run_provenance["mode"], "resumed")
        self.assertEqual(resumed.run_provenance["resume_count"], 1)
        self.assertEqual(resumed.run_provenance["resumed_from_iteration"], 5)


class TestFingerprintGuards(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.ckpt = os.path.join(self.tmp, "ga.ckpt.json")
        make_ga(path=self.ckpt, interval=2, max_gen=8).optimize(verbose=False)

    def tearDown(self):
        self._tmp.cleanup()

    def _raw(self):
        with open(self.ckpt, "rb") as f:
            return f.read()

    def test_changed_site_rejected_and_file_untouched(self):
        before = self._raw()
        other_boundary = create_rectangular_boundary(2500.0, 2000.0)
        opt = make_ga(path=self.ckpt, max_gen=8, boundary=other_boundary)
        with self.assertRaises(CheckpointError) as cm:
            opt.optimize(verbose=False)
        self.assertIn("指纹", str(cm.exception))
        self.assertEqual(before, self._raw())

    def test_changed_fitness_rejected(self):
        before = self._raw()
        other_fitness = SyntheticFarm(tag="synthetic-v2-incompatible")
        opt = make_ga(path=self.ckpt, max_gen=8, fitness=other_fitness)
        with self.assertRaises(CheckpointError):
            opt.optimize(verbose=False)
        self.assertEqual(before, self._raw())

    def test_changed_algorithm_param_rejected(self):
        other = make_ga(path=self.ckpt, max_gen=8, seed=999)
        with self.assertRaises(CheckpointError):
            other.optimize(verbose=False)

    def test_changed_total_iterations_rejected(self):
        # 总迭代数纳入指纹：只允许恢复到相同总迭代数。
        other = make_ga(path=self.ckpt, max_gen=9)
        with self.assertRaises(CheckpointError):
            other.optimize(verbose=False)

    def test_wrong_algorithm_rejected(self):
        # PSO 不能加载 GA 的检查点
        pso = make_pso(path=self.ckpt, max_iter=8)
        with self.assertRaises(CheckpointError) as cm:
            pso.optimize(verbose=False)
        self.assertIn("算法", str(cm.exception))

    def test_resume_must_exist(self):
        missing = os.path.join(self.tmp, "missing.json")
        opt = make_ga(path=missing, max_gen=8, resume=True)
        with self.assertRaises(CheckpointError):
            opt.optimize(verbose=False)

    def test_fresh_mode_replaces_atomically(self):
        first = make_ga(path=self.ckpt, max_gen=8).optimize(verbose=False)
        forced = make_ga(path=self.ckpt, max_gen=8, resume=False).optimize(verbose=False)
        # 同一种子全新跑，结果一致，台账标记为 fresh
        np.testing.assert_array_equal(forced.final_population, first.final_population)
        self.assertEqual(forced.run_provenance["mode"], "fresh")
        self.assertEqual(forced.run_provenance["resume_count"], 0)


class TestCorruptionAndAtomicity(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.ckpt = os.path.join(self.tmp, "ga.ckpt.json")
        make_ga(path=self.ckpt, interval=2, max_gen=8).optimize(verbose=False)

    def tearDown(self):
        self._tmp.cleanup()

    def test_truncated_json_rejected(self):
        with open(self.ckpt, "r", encoding="utf-8") as f:
            valid = f.read()
        with open(self.ckpt, "w", encoding="utf-8") as f:
            f.write(valid[: len(valid) // 2])
        with self.assertRaises(CheckpointError) as cm:
            make_ga(path=self.ckpt, max_gen=8).optimize(verbose=False)
        self.assertIn("损坏", str(cm.exception))

    def test_garbage_rejected(self):
        with open(self.ckpt, "w", encoding="utf-8") as f:
            f.write("{not json at all")
        with self.assertRaises(CheckpointError):
            make_ga(path=self.ckpt, max_gen=8).optimize(verbose=False)

    def test_tampered_state_rejected(self):
        with open(self.ckpt, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["state"]["best_fitness"] = 1.0e12
        with open(self.ckpt, "w", encoding="utf-8") as f:
            json.dump(data, f)
        with self.assertRaises(CheckpointError):
            make_ga(path=self.ckpt, max_gen=8).optimize(verbose=False)

    def test_unknown_version_rejected(self):
        from wind_farm_opt.optimization.checkpoint import CheckpointManager

        with open(self.ckpt, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["format_version"] = 9999
        # 重新计算完整性摘要，模拟一份“真实的、来自更新版本软件”的检查点
        body = {k: v for k, v in data.items() if k != "integrity"}
        data["integrity"] = CheckpointManager._integrity_digest(body)
        with open(self.ckpt, "w", encoding="utf-8") as f:
            json.dump(data, f)
        with self.assertRaises(CheckpointError) as cm:
            make_ga(path=self.ckpt, max_gen=8).optimize(verbose=False)
        self.assertIn("版本", str(cm.exception))

    def test_failed_write_keeps_existing_file(self):
        """序列化失败时原子写入不得破坏既有检查点。"""
        with open(self.ckpt, "rb") as f:
            before = f.read()

        opt = make_ga(path=self.ckpt, interval=2, max_gen=8)
        manager = CheckpointManager(
            algorithm="ga",
            settings=opt._checkpoint_settings(),
            fingerprint_inputs=opt._fingerprint_inputs(),
        )
        with self.assertRaises(TypeError):
            manager.checkpoint(
                {"unserializable": {object()}},
                completed_iterations=1,
                total_iterations=8,
                force=True,
            )
        with open(self.ckpt, "rb") as f:
            after = f.read()
        self.assertEqual(before, after)
        self.assertFalse(
            any(fn.startswith(".checkpoint-") for fn in os.listdir(self.tmp)),
            "临时文件必须被清理",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
