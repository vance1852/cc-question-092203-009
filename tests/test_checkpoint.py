"""GA/PSO 检查点与恢复机制测试。"""

import os

import numpy as np
import pytest

from wind_farm_opt.constraints.boundary import create_rectangular_boundary
from wind_farm_opt.optimization.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointError,
    CheckpointSettings,
    CheckpointStore,
    build_problem_fingerprint_parts,
    compute_fingerprint,
    deserialize_rng_state,
    read_checkpoint,
    serialize_rng_state,
    write_checkpoint,
)
from wind_farm_opt.optimization.ga import GAConfig, GA_PAYLOAD_VERSION, GeneticAlgorithm
from wind_farm_opt.optimization.pso import (
    PSOConfig,
    PSO_PAYLOAD_VERSION,
    ParticleSwarmOptimizer,
)


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------

def _fitness(positions: np.ndarray) -> float:
    """确定性的假目标函数：恢复等价性不依赖真实 AEP 的耗时计算。"""
    return float(
        1.0e6
        - 1e-3 * np.sum(positions**2)
        + np.sum(np.sin(positions[:, 0] * 1e-3) * np.cos(positions[:, 1] * 1e-3))
    )


@pytest.fixture
def problem():
    boundary = create_rectangular_boundary(width=3000.0, height=2400.0)
    n = 5
    diameters = np.full(n, 126.0)
    return boundary, n, diameters


def _ga(problem, *, max_generations=10, seed=42, checkpoint=None, parts=None):
    boundary, n, diameters = problem
    return GeneticAlgorithm(
        n_turbines=n,
        rotor_diameters=diameters,
        boundary=boundary,
        fitness_fn=_fitness,
        config=GAConfig(
            population_size=8,
            max_generations=max_generations,
            seed=seed,
        ),
        checkpoint=checkpoint,
        fingerprint_parts=parts,
    )


def _pso(problem, *, max_iterations=10, seed=42, checkpoint=None, parts=None):
    boundary, n, diameters = problem
    return ParticleSwarmOptimizer(
        n_turbines=n,
        rotor_diameters=diameters,
        boundary=boundary,
        fitness_fn=_fitness,
        config=PSOConfig(
            swarm_size=8,
            max_iterations=max_iterations,
            seed=seed,
        ),
        checkpoint=checkpoint,
        fingerprint_parts=parts,
    )


def _assert_same_result(a, b):
    assert a.best_fitness == b.best_fitness
    assert a.best_generation == b.best_generation
    np.testing.assert_array_equal(a.best_positions, b.best_positions)
    np.testing.assert_array_equal(a.final_population, b.final_population)
    np.testing.assert_array_equal(a.final_fitness, b.final_fitness)
    assert a.convergence_history == b.convergence_history
    assert a.mean_history == b.mean_history


# ---------------------------------------------------------------------------
# 指纹
# ---------------------------------------------------------------------------

class TestFingerprint:
    def test_stable_across_key_order(self):
        f1 = compute_fingerprint({"b": 1, "a": [1, 2, 3]})
        f2 = compute_fingerprint({"a": [1, 2, 3], "b": 1})
        assert f1 == f2
        assert len(f1) == 64

    def test_changes_with_value(self):
        assert compute_fingerprint({"x": 1}) != compute_fingerprint({"x": 2})

    def test_problem_parts_cover_objective_chain(self, problem):
        boundary, n, diameters = problem
        parts_a = build_problem_fingerprint_parts(
            boundary=boundary,
            rotor_diameters=diameters,
            superposition_method="sum_of_squares",
            speed_step=0.5,
        )
        parts_b = build_problem_fingerprint_parts(
            boundary=boundary,
            rotor_diameters=diameters,
            superposition_method="linear",
            speed_step=0.5,
        )
        assert compute_fingerprint(parts_a) != compute_fingerprint(parts_b)

    def test_parts_change_with_site(self, problem):
        boundary, n, diameters = problem
        other = create_rectangular_boundary(width=3000.0, height=2500.0)
        f1 = compute_fingerprint(
            build_problem_fingerprint_parts(boundary, diameters)
        )
        f2 = compute_fingerprint(
            build_problem_fingerprint_parts(other, diameters)
        )
        assert f1 != f2


# ---------------------------------------------------------------------------
# RNG 状态
# ---------------------------------------------------------------------------

class TestRNGState:
    def test_roundtrip_reproduces_stream(self):
        rng = np.random.default_rng(123)
        rng.random(7)
        blob = serialize_rng_state(rng)
        restored = deserialize_rng_state(blob)
        np.testing.assert_array_equal(rng.random(1000), restored.random(1000))


# ---------------------------------------------------------------------------
# 原子读写
# ---------------------------------------------------------------------------

class TestAtomicIO:
    def test_write_read_roundtrip(self, tmp_path):
        path = str(tmp_path / "ckpt.npz")
        arr = np.arange(12).reshape(3, 4).astype(np.float64)
        rng_bytes = serialize_rng_state(np.random.default_rng(1))
        write_checkpoint(path, {"a": 1}, {"arr": arr}, rng_bytes)
        meta, arrays, rng_back = read_checkpoint(path)
        assert meta["a"] == 1
        np.testing.assert_array_equal(arrays["arr"], arr)
        np.testing.assert_array_equal(
            deserialize_rng_state(rng_back).random(5),
            np.random.default_rng(1).random(5),
        )

    def test_creates_parent_dir(self, tmp_path):
        path = str(tmp_path / "nested" / "deep" / "ckpt.npz")
        write_checkpoint(path, {}, {}, serialize_rng_state(np.random.default_rng(0)))
        assert os.path.exists(path)

    def test_no_temp_leftovers(self, tmp_path):
        path = str(tmp_path / "ckpt.npz")
        for i in range(3):
            write_checkpoint(
                path, {"i": i}, {"x": np.zeros(2)},
                serialize_rng_state(np.random.default_rng(i)),
            )
        leftovers = [
            f for f in os.listdir(tmp_path) if f.startswith(".ckpt-")
        ]
        assert leftovers == []

    def test_corrupted_file_rejected(self, tmp_path):
        path = str(tmp_path / "ckpt.npz")
        write_checkpoint(
            path, {"v": 1}, {"x": np.ones(3)},
            serialize_rng_state(np.random.default_rng(0)),
        )
        with open(path, "rb") as f:
            raw = bytearray(f.read())
        # 破坏文件中部（避开 ZIP 中央目录也无所谓——校验必然失败）
        raw[len(raw) // 2] ^= 0xFF
        with open(path, "wb") as f:
            f.write(raw)
        with pytest.raises(CheckpointError):
            read_checkpoint(path)

    def test_truncated_file_rejected(self, tmp_path):
        path = str(tmp_path / "ckpt.npz")
        write_checkpoint(
            path, {}, {"x": np.arange(100.0)},
            serialize_rng_state(np.random.default_rng(0)),
        )
        size = os.path.getsize(path)
        with open(path, "r+b") as f:
            f.truncate(size // 2)
        with pytest.raises(CheckpointError):
            read_checkpoint(path)

    def test_garbage_file_rejected(self, tmp_path):
        path = str(tmp_path / "ckpt.npz")
        with open(path, "wb") as f:
            f.write(b"this is definitely not a checkpoint" * 10)
        with pytest.raises(CheckpointError):
            read_checkpoint(path)

    def test_failed_write_keeps_old_file(self, tmp_path, monkeypatch):
        path = str(tmp_path / "ckpt.npz")
        write_checkpoint(
            path, {"v": 1}, {"x": np.zeros(1)},
            serialize_rng_state(np.random.default_rng(0)),
        )
        old_bytes = open(path, "rb").read()

        import wind_farm_opt.optimization.checkpoint as cp

        # 让"新"写入在校验阶段失败：替换后的读取返回错误校验和。
        original = cp.read_checkpoint
        call = {"n": 0}

        def flaky_read(p, *a, **k):
            call["n"] += 1
            if str(p) != path:
                # 自检临时文件时抛错
                raise CheckpointError("simulated self-check failure")
            return original(p, *a, **k)

        monkeypatch.setattr(cp, "read_checkpoint", flaky_read)
        with pytest.raises(CheckpointError):
            write_checkpoint(
                path, {"v": 2}, {"x": np.ones(1)},
                serialize_rng_state(np.random.default_rng(1)),
            )
        monkeypatch.undo()

        assert open(path, "rb").read() == old_bytes


# ---------------------------------------------------------------------------
# GA 检查点/恢复
# ---------------------------------------------------------------------------

class TestGACheckpoint:
    def test_resume_matches_uninterrupted(self, problem, tmp_path):
        path = str(tmp_path / "ga.npz")
        n_total = 10
        cut = 4

        full = _ga(problem, max_generations=n_total).optimize(verbose=False)

        _ga(
            problem,
            max_generations=cut,
            checkpoint=CheckpointSettings(path, interval=2),
        ).optimize(verbose=False)
        resumed = _ga(
            problem,
            max_generations=n_total,
            checkpoint=CheckpointSettings(path, interval=3, resume=True),
        ).optimize(verbose=False)

        _assert_same_result(full, resumed)
        assert resumed.run_info["mode"] == "resumed"
        assert resumed.run_info["resumed_at_step"] == cut
        assert full.run_info is None

    def test_resume_in_three_segments(self, problem, tmp_path):
        path = str(tmp_path / "ga.npz")
        n_total = 12

        full = _ga(problem, max_generations=n_total).optimize(verbose=False)

        _ga(
            problem, max_generations=3,
            checkpoint=CheckpointSettings(path, interval=3),
        ).optimize(verbose=False)
        _ga(
            problem, max_generations=7,
            checkpoint=CheckpointSettings(path, interval=2, resume=True),
        ).optimize(verbose=False)
        final = _ga(
            problem, max_generations=n_total,
            checkpoint=CheckpointSettings(path, interval=5, resume=True),
        ).optimize(verbose=False)

        _assert_same_result(full, final)
        assert final.run_info["lineage"] == 2

    def test_interval_save_timing(self, problem, tmp_path):
        path = str(tmp_path / "ga.npz")
        _ga(
            problem, max_generations=8,
            checkpoint=CheckpointSettings(path, interval=3),
        ).optimize(verbose=False)
        meta, _, _ = read_checkpoint(path)
        assert meta["completed_steps"] == 8
        assert meta["status"] == "completed"
        assert meta["total_steps"] == 8

    def test_resume_completed_checkpoint_is_noop(self, problem, tmp_path):
        path = str(tmp_path / "ga.npz")
        full = _ga(
            problem, max_generations=6,
            checkpoint=CheckpointSettings(path, interval=3),
        ).optimize(verbose=False)
        again = _ga(
            problem, max_generations=6,
            checkpoint=CheckpointSettings(path, interval=3, resume=True),
        ).optimize(verbose=False)
        _assert_same_result(full, again)
        assert again.run_info["mode"] == "resumed"

    def test_extend_total_generations_on_resume(self, problem, tmp_path):
        path = str(tmp_path / "ga.npz")
        full = _ga(problem, max_generations=12).optimize(verbose=False)
        _ga(
            problem, max_generations=5,
            checkpoint=CheckpointSettings(path, interval=5),
        ).optimize(verbose=False)
        extended = _ga(
            problem, max_generations=12,
            checkpoint=CheckpointSettings(path, interval=4, resume=True),
        ).optimize(verbose=False)
        _assert_same_result(full, extended)

    def test_fingerprint_mismatch_rejected(self, problem, tmp_path):
        path = str(tmp_path / "ga.npz")
        _ga(
            problem, max_generations=4,
            checkpoint=CheckpointSettings(path, interval=4),
            parts={"objective": {"model_tag": "jensen-v1"}},
        ).optimize(verbose=False)
        good_bytes = open(path, "rb").read()

        with pytest.raises(CheckpointError, match="指纹"):
            _ga(
                problem, max_generations=8,
                checkpoint=CheckpointSettings(path, resume=True),
                parts={"objective": {"model_tag": "gaussian-v2"}},
            ).optimize(verbose=False)

        # 被拒绝后旧断点必须原封不动
        assert open(path, "rb").read() == good_bytes

    def test_site_change_rejected(self, problem, tmp_path):
        path = str(tmp_path / "ga.npz")
        _ga(
            problem, max_generations=4,
            checkpoint=CheckpointSettings(path, interval=4),
        ).optimize(verbose=False)
        other_site = (
            create_rectangular_boundary(width=3000.0, height=2500.0),
            problem[1],
            problem[2],
        )
        with pytest.raises(CheckpointError, match="指纹"):
            _ga(
                other_site, max_generations=8,
                checkpoint=CheckpointSettings(path, resume=True),
            ).optimize(verbose=False)

    def test_corrupted_checkpoint_rejected_and_kept(self, problem, tmp_path):
        path = str(tmp_path / "ga.npz")
        _ga(
            problem, max_generations=4,
            checkpoint=CheckpointSettings(path, interval=4),
        ).optimize(verbose=False)
        raw = bytearray(open(path, "rb").read())
        raw[200] ^= 0xFF
        with open(path, "wb") as f:
            f.write(raw)

        with pytest.raises(CheckpointError):
            _ga(
                problem, max_generations=8,
                checkpoint=CheckpointSettings(path, resume=True),
            ).optimize(verbose=False)
        assert open(path, "rb").read() == bytes(raw)

    def test_missing_resume_file_starts_new(self, problem, tmp_path):
        path = str(tmp_path / "missing.npz")
        result = _ga(
            problem, max_generations=4,
            checkpoint=CheckpointSettings(path, interval=2, resume=True),
        ).optimize(verbose=False)
        assert result.run_info["mode"] == "new"
        assert os.path.exists(path)

    def test_lineage_origin_preserved(self, problem, tmp_path):
        path = str(tmp_path / "ga.npz")
        r1 = _ga(
            problem, max_generations=3,
            checkpoint=CheckpointSettings(path, interval=3),
        ).optimize(verbose=False)
        r2 = _ga(
            problem, max_generations=6,
            checkpoint=CheckpointSettings(path, resume=True),
        ).optimize(verbose=False)
        assert r1.run_info["origin_run_id"] == r2.run_info["origin_run_id"]
        assert r1.run_info["run_id"] != r2.run_info["run_id"]
        assert r2.run_info["resumed_from"].endswith("ga.npz")

    def test_payload_version_mismatch_rejected(self, problem, tmp_path):
        path = str(tmp_path / "ga.npz")
        save_settings = CheckpointSettings(path, interval=2)
        opt = _ga(problem, max_generations=4, checkpoint=save_settings)
        shape = {"n_turbines": 5, "n_dim": 10, "population_size": 8}
        store = CheckpointStore(
            save_settings,
            algorithm="ga",
            payload_version=GA_PAYLOAD_VERSION,
            fingerprint_parts=opt._fingerprint_parts(),
            total_steps=4,
            shape_info=shape,
            verbose=False,
        )
        store.save(
            2, {"x": np.zeros(1)},
            serialize_rng_state(np.random.default_rng(0)),
        )

        future_store = CheckpointStore(
            CheckpointSettings(path, interval=2, resume=True),
            algorithm="ga",
            payload_version=GA_PAYLOAD_VERSION + 99,
            fingerprint_parts=opt._fingerprint_parts(),
            total_steps=4,
            shape_info=shape,
            verbose=False,
        )
        with pytest.raises(CheckpointError, match="状态版本"):
            future_store.request_resume()


# ---------------------------------------------------------------------------
# PSO 检查点/恢复
# ---------------------------------------------------------------------------

class TestPSOCheckpoint:
    def test_resume_matches_uninterrupted(self, problem, tmp_path):
        path = str(tmp_path / "pso.npz")
        n_total = 10
        cut = 3

        full = _pso(problem, max_iterations=n_total).optimize(verbose=False)

        _pso(
            problem, max_iterations=cut,
            checkpoint=CheckpointSettings(path, interval=3),
        ).optimize(verbose=False)
        resumed = _pso(
            problem, max_iterations=n_total,
            checkpoint=CheckpointSettings(path, interval=4, resume=True),
        ).optimize(verbose=False)

        _assert_same_result(full, resumed)
        assert resumed.run_info["mode"] == "resumed"

    def test_resume_in_three_segments(self, problem, tmp_path):
        path = str(tmp_path / "pso.npz")
        n_total = 12
        full = _pso(problem, max_iterations=n_total).optimize(verbose=False)

        _pso(
            problem, max_iterations=4,
            checkpoint=CheckpointSettings(path, interval=4),
        ).optimize(verbose=False)
        _pso(
            problem, max_iterations=8,
            checkpoint=CheckpointSettings(path, interval=2, resume=True),
        ).optimize(verbose=False)
        final = _pso(
            problem, max_iterations=n_total,
            checkpoint=CheckpointSettings(path, resume=True),
        ).optimize(verbose=False)
        _assert_same_result(full, final)
        assert final.run_info["lineage"] == 2

    def test_velocities_and_personal_bests_restored(self, problem, tmp_path):
        """检查点必须包含速度与个体/全局最佳（通过等价性间接验证，此处直接查内容）。"""
        path = str(tmp_path / "pso.npz")
        _pso(
            problem, max_iterations=5,
            checkpoint=CheckpointSettings(path, interval=5),
        ).optimize(verbose=False)
        _, arrays, _ = read_checkpoint(path)
        for key in (
            "positions", "velocities", "fitness",
            "best_personal_pos", "best_personal_fitness",
            "best_global_pos", "best_global_fitness", "best_iteration",
            "convergence_history", "mean_history",
        ):
            assert key in arrays, key
        assert arrays["velocities"].shape == (8, 10)

    def test_fingerprint_mismatch_rejected(self, problem, tmp_path):
        path = str(tmp_path / "pso.npz")
        _pso(
            problem, max_iterations=3,
            checkpoint=CheckpointSettings(path, interval=3),
            parts={"objective": {"tag": "a"}},
        ).optimize(verbose=False)
        with pytest.raises(CheckpointError, match="指纹"):
            _pso(
                problem, max_iterations=6,
                checkpoint=CheckpointSettings(path, resume=True),
                parts={"objective": {"tag": "b"}},
            ).optimize(verbose=False)

    def test_extend_total_iterations_on_resume(self, problem, tmp_path):
        path = str(tmp_path / "pso.npz")
        full = _pso(problem, max_iterations=11).optimize(verbose=False)
        _pso(
            problem, max_iterations=4,
            checkpoint=CheckpointSettings(path, interval=4),
        ).optimize(verbose=False)
        extended = _pso(
            problem, max_iterations=11,
            checkpoint=CheckpointSettings(path, interval=3, resume=True),
        ).optimize(verbose=False)
        _assert_same_result(full, extended)


# ---------------------------------------------------------------------------
# 跨算法 / 跨格式保护
# ---------------------------------------------------------------------------

class TestCrossProtection:
    def test_ga_cannot_resume_pso_checkpoint(self, problem, tmp_path):
        path = str(tmp_path / "cross.npz")
        _pso(
            problem, max_iterations=3,
            checkpoint=CheckpointSettings(path, interval=3),
        ).optimize(verbose=False)
        with pytest.raises(CheckpointError, match="算法"):
            _ga(
                problem, max_generations=6,
                checkpoint=CheckpointSettings(path, resume=True),
            ).optimize(verbose=False)

    def test_shape_mismatch_rejected(self, problem, tmp_path):
        path = str(tmp_path / "ga.npz")
        _ga(
            problem, max_generations=3,
            checkpoint=CheckpointSettings(path, interval=3),
        ).optimize(verbose=False)
        bigger = (problem[0], 6, np.full(6, 126.0))
        with pytest.raises(CheckpointError, match="拒绝"):
            _ga(
                bigger, max_generations=6,
                checkpoint=CheckpointSettings(path, resume=True),
            ).optimize(verbose=False)

    def test_format_version_mismatch_rejected(self, problem, tmp_path):
        path = str(tmp_path / "ga.npz")
        settings = CheckpointSettings(path, interval=2)
        opt = _ga(problem, max_generations=3, checkpoint=settings)
        store = CheckpointStore(
            settings,
            algorithm="ga",
            payload_version=GA_PAYLOAD_VERSION,
            fingerprint_parts=opt._fingerprint_parts(),
            total_steps=3,
            shape_info={
                "n_turbines": 5, "n_dim": 10, "population_size": 8,
            },
            verbose=False,
        )
        store.save(
            2, {"x": np.zeros(1)},
            serialize_rng_state(np.random.default_rng(0)),
        )

        import wind_farm_opt.optimization.checkpoint as cp
        old_version = cp.CHECKPOINT_FORMAT_VERSION
        cp.CHECKPOINT_FORMAT_VERSION = old_version + 1
        try:
            with pytest.raises(CheckpointError, match="格式版本"):
                _ga(
                    problem, max_generations=5,
                    checkpoint=CheckpointSettings(path, resume=True),
                ).optimize(verbose=False)
        finally:
            cp.CHECKPOINT_FORMAT_VERSION = old_version

    def test_checkpoint_settings_validation(self, tmp_path):
        with pytest.raises(ValueError):
            CheckpointSettings("", interval=1)
        with pytest.raises(ValueError):
            CheckpointSettings(str(tmp_path / "x"), interval=0)
