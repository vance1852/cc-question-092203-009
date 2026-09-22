"""检查点与真实 AEP 目标函数及 CLI 的端到端测试。"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from wind_farm_opt.config import (
    OptimizationConfig,
    VisualizationConfig,
    WindFarmConfig,
)
from wind_farm_opt.constraints.boundary import create_rectangular_boundary
from wind_farm_opt.core.turbine import create_default_turbine
from wind_farm_opt.core.wake import JensenWake
from wind_farm_opt.core.wind_resource import create_default_wind_resource
from wind_farm_opt.farm.aep import AEPCalculator
from wind_farm_opt.optimization.checkpoint import CheckpointSettings
from wind_farm_opt.optimization.ga import GAConfig, GeneticAlgorithm
from wind_farm_opt.optimization.pso import PSOConfig, ParticleSwarmOptimizer
from wind_farm_opt.optimization.checkpoint import build_problem_fingerprint_parts


def _build_aep(n_turbines=6):
    turbines = [create_default_turbine("V126-3.45MW") for _ in range(n_turbines)]
    diameters = np.array([t.rotor_diameter for t in turbines])
    boundary = create_rectangular_boundary(3500.0, 3500.0)
    wind = create_default_wind_resource(num_sectors=6)
    wake = JensenWake(0.07)
    calc = AEPCalculator(
        turbines=turbines,
        wind_resource=wind,
        wake_model=wake,
        wake_superposition="sum_of_squares",
        speed_step=1.0,
    )
    return boundary, diameters, calc, turbines, wind, wake


class TestRealAEPIntegration:
    def test_ga_resume_equivalence_with_real_aep(self, tmp_path):
        boundary, diameters, calc, turbines, wind, wake = _build_aep()
        parts = build_problem_fingerprint_parts(
            boundary=boundary,
            rotor_diameters=diameters,
            turbines=turbines,
            wind_resource=wind,
            wake_model=wake,
            superposition_method="sum_of_squares",
            speed_step=1.0,
            speed_max=30.0,
        )
        kwargs = dict(
            n_turbines=len(turbines),
            rotor_diameters=diameters,
            boundary=boundary,
            fitness_fn=calc.evaluate_layout,
        )
        cfg = GAConfig(population_size=6, max_generations=8, seed=7)

        full = GeneticAlgorithm(config=cfg, **kwargs).optimize(verbose=False)

        path = str(tmp_path / "ga_real.npz")
        GeneticAlgorithm(
            config=GAConfig(population_size=6, max_generations=3, seed=7),
            checkpoint=CheckpointSettings(path, interval=3),
            fingerprint_parts=parts,
            **kwargs,
        ).optimize(verbose=False)
        resumed = GeneticAlgorithm(
            config=GAConfig(population_size=6, max_generations=8, seed=7),
            checkpoint=CheckpointSettings(path, interval=2, resume=True),
            fingerprint_parts=parts,
            **kwargs,
        ).optimize(verbose=False)

        assert resumed.best_fitness == full.best_fitness
        np.testing.assert_array_equal(resumed.best_positions, full.best_positions)
        assert resumed.convergence_history == full.convergence_history
        assert resumed.final_fitness == pytest.approx(full.final_fitness)

    def test_pso_resume_equivalence_with_real_aep(self, tmp_path):
        boundary, diameters, calc, turbines, wind, wake = _build_aep()
        parts = build_problem_fingerprint_parts(
            boundary=boundary,
            rotor_diameters=diameters,
            turbines=turbines,
            wind_resource=wind,
            wake_model=wake,
            superposition_method="sum_of_squares",
            speed_step=1.0,
            speed_max=30.0,
        )
        kwargs = dict(
            n_turbines=len(turbines),
            rotor_diameters=diameters,
            boundary=boundary,
            fitness_fn=calc.evaluate_layout,
        )
        full = ParticleSwarmOptimizer(
            config=PSOConfig(swarm_size=6, max_iterations=8, seed=11),
            **kwargs,
        ).optimize(verbose=False)

        path = str(tmp_path / "pso_real.npz")
        ParticleSwarmOptimizer(
            config=PSOConfig(swarm_size=6, max_iterations=4, seed=11),
            checkpoint=CheckpointSettings(path, interval=4),
            fingerprint_parts=parts,
            **kwargs,
        ).optimize(verbose=False)
        resumed = ParticleSwarmOptimizer(
            config=PSOConfig(swarm_size=6, max_iterations=8, seed=11),
            checkpoint=CheckpointSettings(path, interval=3, resume=True),
            fingerprint_parts=parts,
            **kwargs,
        ).optimize(verbose=False)

        assert resumed.best_fitness == full.best_fitness
        np.testing.assert_array_equal(resumed.best_positions, full.best_positions)
        assert resumed.convergence_history == full.convergence_history

    def test_wake_model_change_blocks_resume(self, tmp_path):
        from wind_farm_opt.core.wake import GaussianWake
        from wind_farm_opt.optimization.checkpoint import CheckpointError

        boundary, diameters, calc_a, turbines, wind, wake_a = _build_aep()
        parts_a = build_problem_fingerprint_parts(
            boundary=boundary, rotor_diameters=diameters, turbines=turbines,
            wind_resource=wind, wake_model=wake_a,
            superposition_method="sum_of_squares", speed_step=1.0,
        )
        path = str(tmp_path / "ga.npz")
        GeneticAlgorithm(
            n_turbines=6, rotor_diameters=diameters, boundary=boundary,
            fitness_fn=calc_a.evaluate_layout,
            config=GAConfig(population_size=5, max_generations=2, seed=1),
            checkpoint=CheckpointSettings(path, interval=2),
            fingerprint_parts=parts_a,
        ).optimize(verbose=False)

        wake_b = GaussianWake(0.035)
        calc_b = AEPCalculator(
            turbines=turbines, wind_resource=wind, wake_model=wake_b,
            wake_superposition="sum_of_squares", speed_step=1.0,
        )
        parts_b = build_problem_fingerprint_parts(
            boundary=boundary, rotor_diameters=diameters, turbines=turbines,
            wind_resource=wind, wake_model=wake_b,
            superposition_method="sum_of_squares", speed_step=1.0,
        )
        with pytest.raises(CheckpointError, match="指纹"):
            GeneticAlgorithm(
                n_turbines=6, rotor_diameters=diameters, boundary=boundary,
                fitness_fn=calc_b.evaluate_layout,
                config=GAConfig(population_size=5, max_generations=4, seed=1),
                checkpoint=CheckpointSettings(path, resume=True),
                fingerprint_parts=parts_b,
            ).optimize(verbose=False)

    def test_default_run_has_no_run_info(self):
        boundary, diameters, calc, *_ = _build_aep(n_turbines=5)
        result = GeneticAlgorithm(
            n_turbines=5, rotor_diameters=diameters, boundary=boundary,
            fitness_fn=calc.evaluate_layout,
            config=GAConfig(population_size=5, max_generations=2, seed=3),
        ).optimize(verbose=False)
        assert result.run_info is None


def _write_config(path: str, out_dir: str, algo: str) -> str:
    cfg = WindFarmConfig(
        n_turbines=6,
        turbine_model="V126-3.45MW",
        wake_model="jensen",
        wake_decay=0.07,
        boundary_type="rectangular",
        boundary_params={"width": 3500, "height": 3500, "center_x": 0, "center_y": 0},
        wind_resource_params={"num_sectors": 6, "dominant_direction": 270.0, "mean_speed": 8.5},
        optimization=OptimizationConfig(
            algorithm=algo, population_size=6, max_iterations=6, seed=42
        ),
        visualization=VisualizationConfig(
            save_dir=out_dir, save_plots=False, show_plots=False,
            plot_wake_heatmap=False,
        ),
    )
    cfg_path = os.path.join(path, "config.json")
    cfg.to_json(cfg_path)
    return cfg_path


class TestCLI:
    def run_cli(self, cfg_path, *extra):
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH="/workspace")
        return subprocess.run(
            [sys.executable, "-m", "wind_farm_opt", "--config", cfg_path, *extra],
            capture_output=True, text=True, env=env, timeout=600,
        )

    def test_new_then_resume_marks_results_json(self, tmp_path):
        out_dir = str(tmp_path / "out")
        cfg_path = _write_config(str(tmp_path), out_dir, "ga")
        ckpt = str(tmp_path / "ga.npz")

        r1 = self.run_cli(
            cfg_path, "--no-economic", "--no-plots",
            "--checkpoint", ckpt, "--checkpoint-interval", "3",
        )
        assert r1.returncode == 0, r1.stderr
        assert "全新运行" in r1.stdout
        assert os.path.exists(ckpt)

        r2 = self.run_cli(
            cfg_path, "--no-economic", "--no-plots",
            "--checkpoint", ckpt, "--resume",
        )
        assert r2.returncode == 0, r2.stderr
        assert "断点恢复" in r2.stdout
        assert "检查点来源运行" in r2.stdout

        with open(os.path.join(out_dir, "results.json"), encoding="utf-8") as f:
            results = json.load(f)
        run = results["optimization_run"]
        assert run["mode"] == "resumed"
        assert run["mode_label"] == "断点恢复"
        assert run["resumed_at_step"] == 6
        assert run["resumed_from"].endswith("ga.npz")
        assert run["lineage"] == 1

    def test_default_cli_run_has_no_run_section(self, tmp_path):
        out_dir = str(tmp_path / "out2")
        cfg_path = _write_config(str(tmp_path), out_dir, "pso")
        r = self.run_cli(cfg_path, "--no-economic", "--no-plots")
        assert r.returncode == 0, r.stderr
        with open(os.path.join(out_dir, "results.json"), encoding="utf-8") as f:
            results = json.load(f)
        assert "optimization_run" not in results

    def test_resume_missing_file_starts_new(self, tmp_path):
        out_dir = str(tmp_path / "out3")
        cfg_path = _write_config(str(tmp_path), out_dir, "ga")
        r = self.run_cli(
            cfg_path, "--no-economic", "--no-plots",
            "--checkpoint", str(tmp_path / "nope.npz"),
            "--resume",
        )
        assert r.returncode == 0, r.stderr
        assert "未找到" in r.stdout
