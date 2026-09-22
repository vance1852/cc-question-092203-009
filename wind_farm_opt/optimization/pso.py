"""粒子群优化器。"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..constraints.boundary import SiteBoundary
from ..constraints.spacing import (
    check_min_spacing,
    compute_min_spacing_from_diameters,
    enforce_min_spacing,
)
from .checkpoint import (
    CheckpointError,
    CheckpointManager,
    CheckpointSettings,
    require_array,
    require_scalar,
    restore_rng_state,
)


@dataclass
class PSOConfig:
    """粒子群算法配置参数。

    Parameters
    ----------
    swarm_size : int
        粒子群大小
    max_iterations : int
        最大迭代次数
    inertia_weight : float
        惯性权重 w
    cognitive_coeff : float
        认知系数 c1
    social_coeff : float
        社会系数 c2
    max_velocity : float
        最大速度（占场地范围的比例）
    min_spacing_multiple : float
        最小间距倍数（相对于转子直径）
    penalty_factor : float
        约束违反惩罚因子
    seed : Optional[int]
        随机种子
    checkpoint_path : Optional[str]
        周期性检查点文件路径；None（默认）表示不启用检查点，
        保持无断点的默认运行流程不变。
    checkpoint_interval : int
        每隔多少次迭代保存一次检查点（初始状态与最终状态会额外强制保存）。
    resume : Optional[bool]
        恢复策略：None 表示存在检查点则自动恢复、否则新跑；
        True 表示必须从检查点恢复（缺失即报错）；
        False 表示必须新跑（首次保存时原子替换旧文件）。
    """

    swarm_size: int = 40
    max_iterations: int = 150
    inertia_weight: float = 0.7
    cognitive_coeff: float = 1.49
    social_coeff: float = 1.49
    max_velocity: float = 0.2
    min_spacing_multiple: float = 5.0
    penalty_factor: float = 1e6
    seed: Optional[int] = None
    checkpoint_path: Optional[str] = None
    checkpoint_interval: int = 10
    resume: Optional[bool] = None


class ParticleSwarmOptimizer:
    """粒子群算法机位优化器。"""

    def __init__(
        self,
        n_turbines: int,
        rotor_diameters: np.ndarray,
        boundary: SiteBoundary,
        fitness_fn: Callable[[np.ndarray], float],
        config: Optional[PSOConfig] = None,
    ) -> None:
        self.n_turbines = n_turbines
        self.rotor_diameters = np.asarray(rotor_diameters, dtype=np.float64)
        self.boundary = boundary
        self.fitness_fn = fitness_fn
        self.config = config if config is not None else PSOConfig()

        self.rng = np.random.default_rng(self.config.seed)

        self.min_spacing = compute_min_spacing_from_diameters(
            self.rotor_diameters,
            self.config.min_spacing_multiple,
        )

        self.n_dim = n_turbines * 2
        self.x_range = boundary.x_max - boundary.x_min
        self.y_range = boundary.y_max - boundary.y_min

        self.vel_range = np.zeros(self.n_dim, dtype=np.float64)
        for i in range(self.n_dim):
            self.vel_range[i] = (
                self.x_range if i % 2 == 0 else self.y_range
            ) * self.config.max_velocity

        self.pos_bounds = np.zeros((self.n_dim, 2), dtype=np.float64)
        for i in range(self.n_dim):
            if i % 2 == 0:
                self.pos_bounds[i] = [boundary.x_min, boundary.x_max]
            else:
                self.pos_bounds[i] = [boundary.y_min, boundary.y_max]

        self._best_global_pos = None
        self._best_global_fitness = -np.inf
        self._best_iteration = 0

        self.convergence_history: list[float] = []
        self.mean_history: list[float] = []

    # ------------------------------------------------------------------
    # 检查点支持
    # ------------------------------------------------------------------

    def _checkpoint_settings(self) -> CheckpointSettings:
        return CheckpointSettings(
            path=self.config.checkpoint_path,
            interval=self.config.checkpoint_interval,
        )

    def _fingerprint_inputs(self) -> dict:
        """构造场地、风机、目标函数及算法行为相关配置的指纹输入。"""
        from .checkpoint import describe_fitness_function

        return {
            "problem": {
                "n_turbines": int(self.n_turbines),
                "rotor_diameters": self.rotor_diameters.tolist(),
                "min_spacing_multiple": float(self.config.min_spacing_multiple),
                "min_spacing": float(self.min_spacing),
                "boundary_vertices": self.boundary.vertices.tolist(),
            },
            "fitness_function": describe_fitness_function(self.fitness_fn),
            "algorithm": {
                "name": "pso",
                "swarm_size": int(self.config.swarm_size),
                "max_iterations": int(self.config.max_iterations),
                "inertia_weight": float(self.config.inertia_weight),
                "cognitive_coeff": float(self.config.cognitive_coeff),
                "social_coeff": float(self.config.social_coeff),
                "max_velocity": float(self.config.max_velocity),
                "penalty_factor": float(self.config.penalty_factor),
                "seed": self.config.seed,
            },
        }

    def _snapshot_state(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        fitness: np.ndarray,
        best_personal_pos: np.ndarray,
        best_personal_fitness: np.ndarray,
    ) -> dict:
        """捕获足以无损继续计算的全部算法状态。"""
        return {
            "positions": positions.tolist(),
            "velocities": velocities.tolist(),
            "fitness": fitness.tolist(),
            "best_personal_pos": best_personal_pos.tolist(),
            "best_personal_fitness": best_personal_fitness.tolist(),
            "best_global_pos": np.asarray(self._best_global_pos, dtype=np.float64).tolist(),
            "best_global_fitness": float(self._best_global_fitness),
            "best_iteration": int(self._best_iteration),
            "convergence_history": list(self.convergence_history),
            "mean_history": list(self.mean_history),
            "rng_state": self.rng.bit_generator.state,
        }

    def _restore_state(
        self, payload: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """从检查点恢复全部状态。"""
        state = payload["state"]
        swarm_size = self.config.swarm_size
        completed = int(payload["completed_iterations"])

        positions = require_array(state, "positions", (swarm_size, self.n_dim))
        velocities = require_array(state, "velocities", (swarm_size, self.n_dim))
        fitness = require_array(state, "fitness", (swarm_size,))
        best_personal_pos = require_array(
            state, "best_personal_pos", (swarm_size, self.n_dim)
        )
        best_personal_fitness = require_array(
            state, "best_personal_fitness", (swarm_size,)
        )
        best_global_pos = require_array(state, "best_global_pos", (self.n_turbines, 2))

        n_hist = completed
        conv = state.get("convergence_history")
        mean = state.get("mean_history")
        if not isinstance(conv, list) or len(conv) != n_hist:
            raise CheckpointError(
                f"检查点收敛历史长度与迭代位置不一致: {None if conv is None else len(conv)} != {n_hist}"
            )
        if not isinstance(mean, list) or len(mean) != n_hist:
            raise CheckpointError(
                f"检查点均值历史长度与迭代位置不一致: {None if mean is None else len(mean)} != {n_hist}"
            )

        restore_rng_state(self.rng, state.get("rng_state"))

        self._best_global_pos = best_global_pos.copy()
        self._best_global_fitness = float(
            require_scalar(state, "best_global_fitness", (int, float))
        )
        self._best_iteration = int(require_scalar(state, "best_iteration", int))
        self.convergence_history = [float(v) for v in conv]
        self.mean_history = [float(v) for v in mean]

        return (
            positions,
            velocities,
            fitness,
            best_personal_pos,
            best_personal_fitness,
            completed,
        )

    def _initialize_swarm(self, swarm_size: int) -> tuple[np.ndarray, np.ndarray]:
        """初始化粒子群。"""
        positions = np.zeros((swarm_size, self.n_dim), dtype=np.float64)
        velocities = np.zeros((swarm_size, self.n_dim), dtype=np.float64)

        for i in range(swarm_size):
            pos = self._generate_valid_layout()
            positions[i] = pos.flatten()
            velocities[i] = self.rng.uniform(
                -self.vel_range, self.vel_range, self.n_dim
            )

        return positions, velocities

    def _generate_valid_layout(self) -> np.ndarray:
        """生成一个满足约束的初始布局。"""
        max_attempts = 100

        for _ in range(max_attempts):
            try:
                positions = self.boundary.sample_random_points(
                    self.n_turbines, self.rng, max_attempts=50
                )
                valid, _ = check_min_spacing(positions, self.min_spacing)
                if valid:
                    return positions
            except RuntimeError:
                continue

            try:
                positions = self.boundary.sample_random_points(
                    self.n_turbines, self.rng, max_attempts=50
                )
                positions = enforce_min_spacing(
                    positions, self.min_spacing, self.boundary, self.rng
                )
                return positions
            except RuntimeError:
                continue

        raise RuntimeError("无法生成满足约束的初始布局")

    def _compute_penalty(self, positions_flat: np.ndarray) -> float:
        """计算约束违反惩罚。"""
        positions = positions_flat.reshape(self.n_turbines, 2)

        penalty = 0.0

        inside = self.boundary.contains_all(positions)
        if not inside.all():
            n_violations = np.sum(~inside)
            penalty += n_violations * self.config.penalty_factor

        valid, violations = check_min_spacing(positions, self.min_spacing)
        if not valid:
            for i, j in violations:
                dist = np.linalg.norm(positions[i] - positions[j])
                penalty += (self.min_spacing - dist) * self.config.penalty_factor

        return penalty

    def _evaluate_particles(self, positions: np.ndarray) -> np.ndarray:
        """评估所有粒子的适应度。"""
        swarm_size = positions.shape[0]
        fitness = np.zeros(swarm_size, dtype=np.float64)

        for i in range(swarm_size):
            penalty = self._compute_penalty(positions[i])

            if penalty > 0:
                fitness[i] = -penalty
            else:
                pos_reshaped = positions[i].reshape(self.n_turbines, 2)
                try:
                    fitness[i] = self.fitness_fn(pos_reshaped)
                except Exception:
                    fitness[i] = -self.config.penalty_factor

        return fitness

    def _repair(self, positions_flat: np.ndarray) -> np.ndarray:
        """修复违反约束的粒子。"""
        positions = positions_flat.reshape(self.n_turbines, 2)

        for i in range(self.n_turbines):
            if not self.boundary.contains_point(positions[i]):
                positions[i] = self.boundary.project_to_boundary(positions[i])

        valid, _ = check_min_spacing(positions, self.min_spacing)
        inside = self.boundary.contains_all(positions).all()

        if not (valid and inside):
            try:
                positions = enforce_min_spacing(
                    positions, self.min_spacing, self.boundary, self.rng
                )
            except RuntimeError:
                pass

        return positions.flatten()

    def optimize(self, verbose: bool = True) -> "OptimizeResult":
        """执行优化。

        配置 ``checkpoint_path`` 后会周期性保存检查点；再次运行并指向同一
        文件时从断点恢复。恢复时校验场地/风机/目标函数/算法指纹，不兼容则
        报错且不覆盖旧文件。

        Returns
        -------
        OptimizeResult
            优化结果
        """
        from .ga import OptimizeResult

        swarm_size = self.config.swarm_size
        max_iter = self.config.max_iterations

        w = self.config.inertia_weight
        c1 = self.config.cognitive_coeff
        c2 = self.config.social_coeff

        manager = CheckpointManager(
            algorithm="pso",
            settings=self._checkpoint_settings(),
            fingerprint_inputs=self._fingerprint_inputs(),
        )
        loaded = manager.start(self.config.resume)

        if verbose:
            print(f"\n=== 粒子群优化开始 ===")
            print(f"风机台数: {self.n_turbines}")
            print(f"粒子群大小: {swarm_size}")
            print(f"最大迭代: {max_iter}")
            print(f"最小间距: {self.min_spacing:.1f} m "
                  f"({self.config.min_spacing_multiple:.1f}倍转子直径)")
            print(f"w={w}, c1={c1}, c2={c2}")
            if manager.enabled:
                print(f"检查点文件: {manager.abspath()} (每 {self.config.checkpoint_interval} 次迭代保存)")
                if manager.mode == "resumed":
                    print(
                        f"运行方式: 从断点恢复（第 {manager.resumed_from_iteration} 次迭代，"
                        f"第 {manager.resume_count} 次恢复，run_id={manager.run_id[:8]}）"
                    )
                else:
                    print(f"运行方式: 全新运行 (run_id={manager.run_id[:8]})")
            print("=" * 35)

        if loaded is not None:
            (
                positions,
                velocities,
                fitness,
                best_personal_pos,
                best_personal_fitness,
                start_iter,
            ) = self._restore_state(loaded)
            if start_iter >= max_iter:
                if verbose:
                    print(f"检查点已完成 {start_iter} 次迭代（>= 目标 {max_iter} 次），直接产出结果")
            if verbose:
                print(
                    f"[恢复] 已还原第 {start_iter} 次迭代粒子状态与随机数状态，"
                    f"继续执行第 {start_iter + 1}~{max_iter} 次迭代"
                )
        else:
            positions, velocities = self._initialize_swarm(swarm_size)
            fitness = self._evaluate_particles(positions)

            best_personal_pos = positions.copy()
            best_personal_fitness = fitness.copy()

            best_global_idx = np.argmax(fitness)
            self._best_global_pos = positions[best_global_idx].reshape(self.n_turbines, 2).copy()
            self._best_global_fitness = float(fitness[best_global_idx])
            self._best_iteration = 0
            start_iter = 0

            # 初始状态强制落盘：昂贵的初始粒子生成也不会因算力回收而丢失。
            manager.checkpoint(
                self._snapshot_state(
                    positions,
                    velocities,
                    fitness,
                    best_personal_pos,
                    best_personal_fitness,
                ),
                completed_iterations=0,
                total_iterations=max_iter,
                force=True,
                verbose=verbose,
            )

        for iteration in range(start_iter, max_iter):
            self.convergence_history.append(float(self._best_global_fitness))
            self.mean_history.append(float(np.mean(fitness)))

            r1 = self.rng.random((swarm_size, self.n_dim))
            r2 = self.rng.random((swarm_size, self.n_dim))

            best_global_flat = self._best_global_pos.flatten()

            velocities = (
                w * velocities
                + c1 * r1 * (best_personal_pos - positions)
                + c2 * r2 * (best_global_flat - positions)
            )

            velocities = np.clip(velocities, -self.vel_range, self.vel_range)

            positions = positions + velocities

            positions = np.clip(
                positions,
                self.pos_bounds[:, 0],
                self.pos_bounds[:, 1],
            )

            for i in range(swarm_size):
                positions[i] = self._repair(positions[i])

            fitness = self._evaluate_particles(positions)

            improved_mask = fitness > best_personal_fitness
            best_personal_pos[improved_mask] = positions[improved_mask].copy()
            best_personal_fitness[improved_mask] = fitness[improved_mask].copy()

            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > self._best_global_fitness:
                self._best_global_fitness = float(fitness[current_best_idx])
                self._best_global_pos = positions[current_best_idx].reshape(
                    self.n_turbines, 2
                ).copy()
                self._best_iteration = iteration + 1

            completed = iteration + 1
            manager.checkpoint(
                self._snapshot_state(
                    positions,
                    velocities,
                    fitness,
                    best_personal_pos,
                    best_personal_fitness,
                ),
                completed_iterations=completed,
                total_iterations=max_iter,
                force=(completed == max_iter),
                verbose=False,
            )

            if verbose and (iteration % 5 == 0 or iteration == max_iter - 1):
                print(
                    f"Iter {iteration+1:3d} | "
                    f"Best: {self._best_global_fitness/1e3:8.2f} GWh | "
                    f"Mean: {np.mean(fitness)/1e3:8.2f} GWh | "
                    f"Found@Iter {self._best_iteration}"
                )

        if verbose:
            print("=" * 35)
            print(f"优化完成!")
            print(f"最优净AEP: {self._best_global_fitness/1e3:.2f} GWh")
            print(f"找到最优解的迭代: {self._best_iteration}")
            if manager.enabled and manager.mode == "resumed":
                print(
                    f"本次为断点恢复运行（来源: {manager.abspath()}，"
                    f"自第 {manager.resumed_from_iteration} 次迭代续算，共恢复 {manager.resume_count} 次）"
                )

        return OptimizeResult(
            best_positions=self._best_global_pos.copy(),
            best_fitness=float(self._best_global_fitness),
            best_generation=self._best_iteration,
            convergence_history=self.convergence_history.copy(),
            mean_history=self.mean_history.copy(),
            final_population=positions.copy(),
            final_fitness=fitness.copy(),
            run_provenance=manager.provenance() if manager.enabled else None,
        )
