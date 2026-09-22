"""粒子群优化器。"""

import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

import numpy as np

from ..constraints.boundary import SiteBoundary
from ..constraints.spacing import (
    check_min_spacing,
    compute_min_spacing_from_diameters,
    enforce_min_spacing,
)
from .checkpoint import (
    CheckpointError,
    CheckpointSettings,
    CheckpointStore,
    deserialize_rng_state,
    serialize_rng_state,
)
from .ga import OptimizeResult

#: PSO 检查点状态结构版本；保存的数组集合/含义变化时递增。
PSO_PAYLOAD_VERSION = 1


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


class ParticleSwarmOptimizer:
    """粒子群算法机位优化器。"""

    def __init__(
        self,
        n_turbines: int,
        rotor_diameters: np.ndarray,
        boundary: SiteBoundary,
        fitness_fn: Callable[[np.ndarray], float],
        config: Optional[PSOConfig] = None,
        checkpoint: Optional[CheckpointSettings] = None,
        fingerprint_parts: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.n_turbines = n_turbines
        self.rotor_diameters = np.asarray(rotor_diameters, dtype=np.float64)
        self.boundary = boundary
        self.fitness_fn = fitness_fn
        self.config = config if config is not None else PSOConfig()
        self.checkpoint_settings = checkpoint
        self.extra_fingerprint_parts = dict(fingerprint_parts) if fingerprint_parts else {}

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

    def _fingerprint_parts(self) -> dict:
        """组装完整指纹：问题物理配置 + PSO 算法配置。

        总迭代数 ``max_iterations`` 不参与指纹：它属于运行预算而非问题定义，
        允许在恢复时调整（断点文件中另行记录 total_steps）。
        """
        algo_config = dataclasses.asdict(self.config)
        algo_config.pop("max_iterations", None)
        parts: dict = dict(self.extra_fingerprint_parts)
        parts.setdefault("site", {}).update(
            {
                "boundary_type": type(self.boundary).__name__,
                "vertices": np.asarray(
                    self.boundary.vertices, dtype=np.float64
                ).tolist(),
            }
        )
        parts.setdefault("turbines", {}).update(
            {
                "count": int(self.n_turbines),
                "rotor_diameters": self.rotor_diameters.tolist(),
                "min_spacing_multiple": float(self.config.min_spacing_multiple),
                "min_spacing": float(self.min_spacing),
            }
        )
        parts["algorithm"] = {
            "name": "pso",
            "payload_version": PSO_PAYLOAD_VERSION,
            "config": algo_config,
        }
        return parts

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

    # -- 检查点状态 -------------------------------------------------------

    def _state_arrays(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        fitness: np.ndarray,
        best_personal_pos: np.ndarray,
        best_personal_fitness: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """导出迭代边界处的完整状态（此时下一次迭代尚未消耗任何随机数）。"""
        return {
            "positions": positions,
            "velocities": velocities,
            "fitness": fitness,
            "best_personal_pos": best_personal_pos,
            "best_personal_fitness": best_personal_fitness,
            "best_global_pos": np.asarray(self._best_global_pos, dtype=np.float64),
            "best_global_fitness": np.asarray(
                self._best_global_fitness, dtype=np.float64
            ),
            "best_iteration": np.asarray(self._best_iteration, dtype=np.int64),
            "convergence_history": np.asarray(
                self.convergence_history, dtype=np.float64
            ),
            "mean_history": np.asarray(self.mean_history, dtype=np.float64),
        }

    def _restore_state(self, arrays: dict[str, np.ndarray], rng_state: bytes) -> int:
        """从检查点恢复全部状态，返回已完成的迭代数。"""
        self.rng = deserialize_rng_state(rng_state)
        self._best_global_pos = arrays["best_global_pos"].reshape(
            self.n_turbines, 2
        ).copy()
        self._best_global_fitness = float(arrays["best_global_fitness"])
        self._best_iteration = int(arrays["best_iteration"])
        self.convergence_history = arrays["convergence_history"].astype(
            np.float64
        ).tolist()
        self.mean_history = arrays["mean_history"].astype(np.float64).tolist()
        return len(self.convergence_history)

    def optimize(self, verbose: bool = True) -> OptimizeResult:
        """执行优化。

        启用检查点且 ``resume=True`` 时，从兼容断点继续；否则从头新跑。
        续算到与不中断运行相同的总迭代数时，两者的结果逐位一致。

        Returns
        -------
        OptimizeResult
            优化结果
        """
        swarm_size = self.config.swarm_size
        max_iter = self.config.max_iterations

        w = self.config.inertia_weight
        c1 = self.config.cognitive_coeff
        c2 = self.config.social_coeff

        if verbose:
            print(f"\n=== 粒子群优化开始 ===")
            print(f"风机台数: {self.n_turbines}")
            print(f"粒子群大小: {swarm_size}")
            print(f"最大迭代: {max_iter}")
            print(f"最小间距: {self.min_spacing:.1f} m "
                  f"({self.config.min_spacing_multiple:.1f}倍转子直径)")
            print(f"w={w}, c1={c1}, c2={c2}")
            if self.checkpoint_settings is not None:
                print(f"检查点文件: {self.checkpoint_settings.path} "
                      f"(每 {self.checkpoint_settings.interval} 次迭代保存)")
            print("=" * 35)

        store: Optional[CheckpointStore] = None
        if self.checkpoint_settings is not None:
            store = CheckpointStore(
                self.checkpoint_settings,
                algorithm="pso",
                payload_version=PSO_PAYLOAD_VERSION,
                fingerprint_parts=self._fingerprint_parts(),
                total_steps=max_iter,
                shape_info={
                    "n_turbines": int(self.n_turbines),
                    "n_dim": int(self.n_dim),
                    "swarm_size": int(swarm_size),
                },
                verbose=verbose,
            )

        resumed = store.request_resume() if store is not None else None

        if resumed is not None:
            arrays = resumed["arrays"]
            start_iter = self._restore_state(arrays, resumed["rng_state"])
            if start_iter != resumed["completed_steps"]:
                raise CheckpointError(
                    f"检查点历史长度 {start_iter} 与记录进度 "
                    f"{resumed['completed_steps']} 不一致，拒绝恢复"
                )
            positions = arrays["positions"]
            velocities = arrays["velocities"]
            fitness = arrays["fitness"]
            best_personal_pos = arrays["best_personal_pos"]
            best_personal_fitness = arrays["best_personal_fitness"]
        else:
            positions, velocities = self._initialize_swarm(swarm_size)
            fitness = self._evaluate_particles(positions)

            best_personal_pos = positions.copy()
            best_personal_fitness = fitness.copy()

            best_global_idx = np.argmax(fitness)
            self._best_global_pos = positions[best_global_idx].reshape(
                self.n_turbines, 2
            ).copy()
            self._best_global_fitness = float(fitness[best_global_idx])
            self._best_iteration = 0
            start_iter = 0

        if start_iter >= max_iter:
            if verbose:
                print("[检查点] 断点已达到总迭代数，直接输出既有结果")
            result = OptimizeResult(
                best_positions=self._best_global_pos.copy(),
                best_fitness=float(self._best_global_fitness),
                best_generation=self._best_iteration,
                convergence_history=self.convergence_history.copy(),
                mean_history=self.mean_history.copy(),
                final_population=positions.copy(),
                final_fitness=fitness.copy(),
                run_info=store.result_info() if store is not None else None,
            )
            return result

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

            if store is not None and store.should_save(completed):
                store.save(
                    completed,
                    self._state_arrays(
                        positions,
                        velocities,
                        fitness,
                        best_personal_pos,
                        best_personal_fitness,
                    ),
                    serialize_rng_state(self.rng),
                    status=("completed" if completed >= max_iter else "running"),
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

        return OptimizeResult(
            best_positions=self._best_global_pos.copy(),
            best_fitness=float(self._best_global_fitness),
            best_generation=self._best_iteration,
            convergence_history=self.convergence_history.copy(),
            mean_history=self.mean_history.copy(),
            final_population=positions.copy(),
            final_fitness=fitness.copy(),
            run_info=store.result_info() if store is not None else None,
        )
