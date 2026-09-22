"""遗传算法优化器。"""

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
class GAConfig:
    """遗传算法配置参数。

    Parameters
    ----------
    population_size : int
        种群大小
    max_generations : int
        最大迭代代数
    crossover_rate : float
        交叉概率
    mutation_rate : float
        变异概率
    mutation_strength : float
        变异强度（坐标标准差占场地范围的比例）
    elite_ratio : float
        精英保留比例
    tournament_size : int
        锦标赛选择的规模
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
        每隔多少代保存一次检查点（初始状态与最终状态会额外强制保存）。
    resume : Optional[bool]
        恢复策略：None 表示存在检查点则自动恢复、否则新跑；
        True 表示必须从检查点恢复（缺失即报错）；
        False 表示必须新跑（首次保存时原子替换旧文件）。
    """

    population_size: int = 50
    max_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    mutation_strength: float = 0.1
    elite_ratio: float = 0.1
    tournament_size: int = 3
    min_spacing_multiple: float = 5.0
    penalty_factor: float = 1e6
    seed: Optional[int] = None
    checkpoint_path: Optional[str] = None
    checkpoint_interval: int = 10
    resume: Optional[bool] = None


@dataclass
class OptimizeResult:
    """优化结果。

    Parameters
    ----------
    best_positions : np.ndarray
        最优风机位置 (N_turb, 2)
    best_fitness : float
        最优适应度（净AEP，MWh/year）
    best_generation : int
        找到最优解的代数
    convergence_history : list[float]
        每代最优适应度历史
    mean_history : list[float]
        每代平均适应度历史
    final_population : np.ndarray
        最终种群 (pop_size, N_turb*2)
    final_fitness : np.ndarray
        最终种群适应度 (pop_size,)
    run_provenance : Optional[dict]
        运行台账：标明新跑（fresh）/恢复（resumed）、运行 ID、检查点来源等；
        未启用检查点时为 None，默认无断点流程不受影响。
    """

    best_positions: np.ndarray
    best_fitness: float
    best_generation: int
    convergence_history: list[float]
    mean_history: list[float]
    final_population: np.ndarray
    final_fitness: np.ndarray
    run_provenance: Optional[dict] = None


class GeneticAlgorithm:
    """遗传算法机位优化器。

    优化目标：最大化年净发电量（等价于最小化尾流损失）。
    约束：最小间距、场地边界内。
    """

    def __init__(
        self,
        n_turbines: int,
        rotor_diameters: np.ndarray,
        boundary: SiteBoundary,
        fitness_fn: Callable[[np.ndarray], float],
        config: Optional[GAConfig] = None,
    ) -> None:
        """
        Parameters
        ----------
        n_turbines : int
            风机台数
        rotor_diameters : np.ndarray
            每台风机的转子直径
        boundary : SiteBoundary
            场地边界
        fitness_fn : Callable[[np.ndarray], float]
            适应度函数，输入位置数组 (N_turb, 2)，返回净AEP
        config : Optional[GAConfig]
            算法配置参数
        """
        self.n_turbines = n_turbines
        self.rotor_diameters = np.asarray(rotor_diameters, dtype=np.float64)
        self.boundary = boundary
        self.fitness_fn = fitness_fn
        self.config = config if config is not None else GAConfig()

        self.rng = np.random.default_rng(self.config.seed)

        self.min_spacing = compute_min_spacing_from_diameters(
            self.rotor_diameters,
            self.config.min_spacing_multiple,
        )

        self.n_dim = n_turbines * 2
        self.x_range = boundary.x_max - boundary.x_min
        self.y_range = boundary.y_max - boundary.y_min

        self._best_positions = None
        self._best_fitness = -np.inf
        self._best_generation = 0

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
        """构造场地、风机、目标函数及算法行为相关配置的指纹输入。

        不含检查点路径/间隔这类与续算正确性无关的运行选项；
        必须包含 max_generations，因为 RNG 消耗序列与迭代次数绑定，
        恢复到不同总迭代数的“等价性”无从保证，按不兼容处理更安全。
        """
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
                "name": "ga",
                "population_size": int(self.config.population_size),
                "max_generations": int(self.config.max_generations),
                "crossover_rate": float(self.config.crossover_rate),
                "mutation_rate": float(self.config.mutation_rate),
                "mutation_strength": float(self.config.mutation_strength),
                "elite_ratio": float(self.config.elite_ratio),
                "tournament_size": int(self.config.tournament_size),
                "penalty_factor": float(self.config.penalty_factor),
                "seed": self.config.seed,
            },
        }

    def _snapshot_state(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
    ) -> dict:
        """捕获足以无损继续计算的全部算法状态。"""
        return {
            "population": population.tolist(),
            "fitness": fitness.tolist(),
            "best_positions": np.asarray(self._best_positions, dtype=np.float64).tolist(),
            "best_fitness": float(self._best_fitness),
            "best_generation": int(self._best_generation),
            "convergence_history": list(self.convergence_history),
            "mean_history": list(self.mean_history),
            "rng_state": self.rng.bit_generator.state,
        }

    def _restore_state(self, payload: dict) -> tuple[np.ndarray, np.ndarray, int]:
        """从检查点恢复全部状态，返回 (种群, 适应度, 已完成代数)。"""
        state = payload["state"]
        pop_size = self.config.population_size
        completed = int(payload["completed_iterations"])

        population = require_array(state, "population", (pop_size, self.n_dim))
        fitness = require_array(state, "fitness", (pop_size,))
        best_positions = require_array(state, "best_positions", (self.n_turbines, 2))

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

        self._best_positions = best_positions.copy()
        self._best_fitness = float(require_scalar(state, "best_fitness", (int, float)))
        self._best_generation = int(require_scalar(state, "best_generation", int))
        self.convergence_history = [float(v) for v in conv]
        self.mean_history = [float(v) for v in mean]

        return population, fitness, completed

    def _initialize_population(self, pop_size: int) -> np.ndarray:
        """初始化种群。

        每个个体是展平的位置向量：[x1, y1, x2, y2, ..., xn, yn]
        """
        population = np.zeros((pop_size, self.n_dim), dtype=np.float64)

        for i in range(pop_size):
            positions = self._generate_valid_layout()
            population[i] = positions.flatten()

        return population

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

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """评估整个种群的适应度（带惩罚）。"""
        pop_size = population.shape[0]
        fitness = np.zeros(pop_size, dtype=np.float64)

        for i in range(pop_size):
            positions = population[i].reshape(self.n_turbines, 2)

            penalty = self._compute_penalty(population[i])

            if penalty > 0:
                fitness[i] = -penalty
            else:
                try:
                    fitness[i] = self.fitness_fn(positions)
                except Exception:
                    fitness[i] = -self.config.penalty_factor

        return fitness

    def _tournament_selection(
        self, population: np.ndarray, fitness: np.ndarray, n_select: int
    ) -> np.ndarray:
        """锦标赛选择。"""
        pop_size = population.shape[0]
        selected = np.zeros((n_select, self.n_dim), dtype=np.float64)

        for i in range(n_select):
            candidates = self.rng.integers(0, pop_size, size=self.config.tournament_size)
            best_idx = candidates[np.argmax(fitness[candidates])]
            selected[i] = population[best_idx]

        return selected

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """均匀交叉。"""
        if self.rng.random() > self.config.crossover_rate:
            return parent1.copy()

        mask = self.rng.integers(0, 2, size=self.n_dim, dtype=bool)
        child = np.where(mask, parent1, parent2)

        return child

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """高斯变异。"""
        mutated = individual.copy()

        for i in range(self.n_dim):
            if self.rng.random() < self.config.mutation_rate:
                range_sigma = (
                    self.x_range if i % 2 == 0 else self.y_range
                ) * self.config.mutation_strength
                mutated[i] += self.rng.normal(0.0, range_sigma)

        return mutated

    def _repair(self, individual: np.ndarray) -> np.ndarray:
        """修复违反约束的个体。"""
        positions = individual.reshape(self.n_turbines, 2)

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

    def optimize(self, verbose: bool = True) -> OptimizeResult:
        """执行优化。

        Parameters
        ----------
        verbose : bool
            是否打印进度信息

        Returns
        -------
        OptimizeResult
            优化结果
        """
        pop_size = self.config.population_size
        max_gen = self.config.max_generations

        n_elite = max(1, int(pop_size * self.config.elite_ratio))

        manager = CheckpointManager(
            algorithm="ga",
            settings=self._checkpoint_settings(),
            fingerprint_inputs=self._fingerprint_inputs(),
        )
        loaded = manager.start(self.config.resume)

        if verbose:
            print(f"\n=== 遗传算法优化开始 ===")
            print(f"风机台数: {self.n_turbines}")
            print(f"种群大小: {pop_size}")
            print(f"最大代数: {max_gen}")
            print(f"最小间距: {self.min_spacing:.1f} m "
                  f"({self.config.min_spacing_multiple:.1f}倍转子直径)")
            print(f"场地面积: {self.boundary.area / 1e6:.2f} km²")
            if manager.enabled:
                print(f"检查点文件: {manager.abspath()} (每 {self.config.checkpoint_interval} 代保存)")
                if manager.mode == "resumed":
                    print(
                        f"运行方式: 从断点恢复（第 {manager.resumed_from_iteration} 代，"
                        f"第 {manager.resume_count} 次恢复，run_id={manager.run_id[:8]}）"
                    )
                else:
                    print(f"运行方式: 全新运行 (run_id={manager.run_id[:8]})")
            print("=" * 35)

        if loaded is not None:
            population, fitness, start_gen = self._restore_state(loaded)
            if start_gen >= max_gen:
                if verbose:
                    print(f"检查点已完成 {start_gen} 代（>= 目标 {max_gen} 代），直接产出结果")
            if verbose:
                print(
                    f"[恢复] 已还原第 {start_gen} 代种群与随机数状态，"
                    f"继续执行第 {start_gen + 1}~{max_gen} 代"
                )
        else:
            population = self._initialize_population(pop_size)
            fitness = self._evaluate_population(population)

            best_idx = np.argmax(fitness)
            self._best_fitness = float(fitness[best_idx])
            self._best_positions = population[best_idx].reshape(self.n_turbines, 2)
            self._best_generation = 0
            start_gen = 0

            # 初始状态强制落盘：昂贵的初始种群生成也不会因算力回收而丢失。
            manager.checkpoint(
                self._snapshot_state(population, fitness),
                completed_iterations=0,
                total_iterations=max_gen,
                force=True,
                verbose=verbose,
            )

        for gen in range(start_gen, max_gen):
            self.convergence_history.append(float(self._best_fitness))
            self.mean_history.append(float(np.mean(fitness)))

            elite_idx = np.argsort(fitness)[-n_elite:]
            elites = population[elite_idx].copy()

            parents = self._tournament_selection(population, fitness, pop_size - n_elite)

            offspring = np.zeros((pop_size - n_elite, self.n_dim), dtype=np.float64)
            for i in range(0, pop_size - n_elite, 2):
                p1 = parents[i]
                p2 = parents[(i + 1) % (pop_size - n_elite)]
                c1 = self._crossover(p1, p2)
                c2 = self._crossover(p2, p1)
                offspring[i] = self._mutate(c1)
                if i + 1 < pop_size - n_elite:
                    offspring[i + 1] = self._mutate(c2)

            for i in range(len(offspring)):
                offspring[i] = self._repair(offspring[i])

            population[:n_elite] = elites
            population[n_elite:] = offspring

            fitness = self._evaluate_population(population)

            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > self._best_fitness:
                self._best_fitness = float(fitness[current_best_idx])
                self._best_positions = population[current_best_idx].reshape(
                    self.n_turbines, 2
                ).copy()
                self._best_generation = gen + 1

            completed = gen + 1
            manager.checkpoint(
                self._snapshot_state(population, fitness),
                completed_iterations=completed,
                total_iterations=max_gen,
                force=(completed == max_gen),
                verbose=False,
            )

            if verbose and (gen % 5 == 0 or gen == max_gen - 1):
                print(
                    f"Gen {gen+1:3d} | "
                    f"Best: {self._best_fitness/1e3:8.2f} GWh | "
                    f"Mean: {np.mean(fitness)/1e3:8.2f} GWh | "
                    f"Found@Gen {self._best_generation}"
                )

        if verbose:
            print("=" * 35)
            print(f"优化完成!")
            print(f"最优净AEP: {self._best_fitness/1e3:.2f} GWh")
            print(f"找到最优解的代数: {self._best_generation}")
            if manager.enabled and manager.mode == "resumed":
                print(
                    f"本次为断点恢复运行（来源: {manager.abspath()}，"
                    f"自第 {manager.resumed_from_iteration} 代续算，共恢复 {manager.resume_count} 次）"
                )

        return OptimizeResult(
            best_positions=self._best_positions.copy(),
            best_fitness=float(self._best_fitness),
            best_generation=self._best_generation,
            convergence_history=self.convergence_history.copy(),
            mean_history=self.mean_history.copy(),
            final_population=population.copy(),
            final_fitness=fitness.copy(),
            run_provenance=manager.provenance() if manager.enabled else None,
        )
