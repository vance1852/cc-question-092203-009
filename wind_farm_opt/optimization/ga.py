"""遗传算法优化器。"""

from dataclasses import asdict, dataclass, field
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

#: GA 检查点状态结构版本；保存的数组集合/含义变化时递增。
GA_PAYLOAD_VERSION = 1


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
    run_info : Optional[dict]
        运行信息（新跑/恢复、检查点来源等）；未启用检查点时为 None。
    """

    best_positions: np.ndarray
    best_fitness: float
    best_generation: int
    convergence_history: list[float]
    mean_history: list[float]
    final_population: np.ndarray
    final_fitness: np.ndarray
    run_info: Optional[dict] = None


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
        checkpoint: Optional[CheckpointSettings] = None,
        fingerprint_parts: Optional[Mapping[str, object]] = None,
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
        checkpoint : Optional[CheckpointSettings]
            检查点设置；为 None 时不保存/恢复，行为与默认流程完全一致。
        fingerprint_parts : Optional[Mapping]
            场地/风机/目标函数相关的指纹片段（通常由
            :func:`build_problem_fingerprint_parts` 构造）。检查点只会在
            指纹完全一致时恢复。
        """
        self.n_turbines = n_turbines
        self.rotor_diameters = np.asarray(rotor_diameters, dtype=np.float64)
        self.boundary = boundary
        self.fitness_fn = fitness_fn
        self.config = config if config is not None else GAConfig()
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

        self._best_positions = None
        self._best_fitness = -np.inf
        self._best_generation = 0

        self.convergence_history: list[float] = []
        self.mean_history: list[float] = []

    def _fingerprint_parts(self) -> dict:
        """组装完整指纹：问题物理配置 + GA 算法配置。

        总代数 ``max_generations`` 不参与指纹：它属于运行预算而非问题定义，
        允许在恢复时调整（断点文件中另行记录 total_steps）。
        """
        algo_config = asdict(self.config)
        algo_config.pop("max_generations", None)
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
            "name": "ga",
            "payload_version": GA_PAYLOAD_VERSION,
            "config": algo_config,
        }
        return parts

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

    # -- 检查点状态 -------------------------------------------------------

    def _state_arrays(
        self, population: np.ndarray, fitness: np.ndarray
    ) -> dict[str, np.ndarray]:
        """导出代数边界处的完整状态（此时下一代尚未消耗任何随机数）。"""
        return {
            "population": population,
            "fitness": fitness,
            "best_positions": np.asarray(self._best_positions, dtype=np.float64),
            "best_fitness": np.asarray(self._best_fitness, dtype=np.float64),
            "best_generation": np.asarray(self._best_generation, dtype=np.int64),
            "convergence_history": np.asarray(
                self.convergence_history, dtype=np.float64
            ),
            "mean_history": np.asarray(self.mean_history, dtype=np.float64),
        }

    def _restore_state(self, arrays: dict[str, np.ndarray], rng_state: bytes) -> int:
        """从检查点恢复全部状态，返回已完成的代数。"""
        self.rng = deserialize_rng_state(rng_state)
        self._best_positions = arrays["best_positions"].reshape(
            self.n_turbines, 2
        ).copy()
        self._best_fitness = float(arrays["best_fitness"])
        self._best_generation = int(arrays["best_generation"])
        self.convergence_history = arrays["convergence_history"].astype(
            np.float64
        ).tolist()
        self.mean_history = arrays["mean_history"].astype(np.float64).tolist()
        return len(self.convergence_history)

    def _build_result(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        store: Optional[CheckpointStore],
    ) -> OptimizeResult:
        return OptimizeResult(
            best_positions=self._best_positions.copy(),
            best_fitness=float(self._best_fitness),
            best_generation=self._best_generation,
            convergence_history=self.convergence_history.copy(),
            mean_history=self.mean_history.copy(),
            final_population=population.copy(),
            final_fitness=fitness.copy(),
            run_info=store.result_info() if store is not None else None,
        )

    def optimize(self, verbose: bool = True) -> OptimizeResult:
        """执行优化。

        启用检查点且 ``resume=True`` 时，从兼容断点继续；否则从头新跑。
        续算到与不中断运行相同的总代数时，两者的结果逐位一致。

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

        if verbose:
            print(f"\n=== 遗传算法优化开始 ===")
            print(f"风机台数: {self.n_turbines}")
            print(f"种群大小: {pop_size}")
            print(f"最大代数: {max_gen}")
            print(f"最小间距: {self.min_spacing:.1f} m "
                  f"({self.config.min_spacing_multiple:.1f}倍转子直径)")
            print(f"场地面积: {self.boundary.area / 1e6:.2f} km²")
            if self.checkpoint_settings is not None:
                print(f"检查点文件: {self.checkpoint_settings.path} "
                      f"(每 {self.checkpoint_settings.interval} 代保存)")
            print("=" * 35)

        store: Optional[CheckpointStore] = None
        if self.checkpoint_settings is not None:
            store = CheckpointStore(
                self.checkpoint_settings,
                algorithm="ga",
                payload_version=GA_PAYLOAD_VERSION,
                fingerprint_parts=self._fingerprint_parts(),
                total_steps=max_gen,
                shape_info={
                    "n_turbines": int(self.n_turbines),
                    "n_dim": int(self.n_dim),
                    "population_size": int(pop_size),
                },
                verbose=verbose,
            )

        resumed = store.request_resume() if store is not None else None

        if resumed is not None:
            start_gen = self._restore_state(
                resumed["arrays"], resumed["rng_state"]
            )
            population = resumed["arrays"]["population"]
            fitness = resumed["arrays"]["fitness"]
            if start_gen != resumed["completed_steps"]:
                from .checkpoint import CheckpointError
                raise CheckpointError(
                    f"检查点历史长度 {start_gen} 与记录进度 "
                    f"{resumed['completed_steps']} 不一致，拒绝恢复"
                )
        else:
            population = self._initialize_population(pop_size)
            fitness = self._evaluate_population(population)

            best_idx = np.argmax(fitness)
            self._best_fitness = float(fitness[best_idx])
            self._best_positions = population[best_idx].reshape(
                self.n_turbines, 2
            ).copy()
            self._best_generation = 0
            start_gen = 0

        if start_gen >= max_gen:
            if verbose:
                print("[检查点] 断点已达到总代数，直接输出既有结果")
            return self._build_result(population, fitness, store)

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

            if store is not None and store.should_save(completed):
                store.save(
                    completed,
                    self._state_arrays(population, fitness),
                    serialize_rng_state(self.rng),
                    status=("completed" if completed >= max_gen else "running"),
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

        return self._build_result(population, fitness, store)
