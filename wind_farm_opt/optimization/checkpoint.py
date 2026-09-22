"""优化检查点：周期性保存与断点恢复。

为 GA/PSO 提供原子写入的周期性检查点，完整保存继续计算所需的状态
（种群或粒子、速度、个体及全局最佳、迭代位置、收敛历史与随机数状态），
并通过场地、风机、目标函数及算法相关配置的稳定指纹（SHA-256）阻止错误续算。

设计要点
--------
* 写入采用“同目录临时文件 + fsync + os.replace”，崩溃或磁盘写满都不会
  截断已有检查点。
* 加载时依次校验：文件可读性/JSON 完整性、整包完整性摘要（防任何字段被
  篡改或字节级损坏）、格式版本、算法名、指纹自洽（防指纹字段损坏）、运行时
  指纹一致（场地/风机/目标/算法配置兼容）。任何一项失败都抛出
  :class:`CheckpointError`，且不会触发写回，旧文件保持原样。
* 指纹输入使用规范化 JSON（sort_keys、固定分隔符、float 走 repr 往返），
  同一份配置在不同进程中得到相同指纹。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np


#: 检查点格式版本。结构或状态语义发生不兼容变化时递增。
FORMAT_VERSION = 1
SUPPORTED_VERSIONS = (1,)


class CheckpointError(RuntimeError):
    """检查点缺失、已损坏、版本过旧或与当前场地/模型配置不兼容。"""


@dataclass(frozen=True)
class CheckpointSettings:
    """检查点设置。

    Parameters
    ----------
    path : Optional[str]
        检查点文件路径；为 None 时完全停用检查点（默认无断点流程）。
    interval : int
        每隔多少代/次迭代保存一次（首次初始化状态与末次状态会额外强制保存）。
    """

    path: Optional[str] = None
    interval: int = 10

    def __post_init__(self) -> None:
        if self.path is not None and self.interval < 1:
            raise ValueError("检查点保存间隔 checkpoint_interval 必须 >= 1")

    @property
    def enabled(self) -> bool:
        return self.path is not None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_native(obj: Any) -> Any:
    """递归把 numpy 标量/数组转换为可 JSON 序列化的原生 Python 类型。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    return obj


def compute_fingerprint(inputs: dict) -> str:
    """根据指纹输入计算稳定的 SHA-256 指纹。

    float 通过 Python repr 规范化输出，可精确往返；字典键排序后再序列化，
    保证同一组配置在任意进程中得到相同的十六进制摘要。
    """
    canonical = json.dumps(
        to_native(inputs),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_write_json(path: str, payload: dict) -> None:
    """原子写入 JSON：同目录临时文件 -> fsync -> os.replace。

    只有完整、可刷新的文件才会替换目标路径，因此写入中途被回收算力窗口
    不会损坏既有检查点。
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=".checkpoint-",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(directory)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _fsync_directory(directory: str) -> None:
    """尽力持久化目录项（replace 结果），不支持的平台上静默跳过。"""
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


class CheckpointManager:
    """管理一次优化运行的检查点起始（新跑/恢复）与周期性写入。

    Parameters
    ----------
    algorithm : str
        ``"ga"`` 或 ``"pso"``。
    settings : CheckpointSettings
        路径与保存间隔。
    fingerprint_inputs : dict
        场地、风机、目标函数、算法行为相关的配置（含总迭代数，以保证
        “恢复到相同总迭代数”这一可复现前提；不含检查点路径、保存间隔、
        resume 标志等与续算正确性无关的运行选项）。
    """

    def __init__(
        self,
        *,
        algorithm: str,
        settings: CheckpointSettings,
        fingerprint_inputs: dict,
    ) -> None:
        self.algorithm = algorithm
        self.settings = settings
        self.fingerprint_inputs = to_native(fingerprint_inputs)
        self.runtime_fingerprint = compute_fingerprint(self.fingerprint_inputs)

        # 全新运行的台账；start() 恢复成功后会采用旧运行的身份。
        self.mode: str = "fresh"
        self.run_id: str = uuid.uuid4().hex
        self.resume_count: int = 0
        self.created_at: str = _now_iso()
        self.resumed_from_iteration: int = 0

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    @property
    def path(self) -> Optional[str]:
        return self.settings.path

    def abspath(self) -> Optional[str]:
        return os.path.abspath(self.settings.path) if self.settings.path else None

    def start(self, resume: Optional[bool]) -> Optional[dict]:
        """决定新跑还是恢复，并返回已加载的检查点载荷（新跑时为 None）。

        Parameters
        ----------
        resume : Optional[bool]
            - None（默认）：配置了路径且文件存在则恢复，否则新跑；
            - True：必须恢复，文件缺失时报错；
            - False：必须新跑，已有文件将在首次保存时被原子替换。

        任何损坏或不兼容都会抛出 :class:`CheckpointError`，且不会写文件。
        """
        if not self.enabled:
            if resume is True:
                raise CheckpointError("已要求 --resume，但未配置检查点路径")
            return None

        path = self.settings.path
        file_exists = os.path.exists(path)

        if not file_exists:
            if resume is True:
                raise CheckpointError(f"找不到用于恢复的检查点文件: {path}")
            return None

        payload = self._read_payload(path)
        self._validate_payload(payload)

        if resume is False:
            # 操作员显式要求新跑：保留旧文件直到第一次完整保存后再原子替换。
            return None

        self._adopt(payload)
        return payload

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------

    def checkpoint(
        self,
        state: dict,
        *,
        completed_iterations: int,
        total_iterations: int,
        force: bool = False,
        verbose: bool = False,
    ) -> bool:
        """按间隔保存检查点；返回本次是否实际写盘。"""
        if not self.enabled:
            return False

        due = (
            force
            or completed_iterations >= total_iterations
            or completed_iterations % self.settings.interval == 0
        )
        if not due:
            return False

        payload = {
            "format_version": FORMAT_VERSION,
            "algorithm": self.algorithm,
            "run_id": self.run_id,
            "resume_count": self.resume_count,
            "created_at": self.created_at,
            "updated_at": _now_iso(),
            "completed_iterations": int(completed_iterations),
            "total_iterations": int(total_iterations),
            "fingerprint": self.runtime_fingerprint,
            "fingerprint_inputs": self.fingerprint_inputs,
            "state": to_native(state),
        }
        # 整包完整性摘要：任何字节级损坏或对状态/迭代位置/指纹字段的篡改，
        # 都会在加载时被发现。指纹负责“兼容性”，完整性摘要负责“未损坏”。
        payload["integrity"] = self._integrity_digest(payload)
        _atomic_write_json(self.settings.path, payload)

        if verbose:
            print(
                f"[检查点] 已保存第 {completed_iterations} 代/次迭代状态 "
                f"-> {os.path.abspath(self.settings.path)}"
            )
        return True

    # ------------------------------------------------------------------
    # 加载与校验
    # ------------------------------------------------------------------

    def _read_payload(self, path: str) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            raise CheckpointError(f"检查点文件无法读取: {path} ({e})") from e

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            raise CheckpointError(
                f"检查点文件已损坏（JSON 解析失败），为保留现有结果已拒绝加载: "
                f"{path} ({e})"
            ) from e

        return payload

    @staticmethod
    def _integrity_digest(body: dict) -> str:
        """对除 integrity 字段外的整包内容计算 SHA-256 摘要。"""
        canonical = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _validate_payload(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            raise CheckpointError("检查点内容结构无效：顶层不是对象")

        required = (
            "format_version",
            "algorithm",
            "run_id",
            "resume_count",
            "completed_iterations",
            "fingerprint",
            "fingerprint_inputs",
            "state",
            "integrity",
        )
        missing = [k for k in required if k not in payload]
        if missing:
            raise CheckpointError(f"检查点内容结构无效：缺少字段 {missing}")

        # 先做整包完整性校验：任何对状态、迭代位置或指纹字段的字节级损坏/篡改，
        # 都会使摘要不一致。
        stored_integrity = payload["integrity"]
        if not isinstance(stored_integrity, str):
            raise CheckpointError("检查点完整性摘要结构无效")
        body = {k: v for k, v in payload.items() if k != "integrity"}
        expected_integrity = self._integrity_digest(body)
        if not hmac.compare_digest(expected_integrity, stored_integrity):
            raise CheckpointError(
                "检查点完整性校验失败：文件可能已损坏或被修改，已拒绝加载"
            )

        version = payload["format_version"]
        if version not in SUPPORTED_VERSIONS:
            raise CheckpointError(
                f"检查点版本不兼容: 文件版本={version}，当前软件支持 "
                f"{SUPPORTED_VERSIONS}；为避免错误续算已拒绝加载，"
                f"旧检查点保留在原位: {self.settings.path}"
            )

        if payload["algorithm"] != self.algorithm:
            raise CheckpointError(
                f"检查点算法不匹配: 文件为 {payload['algorithm']!r}，"
                f"当前运行为 {self.algorithm!r}"
            )

        stored_inputs = payload.get("fingerprint_inputs")
        if not isinstance(stored_inputs, dict):
            raise CheckpointError("检查点指纹输入结构无效")

        # 用文件自带输入重算指纹，识别截断、损坏或被篡改的文件。
        stored_fingerprint = compute_fingerprint(stored_inputs)
        if not hmac.compare_digest(stored_fingerprint, str(payload["fingerprint"])):
            raise CheckpointError(
                "检查点指纹校验失败：文件可能已损坏或被修改，已拒绝加载"
            )

        # 用当前场地/风机/目标函数配置重算指纹，识别“旧断点与当前场地和模型不兼容”。
        if not hmac.compare_digest(stored_fingerprint, self.runtime_fingerprint):
            raise CheckpointError(
                "检查点与当前场地、风机或目标函数配置不一致（指纹不匹配），"
                "已拒绝恢复以免错误续算。请确认场地边界、风机型号、尾流/风资源 "
                "及算法参数与生成检查点时完全相同，或显式新跑并另选检查点路径。"
            )

        completed = payload["completed_iterations"]
        if not isinstance(completed, int) or completed < 0:
            raise CheckpointError("检查点迭代位置无效")
        if not isinstance(payload["state"], dict):
            raise CheckpointError("检查点算法状态结构无效")

    def _adopt(self, payload: dict) -> None:
        self.mode = "resumed"
        self.run_id = str(payload["run_id"])
        self.resume_count = int(payload.get("resume_count", 0)) + 1
        self.created_at = str(payload.get("created_at", self.created_at))
        self.resumed_from_iteration = int(payload["completed_iterations"])

    # ------------------------------------------------------------------
    # 运行 provenance
    # ------------------------------------------------------------------

    def provenance(self) -> dict:
        """返回标明新跑/恢复及检查点来源的运行摘要信息。"""
        return {
            "mode": self.mode,
            "run_id": self.run_id,
            "resume_count": self.resume_count,
            "checkpoint_path": self.abspath(),
            "resumed_from_iteration": self.resumed_from_iteration,
        }


def require_array(
    state: dict,
    key: str,
    shape: tuple,
    dtype=np.float64,
) -> np.ndarray:
    """从检查点状态中取出并校验数组形状，失败时抛出 CheckpointError。"""
    if key not in state:
        raise CheckpointError(f"检查点算法状态缺少字段: {key!r}")
    try:
        arr = np.asarray(state[key], dtype=dtype)
    except (TypeError, ValueError) as e:
        raise CheckpointError(f"检查点字段 {key!r} 无法解析为数组") from e
    if arr.shape != shape:
        raise CheckpointError(
            f"检查点字段 {key!r} 形状不匹配: 期望 {shape}，实际 {arr.shape}"
        )
    return arr


def require_scalar(state: dict, key: str, expected_type: type | tuple):
    """从检查点状态中取出标量字段并校验类型，失败时抛出 CheckpointError。"""
    if key not in state:
        raise CheckpointError(f"检查点算法状态缺少字段: {key!r}")
    value = state[key]
    # bool 是 int 的子类型，期望 int 时显式拒绝 bool。
    if isinstance(value, bool) and expected_type is not bool:
        raise CheckpointError(f"检查点字段 {key!r} 类型无效: bool")
    if not isinstance(value, expected_type):
        raise CheckpointError(f"检查点字段 {key!r} 类型无效: {type(value).__name__}")
    return value


def restore_rng_state(rng: np.random.Generator, state: dict) -> None:
    """把随机数发生器恢复到检查点记录的位生成器状态。"""
    if not isinstance(state, dict):
        raise CheckpointError("检查点随机数状态结构无效")
    try:
        rng.bit_generator.state = state
    except (ValueError, TypeError) as e:
        raise CheckpointError(
            f"随机数状态与当前 NumPy 位生成器不兼容: {e}"
        ) from e


def describe_fitness_function(fitness_fn: Callable) -> dict:
    """提取目标函数的稳定指纹描述。

    优化器的目标函数通常是绑定方法 ``AEPCalculator.evaluate_layout``，其返回值
    由风机、风资源、尾流模型及叠加方式决定。这里用鸭子类型把这些数据全部纳入
    指纹；任何一项变化都会导致旧断点无法恢复。

    若目标函数对象（或其绑定实例）提供 ``checkpoint_fingerprint()`` 方法，
    则优先采用其返回的稳定字典，便于用户自定义目标函数参与校验。
    """
    owner = getattr(fitness_fn, "__self__", None)

    if hasattr(fitness_fn, "checkpoint_fingerprint"):
        return {"custom": True, "detail": to_native(fitness_fn.checkpoint_fingerprint())}
    if owner is not None and hasattr(owner, "checkpoint_fingerprint"):
        return {
            "method": f"{type(owner).__name__}.{getattr(fitness_fn, '__name__', 'fitness')}",
            "custom": True,
            "detail": to_native(owner.checkpoint_fingerprint()),
        }

    description: dict[str, Any] = {
        "callable": getattr(fitness_fn, "__qualname__", type(fitness_fn).__name__),
    }

    if owner is None:
        # 普通函数/可调用对象没有可内省的数据依赖，只能记录其身份。
        description["note"] = (
            "未绑定的目标函数无法自动捕获数据依赖；"
            "请为其提供 checkpoint_fingerprint() 方法以获得严格的兼容性校验"
        )
        return description

    description["owner_class"] = (
        f"{type(owner).__module__}.{type(owner).__qualname__}"
    )

    wake_model = getattr(owner, "wake_model", None)
    if wake_model is not None:
        description["wake_model"] = {
            "type": f"{type(wake_model).__module__}.{type(wake_model).__qualname__}",
            "params": {
                k: v
                for k, v in sorted(vars(wake_model).items())
                if not k.startswith("_")
            },
        }

    wind_resource = getattr(owner, "wind_resource", None)
    if wind_resource is not None and hasattr(wind_resource, "sectors"):
        description["wind_resource"] = {
            "num_sectors": len(wind_resource.sectors),
            "sectors": [
                {
                    "direction_center": s.direction_center,
                    "direction_width": s.direction_width,
                    "frequency": s.frequency,
                    "mean_speed": s.mean_speed,
                    "weibull_k": s.weibull_k,
                    "weibull_c": s.weibull_c,
                }
                for s in wind_resource.sectors
            ],
        }

    for attr in ("wake_superposition", "speed_step", "speed_max"):
        if hasattr(owner, attr):
            description[attr] = getattr(owner, attr)

    turbines = getattr(owner, "turbines", None)
    if turbines is not None:
        description["turbines"] = {
            "count": len(turbines),
            "names": [getattr(t, "name", type(t).__name__) for t in turbines],
            "rotor_diameters": [float(t.rotor_diameter) for t in turbines],
            "hub_heights": [float(t.hub_height) for t in turbines],
            "thrust_coefficients": [float(t.thrust_coefficient) for t in turbines],
            "rated_powers": [float(t.rated_power) for t in turbines],
            "power_curves": [
                np.asarray(t.power_curve, dtype=np.float64).tolist()
                for t in turbines
            ],
        }

    return description
