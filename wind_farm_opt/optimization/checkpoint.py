"""优化检查点：周期性原子保存与断点恢复。

为 GA / PSO 提供统一的检查点机制：

- ``CheckpointSettings`` 描述检查点路径、保存周期与是否恢复；
- :func:`compute_fingerprint` 对场地、风机、目标函数及算法配置计算稳定指纹，
  指纹不匹配的旧断点一律拒绝，阻止错误续算；
- :func:`write_checkpoint` / :func:`read_checkpoint` 使用 ``.npz`` 容器，
  写入采用「临时文件 + 校验 + 原子替换」，崩溃或磁盘写满都不会破坏既有断点；
- :class:`CheckpointStore` 封装保存/恢复时的版本、指纹、状态校验逻辑。

默认（不传入 ``CheckpointSettings``）时优化器行为与以往完全一致。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

import numpy as np

#: 检查点容器格式版本。结构发生不兼容变化时必须递增。
CHECKPOINT_FORMAT_VERSION = 1

#: npz 内的固定键名。
_META_KEY = "__meta__"
_CHECKSUM_KEY = "__checksum__"
_RNG_KEY = "__rng_state__"


class CheckpointError(RuntimeError):
    """检查点无法保存、读取或与当前问题不兼容时抛出。"""


@dataclass
class CheckpointSettings:
    """检查点配置。

    Parameters
    ----------
    path : str
        检查点文件路径（GA 与 PSO 请使用不同文件）。
    interval : int
        每隔多少代/次迭代保存一次；结束时总会额外保存一次最终状态。
    resume : bool
        为 True 时尝试从 ``path`` 恢复；文件不存在则从头新跑，
        文件损坏或指纹/版本不匹配时抛出 :class:`CheckpointError`，绝不覆盖。
    """

    path: str
    interval: int = 10
    resume: bool = False

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("检查点路径不能为空")
        self.interval = int(self.interval)
        if self.interval < 1:
            raise ValueError("检查点保存周期必须 >= 1")


# ---------------------------------------------------------------------------
# 指纹
# ---------------------------------------------------------------------------

def _json_scalar(obj: Any) -> Any:
    """json.dumps 的兜底转换：仅允许 numpy 标量，拒绝其他不可序列化对象。"""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"无法序列化为稳定指纹的对象: {type(obj)!r}")


def canonical_json(obj: Any) -> str:
    """以键排序、紧凑分隔生成规范 JSON（浮点往返保证一致）。"""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_scalar,
    )


def compute_fingerprint(pieces: Mapping[str, Any]) -> str:
    """对一组配置片段计算 SHA-256 稳定指纹（十六进制字符串）。"""
    return hashlib.sha256(canonical_json(pieces).encode("utf-8")).hexdigest()


def build_problem_fingerprint_parts(
    boundary,
    rotor_diameters: np.ndarray,
    *,
    turbines: Optional[list] = None,
    wind_resource=None,
    wake_model=None,
    superposition_method: Optional[str] = None,
    speed_step: Optional[float] = None,
    speed_max: Optional[float] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict:
    """构造场地 / 风机 / 目标函数（AEP 链路）相关的指纹片段。

    优化器会在此基础上再合并算法自身的配置（种群规模、概率系数等）。
    所有参与适应度计算的物理参数都应进入该片段，避免换场地、换风机或
    换尾流模型后误用旧断点。
    """
    rotor_diameters = np.asarray(rotor_diameters, dtype=np.float64)

    parts: dict[str, Any] = {
        "site": {
            "boundary_type": type(boundary).__name__,
            "vertices": np.asarray(boundary.vertices, dtype=np.float64).tolist(),
            "area": float(boundary.area),
        },
        "turbines": {
            "count": int(len(rotor_diameters)),
            "rotor_diameters": rotor_diameters.tolist(),
        },
    }

    if turbines is not None:
        parts["turbines"]["models"] = [
            {
                "name": getattr(t, "name", None),
                "hub_height": float(t.hub_height),
                "rotor_diameter": float(t.rotor_diameter),
                "thrust_coefficient": float(t.thrust_coefficient),
                "power_curve": np.asarray(t.power_curve, dtype=np.float64).tolist(),
            }
            for t in turbines
        ]

    objective: dict[str, Any] = {}
    if wake_model is not None:
        params = {
            k: (v.item() if isinstance(v, np.generic) else v)
            for k, v in vars(wake_model).items()
            if not k.startswith("_")
        }
        objective["wake_model"] = {
            "type": type(wake_model).__name__,
            "params": params,
        }
    if superposition_method is not None:
        objective["wake_superposition"] = str(superposition_method)
    if speed_step is not None:
        objective["speed_step"] = float(speed_step)
    if speed_max is not None:
        objective["speed_max"] = float(speed_max)
    if wind_resource is not None:
        objective["wind_resource"] = {
            "num_sectors": int(wind_resource.num_sectors),
            "sectors": [
                {
                    "direction_center": float(s.direction_center),
                    "direction_width": float(s.direction_width),
                    "frequency": float(s.frequency),
                    "mean_speed": float(s.mean_speed),
                    "weibull_k": float(s.weibull_k),
                    "weibull_c": float(s.weibull_c),
                }
                for s in wind_resource.sectors
            ],
        }
    if extra:
        objective["extra"] = dict(extra)
    parts["objective"] = objective
    return parts


# ---------------------------------------------------------------------------
# RNG 状态序列化（兼容 numpy 各 BitGenerator）
# ---------------------------------------------------------------------------

_NDARRAY_MARKER = "__ndarray__"


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return {
            _NDARRAY_MARKER: obj.tolist(),
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
        }
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _from_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict) and _NDARRAY_MARKER in obj:
        return np.asarray(obj[_NDARRAY_MARKER], dtype=np.dtype(obj["dtype"])).reshape(
            obj["shape"]
        )
    if isinstance(obj, dict):
        return {k: _from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_jsonable(v) for v in obj]
    return obj


def serialize_rng_state(rng: np.random.Generator) -> bytes:
    """将 Generator 的 BitGenerator 状态序列化为 JSON 字节。"""
    return json.dumps(_to_jsonable(rng.bit_generator.state), ensure_ascii=False).encode(
        "utf-8"
    )


def deserialize_rng_state(blob: bytes) -> np.random.Generator:
    """从 JSON 字节重建与保存时同类型 BitGenerator 的 Generator。"""
    state = _from_jsonable(json.loads(blob.decode("utf-8")))
    kind = state.get("bit_generator")
    builders = {
        "PCG64": lambda: np.random.Generator(np.random.PCG64()),
        "PCG64DXSM": lambda: np.random.Generator(np.random.PCG64DXSM()),
        "Philox": lambda: np.random.Generator(np.random.Philox()),
        "SFC64": lambda: np.random.Generator(np.random.SFC64()),
    }
    if kind not in builders:
        raise CheckpointError(f"不支持的随机数生成器类型: {kind!r}")
    rng = builders[kind]()
    rng.bit_generator.state = state
    return rng


# ---------------------------------------------------------------------------
# 原子读写
# ---------------------------------------------------------------------------

def _as_uint8_array(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.uint8)


def _payload_checksum(meta: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    h.update(canonical_json(meta).encode("utf-8"))
    for key in sorted(arrays):
        arr = np.ascontiguousarray(arrays[key])
        h.update(key.encode("utf-8"))
        h.update(f":{arr.dtype}:{arr.shape}:".encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def write_checkpoint(
    path: str,
    meta: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    rng_state: bytes,
) -> None:
    """原子写入检查点。

    流程：写同目录临时文件 → 关闭刷盘 → 重新打开校验校验和 → ``os.replace``
    原子替换 → 目录 fsync。任何一步失败都会删除临时文件，原有断点保持不变。

    Parameters
    ----------
    path : str
        目标检查点路径。
    meta : Mapping[str, Any]
        JSON 可序列化的元信息（指纹、版本、进度等）。
    arrays : Mapping[str, np.ndarray]
        需要保存的全部状态数组，键名不能以 ``__`` 开头。
    rng_state : bytes
        :func:`serialize_rng_state` 产生的随机数生成器状态字节。
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    # 注意：不能用 np.ascontiguousarray，它会把 0 维标量数组提升为 1 维。
    payload: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        arr = np.asarray(value)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        payload[key] = arr
    payload[_RNG_KEY] = _as_uint8_array(bytes(rng_state))

    checksum = _payload_checksum(meta, payload)
    full_meta = dict(meta)
    full_meta["checksum"] = checksum

    entries = {
        _META_KEY: _as_uint8_array(
            canonical_json(full_meta).encode("utf-8")
        ),
        _CHECKSUM_KEY: _as_uint8_array(checksum.encode("utf-8")),
    }
    entries.update(payload)

    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=".ckpt-", suffix=".tmp", dir=parent
    )
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            np.savez(f, **entries)
            f.flush()
            os.fsync(f.fileno())

        # 重新完整读取临时文件并校验，确认落盘内容可用后才替换旧文件。
        loaded_meta, loaded_arrays, loaded_rng = read_checkpoint(tmp_path)
        if loaded_meta.get("checksum") != checksum:
            raise CheckpointError("临时检查点自检失败（校验和不一致）")
        if set(loaded_arrays) != set(arrays) or not loaded_rng:
            raise CheckpointError("临时检查点自检失败（状态数据缺失）")

        os.replace(tmp_path, path)
        tmp_path = None
        try:
            dir_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            # 某些文件系统不支持目录 fsync，替换已完成，可忽略。
            pass
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def read_checkpoint(path: str) -> tuple[dict, dict[str, np.ndarray]]:
    """读取并校验检查点。

    Returns
    -------
    (meta, arrays, rng_state) : tuple[dict, dict, bytes]
        ``meta`` 含指纹、版本、进度等元信息；``arrays`` 含全部状态数组；
        ``rng_state`` 为随机数生成器状态的原始 JSON 字节。

    Raises
    ------
    CheckpointError
        文件缺失以外的任何问题：无法打开、非 npz、键缺失、校验和不符等。
    """
    try:
        with np.load(path, allow_pickle=False) as npz:
            keys = set(npz.files)
            if _META_KEY not in keys or _CHECKSUM_KEY not in keys:
                raise CheckpointError(
                    f"检查点 {path} 结构不完整，可能已损坏或不是本工具的检查点"
                )
            if _RNG_KEY not in keys:
                raise CheckpointError(f"检查点 {path} 缺少随机数状态")
            meta_raw = npz[_META_KEY].tobytes().decode("utf-8")
            checksum = npz[_CHECKSUM_KEY].tobytes().decode("utf-8")
            rng_state = npz[_RNG_KEY].tobytes()
            arrays = {
                k: npz[k]
                for k in keys
                if k not in (_META_KEY, _CHECKSUM_KEY, _RNG_KEY)
            }
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(f"无法读取检查点 {path}: {exc}") from exc

    try:
        meta = json.loads(meta_raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise CheckpointError(f"检查点 {path} 元信息损坏: {exc}") from exc

    payload_for_checksum = dict(arrays)
    payload_for_checksum[_RNG_KEY] = _as_uint8_array(rng_state)
    actual = _payload_checksum(
        {k: v for k, v in meta.items() if k != "checksum"},
        payload_for_checksum,
    )
    if actual != checksum or meta.get("checksum") != checksum:
        raise CheckpointError(
            f"检查点 {path} 校验和不匹配，文件可能已损坏或被截断；"
            "为安全起见拒绝加载（现有文件未被修改）"
        )

    return meta, arrays, rng_state


# ---------------------------------------------------------------------------
# 优化器侧的保存/恢复协调器
# ---------------------------------------------------------------------------

class CheckpointStore:
    """协调 GA/PSO 与检查点文件之间的保存与恢复。"""

    def __init__(
        self,
        settings: CheckpointSettings,
        *,
        algorithm: str,
        payload_version: int,
        fingerprint_parts: Mapping[str, Any],
        total_steps: int,
        shape_info: Mapping[str, Any],
        verbose: bool = True,
    ) -> None:
        self.settings = settings
        self.algorithm = algorithm
        self.payload_version = int(payload_version)
        self.total_steps = int(total_steps)
        self.shape_info = dict(shape_info)
        self.verbose = verbose

        self.fingerprint = compute_fingerprint(fingerprint_parts)
        self.fingerprint_parts = fingerprint_parts

        # 运行谱系：新跑时创建 origin_run_id；恢复时沿用断点中的 origin。
        self.run_id = uuid.uuid4().hex
        self.origin_run_id = self.run_id
        self.lineage = 0
        self.run_mode = "new"
        self.resumed_from: Optional[str] = None
        self.resumed_at_step = 0
        self.completed_steps = 0

    # -- 恢复 -------------------------------------------------------------

    def request_resume(self) -> Optional[dict]:
        """按设置尝试恢复。

        Returns
        -------
        Optional[dict]
            ``{"completed_steps": int, "meta": dict, "arrays": dict,
            "rng_state": bytes}``；
            未请求恢复或文件不存在时返回 None（文件不存在会打印提示后新跑）。
        """
        if not self.settings.resume:
            return None

        path = self.settings.path
        if not os.path.exists(path):
            if self.verbose:
                print(f"[检查点] 未找到 {path}，将从头开始新跑")
            return None

        meta, arrays, rng_state = read_checkpoint(path)

        if meta.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise CheckpointError(
                f"检查点 {path} 的格式版本为 {meta.get('format_version')!r}，"
                f"当前仅支持版本 {CHECKPOINT_FORMAT_VERSION}，拒绝恢复"
            )
        if meta.get("algorithm") != self.algorithm:
            raise CheckpointError(
                f"检查点 {path} 来自算法 {meta.get('algorithm')!r}，"
                f"不能用于 {self.algorithm.upper()} 恢复"
            )
        if meta.get("payload_version") != self.payload_version:
            raise CheckpointError(
                f"检查点 {path} 的 {self.algorithm.upper()} 状态版本为 "
                f"{meta.get('payload_version')!r}，当前为 {self.payload_version}，"
                "状态结构不兼容，拒绝恢复"
            )
        if meta.get("fingerprint") != self.fingerprint:
            raise CheckpointError(
                f"检查点 {path} 的配置指纹与当前场地/风机/目标函数/算法配置不一致，"
                "拒绝恢复以免错误续算\n"
                f"  断点指纹: {meta.get('fingerprint')}\n"
                f"  当前指纹: {self.fingerprint}"
            )
        if dict(meta.get("shape_info", {})) != self.shape_info:
            raise CheckpointError(
                f"检查点 {path} 的状态维度 {meta.get('shape_info')} 与当前 "
                f"{self.shape_info} 不一致，拒绝恢复"
            )

        completed = int(meta.get("completed_steps", 0))
        if completed < 0:
            raise CheckpointError(f"检查点 {path} 进度非法: {completed}")

        self.run_mode = "resumed"
        self.resumed_from = os.path.abspath(path)
        self.resumed_at_step = completed
        self.completed_steps = completed
        self.origin_run_id = str(meta.get("origin_run_id", self.run_id))
        self.lineage = int(meta.get("lineage", 0)) + 1

        if self.verbose:
            status = meta.get("status", "running")
            tail = "（该断点已跑完总迭代数）" if completed >= self.total_steps else ""
            saved_total = meta.get("total_steps")
            if saved_total is not None and int(saved_total) != self.total_steps:
                tail += (
                    f"；注意: 断点原计划总迭代 {saved_total}，本次为 {self.total_steps}"
                )
            print(f"[检查点] 已从 {path} 恢复: 完成 {completed}/{self.total_steps} "
                  f"代/次，状态={status}{tail}")
            print(f"[检查点] 断点来源运行: {self.origin_run_id} (第 {self.lineage} 次续算)")

        return {
            "completed_steps": completed,
            "meta": meta,
            "arrays": arrays,
            "rng_state": rng_state,
        }

    # -- 保存 -------------------------------------------------------------

    def should_save(self, completed_steps: int) -> bool:
        """是否应在完成 ``completed_steps`` 代/次后保存。"""
        if completed_steps <= 0:
            return False
        if completed_steps >= self.total_steps:
            return True
        return completed_steps % self.settings.interval == 0

    def save(
        self,
        completed_steps: int,
        arrays: Mapping[str, np.ndarray],
        rng_state: bytes,
        *,
        status: str = "running",
    ) -> None:
        """原子保存当前进度。"""
        meta = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "algorithm": self.algorithm,
            "payload_version": self.payload_version,
            "fingerprint": self.fingerprint,
            "shape_info": self.shape_info,
            "total_steps": self.total_steps,
            "completed_steps": int(completed_steps),
            "status": status,
            "interval": self.settings.interval,
            "run_id": self.run_id,
            "origin_run_id": self.origin_run_id,
            "lineage": self.lineage,
            "resumed_from": self.resumed_from,
            "numpy_version": np.__version__,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        for key in arrays:
            if key.startswith("__"):
                raise CheckpointError(f"检查点数组键名 {key!r} 与内部保留键冲突")
        write_checkpoint(self.settings.path, meta, arrays, rng_state)
        self.completed_steps = int(completed_steps)

        if self.verbose:
            tag = "完成" if status == "completed" else "已保存"
            print(f"[检查点] {tag}: {self.settings.path} "
                  f"({completed_steps}/{self.total_steps} 代/次)")

    # -- 结果摘要 ---------------------------------------------------------

    def result_info(self) -> dict:
        """供运行摘要（日志 / results.json）使用的运行信息。"""
        return {
            "mode": self.run_mode,
            "checkpoint_path": os.path.abspath(self.settings.path),
            "resumed_from": self.resumed_from,
            "resumed_at_step": self.resumed_at_step if self.run_mode == "resumed" else None,
            "origin_run_id": self.origin_run_id,
            "run_id": self.run_id,
            "lineage": self.lineage,
            "total_steps": self.total_steps,
        }
