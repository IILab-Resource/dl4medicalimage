"""第 1 层:影像 I/O。

职责:把磁盘上的医学影像文件变成带有完整空间信息的 numpy 数组。

本层教学重点(阶段二 D8):
  - NIfTI 的 header / affine / 方向码到底在说什么
  - 为什么 resize 之后必须同步 affine
  - CT 的 HU 值为什么要先做窗宽窗位再归一化
"""
from __future__ import annotations

import numpy as np


class Volume:
    """一个三维体数据。

    Attributes:
        data: (Z, Y, X) float32 体素数组
        affine: (4, 4) 仿射矩阵,描述体素索引到物理坐标的映射
        spacing: (Z, Y, X) 体素间距(mm)
        origin: 物理坐标原点(mm)
    """

    __slots__ = ("data", "affine", "spacing", "origin", "path")

    def __init__(self, data, affine, spacing, origin, path=""):
        self.data = data
        self.affine = affine
        self.spacing = tuple(float(s) for s in spacing)
        self.origin = np.asarray(origin, dtype=np.float64)
        self.path = path

    @property
    def shape(self):
        return self.data.shape

    def nbytes_mb(self) -> float:
        return self.data.nbytes / 1e6

    def physical_extent_mm(self):
        """整个体数据在真实世界中的物理尺寸。"""
        return tuple(round(s * n, 2) for s, n in zip(self.spacing, self.data.shape))

    def anisotropy_ratio(self) -> float:
        """各向异性比 = 最大间距 / 最小间距。接近 1 表示近似各向同性。"""
        return max(self.spacing) / min(self.spacing)

    def summary(self) -> str:
        return (
            f"shape={self.shape} spacing=({self.spacing[0]:.2f}, {self.spacing[1]:.2f}, "
            f"{self.spacing[2]:.2f}) mm  物理尺寸={self.physical_extent_mm()} mm  "
            f"各向异性比={self.anisotropy_ratio():.1f}x  内存={self.nbytes_mb():.0f} MB"
        )


def read_nifti(path) -> Volume:
    """读取 NIfTI / NIfTI.gz,返回 Volume。

    注意数组顺序:nibabel 返回 (X, Y, Z),本项目统一转为 (Z, Y, X)。
    """
    import nibabel as nib

    img = nib.load(str(path))
    # header 的 zooms 在 (i, j, k) 顺序上;转成 (Z, Y, X) 需反序
    spacing = tuple(reversed(img.header.get_zooms()[:3]))
    affine = img.affine

    data = np.asarray(img.dataobj, dtype=np.float32)
    data = np.transpose(data, (2, 1, 0))  # (X,Y,Z) -> (Z,Y,X)
    data = np.ascontiguousarray(data)

    # 原点:仿射矩阵把索引 (0,0,0) 映射到物理坐标
    origin = affine[:3, :3] @ np.array([0.0, 0.0, 0.0]) + affine[:3, 3]

    return Volume(data=data, affine=affine, spacing=spacing, origin=origin, path=str(path))


# ---------------------------------------------------------------- 强度处理

def window_ct(volume: Volume, level: float = 50.0, width: float = 400.0) -> Volume:
    """CT 窗宽窗位。

    放射科医生看 CT 不是看原始数值,而是选一个"窗"来看:
      下界 = level - width / 2,上界 = level + width / 2,窗外压成 0 或 1。

    常用窗:
      纵隔窗 level=40  width=400   (看肺门、纵隔淋巴结)
      腹部窗 level=50  width=400   (看肝、脾、胰)
      肺窗   level=-600 width=1500  (看肺纹理)

    这一步之前不要直接做 z-score —— 那会破坏组织间的相对对比关系。
    """
    lo, hi = level - width / 2.0, level + width / 2.0
    out = np.clip((volume.data - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return Volume(out.astype(np.float32), volume.affine, volume.spacing, volume.origin, volume.path)


def zscore(volume: Volume, eps: float = 1e-6) -> Volume:
    """z-score 归一化。用于 MRI 等强度本身无物理含义的模态。"""
    m, s = volume.data.mean(), volume.data.std()
    out = ((volume.data - m) / max(s, eps)).astype(np.float32)
    return Volume(out, volume.affine, volume.spacing, volume.origin, volume.path)


def foreground_bbox(mask: np.ndarray, margin: int = 0):
    """返回掩码中前景的包围盒 (z0, z1, y0, y1, x0, x1)。

    前景感知采样的基础:先定位再采样,而不是盲目随机。

    注意返回顺序是"每个维度先下界后上界",不是按轴堆叠
    (即不是 (z0,y0,x0,z1,y1,x1))。调用方必须按同一顺序解包。
    """
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        return None
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + 1 + margin, np.asarray(mask.shape))
    return tuple(int(v) for pair in zip(lo, hi) for v in pair)