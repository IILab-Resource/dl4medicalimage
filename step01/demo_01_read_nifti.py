
from __future__ import annotations
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)


def _find_data_root(start):
    """从 start 向上找第一个含 data/ 的目录,就是项目根;找不到就用 start。"""
    prev, cur = None, _os.path.abspath(start)
    while cur != prev:
        if _os.path.isdir(_os.path.join(cur, "data")):
            return cur
        prev, cur = cur, _os.path.dirname(cur)
    return _os.path.abspath(start)


_ROOT = _find_data_root(_HERE)
_os.chdir(_ROOT)          # 脚本里的 "data/xxx" 相对路径直接可用
# ----------------------------------------------------------------------------

import glob
import os

import numpy as np

from nifti_io import read_nifti, window_ct

ROOT = "data/02_segmentation_3d/Task09_Spleen"


def section(title: str) -> None:
    print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")


def demo_1_目录里有什么() -> None:
    """先看数据在磁盘上长什么样 —— 数据集的目录结构。"""
    section("演示 1 · 数据集目录结构")
    for sub in ["imagesTr", "labelsTr", "imagesTs"]:
        files = sorted(glob.glob(os.path.join(ROOT, sub, "*")))
        print(f"{sub:<10} {len(files):>3} 个文件   例: {os.path.basename(files[0]) if files else '-'}")
    print("\n注意 imagesTs 是测试集,只有图像没有标签 —— 这是评测的标准做法。")
    print("MSD 数据集的目录约定:images<集合><样例>.nii.gz,标签与图像同名同目录并列。")
    print("这套约定叫 MSD 风格,你在后面所有 3D 数据集上都会再遇到它。")


def demo_2_第一个体数据() -> None:
    """打开一个真实的脾脏 CT,打印它的一切。"""
    section("演示 2 · 打开一个体数据")
    vol = read_nifti(os.path.join(ROOT, "imagesTr", "spleen_10.nii.gz"))

    print(f"数组形状     {vol.shape}          ← (Z, Y, X),41 例 CT 都是 512×512×N")
    print(f"体素间距     {vol.spacing} mm      ← Z 方向是层厚,XY 是探测器分辨率")
    print(f"物理尺寸     {vol.physical_extent_mm()} mm  ← 乘出来的,是病人身上真实的大小")
    print(f"各向异性比   {vol.anisotropy_ratio():.2f}x")
    print(f"占用内存     {vol.nbytes_mb():.0f} MB   ← float32 后的体积")

    print("\n★ 停下来想一想:512×512×55 = 1,441 万个数。")
    print("  如果把整卷送进网络,光输入张量就要 1.4 亿参数的空间 —— 这是阶段二的核心矛盾。")


def demo_3_方向与仿射() -> None:
    """affine 矩阵:体素索引 -> 物理坐标 的映射。"""
    section("演示 3 · 仿射矩阵与方向")
    vol = read_nifti(os.path.join(ROOT, "imagesTr", "spleen_10.nii.gz"))
    print("仿射矩阵(4×4),把 (Z,Y,X) 体素坐标映射到毫米坐标:")
    print(np.array2string(vol.affine, precision=2, suppress_small=True))
    print(f"\n原点(0,0,0) 对应的物理坐标: {np.round(vol.origin, 2)}")

    print("\n为什么要关心这个:")
    print("  1) 两台不同机器扫出来的 CT,spacing 不一样,不能直接逐元素相加")
    print("  2) 做重采样/resize 之后,必须同步更新 affine,否则后续所有空间操作都错位")
    print("  3) 配准、检测框坐标回写,全都依赖 affine —— 这是医学影像和自然图像最大的不同之一")


def demo_4_强度与窗宽窗位() -> None:
    """HU 值:CT 的强度不是 0~255,而是有物理含义的毫米数。"""
    section("演示 4 · HU 值与窗宽窗位")
    raw = read_nifti(os.path.join(ROOT, "imagesTr", "spleen_10.nii.gz"))

    print(f"原始强度范围: {raw.data.min():.0f} ~ {raw.data.max():.0f}")
    print("这不是 0~255,是 HU(Hounsfield Unit)。参考值:")
    for name, hu in [("空气", -1000), ("脂肪", -100), ("水", 0), ("脾脏(门脉期)", 110), ("骨", 700)]:
        print(f"    {name:<16} {hu:>6} HU")

    print("\n★ 关键结论:直接把这组数除以 255 归一化,会彻底毁掉组织之间的相对关系。")
    print("  放射科医生看 CT 不是看灰度值,而是选一个「窗」来看。")

    for level, width, name in [(50, 400, "腹部窗"), (-600, 1500, "肺窗")]:
        w = window_ct(raw, level=level, width=width)
        lo, hi = level - width / 2, level + width / 2
        sat = float(((raw.data <= lo) | (raw.data >= hi)).mean())
        print(f"\n{name} (level={level}, width={width}) -> 窗口 [{lo:.0f}, {hi:.0f}]")
        print(f"    归一化后范围 {w.data.min():.2f} ~ {w.data.max():.2f},被压到边界的体素占 {sat*100:.1f}%")


def demo_5_多对比所有各向异性() -> None:
    """同一个数据集里,每一例的层厚都不一样。"""
    section("演示 5 · 41 例 CT 的各向异性分布")
    files = sorted(glob.glob(os.path.join(ROOT, "imagesTr", "*.nii.gz")))[:12]
    print(f"{'文件名':<20}{'形状':>20}{'Z间距':>8}{'XY间距':>8}{'各向异性':>10}")
    ratios = []
    for f in files:
        v = read_nifti(f)
        ratios.append(v.anisotropy_ratio())
        print(f"{os.path.basename(f):<20}{str(v.shape):>20}{v.spacing[0]:>8.2f}"
              f"{v.spacing[1]:>8.2f}{v.anisotropy_ratio():>9.1f}x")
    print(f"\n各向异性比范围: {min(ratios):.1f}x ~ {max(ratios):.1f}x")
    print("★ 每例的 Z 分辨率都不一样。这直接决定了阶段二要不要做重采样、怎么采样。")


def main() -> None:
    print("Day 01 · 影像科的一天 —— 用代码看清一份真实 CT")
    demo_1_目录里有什么()
    demo_2_第一个体数据()
    demo_3_方向与仿射()
    demo_4_强度与窗宽窗位()
    demo_5_多对比所有各向异性()
    print(f"\n{'=' * 66}\n演示结束。课后练习见 exercises.py\n{'=' * 66}")


if __name__ == "__main__":
    main()