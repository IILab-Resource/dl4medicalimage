""" 学生练习

规则:
  - 每个 TODO 都写了要做什么,以及「怎么知道自己做对了」
  - 做完一个就运行一次,不要攒到最后


运行:
    python exercises.py
"""
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

from nifti_io import read_nifti

ROOT = "data/02_segmentation_3d/Task09_Spleen"
YOUR_NAME = "学号_姓名"


def check(name: str, ok, detail: str = "") -> None:
    """ok 可以是布尔值,也可以是一个 lambda —— 未完成的 TODO 不会让脚本崩掉。"""
    try:
        passed = bool(ok() if callable(ok) else ok)
    except Exception:
        passed = False
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))


def g(value):
    """安全取值:TODO 未完成时返回占位符而不是崩掉。"""
    return value if value is not None else "<未完成>"


# ==================================================================== 练习 1
def 练习1_打印基础信息() -> None:
    """打印 spleen_12 的形状、间距、仿射,并算出它的体素总数。"""
    print("\n" + "-" * 60)
    print("练习 1 · 打印基础信息")
    print("-" * 60)

    vol = read_nifti(f"{ROOT}/imagesTr/spleen_12.nii.gz")
    print(f"形状 {vol.shape}, 间距 {vol.spacing}")

    # TODO-1.1 算出这个体数据一共有多少个体素(提示:np.prod(vol.shape))
    n_voxels = None
    # ------------------------------------------------------------------

    print(f"体素总数 {g(n_voxels)}")
    # ✓ 对不对:答案是 512 × 512 × 168 = 44,040,192
    check("体素总数", lambda: n_voxels == 512 * 512 * 168, f"你算的是 {n_voxels}")


# ==================================================================== 练习 2
def 练习2_物理尺寸与各向异性() -> None:
    """自己实现物理尺寸和各向异性比,不要用现成方法。"""
    print("\n" + "-" * 60)
    print("练习 2 · 物理尺寸与各向异性比(自己实现)")
    print("-" * 60)

    vol = read_nifti(f"{ROOT}/imagesTr/spleen_12.nii.gz")
    shape, spacing = vol.shape, vol.spacing

    # TODO-2.1 计算物理尺寸(毫米):每个维度 = 形状 × 间距
    extent = None
    # ------------------------------------------------------------------

    # TODO-2.2 计算各向异性比 = 最大间距 / 最小间距
    aniso = None
    # ------------------------------------------------------------------

    print(f"物理尺寸 {g(extent)} mm,各向异性比 {g(aniso)}")
    check("物理尺寸", lambda: np.allclose(extent, (252.0, 386.0, 386.0), atol=0.5), f"你算的是 {extent}")
    check("各向异性比", lambda: abs(aniso - 1.99) < 0.01, f"你算的是 {aniso} (注意 XY 间距是 0.7539 不是 0.75)")


# ==================================================================== 练习 3
def 练习3_方向码() -> None:
    """判断这份数据的轴方向是 RAS 还是 LPS,并解释为什么重要。"""
    print("\n" + "-" * 60)
    print("练习 3 · 方向码")
    print("-" * 60)

    import nibabel as nib

    img = nib.load(f"{ROOT}/imagesTr/spleen_12.nii.gz")
    print("nibabel 直接读出的轴方向码:", nib.aff2axcodes(img.affine))

    # TODO-3.1 取出 Z 方向对应的字母,并用中文说明它代表哪个方向
    #         提示:aff2axcodes 返回 3 个字母,第 3 个对应第 3 个体素轴(即 Z)
    #         注意:不同数据集、甚至同一数据集的不同病例,方向码都可能不同
    axis_code = nib.aff2axcodes(img.affine)[2]
    z_meaning = None  # TODO: 填一个字符串,如 "从后往前"
    # ------------------------------------------------------------------

    print(f"Z 轴方向码 {axis_code},表示{ z_meaning }")
    check("Z 轴方向", z_meaning is not None and "后" in z_meaning,
          "记住:A=前 P=后 L=左 R=右 S=上 I=下")

    print("\n★ 为什么重要:如果你把一份 LPS 数据和一份 RAS 数据直接拼成多通道,")
    print("  模型的输入就是把「左肺」和「右肺」当成了同一个东西。")


# ==================================================================== 练习 4
def 练习4_自己实现窗宽窗位() -> None:
    """不调用 window_ct,自己写一遍。这是 D8 的预演。"""
    print("\n" + "-" * 60)
    print("练习 4 · 自己实现窗宽窗位")
    print("-" * 60)

    vol = read_nifti(f"{ROOT}/imagesTr/spleen_12.nii.gz")
    level, width = 50, 400
    lo, hi = level - width / 2, level + width / 2
    print(f"窗口 = [{lo}, {hi}]")

    # TODO-4.1 把 HU 值裁剪到 [lo, hi],再线性映射到 [0, 1]
    #         提示:np.clip 后再 (x - lo) / (hi - lo)
    normalized = None
    # ------------------------------------------------------------------

    print(f"结果范围 {g(normalized.min() if normalized is not None else None)}")
    check("归一化范围", lambda: abs(float(normalized.min())) < 1e-3,
          f"最小值应该接近 0,你的是 {g(round(float(normalized.min()),3) if normalized is not None else None)}")

    # TODO-4.2 算出有多少比例的体素被压到了边界(饱和)
    saturated = None
    # ------------------------------------------------------------------
    print(f"饱和体素占比 {g(round(saturated*100,1) if saturated is not None else None)}%")
    check("饱和比例", lambda: 0.30 < saturated < 0.70, f"腹部窗通常在 40~60%,你算的是 {g(round(saturated*100,1) if saturated is not None else None)}%")


# ==================================================================== 练习 5
def 练习5_数据集体检() -> None:
    """给整个数据集做一次体检,输出一张表。这是每天收尾的标准动作。"""
    print("\n" + "-" * 60)
    print("练习 5 · 数据集体检")
    print("-" * 60)

    files = sorted(glob.glob(f"{ROOT}/imagesTr/*.nii.gz"))
    print(f"共 {len(files)} 例\n")
    print(f"{'文件名':<20}{'形状':>18}{'Z间距':>8}{'各向异性':>10}")

    records = []
    for f in files[:10]:
        v = read_nifti(f)
        records.append(v)
        print(f"{os.path.basename(f):<20}{str(v.shape):>18}{v.spacing[0]:>8.2f}{v.anisotropy_ratio():>9.1f}x")

    # TODO-5.1 统计全部样本中,各向异性比的最大值和最小值
    ratios = [v.anisotropy_ratio() for v in records]
    r_min, r_max = None, None
    # ------------------------------------------------------------------
    print(f"\n各向异性比范围: {g(r_min)} ~ {g(r_max)}")
    check("各向异性范围", lambda: r_min is not None and r_max is not None and r_min > 0, f"{g(r_min)} ~ {g(r_max)}")

    # TODO-5.2 找出 Z 方向体素数最少的那一例(最"薄"的扫描)
    thinnest = None
    # ------------------------------------------------------------------
    print(f"最薄的一例: {g(thinnest)}")
    check("最薄样例", "spleen" in str(thinnest), "")


# ==================================================================== 练习 6
def 练习6_交作业() -> None:
    """把体检结果写进你的数据卡文件。"""
    print("\n" + "-" * 60)
    print("练习 6 · 提交数据卡")
    print("-" * 60)

    out_dir = "docs/data_cards"
    os.makedirs(out_dir, exist_ok=True)

    # TODO-6.1 把你的体检结果写进 docs/data_cards/{你的学号}_spleen.md
    #         至少包含:样本量、形状范围、间距范围、各向异性范围、前景占比
    #         提示:用 open(path, "w", encoding="utf-8").write(text)
    path = None
    # ------------------------------------------------------------------

    print(f"数据卡已写入: {path}" if path else "数据卡未写入 —— 完成 TODO-6.1")
    check("数据卡", os.path.exists(path) if path else False, "")


def main() -> None:
    print(f"Day 01 练习 · {YOUR_NAME}")
    print("提示:每个 TODO 都要先自己想 20 分钟,再翻 solution/")
    for fn in [
        练习1_打印基础信息,
        练习2_物理尺寸与各向异性,
        练习3_方向码,
        练习4_自己实现窗宽窗位,
        练习5_数据集体检,
        练习6_交作业,
    ]:
        fn()
    print("\n" + "=" * 60)
    print("全部 6 个练习跑完。把输出截图连同数据卡一起提交。")


if __name__ == "__main__":
    main()