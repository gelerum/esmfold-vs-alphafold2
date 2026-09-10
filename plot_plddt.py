import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROTEINS = ["Alpha-synuclein", "Calmodulin-1", "Ubiquitin"]
AF2_DIR = Path("alphafold2/out")
ESM_DIR = Path("esmfold/out")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

C_AF2 = "#0053D6"
C_ESM = "#FF7D45"

ZONES = [
    (90, 100, "#0053D6", "very high (>90)"),
    (70, 90, "#7FB8E8", "confident (70–90)"),
    (50, 70, "#F5C242", "low (50–70)"),
    (0, 50, "#F55A3B", "very low (<50)"),
]


def load_af2_plddt(name: str) -> np.ndarray:
    """pLDDT для топовой модели ColabFold из rank_001 JSON."""
    pattern = str(AF2_DIR / f"{name}_scores_rank_001_*.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Не найден rank_001 JSON для {name}: {pattern}")
    with open(files[0]) as f:
        data = json.load(f)
    if "plddt" not in data:
        raise KeyError(f"В {files[0]} нет ключа 'plddt'. Ключи: {list(data.keys())}")
    arr = np.asarray(data["plddt"], dtype=float)
    return arr if arr.max() > 1.0 else arr * 100.0


def load_esm_plddt(name: str) -> np.ndarray:
    """
    pLDDT для ESMFold.
    Сначала пробуем JSON (ключ 'plddt'), если его нет — читаем B-factor из PDB.
    Шкала автоматически приводится к 0–100.
    """
    json_path = ESM_DIR / f"{name}.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        if "plddt" in data:
            arr = np.asarray(data["plddt"], dtype=float)
            return arr if arr.max() > 1.0 else arr * 100.0

    pdb_path = ESM_DIR / f"{name}.pdb"
    if not pdb_path.exists():
        raise FileNotFoundError(f"Нет ни JSON, ни PDB для ESMFold: {name}")
    vals = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                vals.append(float(line[60:66]))
    arr = np.asarray(vals, dtype=float)
    return arr if arr.max() > 1.0 else arr * 100.0


def zone_fractions(plddt: np.ndarray) -> dict:
    n = len(plddt)
    return {
        label: 100.0 * np.sum((plddt >= lo) & (plddt < hi)) / n
        for lo, hi, _, label in ZONES
    }


def plot_protein(name: str, af2: np.ndarray, esm: np.ndarray):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [2.2, 1]}
    )

    for lo, hi, color, _ in ZONES:
        ax1.axhspan(lo, hi, color=color, alpha=0.07, zorder=0)

    ax1.plot(af2, color=C_AF2, lw=1.2, label=f"AlphaFold2 (mean = {af2.mean():.1f})")
    ax1.plot(
        esm, color=C_ESM, lw=1.2, alpha=0.85, label=f"ESMFold (mean = {esm.mean():.1f})"
    )

    ax1.set_xlabel("Номер остатка")
    ax1.set_ylabel("pLDDT")
    ax1.set_ylim(0, 100)
    ax1.set_xlim(1, max(len(af2), len(esm)))
    ax1.set_title(f"{name}: профиль pLDDT по длине белка")
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax1.grid(alpha=0.25)

    bins = np.arange(0, 105, 5)
    ax2.hist(
        af2,
        bins=bins,
        color=C_AF2,
        alpha=0.6,
        label="AlphaFold2",
        edgecolor="white",
        linewidth=0.4,
    )
    ax2.hist(
        esm,
        bins=bins,
        color=C_ESM,
        alpha=0.6,
        label="ESMFold",
        edgecolor="white",
        linewidth=0.4,
    )
    ax2.set_xlabel("pLDDT")
    ax2.set_ylabel("Число остатков")
    ax2.set_title(f"{name}: распределение pLDDT")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    out = OUT_DIR / f"{name}_plddt_compare.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


def plot_summary(all_data: dict):
    """Сводный график: по каждому белку средний pLDDT и распределение по зонам."""
    names = list(all_data.keys())
    x = np.arange(len(names))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    af2_means = [all_data[n]["af2"].mean() for n in names]
    esm_means = [all_data[n]["esm"].mean() for n in names]
    ax1.bar(x - w / 2, af2_means, w, label="AlphaFold2", color=C_AF2)
    ax1.bar(x + w / 2, esm_means, w, label="ESMFold", color=C_ESM)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15)
    ax1.set_ylabel("Средний pLDDT")
    ax1.set_ylim(0, 100)
    ax1.axhline(90, ls="--", c="gray", lw=0.8)
    ax1.axhline(70, ls="--", c="gray", lw=0.8)
    ax1.axhline(50, ls="--", c="gray", lw=0.8)
    ax1.set_title("Средний pLDDT по моделям")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    bottom_af2 = np.zeros(len(names))
    bottom_esm = np.zeros(len(names))
    for lo, hi, color, label in ZONES:
        f_af2 = np.array([all_data[n]["af2_zones"][label] for n in names])
        f_esm = np.array([all_data[n]["esm_zones"][label] for n in names])
        ax2.bar(
            x - w / 2,
            f_af2,
            w,
            bottom=bottom_af2,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax2.bar(
            x + w / 2,
            f_esm,
            w,
            bottom=bottom_esm,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            hatch="//",
        )
        bottom_af2 += f_af2
        bottom_esm += f_esm
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15)
    ax2.set_ylabel("% остатков")
    ax2.set_ylim(0, 100)
    ax2.set_title("Доли остатков по зонам pLDDT\n(слева — AF2, справа — ESMFold)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c, _ in ZONES]
    labels = [l for _, _, _, l in ZONES]
    ax2.legend(handles, labels, fontsize=8, loc="lower right")

    plt.tight_layout()
    out = OUT_DIR / "summary_plddt.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


def main():
    all_data = {}
    print(
        f"{'Белок':<18}{'AF2 mean':>10}{'ESM mean':>10}"
        f"{'AF2 >90 %':>11}{'ESM >90 %':>11}"
    )
    print("-" * 62)

    for name in PROTEINS:
        try:
            af2 = load_af2_plddt(name)
            esm = load_esm_plddt(name)
        except Exception as e:
            print(f"[skip] {name}: {e}")
            continue

        all_data[name] = {
            "af2": af2,
            "esm": esm,
            "af2_zones": zone_fractions(af2),
            "esm_zones": zone_fractions(esm),
        }

        plot_protein(name, af2, esm)

        f_af2 = all_data[name]["af2_zones"]["very high (>90)"]
        f_esm = all_data[name]["esm_zones"]["very high (>90)"]
        print(
            f"{name:<18}{af2.mean():>10.1f}{esm.mean():>10.1f}"
            f"{f_af2:>10.1f}%{f_esm:>10.1f}%"
        )

    if all_data:
        plot_summary(all_data)

    summary = {
        name: {
            "af2_mean": float(d["af2"].mean()),
            "esm_mean": float(d["esm"].mean()),
            "af2_zones_%": d["af2_zones"],
            "esm_zones_%": d["esm_zones"],
        }
        for name, d in all_data.items()
    }
    with open(OUT_DIR / "plddt_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[ok] {OUT_DIR / 'plddt_summary.json'}")


if __name__ == "__main__":
    main()
