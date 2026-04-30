"""
세 모델 (TGNN, DDPG, Hybrid) 결과 통합 비교 그래프 생성
- results/{model}/logs/backtest_metrics.csv 파일에서 자동으로 결과 로드
- 전체 전략 비교 (Benchmark + 12개 전략)
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend (tkinter 불필요)

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import sys

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

# 결과 디렉토리 (프로젝트 루트 기준)
# 스크립트 위치: scripts/
# results 위치: {project_root}/results/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_metrics_from_csv():
    """
    각 모델 폴더에서 backtest_metrics.csv를 읽어와 통합.
    파일 구조: results/{model}/logs/backtest_metrics.csv
    """
    model_dirs = {
        "tgnn": os.path.join(RESULTS_DIR, "tgnn", "logs", "backtest_metrics.csv"),
        "ddpg": os.path.join(RESULTS_DIR, "ddpg", "logs", "backtest_metrics.csv"),
        "hybrid": os.path.join(RESULTS_DIR, "hybrid", "logs", "backtest_metrics.csv"),
    }

    performance_data = {}
    loaded_models = []
    missing_models = []

    for model_name, csv_path in model_dirs.items():
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                loaded_models.append(model_name)

                for _, row in df.iterrows():
                    strategy = row["Strategy"]
                    # Benchmark는 한 번만 추가
                    if strategy == "Benchmark" and "Benchmark" in performance_data:
                        continue

                    performance_data[strategy] = {
                        "CAGR": row.get("CAGR", 0.0),
                        "Sharpe": row.get("Sharpe", 0.0),
                        "MDD": row.get("MDD", 0.0),
                        "Return": row.get("Total_Return", 0.0),
                        "Model": model_name.upper()
                        if strategy != "Benchmark"
                        else "Benchmark",
                    }
            except Exception as e:
                print(f"⚠️ Error loading {csv_path}: {e}")
                missing_models.append(model_name)
        else:
            missing_models.append(model_name)

    return performance_data, loaded_models, missing_models


# 메트릭 로드
print("📊 Loading backtest metrics...")
performance_data, loaded_models, missing_models = load_metrics_from_csv()

if not performance_data:
    print("❌ No metrics files found in results directories.")
    print(
        "   Please run backtests first: python main.py --mode backtest"
    )
    exit(1)
else:
    print(f"✅ Loaded metrics from: {', '.join(loaded_models)}")
    if missing_models:
        print(f"⚠️ Missing metrics for: {', '.join(missing_models)}")
        print("   Run backtest for these models to generate metrics.")

# 전체 전략 (CAGR 순 정렬)
all_strategies = sorted(
    performance_data.keys(), key=lambda x: performance_data[x]["CAGR"], reverse=True
)

print("\n전체 전략 (CAGR 순):")
for s in all_strategies:
    print(
        f"  - {s}: CAGR {performance_data[s]['CAGR']:.2f}%, Sharpe {performance_data[s]['Sharpe']:.2f}"
    )


# 모델별 색상 정의
def get_color(strategy):
    if strategy == "Benchmark":
        return "#888888"
    elif "TGNN" in strategy:
        return "#4ECDC4"
    elif "DDPG" in strategy:
        return "#45B7D1"
    else:  # Hybrid
        return "#FF6B6B"


colors = [get_color(s) for s in all_strategies]

# 1. CAGR 비교 막대그래프 (전체)
fig, ax = plt.subplots(figsize=(16, 8))
x = range(len(all_strategies))
cagr_values = [performance_data[s]["CAGR"] for s in all_strategies]

bars = ax.bar(x, cagr_values, color=colors, edgecolor="white", linewidth=1.5)
ax.set_xticks(x)
ax.set_xticklabels(all_strategies, rotation=45, ha="right", fontsize=10)
ax.set_ylabel("CAGR (%)", fontsize=12)
ax.set_title("All Strategies: CAGR Comparison", fontsize=14, fontweight="bold")
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
if "Benchmark" in performance_data:
    ax.axhline(
        y=performance_data["Benchmark"]["CAGR"],
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Benchmark Level",
    )
ax.grid(axis="y", alpha=0.3)

# 모델별 색상 범례 추가
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [
    Patch(facecolor="#888888", label="Benchmark"),
    Patch(facecolor="#4ECDC4", label="TGNN"),
    Patch(facecolor="#45B7D1", label="DDPG"),
    Patch(facecolor="#FF6B6B", label="Hybrid"),
    Line2D([0], [0], color="red", linestyle="--", alpha=0.7, label="Benchmark Level"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

# 값 표시
for bar, val in zip(bars, cagr_values):
    height = bar.get_height()
    va = "bottom" if height >= 0 else "top"
    offset = 3 if height >= 0 else -3
    ax.annotate(
        f"{val:.1f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, offset),
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=8,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "cagr_comparison_all.png"), dpi=150, bbox_inches="tight"
)
print(f"✅ Saved: {OUTPUT_DIR}/cagr_comparison_all.png")

# 2. Sharpe Ratio 비교 (전체)
# Sharpe 순으로 정렬
sharpe_strategies = sorted(
    performance_data.keys(), key=lambda x: performance_data[x]["Sharpe"], reverse=True
)
fig, ax = plt.subplots(figsize=(16, 8))
sharpe_values = [performance_data[s]["Sharpe"] for s in sharpe_strategies]
colors_sharpe = [get_color(s) for s in sharpe_strategies]

x_sharpe = range(len(sharpe_strategies))
bars = ax.bar(
    x_sharpe, sharpe_values, color=colors_sharpe, edgecolor="white", linewidth=1.5
)
ax.set_xticks(x_sharpe)
ax.set_xticklabels(sharpe_strategies, rotation=45, ha="right", fontsize=10)
ax.set_ylabel("Sharpe Ratio", fontsize=12)
ax.set_title("All Strategies: Sharpe Ratio Comparison", fontsize=14, fontweight="bold")
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
if "Benchmark" in performance_data:
    ax.axhline(
        y=performance_data["Benchmark"]["Sharpe"],
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Benchmark Level",
    )
ax.grid(axis="y", alpha=0.3)

# 모델별 색상 범례 추가
legend_elements = [
    Patch(facecolor="#888888", label="Benchmark"),
    Patch(facecolor="#4ECDC4", label="TGNN"),
    Patch(facecolor="#45B7D1", label="DDPG"),
    Patch(facecolor="#FF6B6B", label="Hybrid"),
    Line2D([0], [0], color="red", linestyle="--", alpha=0.7, label="Benchmark Level"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

for bar, val in zip(bars, sharpe_values):
    height = bar.get_height()
    va = "bottom" if height >= 0 else "top"
    offset = 3 if height >= 0 else -3
    ax.annotate(
        f"{val:.2f}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, offset),
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=8,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "sharpe_comparison_all.png"), dpi=150, bbox_inches="tight"
)
print(f"✅ Saved: {OUTPUT_DIR}/sharpe_comparison_all.png")

# 3. 리스크-수익 산점도 (전체)
fig, ax = plt.subplots(figsize=(12, 10))

for strategy in all_strategies:
    data = performance_data[strategy]
    color = get_color(strategy)
    marker = "s" if strategy == "Benchmark" else "o"
    size = 300 if strategy == "Benchmark" else 150
    ax.scatter(
        data["MDD"],
        data["CAGR"],
        s=size,
        c=color,
        marker=marker,
        edgecolors="white",
        linewidth=2,
        zorder=5,
        alpha=0.8,
    )
    # 주요 전략만 라벨 표시
    if data["CAGR"] > 10 or data["CAGR"] < -2 or strategy == "Benchmark":
        ax.annotate(
            strategy,
            (data["MDD"], data["CAGR"]),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

ax.set_xlabel("Maximum Drawdown (%)", fontsize=12)
ax.set_ylabel("CAGR (%)", fontsize=12)
ax.set_title("Risk-Return Profile: All Strategies", fontsize=14, fontweight="bold")
if "Benchmark" in performance_data:
    ax.axhline(
        y=performance_data["Benchmark"]["CAGR"], color="red", linestyle="--", alpha=0.5
    )
    ax.axvline(
        x=performance_data["Benchmark"]["MDD"], color="red", linestyle="--", alpha=0.5
    )
ax.grid(True, alpha=0.3)

# 범례 생성
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="#888888", label="Benchmark"),
    Patch(facecolor="#4ECDC4", label="TGNN"),
    Patch(facecolor="#45B7D1", label="DDPG"),
    Patch(facecolor="#FF6B6B", label="Hybrid"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "risk_return_scatter_all.png"),
    dpi=150,
    bbox_inches="tight",
)
print(f"✅ Saved: {OUTPUT_DIR}/risk_return_scatter_all.png")

# 4. 종합 비교표 저장 (전체)
summary_df = pd.DataFrame(
    [
        {
            "Strategy": s,
            "Model": performance_data[s].get("Model", "Unknown"),
            "CAGR (%)": performance_data[s]["CAGR"],
            "Sharpe Ratio": performance_data[s]["Sharpe"],
            "MDD (%)": performance_data[s]["MDD"],
            "Total Return (%)": performance_data[s]["Return"],
        }
        for s in all_strategies
    ]
)
summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "all_strategies_comparison.csv"), index=False
)
print(f"✅ Saved: {OUTPUT_DIR}/all_strategies_comparison.csv")

print("\n" + "=" * 50)
print("📊 All comparison charts generated!")
print(f"📁 Output directory: {OUTPUT_DIR}")
