# Backtest Module (백테스팅 엔진)

`src/backtest`는 학습된 모델의 성능을 과거 데이터를 통해 검증하는 모듈입니다.
거래 비용(수수료, 슬리피지)을 고려한 현실적인 수익률 시뮬레이션을 수행합니다.

## 📦 모듈 구조

| 모듈 | 역할 |
|---|---|
| `engine.py` | 백테스트 시뮬레이션의 핵심 엔진. 날짜별로 루프를 돌며 자산(Equity) 변화를 추적합니다. |
| `strategy.py` | 모델 출력(Score/Action)을 포트폴리오 비중(Weights)으로 변환하는 전략 핸들러. |
| `metrics.py` | CAGR, MDD, Sharpe Ratio 등 금융 성과 지표 계산. |
| `visualization.py` | 백테스트 결과 시각화 및 관리. 그래프(`plots/`)와 거래 로그(`logs/`)를 분리하여 저장합니다. |

## 🏗️ 백테스트 프로세스 & Top-K 전략

백테스터는 `StrategyHandler`를 통해 모델의 예측값을 실제 투자 비중으로 변환합니다.

### 1. Buy & Hold (Benchmark)
*   모든 종목에 동일 비중($1/N$)으로 투자하고, 중간 리밸런싱 없이 유지합니다.

### 2. Model Strategy (Dynamic Top-K)
TGNN과 같은 지도 학습 모델의 예측값(Score)을 기반으로 포트폴리오를 구성합니다.

1.  **Scoring**: 모델이 각 종목의 상승 확률(또는 모멘텀) 예측.
2.  **Selection**: 상위 $K$개 종목 선정 (Top 30%, 최소 1개 ~ 최대 10개).
3.  **Weighting**: 선정된 종목들에 대해 **Softmax** 비중 할당.
    *   예측 확신도(Score)가 높을수록 더 많은 비중을 투자.
    *   $w_i = \frac{e^{z_i}}{\sum e^{z_j}}$ (여기서 $z$는 Z-Score input)

### 3. RL Strategy (DDPG)
*   **Action**: Actor 네트워크가 직접 포트폴리오 비중($\sum w_i = 1$)을 출력합니다.
*   **Rebalancing**: `config.yaml` 또는 인자에 따라 월별(Monthly), 분기별(Quarterly) 등으로 리밸런싱 주기를 조절할 수 있습니다.

## 🚀 사용법 (Usage)

```python
from src.backtest import Backtester

# 1. 백테스터 초기화
backtester = Backtester(config, model, dataset)

# 2. 벤치마크 실행
bm_results = backtester.run_strategy("buy_and_hold")

# 3. 모델 전략 실행 (예: Momentum 1개월 예측 헤드 사용)
model_results = backtester.run_strategy("model", target_head="Momentum1M")

# 4. 결과 시각화
from src.backtest.visualization import plot_comparison
plot_comparison({"Benchmark": bm_results, "Model": model_results}, "result.png")
```

## 📊 주요 평가지표

*   **Total Return**: 누적 수익률
*   **CAGR**: 연평균 성장률
*   **MDD (Maximum Drawdown)**: 최대 낙폭 (위험 지표)
*   **Sharpe Ratio**: 위험 대비 수익률 (무위험 수익률 고려)
