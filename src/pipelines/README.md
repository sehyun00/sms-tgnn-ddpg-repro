# Pipelines Module (실행 파이프라인)

`src/pipelines`는 `main.py`에서 호출되는 최상위 비즈니스 로직을 담고 있습니다. 기존의 거대한 `main.py`를 역할별로 분리하여 유지보수성을 높였습니다.

## 📦 모듈 구조

| 모듈 | 역할 |
|---|---|
| `train_pipeline.py` | 모델 학습 전체 과정 (데이터셋 로드 $\rightarrow$ 모델 생성 $\rightarrow$ 학습 루프) |
| `backtest_pipeline.py` | 백테스트, 전이 학습, 결과 저장 및 시각화 프로세스 |

## 🏗️ 상세 워크플로우

### 1. Train Pipeline (`run_train`)
학습 모드(`--mode train`) 실행 시 호출됩니다.

1.  **데이터 준비**: `config.yaml`의 종목 유니버스를 기반으로 `train_data.csv` 로드 및 데이터셋 생성.
2.  **모델 초기화**: TGNN, DDPG, Hybrid 등 설정된 모델 생성.
3.  **학습 실행**: `Trainer`를 통해 에폭 반복 학습 수행.
4.  **정보 저장**: 학습 완료 후 `trained_universe.json` (학습에 사용된 종목 리스트)를 저장하여 이후 백테스트 시 참조하도록 함.

### 2. Backtest Pipeline (`run_backtest`)
검증 모드(`--mode backtest`) 실행 시 호출됩니다.

1.  **테스트 데이터 로드**: `test_data.csv` 로드.
2.  **종목 유니버스 확인 및 전이 학습 (Transfer Learning)**:
    *   사용자가 `config.yaml`에 지정한 **Test Universe(예: 10개)**와 학습된 **Train Universe(예: 55개)**를 비교합니다.
    *   **Mismatch 발생 시**: `strict=False`로 모델 가중치를 부분 로드하고, `Trainer.finetune()`을 자동 실행하여 소수 종목에 모델을 적응시킵니다.
3.  **전략 실행 (Strategy Execution)**:
    *   **Benchmark**: Buy & Hold 전략.
    *   **Model Strategy**:
        *   **TGNN (Horizon Matching)**: 투자 주기와 예측 주기를 일치시키는 전략 (예: 월간 투자 시 1개월 예측값 `Momentum1M` 사용).
        *   **DDPG**: 포트폴리오 최적화 가중치를 사용하며, 다양한 리밸런싱 주기 테스트.
4.  **결과 저장**:
    *   **로그**: `results/{model}/logs/` (거래 내역)
    *   **모델**: `results/{model}/checkpoints/` (가중치 파일)
    *   **시각화**: `results/{model}/plots/` (수익률 비교 그래프)

## 🚀 사용법 (Usage)

이 모듈들은 직접 실행되기보다는 `main.py`를 통해 호출되는 것을 권장합니다.

```python
# main.py 내부 예시
from src.pipelines.train_pipeline import run_train
from src.pipelines.backtest_pipeline import run_backtest

if mode == "train":
    run_train(config)
elif mode == "backtest":
    run_backtest(config)
```
