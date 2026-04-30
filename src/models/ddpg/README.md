# 🤖 DDPG (Deep Deterministic Policy Gradient) Model

> **Status**: Active Research 🧪
> **Task**: Portfolio Optimization (Continuous Control)
> **Input Constraint**: Variable Universe Size Supported (Asset-Agnostic) 🌍

## 1. 개요 (Overview)

이 모듈은 강화학습(DDPG)을 사용하여 포트폴리오 최적화 문제를 해결합니다.
시장 상태(`State`)를 관찰하여 자산 배분 비중(`Action`)을 결정하며, 샤프 비율(Sharpe Ratio) 또는 수익률을 극대화하는 방향으로 학습합니다.

이 구현체는 **Actor-Critic** 아키텍처를 따르며, 안정적인 학습을 위해 **Target Network**와 **Replay Buffer**를 사용합니다.

---

## 2. 아키텍처 (Architecture)

The public package does not include manuscript figure assets. The executable
architecture is defined in `actor.py`, `critic.py`, and `agent.py`.

### 2.1 Actor (Policy Network)

`src/models/ddpg/actor.py`

Actor는 현재 시장 데이터를 받아 최적의 포트폴리오 비중을 출력합니다.

- **Feature Integration (Concatenated)**:
  - **Input**: `Prices`와 `Macro` 정보를 하나의 벡터로 결합(Concatenate)하여 사용.
  - TGNN과 달리 별도의 인코더를 두지 않고, 전체 상태 공간($State Space$)을 통합적으로 해석하여 최적 행동(Action)을 결정.
- **Global Net**: `Linear(N*TF)` -> `LayerNorm` -> `ReLU` -> `Linear` -> `Softmax`
- **Constraints**:
  - **Normalization**: `Sum(Weights) = 1.0`

### 2.2 Critic (Value Network)

`src/models/ddpg/critic.py`

Critic은 (상태, 행동) 쌍을 받아 해당 포트폴리오의 가치(Q-Value)를 평가합니다.

- **Input**: `State Embedding` + `Action Vector`
- **Output**: `Scalar (Q-Value)`

---

## 3. 입력 및 출력 명세 (I/O Specification)

### Input Tensor

- **Shape**: `[Batch_Size, Num_Stocks, Window_Size, Features]`
  - 예: `[32, 10, 12, 5]` (32개 배치, 10개 종목, 12일 윈도우, 5개 지표)
- **Note**: `Num_Stocks` 차원은 학습 시와 추론 시 **달라도 무방합니다**. (Shared Weights & Deep Sets 구조)

### Output Tensor

- **Weights**: `[Batch_Size, Num_Stocks]`
  - 예: `[32, 10]`
  - 각 배치의 종목별 투자 비중. `Sum(dim=1)`은 항상 1.0에 근사합니다.

---

## 4. 설정 (Configuration)

`config.yaml`의 `model.ddpg` 섹션에서 하이퍼파라미터를 제어합니다.

```yaml
data:
  stock_universes: ["AAPL", "MSFT", ...] # 학습할 유니버스 정의 (Test 시엔 달라도 됨)

model:
  ddpg:
    actor_lr: 0.0001 # Actor Learning Rate
    critic_lr: 0.001 # Critic Learning Rate
    gamma: 0.99 # Discount Factor
    tau: 0.001 # Soft Update Ratio
    batch_size: 64 # Mini-batch Size

training:
  buffer_size: 10000 # Replay Buffer Capacity
```

## 5. 재현성 (Reproducibility)

본 모델은 `research_code` 스킬에 의거하여 결정론적(Deterministic) 결과를 보장하도록 설계되었습니다.
`main.py` 실행 시 설정된 Global Seed가 `torch`, `numpy`, `random`에 모두 적용됩니다.

- **Seed**: `config.yaml` -> `project.seed` (Default: 42)
