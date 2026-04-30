# 🔀 Hybrid Model (TGNN + DDPG Composition)

> **Status**: Refactored ✅
> **Task**: Portfolio Optimization via Ensemble
> **Architecture**: Composition (DDPGAgent + TGNN)

## 1. 개요 (Overview)

이 모듈은 **DDPG**와 **TGNN**을 조합하여 포트폴리오 최적화를 수행하는 앙상블 모델입니다.

기존 독립 구현 방식에서 **Composition 방식**으로 리팩토링됨:

- ✅ 코드 중복 제거
- ✅ DDPG/TGNN 버그 수정 시 자동 반영
- ✅ 공정한 Ablation Study 가능

---

## 2. 아키텍처 (Architecture)

The public package does not include manuscript figure assets. The executable
architecture is defined in `agent.py`, `mixing.py`, and `checkpoint.py`.

---

## 3. 입력 및 출력 명세 (I/O Specification)

### Input

- `features`: [Batch, N, T, F]
- `adj`: [Batch, N, N] (인접 행렬)

### Output

- `weights`: [Batch, N] (포트폴리오 비중, sum=1)
- `alpha`: [Batch, 1] (TGNN 비중, 0.2~0.8)

---

## 4. 설정 (Configuration)

```yaml
model:
  softmax_temperature: 10.0 # 포트폴리오 분산도 (높을수록 균등 배분)
  hybrid_alpha_min: 0.2 # Alpha 최소값
  hybrid_alpha_max: 0.8 # Alpha 최대값
  # [Note] Alpha Clamping (0.2 ~ 0.8) 이유:
  # - Mode Collapse 방지: 한 모델이 비중을 100% 가져가면 앙상블 효과가 사라짐.
  # - Robustness: TGNN(예측)과 DDPG(최적화)의 장점을 항상 혼합하여 과적합 방지.
  hybrid_alpha_mode: "dynamic" # "fixed" 또는 "dynamic" (리밸런싱 주기 인식)
  hybrid_horizon_dim: 8 # 리밸런싱 주기 임베딩 차원

training:
  buffer_size: 10000
```

### Alpha 모드

| 모드      | 설명                                      |
| --------- | ----------------------------------------- |
| `fixed`   | 학습된 고정 α 사용 (기존 방식)            |
| `dynamic` | 리밸런싱 주기(horizon)에 따라 α 동적 조정 |

### Horizon 매핑

- 0: Monthly
- 1: Quarterly
- 2: Semiannual
- 3: Annual

---

## 5. 학습 메커니즘 (Training)

### update() 메서드

Hybrid 모델은 **3가지 구성요소**를 동시에 학습합니다:

| 구성요소         | Optimizer            | Loss 함수             |
| ---------------- | -------------------- | --------------------- |
| DDPG Actor       | `actor_optimizer`    | -Q(s, a)              |
| DDPG Critic      | `critic_optimizer`   | MSE(Q, target_Q)      |
| **Ensemble Net** | `ensemble_optimizer` | -Q + α_regularization |

### Ensemble Net 학습 목표

```python
# Q-value 최대화 (수익 극대화)
ensemble_loss = -q_value.mean()

# Alpha 정규화 (극단적인 값 방지)
alpha_reg = 0.01 * ((alpha - 0.5) ** 2).mean()

total_loss = ensemble_loss + alpha_reg
```

### 반환 메트릭

```python
{
    "critic_loss": float,      # DDPG Critic 손실
    "actor_loss": float,       # DDPG Actor 손실
    "ensemble_loss": float,    # Ensemble Net 손실 (NEW)
    "alpha_mean": float,       # 평균 Alpha 값 (NEW)
}
```

---

### 4. Data Flow (Split & Merge)

- **Input Handling**: 입력된 데이터(`x`)를 `Prices`($[..., :5]$)와 `Macro`($[..., 5:]$)로 분리.
- **TGNN Path**: 분리된 `Prices`와 `Macro`를 Dual-Path Encoder에 전달.
- **DDPG Path**: 전체 데이터(`x`)를 통합 상태 벡터로 활용.

## 6. 삭제된 파일

| 파일             | 대체                     |
| ---------------- | ------------------------ |
| `actor.py`       | `DDPGAgent.actor` 사용   |
| `critic.py`      | `DDPGAgent.critic` 사용  |
| `encoders.py`    | `TGNN` 내부 encoder 사용 |
| `constraints.py` | 최종 정규화로 대체       |
