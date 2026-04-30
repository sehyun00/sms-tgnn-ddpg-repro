# 🕸️ TGNN (Temporal Graph Neural Network) Model

> **Status**: Active Research 🧪
> **Task**: Multi-Task Momentum Prediction
> **Key Feature**: Graph-based Spatial Correlation + Temporal Attention

## 1. 개요 (Overview)

**TGNN**은 시계열 데이터와 종목 간 관계(Correlation Graph)를 동시에 활용하여 주가의 미래 모멘텀을 예측하는 모델입니다.
단순히 개별 종목의 차트만 보는 것이 아니라, "관련된 종목들이 어떻게 움직였는가"를 GCN(Graph Convolutional Network)으로 분석합니다.

---

## 2. 아키텍처 (Architecture)

The public package does not include manuscript figure assets. The executable
architecture is defined in `model.py`, `loss.py`, and shared layers.

### 2.1 Main Pipeline (`model.TGNN`)

데이터는 다음 4단계 과정을 거쳐 처리됩니다.

1. **Input Projection**:
   - Raw Features `(F)` -> Hidden Dim `(128)`
   - `Linear` -> `LayerNorm`

2. **Spatial Learning (GCN Layer)**:
   - 종목 간 메시지 패싱(Message Passing).
   - `GraphConvLayer` x 2 (Residual Connection 포함)
   - `H_new = GCN(H, Adj) + H`

3. **Temporal Learning (Attention)**:
   - 과거 `T` 시점의 정보를 집약.
   - `TemporalAttention`: 시간 축에 대한 가중치 계산.
   - Output: `Node Embeddings` (각 종목의 최종 특징 벡터)

4. **Multi-Task Prediction Heads**:
   - 4개의 독립적인 예측 헤드가 각각 다른 기간의 수익률을 예측합니다.
   - `Momentum1M` (1개월 후)
   - `Momentum3M` (3개월 후)
   - `Momentum6M` (6개월 후)
   - `Momentum12M` (12개월 후)

### 2.2 Loss Function (`loss.combined_loss`)

단순한 오차 최소화(MSE)뿐만 아니라, **종목 간의 상대적 순위(Ranking)**를 맞추는 것이 중요합니다.

- **Formula**: `Loss = α * MSE + β * RankingLoss`
  - `α (Alpha)`: 0.7 (MSE 가중치)
  - `β (Beta)`: 0.3 (Ranking 가중치)
- **Pairwise Ranking Loss**:
  - "A가 B보다 수익률이 높다면, 예측값도 A > B여야 한다"를 수식화.
  - `d_pred < margin` 일 때 페널티 부과.

---

## 3. 입력 및 출력 명세 (I/O Specification)

### Input Tensor

- **Features (`x`)**: `[Batch, N, T, F]`
  - 예: `[32, 55, 12, 5]`
- **Adjacency (`adj`)**: `[Batch, N, N]` (Correlation Matrix)
- **Target Type**: 예측할 기간 (`"Momentum1M"` 등)

### Output Tensor

- **Predictions**: `[Batch, N]` (Scalar Scores)
- **Embeddings**: `[Batch, N, Hidden]` (For Hybrid Model)

---

## 4. 설정 (Configuration)

`config.yaml`의 `model.tgnn` 섹션에서 제어됩니다.

```yaml
model:
  tgnn:
    hidden_dim: 64 # 은닉층 크기
    num_heads: 4 # Attention Head 개수
    dropout: 0.1 # Dropout 비율
    layer_num: 2 # GCN 레이어 수 (Fixed: 2, models.py에서 하드코딩됨)
```

## 5. 재현성 및 특이사항

- **Permutation Invariant**: GNN 특성상 입력 종목 수(N)가 바뀌어도 동작합니다. (Train 55 -> Test 10 가능)
- **Shared Weights**: 모든 종목이 같은 파라미터를 공유하므로, 과적합에 상대적으로 강합니다.
