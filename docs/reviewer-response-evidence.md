# Reviewer Response Evidence Map

This public-safe map identifies the planned disposition and generated evidence.
The result-bearing version is generated at
`results/revision/paper-v1/aggregate/reviewer_response_evidence.md` after all
reportable runs finish.

| Reviewer concern | Disposition | Generated evidence |
| --- | --- | --- |
| Transaction costs | Experiment | Four cost paths derived from identical gross weights |
| Weak baselines | Experiment | Equal Weight, Minimum Variance, Equal Risk Contribution, TD3 |
| Negative DDPG results | Experiment and defense | Reward/loss/noise/turnover/concentration histories and paired result table |
| Statistical reliability | Experiment | Moving-block intervals, seed intervals, adjusted p-values |
| Static graph versus dynamic claim | Experiment | Sector, rolling positive correlation, combined graph ablation |
| Alpha optimization and contribution | Method correction and experiment | Frozen source hashes, alpha trajectories, branch-return correlation, fixed-0.5 ablation |
| Unseen or variable N | Claim narrowing | Seed-42 N=5/10/15 functional runs only |
| Attention explainability | Experiment or claim narrowing | Top-attention versus matched-random masking |
| Real-time DSS | Claim narrowing | Proposed architecture only; no implementation claim |
| Broad horizon superiority | Claim narrowing | Frequency sensitivity, not universal superiority |

Do not copy a number into the manuscript unless its row can be traced to a
`SUCCESS` run in `ledger.csv`. Editorial correspondence, review PDFs, manuscript
IDs, and personal information must not be added to this repository.
