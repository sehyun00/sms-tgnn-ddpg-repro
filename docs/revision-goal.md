# Major Revision Goal and Evidence Gates

## Goal

Produce a defensible major revision in which the manuscript, experimental
evidence, and public reproducibility package agree. Completion means that every
retained quantitative or methodological claim is traceable to a versioned
configuration, machine-readable result, and documented analysis, and that the
complete submission package is ready for journal resubmission.

Acceptance is not a completion criterion because it is outside the authors'
control. A complete, evidence-backed resubmission is the criterion.

## Scope and Safety Boundaries

- Keep editorial correspondence, reviewer reports, response drafts, manuscript
  identifiers, and personal contact details outside this public repository.
- Do not redistribute restricted or derived market datasets. Publish acquisition
  instructions, integrity hashes, schema, and preprocessing provenance instead.
- Do not report paper metrics from the bundled synthetic fixture.
- Do not tune models, hyperparameters, or ensemble weights on the test period.
- Downgrade or remove any claim that cannot be supported by completed evidence.

## Working Deadline

- Journal revision target: 2026-09-30 (the end-of-September deadline).
- Evidence freeze: 2026-09-21.
- Complete manuscript and response draft: 2026-09-25.
- Internal review and corrections: 2026-09-26 through 2026-09-29.
- Submission buffer: 2026-09-30.

## Experiment-versus-Defense Rule

Reviewer suggestions are not an implementation checklist. Handle each point by
choosing one of the following evidence-backed responses:

1. **Run an experiment** when the point challenges a central novelty, a headline
   performance claim, reproducibility, or the validity of the main conclusion.
2. **Clarify or defend** when the existing design is justified by literature,
   scope, or already available evidence and a new experiment would not change
   the conclusion.
3. **Narrow or remove the claim** when the requested validation cannot be
   completed reliably before the deadline.

Core claims about transaction costs, hybrid alpha, DDPG behavior, graph
construction, comparative baselines, and statistical reliability require new
empirical evidence. Broader variable-universe, XAI, and deployable-DSS claims may
be defended only when concrete evidence already exists; otherwise they must be
narrowed or removed.

## Definition of Done

### Gate 1: Experiment provenance

- Add a paper-scale configuration distinct from `config/sample.yaml`.
- Record code commit, data snapshot/hash, selected assets, split dates, package
  versions, hardware, random seeds, and command line for every reported run.
- Generate tables and figures from result files rather than manual transcription.
- Make the risk-free rate, transaction-cost convention, turnover definition, and
  annualization convention explicit and consistent in code and manuscript.

### Gate 2: Core empirical evidence

- Run transaction-cost sensitivity at 0, 5, 10, and 20 bps for every strategy
  and rebalancing frequency.
- Add classical baselines (Markowitz/minimum variance and risk parity) and at
  least one stronger continuous-control RL baseline (SAC or TD3).
- Run at least 10 fixed seeds for stochastic models, preserving paired seed
  comparisons where applicable.
- Report 95% confidence intervals, paired effect sizes, and an appropriate paired
  test or bootstrap comparison. Treat a deterministic benchmark as a fixed
  reference rather than a seed distribution.
- Save DDPG actor/critic losses, reward curves, exploration-noise schedule,
  turnover, concentration, and failure diagnostics.

### Gate 3: Hybrid mechanism evidence

- Specify the full training order and optimization objective for the ensemble.
- Save alpha trajectories and final alpha distributions by seed and rebalancing
  frequency.
- Compare fixed alpha = 0.5 with learned alpha under identical runs.
- Measure correlation and diversification effects between TGNN- and DDPG-implied
  portfolio returns.
- Demonstrate that any claimed hybrid benefit is not merely suppression of a
  failing DDPG branch.

### Gate 4: Graph and robustness evidence

- Reconcile the current static sector graph with the time-varying graph claim.
- Prefer an ablation comparing static sector, rolling-correlation, and combined
  graph construction; otherwise rename and narrow the method claim.
- Confirm training and test asset sets are disjoint.
- Retain the paper's exact ten-stock test universe and add the fixed secondary
  walk-forward window.
- Run N=5, N=10, and N=15 end-to-end functional checks at seed 42, but remove
  performance-generalization and changing-universe claims.

### Gate 5: XAI and DSS evidence

- Add a perturbation or masking-based faithfulness check for attention maps, or
  reduce the explainability claim to attention visualization.
- Reduce the DSS claim to a proposed deployment architecture because the public
  package contains no empirical real-time DSS implementation.

### Gate 6: Manuscript and response package

- Frame the result as horizon-specific unless new evidence proves broader
  superiority.
- Report cross-frequency averages alongside the best quarterly result.
- Add complete algorithm/pseudocode and explain alpha initialization, freezing,
  update rule, loss, and inference behavior.
- Clarify the relation between the implemented GCN plus temporal attention model
  and memory-based Temporal Graph Networks in the cited literature.
- Prepare a clean manuscript, marked-up manuscript, point-by-point response, and
  supplementary experiment package.
- Map every response item to manuscript locations and concrete evidence files.

### Gate 7: Reproducibility release

- Provide exact public commands for preprocessing, training, evaluation,
  aggregation, statistical analysis, and figure generation.
- Publish paper-scale configurations, environment lock information, result
  schemas, and non-sensitive aggregate artifacts needed to validate all tables.
- Run unit tests, smoke tests, configuration validation, leakage checks, and a
  clean-checkout reproduction audit before resubmission.

## Initial Gap Assessment

| Area | Current authoritative state | Required end state |
| --- | --- | --- |
| Public configuration | Synthetic fixture, one epoch, batch size 2, fixed alpha | Paper-scale versioned configuration matching the manuscript |
| Transaction costs | One configured rate; no documented sensitivity result | 0/5/10/20 bps comparison with consistent conventions |
| Baselines | Buy-and-hold, TGNN, DDPG, Hybrid | Classical baselines plus SAC or TD3 |
| Statistical evidence | Five-seed mean and standard deviation | At least 10 seeds, intervals, effect sizes, and paired analysis |
| Alpha evidence | Alpha network exists; final distributions are not reported | Trajectories, per-seed/per-frequency distributions, and ablation |
| Graph construction | Static sector weights | Dynamic/combined ablation or narrowed terminology |
| Robustness | One test universe and one test window | Exact N=10 test universe, one secondary walk-forward, and functional-only N checks with narrowed claims |
| XAI and DSS | Visual/conceptual evidence | Attention faithfulness test; DSS reduced to a proposed architecture |
| Public reproducibility | Code-path smoke package | Traceable paper-scale commands, configs, and aggregate evidence |

## Execution Order

1. **2026-09-07 to 2026-09-09:** Complete the data manifest and correct the
   pipeline defects.
2. **2026-09-10 to 2026-09-13:** Complete unit/integration tests, Colab CUDA
   checks, interruption/resume verification, and deterministic replay checks.
3. **2026-09-14 to 2026-09-18:** Run the primary experiment and secondary
   walk-forward.
4. **2026-09-19 to 2026-09-20:** Run graph, alpha, XAI, and N=5/10/15 checks.
5. **2026-09-21:** Freeze evidence and decide which remaining claims will be
   supported, defended, narrowed, or removed.
6. **2026-09-22 to 2026-09-25:** Regenerate tables and figures and complete the
   point-by-point response evidence draft.
7. **2026-09-26 to 2026-09-29:** Conduct professor/coauthor review, reconcile
   every number against result artifacts, and correct the submission package.
8. **2026-09-30:** Submit the revised manuscript package.

## Locked Decisions

- Rebuild without treating unavailable original checkpoints or tables as an
  authoritative run snapshot; corrected results replace the submitted values.
- Retain the exact ten paper test symbols and deterministically select 55
  sector-balanced training symbols from the archived 2020-12-31 membership.
- Use Google Colab GPU with Drive-backed resume artifacts, ten primary seeds,
  five secondary seeds, and the core-experiment-first schedule.
