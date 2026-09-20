# Professor Local GPU Run

This is the one-command path for running the complete paper-revision experiment
on a Windows workstation with an NVIDIA GPU. It does not use Google Colab and
does not upload results anywhere.

## Requirements

- Windows 11 or Windows 10
- Git and Python 3
- An NVIDIA GPU with a current driver
- Sufficient free disk space for prepared data, checkpoints, and results
- A stable internet connection for the initial dependency and market-data download

## First run

```powershell
git clone https://github.com/sehyun00/sms-tgnn-ddpg-repro.git
cd sms-tgnn-ddpg-repro
powershell -ExecutionPolicy Bypass -File .\scripts\run_professor_windows.ps1
```

The PowerShell wrapper creates `.venv`, installs CUDA-enabled PyTorch and the
locked package ranges, verifies CUDA, runs the test suite, prepares the data,
and executes the complete experiment matrix. The full run includes primary
training/backtesting, graph ablation, the secondary fold, N=5/15 checks, and
aggregation.

If the process is interrupted, run the exact same PowerShell command again.
Every experiment command uses `--resume`; successful runs are skipped and
interrupted model checkpoints continue when available.

## Outputs

- Prepared data: `data/revision/paper-v1`
- Resumable runs: `results/revision/paper-v1/runs`
- Aggregate evidence: `results/revision/paper-v1/aggregate`
- Return package: `results/revision/paper-v1-professor-handoff.zip`

The handoff ZIP contains successful run artifacts, aggregate tables, and data
manifests. It excludes raw market data and large `latest.pt` replay/resume
checkpoints. In the default full mode, the ZIP is created only after
`evidence_readiness.json` reports `READY`. Nothing is uploaded automatically.

## Faster primary-only run

Use this only when the immediate goal is to finish the primary experiment
before the extended checks:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_professor_windows.ps1 -CoreOnly
```

Run the normal command afterward to resume the completed primary work and add
the remaining experiments.

## Reproducibility rule

Do not pull new commits or edit tracked files after a long run has started.
Every `run.json` records the exact Git commit and code fingerprint, and the
runner refuses to start from a dirty checkout. Generated data and results are
ignored by Git and do not make the checkout dirty.
