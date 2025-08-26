# Safety, Provenance, and Reporting

## Provenance Scan

Verify transplanted directories for required artifacts and license hints.

```bash
llm-ripper provenance --scan ./transplanted --fail-on-violation --json
```

Checks: transplant metadata, presence of model/tokenizer files, license files, and SHA-256 hashes of key files.

## Stress & OOD Drift

Compare distributions between a candidate model and a baseline using entropy histograms (PSI) and mean softmax KL.

```bash
llm-ripper stress --model ./transplanted --baseline gpt2 --out ./reports
```

Output: `reports/stress_drift.json` with `psi_entropy`, `kl_model_vs_baseline`, `kl_baseline_vs_model`.

## Reporting

Aggregate results (validation, UQ) into JSON/MD/PDF.

```bash
llm-ripper report --ideal --out ./reports --from ./runs/<stamp>
```

Outputs:
- `reports/report.json`
- `reports/report.md`
- `reports/report.pdf`
