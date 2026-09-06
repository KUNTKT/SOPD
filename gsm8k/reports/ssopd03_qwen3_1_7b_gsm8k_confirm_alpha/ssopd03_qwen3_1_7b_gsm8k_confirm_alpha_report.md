# Truncation-gate α sweep (confirm, mt=2048)

- direction `layer_14_task_balanced_paired_v_cap`, ‖v‖=57.1007
- n=200, batch=48, clean acc=83.00%
- Gate: **both-finished gain ≥ +2 pp**（非 overall gain）

| alpha | overall_gain | both_n | both_gain | hit_clean | hit_steered | steered_only (from hit) |
|------:|-------------:|-------:|----------:|----------:|------------:|------------------------:|
| 0.5 | 4.00 | 171 | -1.17 | 14.0% | 4.0% | 14 (11) |
| 1.5 | -76.00 | 154 | -88.96 | 14.0% | 12.5% | 2 (2) |
| 2.0 | -83.00 | 125 | -95.20 | 14.0% | 31.0% | 0 (0) |

- Best **overall** α=0.5
- Best **both-finished** α=0.5, gain=-1.17 pp
- Truncation-gate (>=2pp both-finished): **False**
