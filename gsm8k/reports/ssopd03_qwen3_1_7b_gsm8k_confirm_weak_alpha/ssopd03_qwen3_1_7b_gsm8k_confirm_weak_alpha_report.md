# Truncation-gate α sweep (confirm, mt=2048)

- direction `layer_14_task_balanced_paired_v_cap`, ‖v‖=57.1007
- n=200, batch=48, clean acc=83.00%
- Gate: **both-finished gain ≥ +2 pp**（非 overall gain）

| alpha | overall_gain | both_n | both_gain | hit_clean | hit_steered | steered_only (from hit) |
|------:|-------------:|-------:|----------:|----------:|------------:|------------------------:|
| 0.15 | 1.50 | 165 | 1.82 | 14.0% | 12.0% | 11 (8) |
| 0.25 | 2.00 | 169 | 0.59 | 14.0% | 10.0% | 10 (7) |
| 0.35 | 3.50 | 166 | 0.60 | 14.0% | 8.0% | 14 (12) |

- Best **overall** α=0.35
- Best **both-finished** α=0.15, gain=1.82 pp
- Truncation-gate (>=2pp both-finished): **False**
