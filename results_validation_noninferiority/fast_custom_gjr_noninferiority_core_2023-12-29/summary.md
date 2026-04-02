# Fast Custom GJR Non-Inferiority Check

Rule:
- custom may not exceed arch by more than one breach-equivalent on breach rate or CVaR breach rate
- custom may not exceed arch by more than 1 breach count
- custom may not have worse fit stability than arch

## Basket Summary

### india_primary

- scenarios: 2
- pass rate: 1.0
- mean breach-rate diff: 0.0
- mean breach-count diff: 0.0
- mean cvar-breach-rate diff: 0.0

### us_stress

- scenarios: 2
- pass rate: 1.0
- mean breach-rate diff: -0.004166666666666667
- mean breach-count diff: -0.5
- mean cvar-breach-rate diff: 0.0

## Scenario Results

- india_primary | eval=120 | 2021-06-28 to 2023-12-29 | arch_breaches=1.0 | custom_breaches=1.0 | noninferior=1
- india_primary | eval=252 | 2020-12-01 to 2023-12-29 | arch_breaches=2.0 | custom_breaches=2.0 | noninferior=1
- us_stress | eval=120 | 2021-07-15 to 2023-12-29 | arch_breaches=1.0 | custom_breaches=0.0 | noninferior=1
- us_stress | eval=252 | 2021-01-05 to 2023-12-29 | arch_breaches=5.0 | custom_breaches=5.0 | noninferior=1
