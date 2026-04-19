# Kuramoto-R phase thresholds

PhaseAdaptiveCARLReward classifies the instantaneous training phase from the
Kuramoto order parameter R in [0, 1] using two empirical thresholds:

- **Gaseous**: R < 0.30
- **Liquid**:  0.30 <= R < 0.70
- **Crystalline**: R >= 0.70

These are not derived from conservation theory (unlike KAPPA and SIGMA). They
are empirical markers calibrated against coherence trajectories across GRPO
ablations. Cite in methods sections as:

> Phase classification uses Kuramoto-R thresholds R_gaseous = 0.30 and
> R_liquid = 0.70 (Desai, 2026; constants: KURAMOTO_R_GASEOUS_MAX,
> KURAMOTO_R_LIQUID_MAX in carl_core.constants).

## Per-phase reward-weight profiles

`PhaseAdaptiveCARLReward` shifts its composite-reward weighting as the
detected phase changes. The tuples below are `(w_multiscale, w_cloud,
w_discontinuity)` and each sums to 1.0.

| Phase        | R range          | w_mc | w_cloud | w_disc | Rationale                                |
| ------------ | ---------------- | ---- | ------- | ------ | ---------------------------------------- |
| Gaseous      | R < 0.30         | 0.20 | 0.30    | 0.50   | Reward commitment -> phase nucleation.   |
| Liquid       | 0.30 <= R < 0.70 | 0.40 | 0.30    | 0.30   | Balanced -- no single observable rules.  |
| Crystalline  | R >= 0.70        | 0.60 | 0.30    | 0.10   | Reward stability -> preserve attractor.  |

The constants live in `carl_core.constants`:

- `KURAMOTO_R_GASEOUS_MAX`
- `KURAMOTO_R_LIQUID_MAX`
- `PHASE_WEIGHTS_GASEOUS`
- `PHASE_WEIGHTS_LIQUID`
- `PHASE_WEIGHTS_CRYSTALLINE`
