# Credits

Compute billing and credit management for CARL training jobs.

## Key Abstractions

- `CreditBalance` -- Pydantic model for a user's credit state: total, remaining, used, included_monthly. Provides `sufficient` and `can_afford(cost)` checks.
- `CreditError` -- Exception for credit operation failures (offline, insufficient balance).
- `CreditEstimate` -- Cost projection for a training job given hardware flavor, step count, and method.
- `estimate_job_cost()` -- Computes estimated credit cost from config. 1 credit = 1 A100-minute at standard rate; hardware multipliers scale from 0.3x (L4) to 2.5x (H200).
- `best_bundle()` -- Recommends the cheapest Stripe bundle that covers a given credit need.
- `CREDIT_RATES` / `BUNDLES` / `METHOD_STEP_SECONDS` -- Rate tables for hardware, bundles, and per-step timing.

## Architecture

All network calls use stdlib `urllib` (zero external deps). Balance is fetched from Supabase edge functions via CampProfile for PAID tier, with local SQLite as an offline display cache. Credits are deducted on job submission and refunded if a job fails before step 1.

## CLI

Registered as `carl credits` (or `carl camp credits`):

```
carl credits balance          # show current balance
carl credits estimate ...     # estimate job cost
carl credits buy              # open Stripe checkout
```
