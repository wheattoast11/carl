"""CARL Studio credits -- compute billing for paid tier.

Users buy credits. Credits are deducted on job submission.
Refunded if job fails before step 1.

Credit balance stored in:
  1. Supabase user_profiles.credits_remaining (cloud, PAID)
  2. Local SQLite (offline cache for display only)
"""

from carl_studio.credits.balance import CreditBalance, CreditError, CreditNetworkError
from carl_studio.credits.estimate import CreditEstimate, estimate_job_cost

__all__ = [
    "CreditBalance",
    "CreditError",
    "CreditNetworkError",
    "CreditEstimate",
    "estimate_job_cost",
]
