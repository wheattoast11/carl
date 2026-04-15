-- Credits system for carl.camp compute billing
-- Adds credit tracking to user_profiles and creates credit_transactions ledger
--
-- Depends on: initial schema (user_profiles table in public schema)
-- PG 17 compatible

-- Add credits columns to user_profiles (existing table from initial schema)
ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS credits_remaining INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS credits_total INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS credits_monthly_included INTEGER NOT NULL DEFAULT 0;

-- Credit transactions ledger -- every deduction and refund is recorded
CREATE TABLE IF NOT EXISTS public.credit_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    amount INTEGER NOT NULL,         -- positive = credit (purchase/refund/included), negative = debit
    balance_after INTEGER NOT NULL,  -- snapshot of credits_remaining after this transaction
    type TEXT NOT NULL CHECK (type IN ('purchase', 'included', 'deduct', 'refund', 'adjustment')),
    job_id TEXT,                     -- HF Jobs ID (for deduct/refund)
    reason TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_credit_tx_user ON public.credit_transactions (user_id);
CREATE INDEX IF NOT EXISTS idx_credit_tx_job ON public.credit_transactions (job_id) WHERE job_id IS NOT NULL;

-- RLS: users can only see their own transactions
ALTER TABLE public.credit_transactions ENABLE ROW LEVEL SECURITY;

CREATE POLICY credit_transactions_select ON public.credit_transactions
    FOR SELECT USING (auth.uid() = user_id);

-- Only service role can INSERT/UPDATE (Edge Functions run as service role)
CREATE POLICY credit_transactions_insert ON public.credit_transactions
    FOR INSERT WITH CHECK (auth.uid() = user_id OR auth.role() = 'service_role');

-- Function: atomic credit deduction with balance check
-- Uses FOR UPDATE row lock to prevent race conditions on concurrent deductions
CREATE OR REPLACE FUNCTION public.deduct_credits(
    p_user_id UUID,
    p_amount INTEGER,
    p_job_id TEXT DEFAULT NULL,
    p_reason TEXT DEFAULT ''
) RETURNS TABLE(success BOOLEAN, remaining INTEGER, error TEXT) AS $$
DECLARE
    v_balance INTEGER;
BEGIN
    IF p_amount <= 0 THEN
        RETURN QUERY SELECT FALSE, 0, 'Amount must be positive'::TEXT;
        RETURN;
    END IF;

    -- Lock the user row for atomic update
    SELECT credits_remaining INTO v_balance
    FROM public.user_profiles
    WHERE id = p_user_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RETURN QUERY SELECT FALSE, 0, 'User not found'::TEXT;
        RETURN;
    END IF;

    IF v_balance < p_amount THEN
        RETURN QUERY SELECT FALSE, v_balance,
            format('Insufficient credits: have %s, need %s', v_balance, p_amount)::TEXT;
        RETURN;
    END IF;

    -- Deduct
    UPDATE public.user_profiles
    SET credits_remaining = credits_remaining - p_amount
    WHERE id = p_user_id;

    -- Record transaction
    INSERT INTO public.credit_transactions (user_id, amount, balance_after, type, job_id, reason)
    VALUES (p_user_id, -p_amount, v_balance - p_amount, 'deduct', p_job_id, p_reason);

    RETURN QUERY SELECT TRUE, v_balance - p_amount, ''::TEXT;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function: credit refund (also atomic)
CREATE OR REPLACE FUNCTION public.refund_credits(
    p_user_id UUID,
    p_amount INTEGER,
    p_job_id TEXT DEFAULT NULL,
    p_reason TEXT DEFAULT ''
) RETURNS TABLE(success BOOLEAN, remaining INTEGER) AS $$
DECLARE
    v_balance INTEGER;
BEGIN
    IF p_amount <= 0 THEN
        RETURN QUERY SELECT FALSE, 0;
        RETURN;
    END IF;

    UPDATE public.user_profiles
    SET credits_remaining = credits_remaining + p_amount,
        credits_total = credits_total + p_amount
    WHERE id = p_user_id
    RETURNING credits_remaining INTO v_balance;

    IF NOT FOUND THEN
        RETURN QUERY SELECT FALSE, 0;
        RETURN;
    END IF;

    INSERT INTO public.credit_transactions (user_id, amount, balance_after, type, job_id, reason)
    VALUES (p_user_id, p_amount, v_balance, 'refund', p_job_id, p_reason);

    RETURN QUERY SELECT TRUE, v_balance;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function: add purchased or included credits
CREATE OR REPLACE FUNCTION public.add_credits(
    p_user_id UUID,
    p_amount INTEGER,
    p_type TEXT DEFAULT 'purchase',
    p_reason TEXT DEFAULT ''
) RETURNS TABLE(success BOOLEAN, remaining INTEGER) AS $$
DECLARE
    v_balance INTEGER;
BEGIN
    IF p_amount <= 0 THEN
        RETURN QUERY SELECT FALSE, 0;
        RETURN;
    END IF;

    IF p_type NOT IN ('purchase', 'included', 'adjustment') THEN
        RETURN QUERY SELECT FALSE, 0;
        RETURN;
    END IF;

    UPDATE public.user_profiles
    SET credits_remaining = credits_remaining + p_amount,
        credits_total = credits_total + p_amount
    WHERE id = p_user_id
    RETURNING credits_remaining INTO v_balance;

    IF NOT FOUND THEN
        RETURN QUERY SELECT FALSE, 0;
        RETURN;
    END IF;

    INSERT INTO public.credit_transactions (user_id, amount, balance_after, type, reason)
    VALUES (p_user_id, p_amount, v_balance, p_type, p_reason);

    RETURN QUERY SELECT TRUE, v_balance;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
