-- Curriculum tracking -- carlito academic progress (cloud mirror of local SQLite)
-- Local is source of truth; this table exists for dashboard display + cross-device sync
--
-- Depends on: initial schema (auth.users)
-- PG 17 compatible

CREATE TABLE IF NOT EXISTS public.curriculum_tracks (
    model_id TEXT PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    phase TEXT NOT NULL DEFAULT 'enrolled'
        CHECK (phase IN ('enrolled', 'drilling', 'evaluated', 'graduated', 'deployed', 'ttt_active')),
    version INTEGER NOT NULL DEFAULT 1,
    milestones JSONB NOT NULL DEFAULT '[]',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_curriculum_user ON public.curriculum_tracks (user_id);

ALTER TABLE public.curriculum_tracks ENABLE ROW LEVEL SECURITY;

CREATE POLICY curriculum_select ON public.curriculum_tracks
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY curriculum_insert ON public.curriculum_tracks
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY curriculum_update ON public.curriculum_tracks
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY curriculum_delete ON public.curriculum_tracks
    FOR DELETE USING (auth.uid() = user_id);

-- Auto-update updated_at on modification
CREATE OR REPLACE FUNCTION public.curriculum_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER curriculum_updated BEFORE UPDATE ON public.curriculum_tracks
    FOR EACH ROW EXECUTE FUNCTION public.curriculum_set_updated_at();
