# View head handoff for the maxwell Work admission binding

Ruling 63 and 66 exclude this binding from carl-14; it starts when the operator asks. Facts the maxwell lane needs:

- Heads ship in carl `experiments/c3-view-breakout-20261010` at 55a9dbd under `src/carl_studio/semantic/heads/`: `embeddinggemma2-view512.json` (default, PCA 512 int8, 512 bytes, man1-256 judge 0.9934) and `embeddinggemma2-view256.json` (KD 256 int8, 256 bytes, judge 0.9203).
- Acceptance contract: `carl_studio.semantic.views.ViewHeadAcceptance`; loader `load_view_head(path)`; refusals: judge failed, candidate under floor, bytes over budget, weights sha mismatch, wrong shape.
- Scoring: `Carrier.scoring(width)` returns the armed int8 view at 256 or 512; `CARL_VIEW_HEAD=prefix` keeps prefixes; `CARL_VIEW_WIDTH` picks the recall start width (default 512).
- Activation record: `activate_view_head(acceptance_path, workspace, effects=(...), db=...)` writes LocalDB config key `view-head:<content_hash(workspace)>` with `{"current": generation, "predecessor": ...}`; `generation` carries width, element_bytes, weights_sha256, space, judge_candidate, judge_margin, rank_arm, source. The declared effect string is `Activate accepted view head <width> for <workspace>`.
- Judge: strix-mind `lib/rank_arm.py` 9a3ac79, `--width <w> --element-bytes 1 --budget-bytes 512`.
- Held-out evidence: `val-anchor-judge.json` (256 head) and `val-anchor-judge-512.json` (512 head), three seeded 256-row draws from the c2 validation split, all passed.
