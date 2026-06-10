# PORTING.md — tennis model from research/tennis_research.ipynb

Goal: production pre-match tennis win-probability model feeding scripts/run_cycle.py.
Do NOT run the notebook. Extract code from the cells listed below only. Ignore all
diffusion/VAE/embedding cells (61-84, 92-151) — research only, not ported.

## Cell map (source of truth)
- Cell 0: Sackmann ATP loader (2000-2024, raw.githubusercontent.com/JeffSackmann/tennis_atp).
- Cells 3-4, 10: cleaning + surface_type classification (indoor/outdoor by tourney name).
- Cell 5: overall Elo. K = 250/(n_matches+5)^0.4, Grand Slam x1.1. PORT AS-IS logic,
  restructure as incremental updater (class with update(match) + persisted JSON/SQLite state).
- Cell 11: surface-specific Elo, same K scheme. Same restructure.
- Cells 13, 29-31: H2H tables + dominant_h2h. Verify strictly past-only (sequential build — looks ok).
- Cells 26-28, 37: lefty matchup features, fillna(0.5).
- Cell 14: A/B random flip (KEEP — kills winner-position leakage) + feature assembly.
- Cell 50: yElo merge. Source was a manual xlsx — SKIP yElo in v1 unless an automated
  Tennis Abstract pull is added. Do not block the port on it.
- Cells 33, 16: feature lists. v1 features: elo_diff, surface_elo_diff, rank_diff, age_diff,
  ht_diff, h2h_total, dominant_h2h, lefty_win_pct, player_A_win_pct (trailing), bp_ratio_diff (FIXED, see below), surface one-hots.
- Cells 55-57: Platt calibration + reliability curve. v1 uses isotonic on a validation fold instead.

## MANDATORY FIX — in-match leakage
Cell 14 computes bp_ratio from w_bpSaved/w_bpFaced OF THE SAME MATCH. Post-match stats;
unavailable at bet time; inflates backtest. Replace with trailing rolling mean over each
player's previous N=15 matches (shift(1) before rolling — current match must be excluded).
Audit player_A_win_pct the same way: must be computed from matches strictly before current date.

## Target files
- models/tennis/elo.py        incremental overall+surface Elo, state persisted to db or json
- models/tennis/features.py   pre-match-only feature builder (one row per upcoming match)
- models/tennis/train.py      GradientBoosting or LogisticRegression + isotonic calibration.
                              Time split: train <=2022, calibrate 2023, test 2024-25.
                              Report: log loss, Brier, reliability curve. Benchmark vs closing odds.
- models/tennis/score.py      score(client) -> list[Signal]  (plugs into run_cycle.get_signals)
- models/tennis/backtest.py   join Sackmann results with historical closing odds from
                              tennis-data.co.uk (free CSVs, Pinnacle/B365 closing).
                              Output: P&L of rule-engine bets 2024-25, edge distribution,
                              calibration vs market. GO/NO-GO gate: model must show
                              fee-adjusted edge pockets vs closing price, else singles stay off.

## Constraints
- Training/backtest runs in Colab or locally, NOT on the VPS. VPS only runs score.py with
  a frozen model artifact (joblib) + incremental Elo updates.
- No pandas on the hot path is fine to relax — pandas allowed in models/, keep execution/ lean.
- Never read research/*.ipynb wholesale; grep for the cell content listed above.
