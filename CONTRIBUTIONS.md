# Base vs our contribution

**Group 19**, B.Tech CSE, Swami Vivekananda University
Members: Anirban Sasmal, Agniva Gorai, Abhishek Ghosh, Krishna Dutta, Mir MD Kaif

Base: driver-monitor by Kingsley Aremu (MIT), commit `8c12c0590bd7b381c44e339beb2e36d9801382ce`, tag `base-v0`.
Run `git diff base-v0` to see every change we made.

| Area | In the base | Our change | Files | Who | Result |
|---|---|---|---|---|---|
| Cleanup | Web demo, AI tooling files | Removed unused parts | (deleted) | Anirban Sasmal, Mir MD Kaif, Abhishek Ghosh | Done |
| Dataset pipeline | None | Labels and per-frame feature CSVs from YawDD and our own clips | `extract.py` | Anirban Sasmal | Done |
| Window features | None | 20 features per 8 s window, shared by training and live use | `features.py` | Anirban Sasmal | Done |
| Learned classifier | Rule-based only | Random Forest in plain NumPy, person-wise cross-validation, compared with the rules | `forest.py`, `train.py` | Anirban Sasmal | Done: 0.910 accuracy, macro-F1 0.896 (93 people, 1548 windows) |
| Live use of the classifier | None | Adds a WARN with a reason, never lowers an alert (`--rf`) | `classifier.py`, `__main__.py` | Anirban Sasmal | Done |
| Per-driver recognition | Per-session calibration only | SQLite profile store, saved baselines, SFace face embeddings, enrolment and recognition flow, HUD badge | `profiles.py`, `embedder.py`, `hud.py`, `state.py`, `__main__.py` | Anirban Sasmal, Agniva Gorai, Krishna Dutta, Sohan, Likhan, Souvik | Done |
| Recognition check | None | Same-person vs different-person distances on our own clips, threshold suggestion | `idcheck.py` | Anirban Sasmal, Agniva, Krishna, Sohan, Likhan, Souvik | Done |
| Own recordings | 7 AI-generated clips | Normal / Yawning / Drowsy clips of 3 people (for the drowsy class and recognition checks) | not in the repository | Anirban Sasmal, Agniva Gorai, Krishna Dutta | Done (small, acted) |
| Tests | 68 unit tests | Tests for profiles, embeddings and the ML pipeline | `tests/test_profiles.py`, `tests/test_embedder.py`, `tests/test_ml.py` | Anirban Sasmal | 118 passed, 1 skipped |
| Low-light handling | None | CLAHE preprocessing | `detector.py` | | Planned |
| Alert tiers | OK / WARN / ALERT | Normal / Warning / Critical with escalation | `state.py`, `alerts.py` | | Planned |
| More drowsy data | None | UTA-RLDD or recordings of more people | | | Planned |