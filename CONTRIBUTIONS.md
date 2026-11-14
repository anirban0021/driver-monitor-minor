# Base vs our contribution

**Group 19**, B.Tech CSE, Swami Vivekananda University
Members: Anirban Sasmal, Agniva Gorai, Abhishek Ghosh, Krishna Dutta, Mir MD Kaif

Base: driver-monitor by Kingsley Aremu (MIT), commit `8c12c0590bd7b381c44e339beb2e36d9801382ce`, tag `base-v0`.
Run `git diff base-v0` to see every change we made.

| Area | In the base | Our change | Files | Who | Result |
|---|---|---|---|---|---|
| Cleanup | Web demo, AI tooling files | Removed unused parts | (deleted) | Anirban Sasmal | Done |
| Baseline measurement | 7 AI-generated clips | Own clips from 5 drivers | data/own_clips | | |
| Learned classifier | Rule-based only | Random Forest on windowed features | | | |
| Low-light handling | None | CLAHE preprocessing | detector.py | | |
| Alert tiers | OK / WARN / ALERT | Normal / Warning / Critical with escalation | state.py, alerts.py | | |
| Evaluation | False alarms per minute on 7 clips | Subject-independent evaluation | | | |