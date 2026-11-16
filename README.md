# Driver Monitor

> **About this repository.** Minor project by **Group 19** (Anirban Sasmal, Agniva Gorai, Abhishek Ghosh, Krishna Dutta, Mir MD Kaif), B.Tech CSE, Swami Vivekananda University. It is a fork of [driver-monitor](https://github.com/iice257/driver-monitor) by Kingsley Aremu (MIT), starting from commit `8c12c05` (tag `base-v0`). Our changes so far: per-driver face recognition with a driver database, and a Random Forest classifier trained on YawDD (see below). Still planned: CLAHE low-light handling, Normal/Warning/Critical alert tiers, and drowsy recordings from more people. See `CONTRIBUTIONS.md` and `THIRD_PARTY_NOTICES.md`. Not a certified safety system.

## What it detects

| Condition | Signal | Rule (defaults) | Level |
|---|---|---|---|
| Microsleep | EAR + eyeBlink blendshape | Eyes closed ≥ 1.2 s | **ALERT** + siren from the first alert until 2 s after the eyes open |
| Phone in use | Phone detector (EfficientDet / YOLO) | Phone near the face ≥ 1 s, or in view while looking down ≥ 1 s | **ALERT** + alarm |
| Phone call | Hand landmarks | Hand held at the ear ≥ 2 s | **ALERT** + alarm |
| Eyes off road | Head pose + iris gaze | Head turned > 25°, down > 15°, or eyes turned for ≥ 2 s | **ALERT** + alarm |
| Fatigue | PERCLOS | Eyes closed > 15% of the last 60 s | WARN |
| Yawning | jawOpen blendshape | Jaw held open ≥ 1 s; 3 within 5 min is fatigue | WARN |
| Driver not visible | Face presence | No face for ≥ 2 s | WARN |

Every threshold lives in [`config.toml`](config.toml), so you can tune it for your camera position without touching code.

### Alerts you can hear and see

Each alert names the problem, then asks the driver to act: "Yawning detected, please drive carefully.", "Driver distracted, please focus on the road.", "Eyes closed, please stay awake.", "Phone detected, please put your phone away.", "Fatigue detected, please take a break soon.". The voice is a natural neural female voice (Microsoft's Ava), pre-rendered once by `python -m driver_monitor.voice` and committed, so playback is offline and runs on a background thread that never stalls the video. If a phrase is edited without re-rendering, or on a machine without the clips, it falls back to system text-to-speech (pyttsx3, a female voice where installed), then to beeps (`--beep` forces beeps). A phrase is said when its condition starts and repeated at a steady pace while it holds (every 4 s for eyes closed), never every frame.

On screen, the preview shows a banner naming the reason (red for ALERT, amber for caution) and the frame border flashes. The web demo does the same with the browser's `speechSynthesis` and a matching banner and border pulse; the speaker button mutes it.

### How it works

```
                ┌─▶ FaceLandmarker (every frame) ─▶ EAR, blendshapes, head pose, iris gaze ─┐
camera frame ───┤                                                                            ├─▶ temporal state machine ─▶ HUD + alarm + log
                └─▶ background thread: phone detector + HandLandmarker (latest frame only) ─┘    (durations, dropout grace,
                                                                                                   hysteresis, calibration)
```

- **EAR** (Soukupová & Čech, 2016): ratio of eyelid opening to eye width. It falls towards 0 as the eye closes.
- **MAR**: inner-lip opening divided by mouth width, which spikes during a yawn.
- **Blendshapes**: MediaPipe's `eyeBlink*` and `jawOpen` scores are pose-robust, so they back up EAR and drive yawn detection. Talking opens the jaw only briefly, which is why a yawn must be held for at least 1 s.
- **Head pose**: the face's own axes are built from 3D landmarks (across the eyes, forehead to chin), and yaw and pitch are read off its normal, with no ±180° wrap-around. An EMA plus hysteresis keeps the state from flickering at the boundary.
- **Gaze**: the iris position between the eye corners catches eyes-only glances that head pose misses.
- **Phone use**: a COCO object detector finds `cell phone`, and hand landmarks catch a hand held at the ear. Both run on a background thread over the latest frame, so the main loop stays at camera rate. Detections tolerate short dropouts.
- **PERCLOS**: the percentage of time the eyes are closed, an established fatigue metric. It is time-weighted, so frame drops don't skew it.
- **Per-driver calibration**: over the first few seconds of good frames (about 2 s, within an 8 s window) the system learns *your* open-eye EAR and neutral head pose. Thresholds are relative to that baseline rather than a fixed value that only suits some faces. Press `c` to recalibrate.

The decision logic in [`state.py`](driver_monitor/state.py) is pure Python with explicit timestamps, so it is unit tested without a camera.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt            # exact pins; or: pip install -e .  for the driver-monitor command
python -m driver_monitor.download_model
python -m driver_monitor                   # or: driver-monitor
```

Keys: `q` quits, `c` recalibrates.

### Options

```bash
python -m driver_monitor --source 1                    # another camera
python -m driver_monitor --source drive.mp4            # a recorded video
python -m driver_monitor --source drive.mp4 --headless --log events.jsonl
python -m driver_monitor --record out.mp4              # annotated video as H.264 with the spoken alerts mixed in (needs ffmpeg)
python -m driver_monitor --no-alarm --mirror
python -m driver_monitor --beep                        # beeps instead of spoken alerts
python -m driver_monitor --name Anirban                # name given to a new driver when one is enrolled
python -m driver_monitor --db data/drivers.db          # driver database (this is the default)
python -m driver_monitor --no-recognition              # skip face recognition, calibrate from scratch every run
python -m driver_monitor --rf models/drowsiness_rf.pkl # add the trained Random Forest (see below)
python -m driver_monitor --phone yolo                  # YOLO phone detector (pip install ultralytics, AGPL-3.0)
python -m driver_monitor --phone off                   # face only, lightest
python -m driver_monitor --config night.toml           # alternative thresholds
```

### Tuning

The HUD shows each live signal with its threshold marker. Watch where your values sit, then adjust `config.toml`:

- Yawns don't register: lower `yawn_jaw` (watch the **Jaw open** bar while yawning).
- False "eyes off road" with a high or offset camera: raise `pitch_down_limit_deg` / `yaw_limit_deg`, or recalibrate with `c` while looking at the road.
- Glasses cause false eye closure: lower `ear_ratio` or raise `blink_score`.

`--log` writes one JSON line per state change, for example `{"t": 42.1, "level": "ALERT", "reasons": ["Eyes closed 1.3s"]}`.

## Driver recognition (per-driver profiles)

Each driver gets their own saved profile, so a returning driver is recognised straight away and judged against their own normal face, not a generic one.

- **Known driver:** after about 2 s of clear frontal frames the face is looked up in `data/drivers.db`. On a match the HUD shows the name ("Recognized"), the console says "Welcome back", and that driver's saved baseline (open-eye EAR, neutral head pose, gaze) is loaded, so there is no calibration wait.
- **New driver:** the face is scanned for 10 s, then saved with its baseline as a new driver (named `Driver N`, or whatever `--name` says).
- **How a face is matched:** about 90 stable landmarks (jaw outline, brows, eyes, nose, lips) are moved into the face's own frame and divided by the distance between the outer eye corners, so distance from the camera and head tilt matter little. The result is a list of numbers, not a photo. A match needs a distance of at most `match_threshold` (0.05) and a lead over the runner-up of `match_margin` (0.015); both live under `[profiles]` in `config.toml`. Only clear frames count (head within 20°, eyes open, mouth closed).
- **Recalibrate:** press `c`; the new baseline is stored for that driver.

```bash
python -m driver_monitor.profiles list                    # all saved drivers
python -m driver_monitor.profiles rename 1 "Anirban"
python -m driver_monitor.profiles delete 2
python -m driver_monitor.profiles compare                 # distances between every pair of drivers
```

The face signature is biometric data. It stays on your machine in `data/drivers.db`, which is git-ignored: never commit it. This is a shape match for personalising thresholds, not an authentication system.

## Learned classifier (Random Forest)

Besides the rule-based alerts, a Random Forest can be trained on public datasets (YawDD, UTA-RLDD) or your own recordings. It looks at 8-second windows (stride 4 s) of 20 features: eye openness relative to the driver's own baseline, PERCLOS, longest eye closure, mouth opening and yawn-like episodes, head pose spread and gaze spread. The forest is written in plain NumPy (`forest.py`), so there is no scikit-learn dependency and it also runs where compiled packages are blocked.

```bash
python -m driver_monitor.extract labels --root C:\Datasets\YawDD --out data/labels.csv     # then open the CSV and fix wrong labels
python -m driver_monitor.extract frames --root C:\Datasets\YawDD --labels data/labels.csv  # videos -> data/frames/*.csv (resumable)
python -m driver_monitor.train --frames data/frames        # scores on people never seen in training, saves models/drowsiness_rf.pkl
python -m driver_monitor --rf models/drowsiness_rf.pkl
```

Our own clips go in the same way: name them `<n>-Normal.mp4`, `<n>-Yawning.mp4`, `<n>-Drowsy.mp4` (one number per person), run `extract labels` on that folder, change `skip` to `drowsy` on the Drowsy rows, then `extract frames --out data/frames` and train again. Every person needs a Normal clip, because the per-driver baseline is learned from it.

The model can only add a WARN ("Yawn: model 91%", "Fatigue: model 87%"); the rule-based ALERTs stay the safety layer. `train` also scores the app's own thresholds on the same windows, so the two can be compared in the report. Saved models are Python pickles: only load files you trained yourself.

### Results so far (YawDD, 90 people, 1473 windows)

Evaluation holds out whole people, so every score is on drivers the model never saw.

| | Accuracy | Macro-F1 | Yawning recall |
|---|---|---|---|
| Always "alert" | 0.738 | | |
| App's rules | 0.857 | 0.783 | 0.49 |
| Random Forest | 0.907 | 0.872 | 0.74 |

The forest finds more yawns than the fixed thresholds. Some of the remaining misses are label noise: a yawning video also contains seconds without a yawn.

**Drowsy is not evaluated yet.** YawDD has no drowsy videos, and with our own drowsy clips from a single person the held-out score for that class is meaningless (0.00). It needs drowsy recordings from at least 3 to 5 people, or UTA-RLDD.

### Data and credit

YawDD: S. Abtahi, M. Omidyeganeh, S. Shirmohammadi and B. Hariri, "YawDD: A Yawning Detection Dataset", ACM Multimedia Systems Conference (MMSys), 2014 (CC BY 4.0). The dataset is acted, not real driving, and its participants are not to be shown in public material unless they agreed (see the dataset's participant tables). Datasets, videos, frame CSVs and trained models are not in this repository (`data/`, `models/` and `*.mp4` are git-ignored). Each person downloads the data themselves, and anyone who records clips of a person needs that person's consent.

## Test reel

`samples/` holds seven AI-generated scenarios: three normal-driving clips that must stay quiet (calm driving, talking and singing, a mirror and shoulder check) and four that must alert (phone use, repeated yawning, microsleep, looking away). The real phone and hand detectors run on every clip, in `evaluate` and in the reel, and looking down at a phone is not mistaken for closed eyes. All were made with Veo in Gemini from one reference image, [`samples/driver_reference.jpg`](samples/driver_reference.jpg), used as the opening frame, so the same driver appears in every clip. Generate new scenarios from that image to keep the reel consistent.

```bash
python -m driver_monitor.evaluate samples/scenarios.json         # each clip raises what it should, and nothing it shouldn't
python -m driver_monitor.reel samples/scenarios.json         # writes web/demo/{reel,showcase}.mp4, poster.jpg, chapters.json
```

`reel` joins the clips in scenario order as one continuous drive, calibrated once on the opening clip (no recalibration at the cuts) (`reel.mp4`, which the web demo analyses live). Each cut is a short dissolve, and a single uniform cabin ambience runs underneath, looped from one clip with crossfades away from the cuts. It also renders `showcase.mp4`: the same footage with the HUD, alert banners and spoken alerts mixed over the ambience. The videos live only in `web/demo`; `samples/` holds the source clips.

## Models and versions

The desktop app and the web demo run the same model files, so they see the same things.

| Job | File | Where it comes from |
|---|---|---|
| Face landmarks + blendshapes | `face_landmarker.task` (float16) | MediaPipe model storage, `face_landmarker/float16/latest` |
| Phone detector | `efficientdet_lite0.tflite` (float16, COCO `cell phone`, score ≥ 0.35) | `object_detector/efficientdet_lite0/float16/latest` |
| Hand landmarks | `hand_landmarker.task` (float16) | `hand_landmarker/float16/latest` |

Lite0 on both sides: Lite2 is 23 MB, too heavy to ship to a browser tab for a small accuracy gain. The desktop copies live in `models/` (fetched by `python -m driver_monitor.download_model`), the web copies in `web/models/`, and a test checks both lists agree.

Runtimes: Python `mediapipe==1.1.1`, browser `@mediapipe/tasks-vision` **0.10.14**, self-hosted under `web/vendor/` (pinned, Apache-2.0, no CDN at runtime). The page shows real download progress, falls back from GPU to CPU if GPU start-up fails or takes over 5 s, and offers a retry if loading stalls.

## Tests

```bash
pip install -r requirements-dev.txt
ruff check .
pytest -q
```

## Project layout

```
driver_monitor/
  metrics.py        EAR, MAR, head pose, iris gaze from landmarks
  detector.py       MediaPipe FaceLandmarker wrapper (landmarks + blendshapes)
  phone.py          phone detector + hand landmarks on a background thread
  state.py          temporal logic, calibration, alert levels
  profiles.py       per-driver face recognition and the driver database (SQLite)
  features.py       window features shared by training and live use
  extract.py        dataset videos -> per-frame feature CSVs
  train.py          trains and scores the Random Forest
  forest.py         the Random Forest itself, in plain NumPy
  classifier.py     runs the trained model live (--rf)
  hud.py            overlay rendering, alert banner and border flash
  alerts.py         spoken phrase and banner text per reason, and when to repeat them
  alarm.py          spoken alerts on a background thread: voice clips, system TTS, then beeps
  voice.py          renders the alert phrases in a neural voice (driver_monitor/voice, web/voice)
  evaluate.py       scenario checks against the sample clips
  reel.py           builds the demo reel and the annotated showcase video
  __main__.py       CLI
tests/              unit tests for metrics, state logic, alert policy, profiles, the Random Forest pipeline, and a Python-vs-JavaScript parity test (needs node)
data/               driver database and extracted features (git-ignored, never commit)
models/             downloaded MediaPipe models and the trained forest (git-ignored)
web/                static browser demo (MediaPipe in the page), deployed on Vercel
```

## Limitations and roadmap

- A single RGB camera struggles at night. An IR camera with IR illumination is the standard fix.
- Heavy sunglasses hide the eyes, so EAR is unreliable. Head pose and yawning still work.
- The stock COCO phone detector misses phones that are mostly hidden by a hand. A YOLO model fine-tuned on in-cabin images would do better.
- Face recognition compares face shape only. Very different head angles, heavy occlusion or very similar faces can cause a miss or a wrong match; adjust `match_threshold` and `match_margin` in `config.toml`.
- The Random Forest was trained on acted yawning videos (YawDD), and its drowsy class still lacks data from enough people.
- Roadmap: drowsy recordings from more people (or UTA-RLDD), CLAHE for low light, Normal/Warning/Critical tiers, a fine-tuned in-cabin detector (phone, cigarette, drinking), seatbelt check, and a Raspberry Pi / Jetson build.

This is a learning and portfolio project, not a certified safety system.

## License

MIT
