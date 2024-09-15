# Fall Detection Evaluation (new_data)

This test suite evaluates the current enhanced detector against labeled ground truth for the six `new_data` recordings.

## Video Descriptions

| Video                               | Label    | Description                                                           |
|-------------------------------------|----------|-----------------------------------------------------------------------|
| video-02-epic-fall                  | Fall     | Dramatic fall, 2.1m pelvis drop, ends lying down                      |
| video-04-short-fall-standing-up-try | Fall     | Quick fall with attempt to stand, 1.18m drop                          |
| video-05-turn                       | Non-fall | Turning motion (negative head height suggests different origin point) |
| video-06-under-the-mattress         | Non-fall | Going under/onto mattress, controlled descent                         |
| video-09-meditation                 | Non-fall | Controlled lying down, ends in meditation position (crossed legs sitting on floor) |
| video-11-dance                      | Non-fall | Dancing, smallest height drop (0.51m), rhythmic motion                |

## What This Does

- Uses the labels above as ground truth
- Runs the detector on each `.c3d` in `new_data/` with those names
- Produces:
  - `tests/reports/<clip>.txt`: per-file report with expected vs predicted, confidence, activity type, and key metrics
  - `tests/results.txt`: aggregate summary (confusion counts, accuracy) and a simple reward score

Reward (for future tuning):
- +2 correct fall, +1 correct non-fall
- −3 false positive, −2 false negative

## How to Run

```bash
source venv/bin/activate
python tests/evaluate_new_data_v3.py
```

## LSTM Evaluation

Run the LSTM detector on the same `new_data` set:

```bash
source venv/bin/activate
python tests/evaluate_new_data_lstm.py
```

By default, the script loads `models/lstm_fall_v1.pt`. You can override:

```bash
LSTM_MODEL_PATH=models/your_model.pt LSTM_THRESHOLD=0.5 python tests/evaluate_new_data_lstm.py
```

## Recent LSTM Runs

- Threshold sweep (v2 model): `LSTM_THRESHOLD=0.7` with `models/lstm_fall_v2.pt` -> Accuracy 0.667 (TP=2 TN=2 FP=2 FN=0). Results: `tests/results_lstm_v2_thr0.7.txt`.
- Lower augmentation (v2_aug10 model): `--augment-multiplier 10` -> Accuracy 0.833 (TP=2 TN=3 FP=1 FN=0). Results: `tests/results_lstm_v2_aug10.txt`.

## Latest Results

See `results_v3.txt` for rules-based results and `results_lstm_v1.txt` for LSTM results.
