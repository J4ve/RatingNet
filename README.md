# Chess Rating Estimation from Moves and Clock Times Using a CNN-LSTM

**Authors**: Michael Omori (ORCID: [0009-0000-4632-9272](https://orcid.org/0009-0000-4632-9272)), Prasad Tadepalli (ORCID: [0000-0003-2736-3912](https://orcid.org/0000-0003-2736-3912))  
Oregon State University, Corvallis OR, USA  
Email: [omorim@oregonstate.edu](mailto:omorim@oregonstate.edu), [prasad.tadepalli@oregonstate.edu](mailto:prasad.tadepalli@oregonstate.edu)

---

The paper can be found here: https://arxiv.org/abs/2409.11506

## Abstract

Current chess rating systems update ratings incrementally and may not always accurately reflect a player's true strength at all times, especially for rapidly improving players or very rusty players. To overcome this, we explore a method to estimate player ratings directly from game moves and clock times. We compiled a benchmark dataset from Lichess with over one million games, encompassing various time controls and including move sequences and clock times. Our model architecture comprises a CNN to learn positional features, which are then integrated with clock-time data into a Bidirectional LSTM, predicting player ratings after each move. The model achieved an MAE of 182 rating points on the test data. Additionally, we applied our model to the 2024 IEEE Big Data Cup Chess Puzzle Difficulty Competition dataset, predicted puzzle ratings and achieved competitive results. This model is the first to use no hand-crafted features to estimate chess ratings and also the first to output a rating prediction after each move. Our method highlights the potential of using move-based rating estimation for enhancing rating systems and potentially other applications such as cheating detection.

## Installation and Setup
```bash
conda create --name rating_env python=3.8
conda activate rating_env
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install scikit-learn==1.3.2
conda install tensorboard
pip install chess==1.10.0
```

You can download games from https://database.lichess.org/ in the .pgn.zst format.
Put them in data/game_zips.
Next run
```bash
sh format.sh
```
You can change the year and months in that file. This will run src/format_data.py which converts the games into a format suitable for the cnn input.
The converted game data will be saved in data/processed_games.
Download model_55.pth and put it in models/cnn_bilstm_clocks_all.
We also provide a direct download from google drive at this link: https://drive.google.com/drive/folders/164qXisHsNAKSM6R7ZMeTeJjPpnZ7s5Rt?usp=sharing
You can run the code with
```bash
python src/chess_rating_net.py
```
python src/game_analysis.py will output analyzed games with the rating predictions.

## Prototype extension (Thesis 2)

This fork is the code base for a BS Computer Science thesis (Camarines Sur
Polytechnic Colleges) that extends the baseline with attention and anomaly
detection. Latest build as of 2026-08-14:

- **Parametrized trainer** (`src/chess_rating_net.py`) — argparse + YAML config
  (`--data_dir --experiment --train --epochs --lr --batch_size --model_dir
  --resume`, `train=False` by default), periodic checkpointing, and resume support.
- **Attention module** (`src/attention.py`) — full query-key additive
  (Bahdanau) attention with a causal-cumulative forward path (ply *t* attends
  only to plies 1..*t*), wired into the rating head; per-move output preserved,
  no lookahead.
- **Anomaly-detection module** (`src/anomaly.py`) — attention-weighted per-move
  deviation, with a defined `R_baseline` and an Elo-scale unit guard.
- **Dense supervision** (`--dense_supervision`) — optional all-ply supervision.
- **FastAPI service** (`src/api.py`) — serves per-move ratings, attention
  weights, and Elo-scale deviations.

### Status

Done:
- Baseline reproduced from scratch (train MAE ~181 ≈ the paper's 182).

In progress / pending:
- [ ] Attention vs baseline ablation on the 170k-game subset (training on the
      school HPC; results pending).
- [ ] Full 1.2M-game corpus run.
- [ ] Anomaly-validation corpus (bot-vs-bot + real-world closed accounts).

The manuscript and experimental plan live in the companion repo
[J4ve/cs_thesis](https://github.com/J4ve/cs_thesis).

### Frozen weights

The released baseline checkpoint (`model_55.pth`, Option C "both" weight
strategy) is required for inference. It is **not committed** to git (see
`.gitignore`).

- Expected location: `models/model_55.pth`
- Download: [Google Drive folder](https://drive.google.com/drive/folders/164qXisHsNAKSM6R7ZMeTeJjPpnZ7s5Rt)

### Quick start (inference API)

PyTorch needs Python 3.12 or 3.13. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy the frozen weights into place
cp /path/to/model_55.pth models/model_55.pth

# Start the FastAPI service
python src/api.py
```

The service binds to `http://0.0.0.0:8000` by default. Set `PORT` to override.

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict/pgn \
  -H "Content-Type: application/json" \
  -d '{"pgn": "[Event \"Demo\"]\n[WhiteElo \"1500\"]\n[BlackElo \"1500\"]\n[TimeControl \"300+0\"]\n\n1. e4 {[%clk 0:05:00]} e5 {[%clk 0:05:00]} *"}'

curl -X POST http://localhost:8000/predict/upload -F "file=@game.pgn"
```

### Training path (optional)

```bash
python src/chess_rating_net.py --train --data_dir data/processed_games \
  --experiment cnn_bilstm_clocks_all --epochs 60 --lr 1e-4 --batch_size 32 \
  --model_dir models --resume models/cnn_bilstm_clocks_all/latest.pth
```

A YAML config is available at `example_config.yaml`.

## License
This project is licensed under the MIT license - see LICENSE.