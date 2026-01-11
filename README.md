# ML-Hackathon

This repository contains code, models and data used for reinforcement learning experiments (a DQN agent) and a small Hidden Markov Model (HMM) experiment. It includes training and evaluation scripts, pretrained model checkpoints, training logs / reward curves, and helper utilities.

## Quick summary

- DQN-based RL training and evaluation implemented in Python (PyTorch).
- A small HMM sequence-modeling component (trained model saved as a pickle).
- Pretrained checkpoints and training artifacts are included for quick inspection / evaluation.

Repository: https://github.com/karthikeya172/ML-Hackathon

## Requirements

- Python 3.8+ (recommended)
- Install required packages:
  - pip install -r [requirements.txt](https://github.com/karthikeya172/ML-Hackathon/blob/main/requirements.txt)

Note: The project uses PyTorch for the DQN agent and common scientific packages (see `requirements.txt`). A GPU is recommended for faster DQN training but not required for loading/evaluating saved models.

## How to run

1. Inspect and edit hyperparameters
   - Hyperparameters (learning rates, number of episodes, checkpoint paths, etc.) are defined in the top of `train.py` and `evaluate.py`. Edit those values or add CLI parsing as needed.

2. Train DQN
   - Run the training script to start training (this will create checkpoints / reward logs):
     - python train.py
   - Training writes model checkpoints and reward logs (see `training_rewards_final.pkl`, `rewards_latest.pkl`, and checkpoint `.pt` files).

3. Evaluate a trained DQN
   - Use the evaluation script to evaluate a saved checkpoint or run the trained agent:
     - python evaluate.py
   - Update the checkpoint path inside `evaluate.py` (or pass a checkpoint path if the script supports CLI flags).

4. HMM model
   - `hmm_model.py` contains code to train / load a Hidden Markov Model.
   - `hmm_model_final.pkl` is the saved HMM model that can be loaded with pickle for inference / inspection.

## Files & purpose

- [train.py](https://github.com/karthikeya172/ML-Hackathon/blob/main/train.py) — Main training loop for the DQN agent.
- [evaluate.py](https://github.com/karthikeya172/ML-Hackathon/blob/main/evaluate.py) — Evaluation script for running a trained agent and computing metrics.
- [dqn_agent.py](https://github.com/karthikeya172/ML-Hackathon/blob/main/dqn_agent.py) — DQN agent implementation (network, replay buffer, training step).
- [env.py](https://github.com/karthikeya172/ML-Hackathon/blob/main/env.py) — Environment wrapper / custom environment used for training and evaluation.
- [hmm_model.py](https://github.com/karthikeya172/ML-Hackathon/blob/main/hmm_model.py) — HMM training / inference utilities.
- [utils.py](https://github.com/karthikeya172/ML-Hackathon/blob/main/utils.py) — Utility helpers used across scripts.
- [requirements.txt](https://github.com/karthikeya172/ML-Hackathon/blob/main/requirements.txt) — Python dependencies.
- [corpus.txt](https://github.com/karthikeya172/ML-Hackathon/blob/main/corpus.txt) — Text / sequence data used by HMM or experiments.
- [test.txt](https://github.com/karthikeya172/ML-Hackathon/blob/main/test.txt) — Example/test data.
- [reward_curve_final.png](https://github.com/karthikeya172/ML-Hackathon/blob/main/reward_curve_final.png) — Training reward curve visualization.
- Model checkpoints and artifacts:
  - `dqn_checkpoint_latest.pt`, `dqn_checkpoint_1000.pt`, `dqn_checkpoint_2000.pt`, ..., `dqn_checkpoint_8000.pt`, `dqn_net_final.pt` — Saved PyTorch checkpoints for the DQN agent.
  - `training_rewards_final.pkl`, `rewards_latest.pkl` — Pickled training reward logs.
  - `hmm_model_final.pkl`, `hmm_checkpoint.pkl` — Saved HMM model(s).

## Notes & recommendations

- Checkpoint sizes: Model files are included and may be large. If you re-run training, consider saving checkpoints selectively (every N episodes) to reduce disk usage.
- If you'd like CLI flags for specifying checkpoint paths, number of episodes, or device selection (cpu/gpu), consider adding argparse to `train.py` and `evaluate.py`.
- For reproducibility, set random seeds in `train.py` and `env.py` (if not already present).

## Reproducing results

1. Install dependencies.
2. If you have a GPU, ensure CUDA/CuDNN and the correct PyTorch build are installed.
3. Run:
   - python train.py
   - After training finishes or at a saved checkpoint, run:
   - python evaluate.py
4. Inspect `reward_curve_final.png` and the `.pkl` reward logs to analyze training progress.

## Contributing

Contributions are welcome. Suggested improvements:
- Add clear CLI arguments for all scripts.
- Add a small config file (YAML/JSON) for hyperparameters.
- Add unit tests for core components (replay buffer, env wrapper).
- Add a short example / notebook demonstrating how to load and run a saved model.

## License

No license file is included in the repository. If you want this project to be open-source, add a LICENSE (MIT, Apache-2.0, etc.) file.

## Contact

If you have questions or want help reproducing results, open an issue in the repository or reach out to the repository owner.
