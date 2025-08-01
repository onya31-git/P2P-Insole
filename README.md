# P2P-Insole: Transformerによる足圧データ、IMUデータからの骨格動作予測モデル
※現在作成中

# Repository Orientation

## Overview
This repository implements a model that predicts human skeleton motion from insole pressure and IMU data using a Transformer-based architecture.  
The README is minimal, simply naming the project and noting it is still under construction.


Most functionality resides in the `processor/` package, with command-line entry via `main.py`. The repository also contains older experimental scripts under `cord_arcive/`, raw data directories, and configuration files.

## Key Components

### Entry Point
- **`main.py`** – Loads sub-modules and dispatches to different processing modes (train, predict, visual) based on CLI arguments.

### Processing Modules (under `processor/`)
- **`train.py`** – Handles data loading, preprocessing, and training of the Transformer model. Reads YAML config parameters, applies scalers, splits datasets, and saves checkpoints.
- **`predict.py`** – Loads the trained model and performs inference over test data, then writes predictions to CSV.
- **`model.py`** – Defines the `Transformer_Encoder` architecture, loss function, and training loop including progress output and checkpoint saving.
- **`loader.py`** – Utilities for reading YAML configs, pairing skeleton and insole files, preprocessing data, and defining custom `Dataset` classes for PyTorch.
- **`util.py`** – Helper for printing configuration settings to the console.
- **`visualization.py`** – Placeholder for future visualization utilities.

### Configuration
YAML files in `config/transformer_encoder/` specify default hyperparameters, dataset paths, and checkpoints. Example parameters in `train.yaml` include model size (`d_model`), number of encoder layers, training epochs, and batch size. `predict.yaml` defines similar settings for inference, such as the checkpoint file to load.

### Data and Experiments
- **`data/`** and **`rawData/`** hold sample datasets. Some directories contain older sensor scripts (e.g., `sensor.py`, `v2f.py`) for converting voltage readings to forces.
- **`cord_arcive/`** contains previous notebooks and scripts that aren’t part of the main pipeline but may serve as references.

### Output
- Training saves models to `./weight/` and scalers to `./scaler/`.
- Predictions are written to `./output/predicted_skeleton.csv`.
- Visualization is handled via `visualization.py` (Plotly-based animation).

## Usage

1. **Training**  
   Edit `config/transformer_encoder/train.yaml` to set dataset directories and hyperparameters. Then run:
   ```bash
   python main.py train --config config/transformer_encoder/train.yaml
   ```
   This reads data from data_path/skeleton and data_path/Insole and trains the Transformer model.

2. **Prediction**
   After training, update predict.yaml with the checkpoint path and desired data directory. Run:
   ```bash
   python main.py predict --config config/transformer_encoder/predict.yaml
   ```
   The predictions CSV can be visualized with:
   ```bash
   python visualization.py
   ```

## Important Notes
- The project relies on PyTorch, pandas, NumPy, scikit-learn, and Plotly; ensure these are installed.
- Training saves a scaler (skeleton_scaler.pkl) for later inverse-transforming predictions in predict.py.
- Evaluation utilities are still under development (the current processor/evaluation.py is empty). Older evaluation scripts can be found under cord_arcive/.

## Suggested Next Steps for New Contributors
1. Understand loader.py
   Study how dataset paths are collected and how data is preprocessed into combined pressure/IMU features.

2. Review Model Implementation
   processor/model.py shows the Transformer architecture and the training routine. Pay attention to how positional encoding and the training loop are implemented.

3. Experiment with Hyperparameters
   Tweak config/transformer_encoder/train.yaml to modify sequence length, dropout, or the number of encoder layers. Re-train to gauge their effect.

4. Implement Evaluation
   Consider enhancing processor/evaluation.py by adapting logic from cord_arcive/evaluation.py or evaluation_rich.py to compute MSE/RMSE metrics on prediction results.

5. Improve Documentation
   The README is currently minimal. Adding setup instructions, dependency lists, and usage examples would help future users.
