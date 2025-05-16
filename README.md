# AI-Driven Fitness Coach: Real-Time Posture Correction & Personalized Training

**Authors:** Sibei Zhou, Xiaohe Hu, Yihe Liu

## Overview

This repository hosts the implementation of our semester project—a real-time AI fitness coach that automatically classifies exercises, counts repetitions, and evaluates movement quality from video input. By combining graph convolutional spatial modeling with bidirectional temporal processing and a “perfect skeleton” reference, users receive instant, frame-by-frame corrective feedback to improve their form and reduce injury risk.

## Key Features

* **Exercise Recognition**: Classifies three common workouts (squats, push‑ups, jumping jacks) with 98.6% accuracy.
* **Repetition Counting**: Estimates rep counts with a mean absolute error of 0.27 and RMSE of 0.52.
* **Motion Scoring**: Compares user skeletons against an averaged “perfect” template, scoring both coordinate and structural (angle-based) deviations to deliver real-time feedback.
* **Multi-Task Learning**: Trains on three simultaneous objectives—action classification (CrossEntropy), rep count regression (MSE), and rep-phase detection (BCE)—to stabilize predictions and guide frame-level timing.

## Method Pipeline

1. **Data & Preprocessing**

   * **Dataset**: UCF101 actions subset (body‑weight squats, push‑ups, jumping jacks).
   * **Skeleton Extraction**: MediaPipe Holistic → 33 joints (x,y,z,visibility).
   * **Normalization**: Center at mid‑hip, scale by shoulder‑hip distance, align axes for view invariance.
   * **Augmentations**: Noise, horizontal flips, time warps, 3D rotations, global scaling.

2. **Model Architecture**

   * **Spatial GCN**: Two-layer graph convolution on 33‑joint skeleton graph to encode structural context.
   * **Temporal BiLSTM**: Bidirectional LSTM processes the sequence of spatial embeddings to capture cyclic motion patterns.

3. **Perfect Skeleton Comparison**

   * Build per-exercise average skeleton from trimmed standard videos.
   * During inference, segment each rep via phase-peak detection.
   * Score by weighted 3D coordinate difference + joint‑angle structural error.

4. **Training & Loss**

   * **Loss** = CE$_	ext{class}$ + α·MSE$_	ext{count}$ + β·BCE$_	ext{phase}$.
   * Best results with α=3, β=1: Val Loss=0.98, Action Acc=98.55%, Rep MAE=0.27, RMSE=0.52, Int-Acc=85.51%.

## Repository Structure

```
├── __pycache__/            # Python cache directory
├── checkpoints/            # Saved training checkpoints
├── result_videos/          # Overlay videos with skeleton and feedback
├── split_data/             # Partitioned dataset for train/val/test
├── test_skeleton/          # Raw skeleton JSON outputs for test videos
├── test_skeleton_json/     # Processed JSON skeleton data
├── test_video/             # Full-size test video files
├── test_video_small/       # Downscaled test videos for quick testing
├── README.md               # Project overview and instructions
├── action_correction.py    # Rep-phase detection and corrective feedback logic
├── dataset.py              # Data loading and preprocessing pipeline
├── fitness_model.onnx      # Exported inference model
├── loss_curve.png          # Training loss curve visualization
├── main.py                 # Entry point for end-to-end execution
├── model.py                # GCN + BiLSTM model definition
├── poseTrackingVideo.py    # MediaPipe-based skeleton extraction script
├── predict.py              # Inference and scoring wrapper
├── train.py                # Multi-task training loop
```

## Installation

```bash
git clone https://github.com/yourusername/fitness-coach.git
cd fitness-coach
env/bin/activate  # or `source env/bin/activate`
pip install -r requirements.txt
```

## Usage

1. **Preprocess Video**

   ```bash
   python src/preprocess.py --video path/to/video.mp4 --out data/skeleton.json
   ```
2. **Train Model**

   ```bash
   python src/train.py --config configs/train.yaml
   ```
3. **Run Inference & Scoring**

   ```bash
   python src/infer.py --skeleton data/skeleton.json --out results/
   ```

## Results

| Metric                         | Value  |
| ------------------------------ | ------ |
| Action Classification Accuracy | 98.55% |
| Rep Count MAE                  | 0.27   |
| Rep Count RMSE                 | 0.52   |
| Integer Count Accuracy         | 85.51% |
