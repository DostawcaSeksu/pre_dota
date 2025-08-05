# Dota 2 Win Prediction Project

## Project Overview

This project serves as a hands-on learning endeavor designed to master fundamental concepts in Data Science and Deep Learning. Created as a personal educational exercise, it's primary purpose was to apply theoretical knowledge to a real-world problem, thereby gaining practical skills in key libraries, architectures, and the end-to-end machine learning workflow.

The project centers on predicting the winning team in the popular online game Dota 2. This challenge was chosen for its data-rich environment and clear objective, providing an ideal testing ground for a variety of predictive modeling techniques. The core of the project involves taking raw match data—detailing the performance of each of the ten players—and transforming it into a powerful set of predictive features. By engineering team-level "advantage" metrics such as gold and experience differentials, the goal was to build a model that could accurately determine the match outcome. 

---

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [1. Data Exploration (EDA)](#1-data-exploration-eda)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Model Training & Evaluation](#3-model-training--evaluation)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)

---

## Dataset

The project utilizes the **Dota 2 Matches** dataset, which contains data from approximately 50,000 matches. The key files used are:

-   `players.csv`: Detailed end-game statistics for each of the 10 players in every match (GPM, XPM, Kills, Deaths, Assists, etc.).
-   `match.csv`: General information about each match, including the final outcome (`radiant_win`), which serves as our target variable.

The full dataset can be found on Kaggle: [Dota 2 Matches](https://www.kaggle.com/datasets/devinanzelmo/dota-2-matches).

Of course you can use your own dataset!

---

## Methodology

### 1. Data Exploration (EDA)

The initial analysis was performed to understand the structure and quality of the raw data.
-   **Key Findings:**
    -   The dataset is well-balanced, with Radiant winning approximately 48.11% of the matches.
    -   The `players.csv` file contains a rich set of features, but many columns (especially `unit_order_...`) have a high number of missing values and were deemed unsuitable for this analysis.
    -   Core statistical indicators like `gold_per_min`, `xp_per_min`, `kills`, etc., are complete and serve as strong candidates for feature engineering.

### 2. Feature Engineering

The raw per-player data was transformed into team-level "advantage" features for each match.
-   **Process:**
    1.  A `team` feature (`radiant` or `dire`) was created based on the `player_slot`.
    2.  Key player stats (GPM, XPM, Kills, Deaths, Hero Damage, Tower Damage, etc.) were grouped by `match_id` and `team`.
    3.  The stats were summed up for each team in every match.
    4.  The table was "unstacked" to have a single row per match.
    5.  Final "advantage" features were calculated as `radiant_stat - dire_stat`. For example: `gold_per_min_adv = radiant_total_gpm - dire_total_gpm`.
    6.  The resulting feature set was merged with the target variable `radiant_win` from `match.csv`.
-   **Result:** A clean, processed dataset (`dota_preprocessed_data.csv`) with 10 powerful features and 1 target column.

### 3. Model Training & Evaluation

Three different models were trained on the processed data to predict `radiant_win`.

1.  **Logistic Regression:** A strong baseline model for binary classification.
2.  **Random Forest Classifier:** A powerful ensemble model known for high accuracy on tabular data.
3.  **Neural Network (PyTorch):** A simple deep learning model with two hidden layers, trained on a CUDA-enabled GPU.

For all models, the data was scaled using `StandardScaler` to ensure stable training.

---

## Setup & Installation

This project was developed using Python. It is recommended to use a virtual environment.

1.  **Clone the repository or download the files.**
2.  **Install CUDA toolkit 12.6 (you can dowload it [here](https://developer.nvidia.com/cuda-12-6-0-download-archive)).**
3.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
4.  **Install the required libraries:**
    ```bash
    pip install -r /path/to/requirements.txt
    ```
    *(Note: Make sure to change `/path/to/requirements.txt` to real file path).*

---

## How to Run

The scripts must be run in the following order:

1.  **Run the feature engineering script:** This will process the raw data and create the `dota_preprocessed_data.csv` file.
    ```bash
    python feature_engineering.py
    ```
2.  **Run the classic ML models script:** This will train and evaluate the Logistic Regression and Random Forest models.
    ```bash
    python train_model.py
    ```
3.  **Run the PyTorch script:** This will train and evaluate the Neural Network.
    ```bash
    python train_nn.py
    ```