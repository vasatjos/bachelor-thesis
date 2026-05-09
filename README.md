# RL Agents for Prší - Evaluation

This repository provides an interactive environment to play against various Reinforcement Learning agents trained for the Czech card game **Prší**.

## Setup

### 1. Install `uv`
This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Install it using:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Setup the Project
Clone the repository and install the dependencies:

```bash
git clone https://github.com/vasatjos/bachelor-thesis.git
cd bachelor-thesis
uv sync --all-groups
```

## How to Play

You can test your skills against a trained agent using the `human.py` script from the `src` directory.

### Quick Start (Evaluate against best agent)
On this branch, reasonable defaults are already set for the best performing agent. Simply specify how many games you want to play:

```bash
cd src
uv run -m agents.human --evaluate_for 5
```

### Statistics & Progress
Your win/loss statistics are automatically saved to `src/human_stats.json`. You can track your overall performance across multiple sessions. To use a different file:
```bash
uv run -m agents.human --stats_path my_results.json
```
Set the argument to `None` to disable log saving.

### Visual Options (Icons)
If you have a [Nerd Font](https://www.nerdfonts.com/) installed and set up in your terminal, you can enable graphical card icons for a better experience:

```bash
export PRSI_USE_ICONS=true
uv run -m agents.human
```
