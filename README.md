# RL Agents for Prší

This project implements various Reinforcement Learning agents for the Czech card game **Prší** (a variant of Mau-Mau or Crazy Eights).

> [!NOTE]
> If you just want to play against a pre-trained agent without setting up training, check out the `evaluation` branch. It comes with a pre-trained model and simplified evaluation defaults.

## Quick Start

### 1. Install `uv`
This project uses [uv](https://github.com/astral-sh/uv) for dependency management. If you don't have it installed, you can install it using:

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

## Training Agents

You can train various agents by running their respective modules from the `src` directory. Available algorithms include:
- **DQN** (`agents.dqn`)
- **Double DQN** (`agents.ddqn`)
- **Monte Carlo** (`agents.monte_carlo`)
- **Q-Learning** (`agents.q_learning`)
- **REINFORCE** (`agents.reinforce`)

### Example: Training a DQN agent
```bash
cd src
uv run -m agents.dqn --episodes 500000 --epsilon 0.1 --gamma 0.99
```

Training progress and models are saved by default in the `src/agent_strategies` and `src/logs` directories (or as specified by `--model_path`).

## Playing Against Agents

You can test your own skills by playing against a trained agent or a baseline agent using the `human.py` script.

### Play against a baseline (Greedy) agent:
```bash
cd src
uv run -m agents.human --opponent greedy
```

### Play against a trained DQN agent:
```bash
cd src
uv run -m agents.human --opponent dqn --model_path path/to/a/model/file
```

### Available Opponents:
- `random`, `greedy` (baselines)
- `dqn`, `ddqn`, `monte_carlo`, `q_learning`, `reinforce` (RL agents)


> [!TIP]
> A trained REINFORCE agent is available on the `evaluation` branch

### Visual Options (Icons)
If you have a [Nerd Font](https://www.nerdfonts.com/) installed and set up in your terminal, you can enable graphical card icons for a better experience:

```bash
export PRSI_USE_ICONS=true
uv run -m agents.human
```

## Project Structure

- `src/agents/`: Implementation of various RL algorithms.
- `src/prsi/`: The game environment and logic.
- `src/tests/`: Unit tests for game logic.
- `paper/`: Typst source files for the bachelor thesis.
- `src/agent_strategies/`: Saved models and training logs.

---

<img src="https://fit.cvut.cz/static/images/fit-cvut-logo-en.svg" alt="FIT CTU logo" height="200">

This software was developed with the support of the **Faculty of Information Technology, Czech Technical University in Prague**.
For more information, visit [fit.cvut.cz](https://fit.cvut.cz).
