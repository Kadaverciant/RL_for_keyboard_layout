""" Module contains functions for obtaining dataset """

import argparse
import os
import sys
from ast import literal_eval
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

sys.path.append("./src")  # workaround to import files from ./src folders


from keyboard import (
    ACTION_SPACE_SIZE,
    ID_TO_PAIR,
    KEYS_NUMBER,
    QWERTY_ENCODED_HIGH,
    QWERTY_ENCODED_LOW,
    KeyboardLayout,
    SwapType,
    convert_int_to_coords,
)

# Paths from root directory
FIGURES_SAVE_PATH = os.path.join(".", "data/figures")
MODELS_SAVE_PATH = os.path.join(".", "data/models")
RAW_DATA_PATH = os.path.join(".", "data/raw/encoded_programs.csv")


class Logger:
    """Manage log messages"""

    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose

    def log(self, message: str):
        """Log message to console

        Args:
            message (str): message to log
        """
        if self.verbose:
            print(message)


""" Collect data functions """


def read_data(path: str, limit: int, logger: Logger):
    logger.log(f"Reading data '{path}'...")
    data_points = pd.read_csv(path, index_col=0)

    # Remove all data from folder
    dataset = np.array(
        [
            np.array(text)
            for text in data_points["encoded_text"].apply(literal_eval).to_list()
        ]
    )

    logger.log("Success!\n")

    return dataset[:limit]


def train_a2c(
    qwerty_score,
    env,
    policy_network,
    policy_optimizer,
    data,
    episodes,
    max_steps,
    discount_factor,
):
    solved_score = qwerty_score * 0.5

    best_score = qwerty_score
    best_keyboard = env
    # log scores
    scores = []

    for episode in tqdm(range(episodes)):
        state = KeyboardLayout(
            QWERTY_ENCODED_LOW, QWERTY_ENCODED_HIGH, Logger(verbose=False)
        )
        is_done = False
        scores.append([qwerty_score])
        cumulative_discount = 1.0
        prev_states = [(state.low_layout, state.high_layout)]

        for _ in range(max_steps):
            action, actions_log_probabilities = select_action(policy_network, state)

            # step with action
            btn1, btn2 = ID_TO_PAIR[action]
            cord1, cord2 = convert_int_to_coords(btn1), convert_int_to_coords(btn2)

            swap_type: SwapType = "low_layout"

            if cord2[0] == 1:
                swap_type = "between_layouts"

            if cord1[0] == 1:
                swap_type = "high_layout"

            pos1 = (cord1[1], cord1[2])
            pos2 = (cord2[1], cord2[2])
            new_state = deepcopy(state)
            new_state.swap_buttons(pos1, pos2, swap_type)

            new_score = estimate_layout(new_state, data)

            if new_score.item() < best_score:
                best_score = new_score.item()
                best_keyboard = deepcopy(new_state)

            scores[episode].append(new_score.item())
            reward = qwerty_score - new_score.item()

            if (
                new_state.low_layout,
                new_state.high_layout,
            ) in prev_states or reward < -1000:
                is_done = True
            else:
                is_done = False

            advantage = reward / qwerty_score
            policy_loss = -actions_log_probabilities * advantage

            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            if is_done:
                break
            state = new_state
            prev_states.append((state.low_layout, state.high_layout))
            cumulative_discount *= discount_factor

        if best_score <= solved_score:
            break
    return scores, best_score, best_keyboard


""" Save functions """


def save_best_layout_info(
    best_keyboard: KeyboardLayout,
    best_score: float,
    save_path: str,
    logger: Logger,
):
    logger.log("Saving data...")

    lines = [str(best_score), "\n\n", best_keyboard.get_string_layouts()]
    with open(save_path, "w") as f:
        f.writelines(lines)

    logger.log("Success!\n")


def save_scores_plot(
    scores: torch.Tensor,
    save_path: str,
    best_score: float,
    logger: Logger,
    figsize: tuple[int, int] = (16, 9),
):
    logger.log("Saving figure...")

    best_scores = [min(x) for x in scores]
    worst_scores = [max(x) for x in scores]
    mean_scores = [np.mean(x) for x in scores]
    qwerty_scores = [x[0] for x in scores]

    plt.subplots(1, 1, figsize=figsize)
    plt.plot(best_scores, label="Best score")
    plt.plot(worst_scores, label="Worst score")
    plt.plot(mean_scores, label="Mean score")
    plt.plot(qwerty_scores, label="QWERTY score")

    plt.legend()
    plt.grid()
    plt.xlim([0, len(scores) - 1])
    plt.xticks(list(range(len(scores))))
    plt.ylim([best_score, 14000])
    plt.ylabel("Score")
    plt.xlabel("Episode")
    plt.title("Scores for each episode")

    plt.savefig(save_path)
    plt.close()
    logger.log("Success!\n")


""" Actor Critic functions"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PolicyNetwork(nn.Module):
    def __init__(self, observation_space: int, action_space: int, hidden_dim: int = 128):
        super(PolicyNetwork, self).__init__()
        self.input_layer = nn.Linear(observation_space, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, action_space)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        actions = self.output_layer(x)
        return F.softmax(actions, dim=0)


class StateValueNetwork(nn.Module):
    def __init__(self, observation_space: int, hidden_dim: int = 128):
        super(StateValueNetwork, self).__init__()

        self.input_layer = nn.Linear(observation_space, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))

        return self.output_layer(x)


def select_action(
    policy_network: nn.Module, layout: KeyboardLayout
) -> tuple[int, torch.Tensor]:
    state = layout.flatten().to(DEVICE)
    action_probabilities = policy_network(state)

    action_probabilities = Categorical(action_probabilities)
    action = action_probabilities.sample()

    return int(action.item()), action_probabilities.log_prob(action)


def estimate_layout(layout: KeyboardLayout, dataset: np.ndarray) -> torch.Tensor:
    score = 0
    loop = dataset
    layout.reset()
    for text in loop:
        layout.type_encoded_text(text)

    score = layout.total_score
    return torch.tensor([score]).float().unsqueeze(0).to(DEVICE)


""" Main function """


def train():
    """Train model"""

    # Parse arguments
    parser = argparse.ArgumentParser(description="Make dataset script")
    parser.add_argument(
        "-f",
        "--figures-save-path",
        type=str,
        dest="figures_path",
        default=FIGURES_SAVE_PATH,
        help="path to save figures",
    )
    parser.add_argument(
        "-m",
        "--models-save-path",
        type=str,
        dest="models_path",
        default=MODELS_SAVE_PATH,
        help="path to save models",
    )
    parser.add_argument(
        "-d",
        "--data-path",
        type=str,
        dest="data_path",
        default=RAW_DATA_PATH,
        help="path to load the raw data",
    )
    parser.add_argument(
        "-l",
        "--data-limit",
        type=int,
        dest="data_limit",
        default=10,
        help="number of used data points",
    )
    parser.add_argument(
        "-df",
        "--discount-factor",
        type=float,
        dest="discount_factor",
        default=0.99,
        help="discount factor",
    )
    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        dest="episodes",
        default=500,
        help="number of train episodes",
    )
    parser.add_argument(
        "-s",
        "--max-steps",
        type=int,
        dest="max_steps",
        default=100,
        help="number max train steps",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="print information (default: True)",
    )

    namespace = parser.parse_args()
    (
        figures_save_path,
        models_save_path,
        data_path,
        data_limit,
        discount_factor,
        episodes,
        max_steps,
        verbose,
    ) = (
        namespace.figures_path,
        namespace.models_path,
        namespace.data_path,
        namespace.data_limit,
        namespace.discount_factor,
        namespace.episodes,
        namespace.max_steps,
        namespace.verbose,
    )
    verbose: bool = bool(verbose)

    # Set up logger
    logger = Logger(verbose)

    # Read data from disc
    data = read_data(data_path, data_limit, logger)

    # Init env and networks
    logger.log("Init\n")
    env = KeyboardLayout(QWERTY_ENCODED_LOW, QWERTY_ENCODED_HIGH, Logger(verbose=False))
    policy_network = PolicyNetwork(KEYS_NUMBER, ACTION_SPACE_SIZE).to(DEVICE)
    policy_optimizer = optim.SGD(policy_network.parameters(), lr=1e-3)

    # Estimate qwerty
    logger.log("Estimate QWERTY layout")
    qwerty_keyboard = KeyboardLayout(
        QWERTY_ENCODED_LOW, QWERTY_ENCODED_HIGH, Logger(verbose=False)
    )
    qwerty_score = estimate_layout(qwerty_keyboard, data).item()

    # Estimate random layouts
    logger.log("Estimate\n")

    scores, best_score, best_keyboard = train_a2c(
        qwerty_score,
        env,
        policy_network,
        policy_optimizer,
        data,
        episodes,
        max_steps,
        discount_factor,
    )

    # Save policy network
    torch.save(policy_network.state_dict(), os.path.join(models_save_path, "policy"))

    # Save keyboard data
    save_best_layout_info(
        best_keyboard,
        best_score,
        os.path.join(figures_save_path, f"best_{episodes}.txt"),
        logger,
    )

    # Save figures
    save_scores_plot(
        scores,
        os.path.join(figures_save_path, f"best_{episodes}.png"),
        best_score,
        logger,
    )

    logger.log("Done!")


if __name__ == "__main__":
    train()
