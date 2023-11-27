""" Module contains functions for obtaining dataset """

import argparse
import os
import random
import sys
from ast import literal_eval
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append("./src")  # workaround to import files from ./src folders


from keyboard import (
    ALL_BUTTONS_ENCODED,
    KEYBOARD_LAYOUT_SHAPE,
    QWERTY_ENCODED_HIGH,
    QWERTY_ENCODED_LOW,
    SHIFT_CODE,
    KeyboardLayout,
    Layout,
)

# Paths from root directory
FIGURES_SAVE_PATH = os.path.join(".", "data/figures")
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


""" Layout generation functions """


def generate_random_layout(
    all_buttons_encoded: list[int],
    keyboard_shape: tuple[int, ...],
    seed: Optional[int] = None,
) -> tuple[Layout, Layout]:
    if seed is not None:
        random.seed(seed)

    all_buttons = all_buttons_encoded.copy()

    low_layout = [[0 for _ in range(row_length)] for row_length in keyboard_shape]
    high_layout = [[0 for _ in range(row_length)] for row_length in keyboard_shape]

    # Push single SHIFT to the low layout
    all_buttons.remove(SHIFT_CODE)
    random.shuffle(all_buttons)

    shift_position_low_layout = random.randint(0, len(all_buttons_encoded) // 2)
    all_buttons.insert(shift_position_low_layout, SHIFT_CODE)
    pointer = 0
    for layout in (low_layout, high_layout):
        for i, row_length in enumerate(keyboard_shape):
            for j in range(row_length):
                layout[i][j] = all_buttons[pointer]
                pointer += 1

    return low_layout, high_layout


def generate_layouts(
    layouts_number: int,
    logger: Logger,
) -> list[KeyboardLayout]:
    layouts = []
    loop = range(layouts_number)
    if logger.verbose:
        loop = tqdm(loop)
        loop.set_description(desc="Generating layouts")

    for i in loop:
        random_low, random_high = generate_random_layout(
            ALL_BUTTONS_ENCODED, KEYBOARD_LAYOUT_SHAPE, seed=i
        )
        layouts.append(KeyboardLayout(random_low, random_high, Logger(verbose=False)))
    logger.log("Success!\n")
    return layouts


def estimate_layouts(
    layouts: list[KeyboardLayout], dataset: np.ndarray, logger: Logger
) -> torch.Tensor:
    loop = dataset
    if logger.verbose:
        loop = tqdm(loop)
        loop.set_description(desc="Estimating layouts")
    for layout in layouts:
        layout.reset()
    for text in loop:
        for layout in layouts:
            layout.type_encoded_text(text)
    logger.log("Success!\n")
    return torch.as_tensor([layout.total_score for layout in layouts])


def save_scores_plot(
    scores: torch.Tensor,
    qwerty_score: float,
    save_path: str,
    logger: Logger,
    figsize: tuple[int, int] = (16, 9),
):
    logger.log("Saving figure...")

    qwerty_scores = [qwerty_score for _ in range(len(scores))]
    random_scores = [x.item() for x in scores]
    plt.subplots(1, 1, figsize=figsize)
    points = list(range(len(random_scores)))
    plt.scatter(points, random_scores, s=10, label="Layout score")
    plt.plot(qwerty_scores, label="QWERTY score", color="red")
    plt.legend()
    plt.grid()
    plt.ylabel("Score")
    plt.xlabel("Random layout #")
    plt.title("Scores for random layouts")
    plt.savefig(save_path)
    plt.close()
    logger.log("Success!\n")


""" Main function """


def train():
    """Train model"""

    # Parse arguments
    parser = argparse.ArgumentParser(description="Make dataset script")
    parser.add_argument(
        "-s",
        "--save-path",
        type=str,
        dest="save_path",
        default=FIGURES_SAVE_PATH,
        help="path to save figures",
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
        "-n",
        "--number-layouts",
        type=int,
        dest="layouts_number",
        default=5000,
        help="number of random layouts",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="print information (default: True)",
    )

    namespace = parser.parse_args()
    save_path, data_path, layouts_number, data_limit, verbose = (
        namespace.save_path,
        namespace.data_path,
        namespace.layouts_number,
        namespace.data_limit,
        namespace.verbose,
    )
    verbose: bool = bool(verbose)

    # Set up logger
    logger = Logger(verbose)

    # Read data from disc
    data = read_data(data_path, data_limit, logger)

    # Generate random layouts
    logger.log("Estimate random layouts")
    layouts = generate_layouts(
        layouts_number,
        logger,
    )

    # Estimate qwerty
    logger.log("Estimate QWERTY layout")
    qwerty_keyboard = KeyboardLayout(
        QWERTY_ENCODED_LOW, QWERTY_ENCODED_HIGH, Logger(verbose=False)
    )
    qwerty_score = estimate_layouts([qwerty_keyboard], data, logger)[0].item()

    # Estimate random layouts
    layouts_scores = estimate_layouts(layouts, data, logger)

    # Save figures
    save_scores_plot(
        layouts_scores,
        qwerty_score,
        os.path.join(save_path, f"baseline_{layouts_number}.png"),
        logger,
    )

    logger.log("Done!")


if __name__ == "__main__":
    train()
