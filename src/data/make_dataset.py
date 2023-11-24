""" Module contains functions for obtaining dataset """

import argparse
import os
import shutil
import sys

import pandas as pd
from tqdm import tqdm

sys.path.append("./src")  # workaround to import files from ./src folders


from keyboard import BUTTONS_SET, ENCODE_DICT, SPECIAL_SYMBOLS  # pylint: disable=E0401

MANUAL_SEED = 42

# Assume it is already downloaded to folder data/interim/validated
DATASET_URL = "https://www.kaggle.com/datasets/joshuwamiller/software-code-dataset-cc"

# Paths from root directory
INTERIM_DATA_PATH = os.path.join(".", "data/interim")
RAW_DATA_PATH = os.path.join(".", "data/raw")


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


def clear_folder(path: str, logger: Logger):
    """Clear folder content and add .gitignore

    Args:
        path (str): path to folder
        logger (Logger): logger class instance
    """

    logger.log(f"Clearing folder `{path}`...")

    # Remove all data from folder
    shutil.rmtree(path)

    # Create empty directory
    os.mkdir(path)

    # Add .gitignore file
    file_path = os.path.join(path, ".gitignore")
    with open(file_path, "a", encoding="utf8"):
        pass
    logger.log("Success!\n")


""" Preprocess data functions """


def combine_raw_into_csv(path: str, save_path: str, logger: Logger) -> pd.DataFrame:
    """Combine raw programs from initial dataset into single csv

    Args:
        path (str): initial data path
        save_path (str): path to save resulting csv
        logger (Logger): logger class instance

    Returns:
        pd.DataFrame: resulting pandas data frame
    """
    loop = os.listdir(path)
    if logger.verbose:
        loop = tqdm(loop, desc="Reading raw data from disk")

    programs = []
    for filename in loop:
        df = pd.read_csv(os.path.join(path, filename), sep="`")
        programs.append("\n".join(df["code"]))

    programs_df = pd.DataFrame(programs, columns=["text"])

    logger.log("Saving raw programs...")
    programs_df.to_csv(save_path)
    logger.log("Success!\n")
    return programs_df


def preprocess_text(text: str, max_length: int, padding_value: int) -> list[int]:
    """Preprocess raw program text

    Args:
        text (str): raw program txt
        max_length (int): max length of text
        padding_value (int): symbol to pad the text

    Returns:
        list[int]: cleaned and encoded program text
    """
    splitted_text = [
        SPECIAL_SYMBOLS[s] if s in SPECIAL_SYMBOLS else s for s in list(text)
    ]
    encoded_text = [
        ENCODE_DICT[symbol]
        for symbol in list(filter(lambda s: s in BUTTONS_SET, splitted_text))[:max_length]
    ]
    return encoded_text + [padding_value for _ in range(max_length - len(encoded_text))]


def encode_dataset(data: pd.DataFrame, max_length: int, save_path: str, logger: Logger):
    """Encode raw dataset

    Args:
        data (pd.DataFrame): raw programs dataset
        max_length (int): max length of text
        save_path (str): path to save encoded dataset
        logger (Logger): logger instance

    """

    padding_value: int = ENCODE_DICT["<space>"]

    encoded_df = data.copy()

    logger.log("Preprocessing data...")
    encoded_df["encoded_text"] = encoded_df["text"].apply(
        lambda t: preprocess_text(t, max_length, padding_value)
    )
    logger.log("Success!")

    logger.log("Saving encoded programs...")
    encoded_df.drop(columns=["text"], inplace=True)
    encoded_df.to_csv(save_path)
    logger.log("Success!\n")


""" Main function """


def make_dataset():
    """Make dataset"""

    # Parse arguments
    parser = argparse.ArgumentParser(description="Make dataset script")
    parser.add_argument(
        "-i",
        "--interim-path",
        type=str,
        dest="interim",
        default=INTERIM_DATA_PATH,
        help="path to save the intermediate data like .zip files",
    )
    parser.add_argument(
        "-r",
        "--raw-path",
        type=str,
        dest="raw",
        default=RAW_DATA_PATH,
        help="path to save the raw data like .tsv files",
    )
    parser.add_argument(
        "-m",
        "--max-length",
        type=int,
        dest="max_length",
        default=400,
        help="max length of program in final dataset",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="print information (default: True)",
    )
    namespace = parser.parse_args()
    interim_path, raw_path, max_length, verbose = (
        namespace.interim,
        namespace.raw,
        namespace.max_length,
        namespace.verbose,
    )
    verbose: bool = bool(verbose)

    # Set up logger
    logger = Logger(verbose)

    # Clear old data
    clear_folder(raw_path, logger)

    # Reading raw data from disk data
    programs_df = combine_raw_into_csv(
        os.path.join(interim_path, "validated"),
        os.path.join(raw_path, "raw_programs.csv"),
        logger,
    )

    # Preprocess dataset - Encoding
    encode_dataset(
        programs_df, max_length, os.path.join(raw_path, "encoded_programs.csv"), logger
    )

    logger.log("Done!")


if __name__ == "__main__":
    make_dataset()
