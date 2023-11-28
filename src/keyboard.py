from copy import deepcopy
from typing import Literal, Optional

import numpy as np
import torch

QWERTY_LOW_LAYOUT: list[list[str]] = [
    ["`", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "=", "<back>"],
    ["<tab>", "q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "[", "]", "\\"],
    [
        "<caps>",
        "a",
        "s",
        "d",
        "f",
        "g",
        "h",
        "j",
        "k",
        "l",
        ";",
        "'",
        "<enter>",
        "<enter>",
    ],
    [
        "<shift>",
        "<shift>",
        "z",
        "x",
        "c",
        "v",
        "b",
        "n",
        "m",
        ",",
        ".",
        "/",
        "<shift>",
        "<shift>",
    ],
    [
        "<ctrl>",
        "<alt>",
        "<space>",
        "<space>",
        "<space>",
        "<space>",
        "<space>",
        "<space>",
        "<space>",
        "<alt>",
        "<ctrl>",
    ],
]

QWERTY_HIGH_LAYOUT: list[list[str]] = [
    ["~", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_", "+", "<back>"],
    ["<tab>", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "{", "}", "|"],
    [
        "<caps>",
        "A",
        "S",
        "D",
        "F",
        "G",
        "H",
        "J",
        "K",
        "L",
        ":",
        '"',
        "<enter>",
        "<enter>",
    ],
    [
        "<shift>",
        "<shift>",
        "Z",
        "X",
        "C",
        "V",
        "B",
        "N",
        "M",
        "<",
        ">",
        "?",
        "<shift>",
        "<shift>",
    ],
    [
        "<ctrl>",
        "<alt>",
        "<space>",
        "<space>",
        "<space>",
        "<space>",
        "<space>",
        "<space>",
        "<space>",
        "<alt>",
        "<ctrl>",
    ],
]


def get_buttons_set(
    low_layout: list[list[str]], high_layout: list[list[str]]
) -> set[str]:
    buttons: set[str] = set()

    for layout in [low_layout, high_layout]:
        for row in layout:
            for btn in row:
                buttons.add(btn)

    return buttons


def get_keyboard_shape(layout: list[list[str]]) -> tuple[int, ...]:
    shape = [len(row) for row in layout]

    return tuple(shape)


def encode_decode_buttons(buttons: set[str]) -> tuple[dict[str, int], dict[int, str]]:
    letters_dict = {}
    for idx, letter in enumerate("abcdefghijklmnopqrstuvwxyz"):
        letters_dict[letter] = idx + 1

    offset = len(letters_dict)
    for idx, letter in enumerate("abcdefghijklmnopqrstuvwxyz".upper()):
        letters_dict[letter] = offset + idx + 1

    encode_value = len(letters_dict) + 1
    encode_dict = {}
    decode_dict = {}
    for btn in sorted(buttons):
        if btn in letters_dict:
            decode_dict[letters_dict[btn]] = btn
            encode_dict[btn] = letters_dict[btn]
        else:
            decode_dict[encode_value] = btn
            encode_dict[btn] = encode_value
            encode_value += 1
    return encode_dict, decode_dict


BUTTONS_SET = get_buttons_set(QWERTY_LOW_LAYOUT, QWERTY_HIGH_LAYOUT)
KEYBOARD_LAYOUT_SHAPE = get_keyboard_shape(QWERTY_LOW_LAYOUT)
KEYBOARD_LAYOUT_CUMSUM_SHAPE = np.cumsum(KEYBOARD_LAYOUT_SHAPE)
KEYS_NUMBER: int = sum(KEYBOARD_LAYOUT_SHAPE) * 2


def convert_int_to_coords(n: int) -> tuple[int, int, int]:
    shift = 0
    row = 0
    column = 0
    if n >= KEYS_NUMBER:
        return 2, 0, 0
    if n < 0:
        return 2, 0, 0
    if n >= KEYS_NUMBER // 2:
        n -= KEYS_NUMBER // 2
        shift = 1
    for i, el in enumerate(KEYBOARD_LAYOUT_CUMSUM_SHAPE):
        if n < el:
            row = i
            break
    if row > 0:
        n -= KEYBOARD_LAYOUT_CUMSUM_SHAPE[row - 1]
    column = n
    return shift, row, column


ENCODE_DICT, DECODE_DICT = encode_decode_buttons(BUTTONS_SET)
ENCODED_BUTTONS_SET = {ENCODE_DICT[btn] for btn in BUTTONS_SET}
SHIFT_CODE = ENCODE_DICT["<shift>"]


SPECIAL_SYMBOLS = {
    "\t": "<tab>",
    "\n": "<enter>",
    " ": "<space>",
}

Layout = list[list[int]]


def encode_layout(layout: list[list[str]]) -> Layout:
    return [[ENCODE_DICT[btn] for btn in layout[i]] for i in range(len(layout))]


def decode_layout(layout: Layout) -> list[list[str]]:
    return [[DECODE_DICT[btn] for btn in layout[i]] for i in range(len(layout))]


QWERTY_ENCODED_HIGH: Layout = encode_layout(QWERTY_HIGH_LAYOUT)
QWERTY_ENCODED_LOW: Layout = encode_layout(QWERTY_LOW_LAYOUT)


def get_all_buttons_encoded(high_layout: Layout, low_layout: Layout) -> list[int]:
    all_buttons = []

    for layout in (low_layout, high_layout):
        for row in layout:
            all_buttons.extend(row)
    return all_buttons


ALL_BUTTONS_ENCODED = get_all_buttons_encoded(QWERTY_ENCODED_HIGH, QWERTY_ENCODED_LOW)


def get_encode_dicts() -> tuple[dict[tuple[int, int], int], dict[int, tuple[int, int]]]:
    pair_to_id = {}
    id_to_pair = {}
    counter = 0
    for i in range(134):
        for j in range(i + 1, 134):
            pair_to_id[(i, j)] = counter
            id_to_pair[counter] = (i, j)
            counter += 1
    return pair_to_id, id_to_pair


PAIR_TO_ID, ID_TO_PAIR = get_encode_dicts()
ACTION_SPACE_SIZE = len(PAIR_TO_ID)


Position = tuple[int, int]


LogType = Literal["basic"] | Literal["debug"] | Literal["error"]


class Logger:
    def __init__(self, verbose: bool = True, hide_types: list[LogType] = []) -> None:
        self.verbose = verbose
        self.hide_types = set(hide_types)

    def log(self, message: str, log_type: LogType = "basic") -> None:
        if self.verbose and log_type not in self.hide_types:
            print(message)


LOGGER = Logger()


class Finger:
    def __init__(
        self, initial_position: Position, name: str, logger: Logger = LOGGER
    ) -> None:
        self.name = name
        self.initial_position = initial_position

        self.logger = logger

        self.reset()

        # Constants

        self.wait_before_return = 4  # in ticks

        self.long_row_move_shift = 3
        self.long_row_move_penalty = 1

        self.row_penalty_coefficient = 1
        self.column_penalty_coefficient = 1.2

    def reset(self):
        self.current_position = self.initial_position
        self.ticks_before_return = 0  # if == 0, returns to the initial position
        self.typed_keys = 0

    def move(self, position: Position):
        self.current_position = position

        self.ticks_before_return = self.wait_before_return
        self.typed_keys += 1

    def tick(self) -> float:
        if self.ticks_before_return > 0:
            self.ticks_before_return -= 1

        if self.ticks_before_return == 0:
            score = self.get_score(self.initial_position)
            self.current_position = self.initial_position
            return score

        return 0

    def get_score(self, target_position: Position) -> float:
        x1, y1 = self.current_position
        x2, y2 = target_position

        row_distance = abs(x1 - x2) ** 2
        column_distance = abs(y1 - y2) ** 2

        penalty = 0
        if row_distance > self.long_row_move_shift:
            penalty = self.long_row_move_penalty
        return (
            row_distance * self.row_penalty_coefficient
            + column_distance * self.column_penalty_coefficient
            + penalty
        )

    def show_statistics(self):
        self.logger.log(
            f"Name: {self.name:22} \
            Typed keys: {self.typed_keys:5} \
            Ticks before return: {self.ticks_before_return:5} \
            Current position: {self.current_position}\t\
            Default position: {self.initial_position}"
        )


DEFAULT_FINGERS: list[Finger] = [
    Finger((2, 1), "левый мизинец"),
    Finger((2, 2), "левый безымянный"),
    Finger((2, 3), "левый средний"),
    Finger((2, 4), "левый указательный"),
    Finger((4, 3), "левый большой"),
    Finger((4, 6), "правый большой"),
    Finger((2, 7), "правый указательный"),
    Finger((2, 8), "правый средний"),
    Finger((2, 9), "правый безымянный"),
    Finger((2, 10), "правый мизинец"),
]

SwapType = Literal["low_layout"] | Literal["high_layout"] | Literal["between_layouts"]


class KeyboardLayout:
    @staticmethod
    def layout_to_dict(
        layout: Layout, unused_layout: Layout
    ) -> dict[int, list[Position]]:
        layout_dict: dict[int, list[Position]] = {}

        for i, row in enumerate(layout):
            for j, element in enumerate(row):
                button = element
                if button in layout_dict:
                    layout_dict[button].append((i, j))
                else:
                    layout_dict[button] = [(i, j)]

        for i, row in enumerate(unused_layout):
            for j, element in enumerate(row):
                button = element
                if button not in layout_dict:
                    layout_dict[button] = []

        return layout_dict

    def _finish_move(self):
        for finger in self.fingers:
            self.total_score += finger.tick()

    def __init__(self, low_layout: Layout, high_layout: Layout, logger: Logger = LOGGER):
        self.low_layout = deepcopy(low_layout)
        self.high_layout = deepcopy(high_layout)

        self.low_layout_dict = KeyboardLayout.layout_to_dict(
            self.low_layout, self.high_layout
        )
        self.high_layout_dict = KeyboardLayout.layout_to_dict(
            self.high_layout, self.low_layout
        )

        self.logger = logger

        self.fingers = deepcopy(DEFAULT_FINGERS)

        self.reset()

    def reset(self):
        self.total_score: float = 0
        self.typed_keys: int = 0
        for f in self.fingers:
            f.reset()

    def move_one_finger(
        self, positions: list[Position], busy_finger_id: Optional[int] = None
    ) -> tuple[tuple[int, Position], float]:
        best_finger_id: int = 0
        best_score = np.inf

        final_position: Position = (0, 0)

        for position in positions:
            scores = [
                finger.get_score(position) if i != busy_finger_id else np.inf
                for i, finger in enumerate(self.fingers)
            ]

            candidate_finger_id = int(np.argmin(scores))
            candidate_score = scores[candidate_finger_id]

            if candidate_score < best_score:
                best_score = candidate_score
                best_finger_id = candidate_finger_id
                final_position = position

        return (best_finger_id, final_position), best_score

    def move_two_fingers(
        self, positions: list[Position]
    ) -> tuple[tuple[int, Position], tuple[int, Position], float]:
        shift_positions = self.low_layout_dict[SHIFT_CODE]
        if len(shift_positions) == 0:
            # ERROR: SHIFT IS UNREACHABLE
            return (0, (0, 0)), (0, (0, 0)), 9999

        # firstly reach SHIFT, then - positions
        finger_shift_info_1, shift_distance_1 = self.move_one_finger(shift_positions)
        finger_btn_info_1, d1_btn = self.move_one_finger(
            positions, finger_shift_info_1[0]
        )
        total_distance_1 = shift_distance_1 + d1_btn

        # firstly reach positions, then - SHIFT
        finger_btn_info_2, d1_btn = self.move_one_finger(positions)
        finger_shift_info_2, shift_distance_2 = self.move_one_finger(
            shift_positions, finger_btn_info_2[0]
        )
        total_distance_2 = shift_distance_2 + d1_btn

        if total_distance_1 < total_distance_2:
            return finger_btn_info_1, finger_shift_info_1, total_distance_1

        return finger_btn_info_2, finger_shift_info_2, total_distance_2

    def find_button(self, button: int):
        if button in self.low_layout_dict and len(self.low_layout_dict[button]) > 0:
            (finger_id, finger_position), score = self.move_one_finger(
                self.low_layout_dict[button]
            )

            self.fingers[finger_id].move(finger_position)
            self.total_score += score
            self.typed_keys += 1

            self.logger.log(f"{button}:\t{self.fingers[finger_id].name}")

        elif button in self.high_layout_dict and len(self.high_layout_dict[button]) > 0:
            (
                (finger_id_1, finger_position_1),
                (finger_id_2, finger_position_2),
                score,
            ) = self.move_two_fingers(self.high_layout_dict[button])

            self.fingers[finger_id_1].move(finger_position_1)
            self.fingers[finger_id_2].move(finger_position_2)
            self.total_score += score
            self.typed_keys += 2

            self.logger.log(
                f"{button}:\t{self.fingers[finger_id_1].name} + {self.fingers[finger_id_2].name}"
            )

        else:
            self.logger.log(f"NO SUCH KEY: {button}")

        self._finish_move()

    def type_text(self, text: list[str]) -> float:
        for button in text:
            self.find_button(ENCODE_DICT[button])
        return self.total_score

    def type_encoded_text(self, encoded_text: list[int]) -> float:
        for button in encoded_text:
            self.find_button(button)
        return self.total_score

    def swap_buttons(self, position1: Position, position2: Position, swap_type: SwapType):
        if swap_type == "high_layout":
            layout_from = layout_to = self.high_layout
            layout_from_dict = layout_to_dict = self.high_layout_dict
        elif swap_type == "low_layout":
            layout_from = layout_to = self.low_layout
            layout_from_dict = layout_to_dict = self.low_layout_dict
        else:  # swap_type == "between_layouts"
            layout_from = self.low_layout
            layout_to = self.high_layout
            layout_from_dict = self.low_layout_dict
            layout_to_dict = self.high_layout_dict

        x1, y1 = position1
        btn1 = layout_from[x1][y1]
        x2, y2 = position2
        btn2 = layout_to[x2][y2]

        layout_from[x1][y1], layout_to[x2][y2] = layout_to[x2][y2], layout_from[x1][y1]

        layout_from_dict[btn1].remove(position1)
        layout_to_dict[btn2].remove(position2)

        layout_from_dict[btn2].append(position1)
        layout_to_dict[btn1].append(position2)

    def decode_layouts(self) -> tuple[list[list[str]], list[list[str]]]:
        return (decode_layout(self.low_layout), decode_layout(self.high_layout))

    def get_string_layouts(self) -> str:
        low_layout, high_layout = self.decode_layouts()
        result_string = "High layout:\n"
        for row in high_layout:
            for s in row:
                result_string += f"{s:8}"
            result_string += "\n"
        result_string += "\n"

        result_string += "\nLow layout:\n"
        for row in low_layout:
            for s in row:
                result_string += f"{s:8}"
            result_string += "\n"
        result_string += "\n"

        return result_string

    def show_statistics(self):
        self.logger.log("\nStatistics:")
        for f in self.fingers:
            f.show_statistics()

    def flatten(self):
        flatten = []
        for row in self.low_layout:
            flatten.extend(row)
        for row in self.high_layout:
            flatten.extend(row)

        flatten = [x / len(BUTTONS_SET) for x in flatten]
        return torch.as_tensor(flatten, dtype=torch.float32)

    def get_average_score(self) -> float:
        return self.total_score / self.typed_keys
