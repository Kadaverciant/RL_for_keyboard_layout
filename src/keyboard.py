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


BUTTONS_SET = get_buttons_set(QWERTY_LOW_LAYOUT, QWERTY_HIGH_LAYOUT)
KEYBOARD_LAYOUT_SHAPE = get_keyboard_shape(QWERTY_LOW_LAYOUT)
KEYS_NUMBER = sum(KEYBOARD_LAYOUT_SHAPE) * 2


def encode_decode_buttons(buttons: set[str]) -> tuple[dict[str, int], dict[int, str]]:
    letters_dict = {}
    for idx, letter in enumerate("abcdefghijklmnopqrsuvwxyz"):
        letters_dict[letter] = idx + 1
        letters_dict[letter.upper()] = -(idx + 1)

    encode_value = (len(letters_dict) // 2) + 1
    encode_dict = {}
    decode_dict = {}
    for btn in buttons:
        if btn in letters_dict:
            decode_dict[letters_dict[btn]] = btn
            encode_dict[btn] = letters_dict[btn]
        else:
            decode_dict[encode_value] = btn
            encode_dict[btn] = encode_value
            encode_value += 1
    return encode_dict, decode_dict


ENCODE_DICT, DECODE_DICT = encode_decode_buttons(BUTTONS_SET)
ENCODED_BUTTONS_SET = {ENCODE_DICT[btn] for btn in BUTTONS_SET}
SHIFT_CODE = ENCODE_DICT["<shift>"]


SPECIAL_SYMBOLS = {
    "\t": "<tab>",
    "\n": "<enter>",
    " ": "<space>",
}
