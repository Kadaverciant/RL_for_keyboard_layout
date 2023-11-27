"""
original ={["~\n`","!\n1","@\n2","#\n3","$\n4","%\n5","^\n6","&\n7","*\n8","(\n9",")\n0","_\n-","+\n=",{w:2},"Backspace"],
[{w:1.5},"Tab","Q","W","E","R","T","Y","U","I","O","P","{\n[","}\n]",{w:1.5},"|\n\\"],
[{w:1.75},"Caps Lock","A","S","D","F","G","H","J","K","L",":\n;","\"\n'",{w:2.25},"Enter"],
[{w:2.25},"Shift","Z","X","C","V","B","N","M","<\n,",">\n.","?\n/",{w:2.75},"Shift"],
[{w:1.25},"Ctrl",{w:1.25},"Win",{w:1.25},"Alt",{a:7,w:6.25},"",{a:4,w:1.25},"Alt",{w:1.25},"Win",{w:1.25},"Menu",{w:1.25},"Ctrl"]}
"""
keys_names = {
    "<shift>": "Shift",
    '<enter>': "Enter",
    "<alt>": "Alt",
    "<back>": "BackSpace",
    "<ctrl>": "Ctrl",
    '<tab>': "Tab",
    '<caps>': "CapsLock",
    "<space>": "Space"
}
keyboard1 = ([['`', '1', '2', 'B', '4', '5', '<ctrl>', '7', '8', '9', '0', '3', '=', 'O'],
  ['S', 'q', 'w', 'e', 'r', 'G', 'Z', 'u', 'i', 'o', 'p', '[', ']', '\\'],
  ['N', 'a', 's', 'd', '6', 'g', 'h', 'V', 'k', 'l', ';', "'", 'U', '<enter>'],
  ['<shift>',
   '<shift>',
   'z',
   'x',
   'c',
   'M',
   'b',
   'n',
   'm',
   ',',
   '.',
   '/',
   '<shift>',
   '<shift>'],
  ['<shift>',
   '<alt>',
   '<space>',
   '<space>',
   '<space>',
   '<space>',
   '<enter>',
   '<space>',
   '<space>',
   '_',
   '<ctrl>']],
 [['~',
   '!',
   '@',
   '<space>',
   '$',
   '%',
   '^',
   '>',
   '*',
   'I',
   'A',
   '<alt>',
   '+',
   '<back>'],
  ['<tab>',
   '-',
   'R',
   'E',
   'W',
   'T',
   'Y',
   '<enter>',
   '(',
   '<shift>',
   'P',
   '{',
   '}',
   '|'],
  ['<caps>',
   ')',
   '<tab>',
   'D',
   'F',
   '<space>',
   '&',
   'J',
   'K',
   'L',
   ':',
   '"',
   '<enter>',
   '<space>'],
  ['<ctrl>',
   '<shift>',
   'y',
   'X',
   't',
   'j',
   'Q',
   '<caps>',
   'v',
   '<',
   'H',
   '?',
   '<back>',
   '<shift>'],
  ['<ctrl>',
   '<alt>',
   'C',
   'f',
   '<space>',
   '<space>',
   '<space>',
   '<space>',
   '<space>',
   '<alt>',
   '#']])
keyboard2 = ([['`', '1', '2', '3', 'Y', '5', '6', 'Q', '8', '9', '0', '-', '=', '!'],
  ['<', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'j', '[', ']', '\\'],
  ['<caps>',
   'a',
   's',
   '<shift>',
   'f',
   'g',
   'h',
   '<enter>',
   'T',
   'S',
   ';',
   "'",
   '<enter>',
   '<enter>'],
  ['<shift>',
   'B',
   'z',
   'x',
   'c',
   'v',
   'b',
   'n',
   'm',
   'F',
   '.',
   '/',
   '<shift>',
   '<shift>'],
  ['<ctrl>',
   '<alt>',
   '<space>',
   '<space>',
   '<space>',
   '"',
   '<space>',
   '<enter>',
   '@',
   '<alt>',
   '<ctrl>']],
 [['~',
   '*',
   '<space>',
   '#',
   '$',
   '%',
   '^',
   '&',
   '<back>',
   '(',
   ')',
   '_',
   '+',
   '<back>'],
  ['<tab>', '7', 'W', 'E', 'R', ':', '4', 'U', 'I', 'O', 'P', '{', '}', '|'],
  ['<caps>',
   'A',
   'l',
   'D',
   ',',
   'G',
   'V',
   'J',
   '<space>',
   'L',
   'k',
   '<space>',
   'K',
   '<space>'],
  ['d',
   '<shift>',
   'Z',
   'X',
   'C',
   'H',
   '<shift>',
   'N',
   'M',
   '<tab>',
   '>',
   '?',
   '<shift>',
   '<shift>'],
  ['<ctrl>',
   '<alt>',
   '<space>',
   '<space>',
   '<space>',
   '<space>',
   'p',
   '<space>',
   '<space>',
   '<alt>',
   '<ctrl>']])


def create_keyboard_json(keyboard, keyboard_name):
    keyboard_arr = []
    for u, d in zip(keyboard[0], keyboard[1]):
        arr = []
        for i in range(len(u)):
            new_line = d[i] if d[i] not in keys_names.keys() else keys_names[d[i]]
            new_line += "\n"
            new_line += u[i] if u[i] not in keys_names.keys() else keys_names[u[i]]
            arr.append(new_line)
        keyboard_arr.append(arr)
    with open(keyboard_name, 'w') as f:
        for elem in keyboard_arr:
            f.write("" + str(elem).replace("'", '"') + "")
            f.write("\n")
    print(keyboard_arr)

create_keyboard_json(keyboard1, "keyboard_layout1.txt")
create_keyboard_json(keyboard2, "keyboard_layout2.txt")
