from itertools import combinations, permutations
import numpy as np

from constants import COLORS
from constants import R, L, U, D, F, B, X, Y, Z
from constants import ROT_X_POS, ROT_X_NEG, ROT_Y_POS, ROT_Y_NEG, ROT_Z_POS, ROT_Z_NEG

corners = [
    R+U+F,
    R+U+B,
    R+D+F,
    R+D+B,
    L+U+F,
    L+U+B,
    L+D+F,
    L+D+B
]

print(' ------------------------ COLOR COMBOS ------------------------ ')
colors = list()
for col in list(combinations(COLORS, 3)):
    if not ('W' in col and 'Y' in col):
        if not ('R' in col and 'O' in col):
            if not ('B' in col and 'G' in col):
                colors += [col]
                print(col)
print(' -------------------------------------------------------------- ')

print(' ---------------------- COLOR-POS-COMBOS ---------------------- ')

with open("core/corner_pattern_database.py", "w+") as f:
    f.write("CORNER_MAP = {\n")
    # for col in permutations(colors):
    #     combos = zip(corners, col)
    #     for combo in combos:
    #         f.write(f'\t')
    f.write("}")
print(' -------------------------------------------------------------- ')

# with open("core/corner_pattern_database.py", "w+") as f:
#     f.write("CORNER_MAP = {\n")
#     # for coso in combinations(position_color_combos, 8):
#     #     f.write(f'{coso}\n')
#     f.write("}")
CORNER_MAP = {
    ''
}