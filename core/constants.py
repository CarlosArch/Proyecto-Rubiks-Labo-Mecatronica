import numpy as np

COLORS = ('W', 'Y', 'R', 'G', 'B', 'O')

R = X = np.array([ 1, 0, 0]) # Right
L     = np.array([-1, 0, 0]) # Left
U = Y = np.array([ 0, 1, 0]) # Up
D     = np.array([ 0,-1, 0]) # Down
F = Z = np.array([ 0, 0, 1]) # Front
B     = np.array([ 0, 0,-1]) # Back

ROT_X_POS = np.array([ # On X axis, counter-clockwise.
    [ 1, 0, 0],
    [ 0, 0, 1],
    [ 0,-1, 0],
    ])
ROT_X_NEG = np.array([ # On X axis, clockwise.
    [ 1, 0, 0],
    [ 0, 0,-1],
    [ 0, 1, 0],
    ])
ROT_Y_POS = np.array([ # On Y axis, counter-clockwise.
    [ 0, 0, 1],
    [ 0, 1, 0],
    [-1, 0, 0],
    ])
ROT_Y_NEG = np.array([ # On Y axis, clockwise.
    [ 0, 0,-1],
    [ 0, 1, 0],
    [ 1, 0, 0],
    ])
ROT_Z_POS = np.array([ # On Z axis, counter-clockwise.
    [ 0,-1, 0],
    [ 1, 0, 0],
    [ 0, 0, 1],
    ])
ROT_Z_NEG = np.array([ # On Z axis, clockwise.
    [ 0, 1, 0],
    [-1, 0, 0],
    [ 0, 0, 1],
    ])
