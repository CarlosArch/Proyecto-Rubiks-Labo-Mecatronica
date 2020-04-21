"""
Rubik's Cube objects "Cubie" and "Cube".

Based on https://github.com/pglass/cube
"""
import string
from typing import Union, Type
from copy import deepcopy

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
    [ 1, 0, 1],
    [ 0, 1, 0],
    [-1, 0, 0],
    ])
ROT_Y_NEG = np.array([ # On Y axis, clockwise.
    [ 1, 0,-1],
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

class Cubie():
    """
    Each piece of the Rubik's cube is a "cubie".

    Parameters:
    -----------
    pos: array-like
        Current cubie's position as a (x, y, z) array.

    col: array-like
        Colors on each of the cubie's visible faces.
        Can only be one of W, Y, R, G, B, O or None.

    Attributes:
    -----------
    pos: np.ndarray
        Current cubie's position as a (x, y, z) array.

    col: np.ndarray
        Colors on each of the cubie's visible faces.

    typ: str
        Type of cubie (face, edge or corner)

    Methods:
    --------
    rotate(rotation_matrix)
    """
    colors = COLORS

    def __init__(self, pos: list, col: list):
        assert len(pos) == 3, \
            f'pos must have 3 elements, it has {len(pos)}.'
        assert all(x in (-1, 0, 1) for x in pos), \
            f'pos not in range (-1, 0, 1), pos is {pos}.'

        assert len(col) == 3, \
            f'col must be a have 3 elements, it has {len(col)}.'
        assert all(c in self.colors or c is None for c in col), \
            f'col has invalid colors (must be one of {self.colors} or None), col is {col}.'

        self.pos = np.array(pos, dtype='i1')
        self.col = np.array(col, dtype='O')

    @property
    def typ(self):
        c = np.count_nonzero(self.col)
        if c == 1:
            return 'face'
        elif c == 2:
            return 'edge'
        elif c == 3:
            return 'corner'
        else:
            raise AttributeError(f'Incorrect cubie, has {c} colors.')

    def rotate(self, rotation_matrix):
        """
        Rotate the cubie.

        Parameters:
        -----------
        rotation_matrix: array-like
            3D rotation matrix.

        https://en.wikipedia.org/wiki/Rotation_matrix
        """
        # Rotate position
        self.pos = np.dot(rotation_matrix, self.pos)

        # Rotate color
        col_x, col_y, col_z = self.col
        self.col[np.dot(rotation_matrix, X).astype(bool)] = col_x
        self.col[np.dot(rotation_matrix, Y).astype(bool)] = col_y
        self.col[np.dot(rotation_matrix, Z).astype(bool)] = col_z

    def __repr__(self):
        return f'Cubie(type: {self.typ}, pos: {self.pos}, col: {self.col})'

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return np.array_equiv(self.pos, other.pos) and np.array_equiv(self.col, other.col)

class Cube():
    """
    Model of the Rubik's cube, composed of 26 cubies.

    Parameters:
    -----------
    cube: str or Cube
        String representation of cube or another Cube to copy.
        format:
                   U U U
                   U U U
                   U U U

            L L L  F F F  R R R  B B B
            L L L  F F F  R R R  B B B
            L L L  F F F  R R R  B B B

                   D D D
                   D D D
                   D D D

    Attributes:
    -----------
    cubies: tuple of Cubie
        Every cubie in the Cube.

    Methods:
    --------
    None
    """
    colors = COLORS

    def __init__(self, cube: str):
        if isinstance(cube, Cube):
            self.cubies = deepcopy(cube).cubies
            return

        for s in string.whitespace:
            cube = cube.replace(s, '')
        for color in self.colors:
            assert ((c := cube.count(color)) == 9), \
                f'Incorrect number of {color}, has {c} but must have 9'
        #            0  1  2
        #            3  4  5
        #            6  7  8

        #  9 10 11  12 13 14  15 16 17  18 19 20
        # 21 22 23  24 25 26  27 28 29  30 31 32
        # 33 34 35  36 37 38  39 40 41  42 43 44

        #           45 46 47
        #           48 49 50
        #           51 52 53
        self.cubies = (
            #Faces
            Cubie(pos=R, col=[cube[28], None, None]),
            Cubie(pos=L, col=[cube[22], None, None]),
            Cubie(pos=U, col=[None, cube[4],  None]),
            Cubie(pos=D, col=[None, cube[49], None]),
            Cubie(pos=F, col=[None, None, cube[25]]),
            Cubie(pos=B, col=[None, None, cube[31]]),
            # Edges
            Cubie(pos=R+U, col=[cube[16], cube[5], None]),
            Cubie(pos=R+D, col=[cube[40], cube[50], None]),
            Cubie(pos=R+F, col=[cube[27], None, cube[26]]),
            Cubie(pos=R+B, col=[cube[29], None, cube[30]]),
            Cubie(pos=L+U, col=[cube[10], cube[3], None]),
            Cubie(pos=L+D, col=[cube[34], cube[48], None]),
            Cubie(pos=L+F, col=[cube[23], None, cube[24]]),
            Cubie(pos=L+B, col=[cube[21], None, cube[32]]),
            Cubie(pos=U+F, col=[None, cube[7], cube[13]]),
            Cubie(pos=U+B, col=[None, cube[1], cube[19]]),
            Cubie(pos=D+F, col=[None, cube[46], cube[37]]),
            Cubie(pos=D+B, col=[None, cube[52], cube[43]]),
            # Corners
            Cubie(pos=R+U+F, col=[cube[15], cube[8], cube[14]]),
            Cubie(pos=R+U+B, col=[cube[17], cube[2], cube[18]]),
            Cubie(pos=R+D+F, col=[cube[39], cube[47], cube[38]]),
            Cubie(pos=R+D+B, col=[cube[41], cube[53], cube[42]]),
            Cubie(pos=L+U+F, col=[cube[11], cube[6], cube[12]]),
            Cubie(pos=L+U+B, col=[cube[9], cube[0], cube[20]]),
            Cubie(pos=L+D+F, col=[cube[35], cube[45], cube[36]]),
            Cubie(pos=L+D+B, col=[cube[33], cube[51], cube[44]]),
        )

    @property
    def is_solved(self):
        all_same = lambda obj: len(set(obj)) == 1
        if not all_same([cubie.col[0] for cubie in self.get_face(R)]):
            return False
        if not all_same([cubie.col[0] for cubie in self.get_face(L)]):
            return False
        if not all_same([cubie.col[1] for cubie in self.get_face(U)]):
            return False
        if not all_same([cubie.col[1] for cubie in self.get_face(D)]):
            return False
        if not all_same([cubie.col[2] for cubie in self.get_face(F)]):
            return False
        if not all_same([cubie.col[2] for cubie in self.get_face(B)]):
            return False
        return True

    def get_face(self, axis):
        mask = axis.astype(bool)
        return [c for c in self.cubies if np.all(c.pos[mask] == axis[mask])]

    def rotate(self, face, rotation_matrix):
        for cubie in self.get_face(face):
            cubie.rotate(rotation_matrix)

    def R(self): self.rotate(R, ROT_X_NEG)
    def L(self): self.rotate(L, ROT_X_POS)
    def U(self): self.rotate(U, ROT_Y_NEG)
    def D(self): self.rotate(D, ROT_Y_POS)
    def F(self): self.rotate(F, ROT_Z_NEG)
    def B(self): self.rotate(B, ROT_Z_POS)

    def Ri(self): self.rotate(R, ROT_X_POS)
    def Li(self): self.rotate(L, ROT_X_NEG)
    def Ui(self): self.rotate(U, ROT_Y_POS)
    def Di(self): self.rotate(D, ROT_Y_NEG)
    def Fi(self): self.rotate(F, ROT_Z_POS)
    def Bi(self): self.rotate(B, ROT_Z_NEG)

    def sequence(self, moves: str):
        """
        Parameters:
        -----------
        moves: str
            Sequence of moves to follow.
            Example: "R' D' R D"
        """
        moves = [move.replace("'", "i") for move in moves.split()]
        for move in moves:
            getattr(self, move)()

    def __str__(self):
        r = [c.col[0] for c in sorted(self.get_face(R), key=lambda c: (-c.pos[1], -c.pos[2]))]
        l = [c.col[0] for c in sorted(self.get_face(L), key=lambda c: (-c.pos[1], c.pos[2]))]
        u = [c.col[1] for c in sorted(self.get_face(U), key=lambda c: (c.pos[2], c.pos[0]))]
        d = [c.col[1] for c in sorted(self.get_face(D), key=lambda c: (-c.pos[2], c.pos[0]))]
        f = [c.col[2] for c in sorted(self.get_face(F), key=lambda c: (-c.pos[1], c.pos[0]))]
        b = [c.col[2] for c in sorted(self.get_face(B), key=lambda c: (-c.pos[1], -c.pos[0]))]

        return \
            f'       {u[0]} {u[1]} {u[2]}\n'\
            f'       {u[3]} {u[4]} {u[5]}\n'\
            f'       {u[6]} {u[7]} {u[8]}\n\n'\
            f'{l[0]} {l[1]} {l[2]}  {f[0]} {f[1]} {f[2]}  {r[0]} {r[1]} {r[2]}  {b[0]} {b[1]} {b[2]}\n'\
            f'{l[3]} {l[4]} {l[5]}  {f[3]} {f[4]} {f[5]}  {r[3]} {r[4]} {r[5]}  {b[3]} {b[4]} {b[5]}\n'\
            f'{l[6]} {l[7]} {l[8]}  {f[6]} {f[7]} {f[8]}  {r[6]} {r[7]} {r[8]}  {b[6]} {b[7]} {b[8]}\n\n'\
            f'       {d[0]} {d[1]} {d[2]}\n'\
            f'       {d[3]} {d[4]} {d[5]}\n'\
            f'       {d[6]} {d[7]} {d[8]}'

    def __repr__(self):
        return f'Cube(solved: {self.is_solved})'

    def __getitem__(self, *args):
        for cubie in self.cubies:
            if np.array_equiv(cubie.pos, args):
                return cubie

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return all(cubie == other[cubie.pos] for cubie in self.cubies)

if __name__ == "__main__":
    def check_cubies():
        cubie_corner = Cubie(
            pos=[1, 1, 1],
            col=['Y', 'B', 'R']
            )
        print(f'{cubie_corner.typ} pos1: {cubie_corner.pos}')
        print(f'{cubie_corner.typ} col1: {cubie_corner.col}')
        cubie_corner.rotate(ROT_Z_POS)
        print(f'{cubie_corner.typ} pos2: {cubie_corner.pos}')
        print(f'{cubie_corner.typ} col2: {cubie_corner.col}')

        cubie_edge = Cubie(
            pos=[0, 1, 1],
            col=[None, 'R', 'B']
            )
        print(f'{cubie_edge.typ} pos1: {cubie_edge.pos}')
        print(f'{cubie_edge.typ} col1: {cubie_edge.col}')
        cubie_edge.rotate(ROT_Z_POS)
        print(f'{cubie_edge.typ} pos2: {cubie_edge.pos}')
        print(f'{cubie_edge.typ} col2: {cubie_edge.col}')

        cubie_face = Cubie(
            pos=[0, 0, 1],
            col=[None, None, 'B']
            )
        print(f'{cubie_face.typ} pos1: {cubie_face.pos}')
        print(f'{cubie_face.typ} col1: {cubie_face.col}')
        cubie_face.rotate(ROT_Y_POS)
        print(f'{cubie_face.typ} pos2: {cubie_face.pos}')
        print(f'{cubie_face.typ} col2: {cubie_face.col}')

    def check_cube():
        print('------------- SOLVED CUBE -------------')
        cube_solved = Cube(
            cube='''
                   R R R
                   R R R
                   R R R
            B B B  W W W  G G G  Y Y Y
            B B B  W W W  G G G  Y Y Y
            B B B  W W W  G G G  Y Y Y
                   O O O
                   O O O
                   O O O
            '''
        )
        print(cube_solved)
        print(f'Cube solved? {cube_solved.is_solved}')
        print('---------------------------------------')

        print()

        print('----------- SCRAMBLED CUBE -----------')
        cube_scrambled = Cube(
            cube='''
                   G G O
                   B R W
                   Y Y Y
            R B Y  B G R  W Y R  W O B
            R B O  B W O  G G R  B Y O
            W R O  G R B  G W Y  W Y O
                   B W G
                   G O W
                   R Y O
            '''
        )
        print(cube_scrambled)
        print(f'Cube solved? {cube_scrambled.is_solved}')
        print('---------------------------------------')

        print()

        print('------- ROTATED SCRAMBLED CUBE -------')
        rot_scrambled = Cube(cube=cube_scrambled)
        rot_scrambled.F()
        print(rot_scrambled)
        print(rot_scrambled[1, 1, 1])
        print(f'Cube solved? {rot_scrambled.is_solved}')
        returned_to_scrambled = Cube(cube=rot_scrambled)
        returned_to_scrambled.Fi()
        print(f'Is equal after return? {returned_to_scrambled == cube_scrambled}')
        print('---------------------------------------')
