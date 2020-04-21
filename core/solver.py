"""
Rubik's Cube Solver using a problem solving algorithm.
"""
from itertools import permutations
from copy import deepcopy

import numpy as np

from constants import COLORS
from constants import R, L, U, D, F, B, X, Y, Z
from constants import ROT_X_POS, ROT_X_NEG, ROT_Y_POS, ROT_Y_NEG, ROT_Z_POS, ROT_Z_NEG

from objects import RubiksCube, Cubie

class RubiksProblem:
    actions = (
        "R", "R'",
        "L", "L'",
        "U", "U'",
        "D", "D'",
        "F", "F'",
        "B", "B'",
        )
    def __init__(self, state, path: str = 'initial', cost: int = 0, **kwargs):
        if isinstance(state, self.__class__):
            problem = deepcopy(state)
            self.state = problem.state
            self.path = problem.path
            self.cost = problem.cost
            return

        self.state = RubiksCube(state)
        self.path = path
        self.cost = cost

    def solve(self):
        self._solve_front_cross()
        # self._solve_front_corners()
        # self._solve_second_layer()
        # self._solve_back_edges()
        # self._solve_back_cross()
        # self._solve_back_position_corners()
        # self._solve_back_orient_corners()
        return

    def _solve_front_cross(self):
        STEPS = {
            'U': {
                'R': "",
                'L': "",
                'F': "",
                'B': "",
            },
            'D': {
                'R': "R R",
                'L': "L R",
                'F': "F R",
                'B': "B R"
            },
            'F': {
                'U': "F R U' R' F'",
                'D': "F' R U' R'",
                'R': "R U R'",
                'L': "L' U' L",
            },
            'B': {
                'U': "B L U' L' B'",
                'D': "B R' U R",
                'R': "R' U R",
                'L': "L U' L'",
            },
            'L': {
                'U': "L F U' F' L'",
                'D': "L' F U' F'",
                'F': "F U' F'",
                'B': "B' U B",
            },
            'R': {
                'U': "R' F' U F R",
                'D': "R F' U F",
                'F': "F' U F",
                'B': "B U' B'",
            }
        }
        for color in 'RGOB':
            cubie_position = self.state.cubie_from_colors('W', color)

            orig_cubie = self.state[cubie_position]
            white_facing = orig_cubie.('W')
            color_facing = orig_cubie.color_facing(color)
            step_solution = WhiteCrossSolver.first_step(white_facing, color_facing)
            # First goal is to put white sticker on top face

            for m in step_solution:
                self.cube.move(Move(m))
            solution.extend(step_solution)

            # Second goal is to place the cubie on the top over its place
            while self.cube.cubies['FU'].facings['U'] != 'W' or self.cube.cubies['FU'].facings['F'] != color:
                solution.append('U')
                self.cube.move(Move('U'))
            # Third goal will be a F2 movement
            solution.append("F2")
            self.cube.move(Move("F2"))
            solution.append('Y')
            self.cube.move(Move("Y"))




    def _solve_front_corners(self):
        pass

    def _solve_second_layer(self):
        pass

    def _solve_back_edges(self):
        pass

    def _solve_back_cross(self):
        pass

    def _solve_back_position_corners(self):
        pass

    def _solve_back_orient_corners(self):
        pass






if __name__ == "__main__":
    print(' -------------------------- SOLVED --------------------------- ')
    solved = RubiksProblem(
        state='''
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
    print(solved.state)
    print(' -------------------------- SOLVING -------------------------- ')
    print(solved.IDA_star().state)
    print(' ------------------------------------------------------------- ')
    print(' ----------------------- ALMOST SOLVED ----------------------- ')
    almost_solved = RubiksProblem(
        state='''
               R R W
               R R W
               R R W
        B B B  W W O  G G G  R Y Y
        B B B  W W O  G G G  R Y Y
        B B B  W W O  G G G  R Y Y
               O O Y
               O O Y
               O O Y
        '''
        )
    print(almost_solved.state)
    print(' -------------------------- SOLVING -------------------------- ')
    print(almost_solved.IDA_star().state)
    print(' ------------------------------------------------------------- ')
    print(' ---------------------- ALMOST SOLVED 2 ---------------------- ')
    almost_solved_2 = RubiksProblem(
        state='''
               W W W
               R R R
               R R R
        R Y Y  B B B  W W O  G G G
        B B B  W W O  G G G  R Y Y
        B B B  W W O  G G G  R Y Y
               O O Y
               O O Y
               O O Y
        '''
        )
    print(almost_solved_2.state)
    print(' -------------------------- SOLVING -------------------------- ')
    print(almost_solved_2.IDA_star().state)
    print(' ------------------------------------------------------------- ')
    print(' ------------------------- SCRAMBLED ------------------------- ')
    scrambled = RubiksProblem(
        state='''
               G G O
               B R W
               Y Y Y
        Y O O  B G R  G G W  B O W
        B B R  B W O  W G Y  O Y B
        R R W  G R B  Y R R  O Y W
               O Y R
               W O G
               G W B
        '''
        )
    print(scrambled.state)
    print(' -------------------------- SOLVING -------------------------- ')
    print(scrambled.IDA_star().state)
    print(' ------------------------------------------------------------- ')
