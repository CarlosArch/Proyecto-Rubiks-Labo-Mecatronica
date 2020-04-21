"""
Rubik's Cube Solver using a problem solving algorithm. (ATTEMPT FAILED AT IDA*)
"""
from itertools import permutations
from copy import deepcopy

import numpy as np

from constants import COLORS
from constants import R, L, U, D, F, B, X, Y, Z
from constants import ROT_X_POS, ROT_X_NEG, ROT_Y_POS, ROT_Y_NEG, ROT_Z_POS, ROT_Z_NEG

from objects import RubiksCube

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
            self._h_func = problem._h_func
            return

        self.state = RubiksCube(state)
        self.path = path
        self.cost = cost
        self._h_func = '_manhattan_distance_edges'
        for name, value in kwargs.items():
            setattr(self, name, value)

    # @property
    # def hval(self):
    #     return getattr(self, self._h_func)()

    # @property
    # def fval(self):
    #     return (self.cost + self.hval)

    # def _manhattan_distance_edges(self):
    #     return

    def solve(self):
        self._solve_front_layer()
        return

    def iterative_deepening_depth_search(self, test, max_depth = 10):
        front = [self]
        current_depth = 0

        while True:
            if not front:
                if current_depth < max_depth:
                    front = []
                    current_depth += 1

                    for seq in permutations(self.actions, current_depth):
                        child = RubiksProblem(self)
                        seq = " ".join(seq)
                        child.state.sequence(seq)
                        child.path += f' {seq}'
                        child.cost += current_depth
                        front += [child]
                else:
                    raise Exception(f"Couldn't find a solution at depth {current_depth}")

            node = front.pop()

            # print(f'Depth: {current_depth},\t cost: {node.cost},\t path: {node.path}')

            if test(node.state):
                return RubiksProblem(node)

    @property
    def h_val(self):
        r_col = self.state.color(R)
        l_col = self.state.color(L)
        u_col = self.state.color(U)
        d_col = self.state.color(D)
        f_col = self.state.color(F)
        b_col = self.state.color(B)

        fr = self.state.cubie_from_colors(f_col, r_col)
        fl = self.state.cubie_from_colors(f_col, l_col)
        fu = self.state.cubie_from_colors(f_col, u_col)
        fd = self.state.cubie_from_colors(f_col, d_col)

        h = 0
        h += 1 if fr == Cubie(pos=F+R, col=[r_col, None, f_col]) else 0
        h += 1 if fl == Cubie(pos=F+L, col=[l_col, None, f_col]) else 0
        h += 1 if fu == Cubie(pos=F+U, col=[None, u_col, f_col]) else 0
        h += 1 if fd == Cubie(pos=F+D, col=[None, d_col, f_col]) else 0

        if h < 4:
            return 26 - h

        return 26 - h


    def IDA_star(self, max_depth = 26):
        front = [self]
        current_depth = 0
        best = None

        while True:
            if not front:
                if current_depth < max_depth:
                    front = [self]
                    current_depth += 1

            node = front.pop()

            print(f'Cost: {node.cost},\t H_Val: {node.h_val},\t Path: {node.path}')

            if node.h_val == 26 - 4:
                return RubiksProblem(node)

            if best is None:
                best = node.h_val


            if current_depth >= max_depth:
                raise Exception(f"Couldn't find a solution at depth {current_depth}")

            if node.cost < current_depth:
                for move in self.actions:
                    child = RubiksProblem(node)
                    child.state.move(move)
                    child.path += f' {move}'
                    child.cost += 1
                    front += [child]

                    if best >= child.h_val:
                        best = child.h_val
                    # else:
                    #     front.sort(key=lambda x: x.h_val, reverse=True)

    def _is_front_cross_solved(self, state):
        front_color = state.color(F)
        right_color = state.color(R)
        left_color = state.color(L)
        up_color = state.color(U)
        down_color = state.color(D)

        front_cubies = state._cubies_from_face(F)
        for cubie in front_cubies:
            if cubie.typ == 'face':
                if not cubie.col[2] == front_color:
                    return False
            elif cubie.typ == 'edge':
                if cubie.pos[0] == 1:
                    if not cubie.col[0] == right_color:
                        return False
                elif cubie.pos[0] == -1:
                    if not cubie.col[0] == left_color:
                        return False
                elif cubie.pos[1] == 1:
                    if not cubie.col[1] == up_color:
                        return False
                elif cubie.pos[1] == -1:
                    if not cubie.col[1] == down_color:
                        return False
        return True

    def _is_front_solved(self, state):
        front_color = state.color(F)
        right_color = state.color(R)
        left_color = state.color(L)
        up_color = state.color(U)
        down_color = state.color(D)

        front_cubies = state._cubies_from_face(F)
        if not all(cubie.col[2] == front_color for cubie in front_cubies):
            return False
        if not all(cubie.col[0] == right_color for cubie in front_cubies if cubie.pos[0] == 1):
            return False
        if not all(cubie.col[0] == left_color for cubie in front_cubies if cubie.pos[0] == -1):
            return False
        if not all(cubie.col[1] == up_color for cubie in front_cubies if cubie.pos[1] == 1):
            return False
        if not all(cubie.col[1] == down_color for cubie in front_cubies if cubie.pos[1] == -1):
            return False
        return True

    def _solve_front_layer(self):
        result = self.iterative_deepening_depth_search(self._is_front_cross_solved, max_depth=10)
        print(f"result after cross: and path {result.path}")
        print(result.state)
        # result = self.iterative_deepening_depth_search(self._is_front_solved, max_depth=10)
        # print(f"result front face:")
        # print(result.state)

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
    solved.solve()
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
    almost_solved.solve()
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
    almost_solved_2.solve()
    print(' ------------------------------------------------------------- ')
    print(' ------------------------- SCRAMBLED ------------------------- ')
    scrambled = RubiksProblem(
        state='''
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
    print(scrambled.state)
    print(' -------------------------- SOLVING -------------------------- ')
    scrambled.solve()
    print(' ------------------------------------------------------------- ')
