import sys

from core import vision
from rubik_solver import utils

# folder = 'imgs/B&W'
# folder = 'imgs/FullColor'
# folder = 'imgs/Mi_cubo'
folder = f'imgs/{sys.argv[1]}'
paths = {
    'Y' : f'{folder}/rubik_amarillo.jpg',
    'W' : f'{folder}/rubik_blanco.jpg',
    'R' : f'{folder}/rubik_rojo.jpg',
    'O' : f'{folder}/rubik_naranja.jpg',
    'B' : f'{folder}/rubik_azul.jpg',
    'G' : f'{folder}/rubik_verde.jpg',
}
print(" ------------ ESCANEANDO CUBO... ------------ ")
cube = vision.RubiksCube(paths)
print(cube)
print(" ------------ RESOLVIENDO CUBO... ----------- ")
solution_list = utils.solve(cube.to_solve_string(), 'Kociemba')
solution_list = [str(move) for move in solution_list]
solution_string = " ".join(solution_list)
print(f'SOLUCIÓN: {solution_string}')
print(f'{len(solution_list)}')

#        O B B
#        O Y B
#        R Y R

# G Y W  G G Y  B O Y  O Y Y
# B B W  G R B  W G O  G O R
# O W W  B Y G  O W R  Y R W

#        R R W
#        O W R
#        B G G
# obboybryrgywbbwowwggygrbbygboywgoowroyygoryrwrrwowrbgg
# [U, D2, R, U2, F', B', L, B, U2, D', R2, F', L2, B2, D', B2, D, R2, U, B2, D', B2]
