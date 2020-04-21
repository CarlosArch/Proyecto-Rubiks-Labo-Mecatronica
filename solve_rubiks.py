from core import vision
from rubik_solver import utils

# folder = 'imgs/B&W'
# folder = 'imgs/FullColor'
folder = 'imgs/Mi_cubo'
paths = {
    'Y' : f'{folder}/rubik_amarillo.jpg',
    'W' : f'{folder}/rubik_blanco.jpg',
    'R' : f'{folder}/rubik_rojo.jpg',
    'O' : f'{folder}/rubik_naranja.jpg',
    'B' : f'{folder}/rubik_azul.jpg',
    'G' : f'{folder}/rubik_verde.jpg',
}
cube = vision.RubiksCube(paths)
print(" ------------ ESCANEANDO CUBO... ------------ ")
print(cube)
print(" ------------ RESOLVIENDO CUBO... ----------- ")
print(utils.solve(cube.to_solve_string(), 'Kociemba'))

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
