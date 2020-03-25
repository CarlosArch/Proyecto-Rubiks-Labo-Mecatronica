import sys
import copy
from operator import itemgetter

import tkinter as tk
import numpy as np

class Cubo:
    colores = {
        0 : 'nada',
        1 : 'blanco',
        2 : 'rojo',
        3 : 'verde',
        4 : 'naranja',
        5 : 'azul',
        6 : 'amarillo',
        }

    acciones = (
        ('F', 'der'),
        ('F', 'izq'),
        ('B', 'der'),
        ('B', 'izq'),
        ('U', 'der'),
        ('U', 'izq'),
        ('D', 'der'),
        ('D', 'izq'),
        ('L', 'der'),
        ('L', 'izq'),
        ('R', 'der'),
        ('R', 'izq'),
        )

    def __init__(self,
                 caras: dict,
                 camino: list = ['inicial'],
                 costo: int = 0,
                 heuristica: str = '_fuera_de_lugar'):
        self.caras = caras
        self.camino = camino
        self.costo = costo
        self.funcion_h = heuristica

    def rotar(self, cara: str, direccion: str):
        '''
        Regresa una copia del cubo original, con una cara rotada.
        '''
        hijo = copy.deepcopy(self)
        aph = "'" if direccion == 'izq' else ""
        hijo.camino += [f'{cara}{aph}']
        hijo.costo += 1

        # Esta es la cosa más horrible que he hecho en un buen rato.
        if cara == 'F':
            if direccion == 'der':
                hijo.caras['F'] = np.rot90(self.caras['F'], 3)

                hijo.caras['U'][2, :] = self.caras['L'][2, :]
                hijo.caras['L'][2, :] = self.caras['D'][2, :]
                hijo.caras['D'][2, :] = self.caras['R'][2, :]
                hijo.caras['R'][2, :] = self.caras['U'][2, :]
            if direccion == 'izq':
                hijo.caras['F'] = np.rot90(self.caras['F'], 1)

                hijo.caras['U'][2, :] = self.caras['R'][2, :]
                hijo.caras['R'][2, :] = self.caras['D'][2, :]
                hijo.caras['D'][2, :] = self.caras['L'][2, :]
                hijo.caras['L'][2, :] = self.caras['U'][2, :]
        if cara == 'B':
            if direccion == 'der':
                hijo.caras['B'] = np.rot90(self.caras['B'], 3)

                hijo.caras['R'][0, :] = self.caras['D'][0, :]
                hijo.caras['D'][0, :] = self.caras['L'][0, :]
                hijo.caras['L'][0, :] = self.caras['U'][0, :]
                hijo.caras['U'][0, :] = self.caras['R'][0, :]
            if direccion == 'izq':
                hijo.caras['B'] = np.rot90(self.caras['B'], 1)

                hijo.caras['R'][0, :] = self.caras['U'][0, :]
                hijo.caras['U'][0, :] = self.caras['L'][0, :]
                hijo.caras['L'][0, :] = self.caras['D'][0, :]
                hijo.caras['D'][0, :] = self.caras['R'][0, :]
        if cara == 'U':
            if direccion == 'der':
                hijo.caras['U'] = np.rot90(self.caras['U'], 3)

                hijo.caras['B'][:, 2] = self.caras['L'][:, 2]
                hijo.caras['L'][:, 2] = self.caras['F'][0, :]
                hijo.caras['F'][0, :] = self.caras['R'][:, 0]
                hijo.caras['R'][:, 0] = self.caras['B'][:, 2]
            if direccion == 'izq':
                hijo.caras['U'] = np.rot90(self.caras['U'], 1)

                hijo.caras['B'][:, 2] = self.caras['R'][:, 0]
                hijo.caras['R'][:, 0] = self.caras['F'][0, :]
                hijo.caras['F'][0, :] = self.caras['L'][:, 2]
                hijo.caras['L'][:, 2] = self.caras['B'][:, 2]
        if cara == 'D':
            if direccion == 'der':
                hijo.caras['D'] = np.rot90(self.caras['D'], 3)

                hijo.caras['B'][:, 0] = self.caras['R'][:, 2]
                hijo.caras['R'][:, 2] = self.caras['F'][2, :]
                hijo.caras['F'][2, :] = self.caras['L'][:, 0]
                hijo.caras['L'][:, 0] = self.caras['B'][:, 0]
            if direccion == 'izq':
                hijo.caras['D'] = np.rot90(self.caras['D'], 1)

                hijo.caras['B'][:, 0] = self.caras['L'][:, 0]
                hijo.caras['L'][:, 0] = self.caras['F'][2, :]
                hijo.caras['F'][2, :] = self.caras['R'][:, 2]
                hijo.caras['R'][:, 2] = self.caras['B'][:, 0]
        if cara == 'L':
            if direccion == 'der':
                hijo.caras['L'] = np.rot90(self.caras['L'], 3)

                hijo.caras['B'][2, :] = self.caras['D'][:, 2]
                hijo.caras['D'][:, 2] = self.caras['F'][:, 0]
                hijo.caras['F'][:, 0] = self.caras['U'][:, 0]
                hijo.caras['U'][:, 0] = self.caras['B'][2, :]
            if direccion == 'izq':
                hijo.caras['L'] = np.rot90(self.caras['L'], 1)

                hijo.caras['B'][2, :] = self.caras['U'][:, 0]
                hijo.caras['U'][:, 0] = self.caras['F'][:, 0]
                hijo.caras['F'][:, 0] = self.caras['D'][:, 2]
                hijo.caras['D'][:, 2] = self.caras['B'][2, :]
        if cara == 'R':
            if direccion == 'der':
                hijo.caras['R'] = np.rot90(self.caras['R'], 3)

                hijo.caras['B'][0, :] = self.caras['U'][:, 2]
                hijo.caras['U'][:, 2] = self.caras['F'][:, 2]
                hijo.caras['F'][:, 2] = self.caras['D'][:, 0]
                hijo.caras['D'][:, 0] = self.caras['B'][0, :]
            if direccion == 'izq':
                hijo.caras['R'] = np.rot90(self.caras['R'], 1)

                hijo.caras['B'][0, :] = self.caras['D'][:, 0]
                hijo.caras['D'][:, 0] = self.caras['F'][:, 2]
                hijo.caras['F'][:, 2] = self.caras['U'][:, 2]
                hijo.caras['U'][:, 2] = self.caras['B'][0, :]
        return hijo

    @property
    def resuelto(self):
        # Si la heurística es 0, ya llegamos a la solución.
        return (self.heuristica == 0)

    @property
    def heuristica(self):
        '''
        Regresa la heurística definida al inicializar la clase.
        '''
        return getattr(self, self.funcion_h)()

    def _fuera_de_lugar(self):
        '''
        Cuenta la cantidad de cuadros fuera de su cara.
        
        Si es 0, el cubo rubik está resuelto.
        '''
        faltantes = 0
        for matriz in self.caras.values():
            central = matriz[1][1]
            faltantes += np.count_nonzero(matriz != central)
        return faltantes

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all(
                np.array_equal(self.caras[k], other.caras[k]) for
                k in self.caras)
        else:
            return False

    def __str__(self):
        texto = str()
        for cara, matriz in self.caras.items():
            texto += f'{cara}: \n'
            texto += str(np.vectorize(self.colores.get)(matriz)) + '\n'
        return texto

def A_star(inicial: Cubo, frontera: list = [], explorados: list = [], costo_max: int = 100):
    '''
    Implementación del algoritmo de A*
    '''
    frontera = [inicial]

    for nodo in frontera:

        # wait = input('Continuar')
        print('última acción:', nodo.camino[-1],
            '\t', 'costo:', nodo.costo,
            '\t', 'heuristica:', nodo.heuristica,
            '\t', 'f-cost:', nodo.heuristica + nodo.costo)

        if nodo.resuelto:
            return f'Solución: {nodo.camino}'

        if nodo.costo >= costo_max:
            mejor = explorados.sort(key=lambda x: x.heuristica, reverse=True)[-1]
            return f'Ninguna solución encontrada. \n mejor: {mejor.camino}'

        explorados += [nodo]

        for accion in nodo.acciones:
            hijo = nodo.rotar(*accion)

            costo_f = 0
            costo_f += hijo.costo
            costo_f += hijo.heuristica

            # Evitar loopear infinitamente.
            nuevo = True
            ultima_accion = nodo.camino[-1]
            if ultima_accion[0] == accion[0]: # Si son 2 rotaciones de la misma cara
                if not ultima_accion[-1] == accion[-1]: # Y en direcciones opuestas
                    nuevo = False
            if len(nodo.camino) >= 4:
                if len(set(nodo.camino[-3:] + [accion])) <= 1: # Si las ultimas 4 son iguales
                    nuevo = False
            if nuevo:
                frontera += [(hijo, costo_f)]

            # Evitar loopear infinitamente (LENTO AS FUCC)
            # if not hijo in explorados:
            #     frontera += [(hijo, costo_f)]
                # print(f'Añadiendo:\t{hijo.camino}')
            # else:
            #     print(f'Ya explorado:\t{hijo.camino}')

        # Ordena por mejor a peor por costo_f
        frontera.sort(key=lambda x: x[1], reverse=True)
 
        mejor = frontera.pop()[0]

def tests():
    resuelto = {
        'F': np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
            ], dtype=np.uint8),
        'U': np.array([
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2]
            ], dtype=np.uint8),
        'R': np.array([
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3]
            ], dtype=np.uint8),
        'D': np.array([
            [4, 4, 4],
            [4, 4, 4],
            [4, 4, 4]
            ], dtype=np.uint8),
        'L': np.array([
            [5, 5, 5],
            [5, 5, 5],
            [5, 5, 5]
            ], dtype=np.uint8),
        'B': np.array([
            [6, 6, 6],
            [6, 6, 6],
            [6, 6, 6]
            ], dtype=np.uint8),
        }
    cubo_prueba = Cubo(caras=resuelto)

    assert cubo_prueba.resuelto

    assert cubo_prueba == cubo_prueba

    assert cubo_prueba == cubo_prueba.rotar('F', 'der').rotar('F', 'izq')
    assert cubo_prueba == cubo_prueba.rotar('U', 'der').rotar('U', 'izq')
    assert cubo_prueba == cubo_prueba.rotar('R', 'der').rotar('R', 'izq')
    assert cubo_prueba == cubo_prueba.rotar('D', 'der').rotar('D', 'izq')
    assert cubo_prueba == cubo_prueba.rotar('L', 'der').rotar('L', 'izq')
    assert cubo_prueba == cubo_prueba.rotar('B', 'der').rotar('B', 'izq')

    assert cubo_prueba == cubo_prueba.rotar('F', 'der').rotar('F', 'der').rotar('F', 'der').rotar('F', 'der')
    assert cubo_prueba == cubo_prueba.rotar('U', 'der').rotar('U', 'der').rotar('U', 'der').rotar('U', 'der')
    assert cubo_prueba == cubo_prueba.rotar('R', 'der').rotar('R', 'der').rotar('R', 'der').rotar('R', 'der')
    assert cubo_prueba == cubo_prueba.rotar('D', 'der').rotar('D', 'der').rotar('D', 'der').rotar('D', 'der')
    assert cubo_prueba == cubo_prueba.rotar('L', 'der').rotar('L', 'der').rotar('L', 'der').rotar('L', 'der')
    assert cubo_prueba == cubo_prueba.rotar('B', 'der').rotar('B', 'der').rotar('B', 'der').rotar('B', 'der')

if __name__ == '__main__':
    tests()

    # sys.setrecursionlimit(5000)

    # inicial = {
            # 'F' : np.array([
            #     [6, 2, 1],
            #     [6, 1, 6],
            #     [2, 4, 6],
            #     ], dtype=np.uint8),
            # 'B' : np.array([
            #     [3, 3, 3],
            #     [4, 2, 6],
            #     [5, 5, 4],
            #     ], dtype=np.uint8),
            # 'U' : np.array([
            #     [1, 3, 6],
            #     [6, 3, 3],
            #     [2, 5, 5],
            #     ], dtype=np.uint8),
            # 'D' : np.array([
            #     [5, 2, 1],
            #     [4, 4, 5],
            #     [4, 5, 5],
            #     ], dtype=np.uint8),
            # 'L' : np.array([
            #     [6, 1, 4],
            #     [1, 5, 3],
            #     [4, 4, 3],
            #     ], dtype=np.uint8),
            # 'R' : np.array([
            #     [1, 2, 2],
            #     [1, 6, 2],
            #     [3, 1, 2],
            #     ], dtype=np.uint8),
            # }
    # inicial = {
            # 'F' : np.array([
            #     [1, 1, 1],
            #     [1, 1, 1],
            #     [1, 1, 1],
            #     ], dtype=np.uint8),
            # 'B' : np.array([
            #     [2, 2, 2],
            #     [2, 2, 2],
            #     [2, 2, 2],
            #     ], dtype=np.uint8),
            # 'U' : np.array([
            #     [3, 3, 3],
            #     [3, 3, 3],
            #     [3, 3, 3],
            #     ], dtype=np.uint8),
            # 'D' : np.array([
            #     [4, 4, 4],
            #     [4, 4, 4],
            #     [4, 4, 4],
            #     ], dtype=np.uint8),
            # 'L' : np.array([
            #     [5, 5, 5],
            #     [5, 5, 5],
            #     [5, 5, 5],
            #     ], dtype=np.uint8),
            # 'R' : np.array([
            #     [6, 6, 6],
            #     [6, 6, 6],
            #     [6, 6, 6],
            #     ], dtype=np.uint8),
            # }
    inicial = {
        'F': np.array([
            [5, 6, 3],
            [4, 1, 2],
            [4, 3, 5]
            ], dtype=np.uint8),
        'U': np.array([
            [3, 5, 6],
            [3, 2, 1],
            [1, 2, 6]
            ], dtype=np.uint8),
        'R': np.array([
            [2, 4, 3],
            [4, 3, 6],
            [4, 1, 6]
            ], dtype=np.uint8),
        'D': np.array([
            [2, 3, 1],
            [5, 4, 3],
            [2, 6, 6]
            ], dtype=np.uint8),
        'L': np.array([
            [5, 2, 4],
            [1, 5, 4],
            [5, 5, 2]
            ], dtype=np.uint8),
        'B': np.array([
            [1, 6, 3],
            [2, 6, 1],
            [4, 5, 1]
            ],dtype=np.uint8),
        }

    cubo = Cubo(caras=inicial, heuristica='_fuera_de_lugar')
    print(A_star(cubo))

