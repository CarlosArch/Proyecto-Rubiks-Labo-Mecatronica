import warnings

import cv2
import numpy as np
from imutils import contours
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

colores = {
    'bgr' : {
        'amarillo' : (8, 159, 176),
        'blanco'   : (150, 150, 150),
        'rojo'     : (0, 6, 140),
        'naranja'  : (5, 57, 194),
        'azul'     : (111, 34, 2),
        'verde'    : (39, 110, 0),
    },
}
COL_TRESHOLD = (0.05, 0.2, 0.25)
VERBOSE = True

def ver_imagen(**imagenes):
    if VERBOSE: print('func: ver_imagen')
    for nombre, imagen in imagenes.items():
        if VERBOSE: print(f'Mostrando {nombre}.')
        cv2.imshow(nombre, imagen)
    cv2.waitKey(0)
    if VERBOSE: print('Destruyendo Ventanas.')
    cv2.destroyAllWindows()
    if VERBOSE: print()

def bgr_a_hsv(bgr, tipo = float):
    if VERBOSE: print('func: bgr_a_hsv')
    b, g, r = (x/255 for x in bgr)

    cmax = max(b, g, r)
    cmin = min(b, g, r)
    diff = cmax - cmin

    if cmax == cmin:
        h = 0
    elif cmax == r:
        h = (60 * ((g - b) / diff) +   0) % 360
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    elif cmax == b:
        h = (60 * ((r - g) / diff) + 240) % 360

    s = 0 if cmax == 0 else (diff / cmax) * 100

    v = cmax * 100

    hsv = tuple(tipo(x) for x in (h, s, v))

    if VERBOSE:
        print(f'bgr : {bgr}.')
        print(f'hsv : {hsv}.')
        print()
    return hsv

def cambiar_rango(color, rango_original, rango_final, tipo = float):
    if VERBOSE: print('func: cambiar_rango')
    zipped = zip(color, rango_original, rango_final)
    final = tuple (tipo(col * fin / ori) for col, ori, fin in zipped)
    if VERBOSE:
        print(f'color : {color}.')
        print(f'final : {final}.')
        print()
    return final

def min_max_color(color, rango, umbral, tipo = float):
    if VERBOSE: print('func: min_max_color')
    mini = list()
    maxi = list()
    for col, ran, umb in zip(color, rango, umbral):
        mini.append(tipo(max(0  , col - (ran * umb))))
        maxi.append(tipo(min(ran, col + (ran * umb))))

    mini = tuple(mini)
    maxi = tuple(maxi)
    if VERBOSE:
        print(f'color : {color}.')
        print(f'minimaxi : {(mini, maxi)}.')
        print()
    return (mini, maxi)

def generar_colores(color_dict, tipo = float):
    if VERBOSE: print('func: generar_colores')
    colores_hsv = dict()
    for color, bgr in color_dict.items():
        colores_hsv[color] = bgr_a_hsv(bgr, tipo)

    colores_hsv_norm = dict()
    for color, hsv in colores_hsv.items():
        colores_hsv_norm[color] = cambiar_rango(hsv,
                                                (360, 100, 100),
                                                (179, 255, 255),
                                                tipo)

    colores_hsv_minmax = dict()
    for color, norm in colores_hsv_norm.items():
        colores_hsv_minmax[color] = min_max_color(norm,
                                                (179, 255, 255),
                                                COL_TRESHOLD,
                                                tipo)
    colores = {
        'hsv'        : colores_hsv,
        'hsv_norm'   : colores_hsv_norm,
        'hsv_minmax' : colores_hsv_minmax,
    }
    if VERBOSE:
        print('color_dict:')
        for col, val in color_dict.items():
            print(f'\t{col}:{val}')
        for tipo, vals in colores.items():
            print(f'{tipo}:')
            for col, val in vals.items():
                print(f'\t{col}:{val}')
        print()
    return colores

class CuboRubik():
    def __init__(self, colores, *args, **kwargs):
        if VERBOSE: print('inicializando cubo rubik...')
        self.colores = colores
        self.caras = dict()
        for nombre in colores['bgr']:
            self.caras[nombre] = np.full((3,3), '---', np.dtype('U10'))
        if VERBOSE:
            print('caras del cubo:')
            for cara, valores in self.caras.items():
                print(f'\t{cara}')
                print(f'{valores}')
            print()

    def escanear_cara(self, cara, ruta_img, ver = False):
        if VERBOSE: print(f'escaneando cara "{cara}"...')

        # Se lee imagen
        if VERBOSE: print(f'\t leyendo "{ruta_img}"...')
        original = cv2.imread(ruta_img)
        if VERBOSE: ver_imagen(**{f'original {cara}' : original})
        # Se genera una copia en HSV (mejor para visión artificial).
        if VERBOSE: print(f'\t convirtiendo "{ruta_img}" a HSV...')
        orig_hsv = cv2.cvtColor(original.copy(), cv2.COLOR_BGR2HSV)

        # Se genera un diccionario con máscaras por cada color.
        # Estas máscaras serán 'blanco' en todos los pixeles dentro de el
        # rango de color, y 'negro' en todo otro pixel
        if VERBOSE: print(f'\t generando máscaras para {cara}...')
        mascaras_colores = dict()
        for color, (mini, maxi) in self.colores['hsv_minmax'].items():
            if VERBOSE: print(f'\t generando máscara para {color}...')
            # Se genera una máscara básica
            masc_col = cv2.inRange(orig_hsv,
                                   np.array(mini, dtype=np.uint8),
                                   np.array(maxi, dtype=np.uint8))
            if VERBOSE:
                img = cv2.merge([masc_col, masc_col, masc_col])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                plt.imshow([[mini]])
                ver_imagen(**{
                    color : img,
                    f'original {cara}' : original,
                    f'hsv {cara}': orig_hsv,
                })

            # Se hacen operaciones morfológicas para limpiar la imagen
            # https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html

            # Se remueven 'huecos', es decir, cualquier área más pequeña de
            # color negro que el kernel cerrado utilizando un 'closing'.
            kernel_cerrar = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
            masc_col = cv2.morphologyEx(src=masc_col,
                                        op=cv2.MORPH_CLOSE,
                                        kernel=kernel_cerrar,
                                        iterations=2)
            # Se remueve el 'ruido', es decir, cualquier área más pequeña de
            # color blanco que el kernel abierto utilizando un 'opening'.
            kernel_abrir = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
            masc_col = cv2.morphologyEx(src=masc_col,
                                        op=cv2.MORPH_OPEN,
                                        kernel=kernel_abrir,
                                        iterations=1)

            # Se convierte la matriz máscara en escala de grises.
            masc_col  = cv2.merge([masc_col, masc_col, masc_col])
            masc_col = cv2.cvtColor(masc_col, cv2.COLOR_BGR2GRAY)

            # Se añade al diccionario de colores con sus máscaras
            mascaras_colores[color] = masc_col
            if VERBOSE:
                ver_imagen(**{
                    color : masc_col,
                    f'original {cara}' : original,
                    f'hsv {cara}': orig_hsv,
                })


        # Por cada máscara, se buscan sólo los contornos y se hace un
        # diccionario de colores con contornos, además de una lista con
        # todos los contornos, para ordenar luego
        contornos_colores = dict()
        contornos_completo = list()
        for color, mascara in mascaras_colores.items():
            # Si la máscara existe (si el color fue encontrado).
            if mascara.any():
                # Se buscan los contornos de tal máscara
                conts = cv2.findContours(mascara,
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
                # Para ser sincero, ni idea de por qué se necesita esto.
                conts = conts[0] if len(conts) == 2 else conts[1]

                # Se añaden los contornos encontrados al diccionario y la
                # lista
                contornos_colores[color] = conts
                contornos_completo += conts

        # Se ordenan todos los contornos de arriba hacia abajo.
        (contornos_completo, _) = contours.sort_contours(contornos_completo,
                                                         method='top-to-bottom')

        # Con los contornos ya ordenados de arriba hacia abajo, se toman en
        # grupos de 3 (cada línea del cubo) y se ordenan de
        # izquierda a derecha.
        cont_cubo = list()
        cont_linea = list()
        for i, contorno in enumerate(contornos_completo, 1):
            # Se añaden los contornos hasta llegar a 3
            cont_linea.append(contorno)
            if (i % 3) == 0:
                # Se ordenan de izquierda a derecha
                (cont_linea, _) = contours.sort_contours(cont_linea,
                                                         method='left-to-right')
                cont_cubo.append(cont_linea)
                cont_linea = list()

        # Se genera la matriz 3x3 de la cara
        matriz_cara = np.full((3,3), 'Nada', np.dtype('U10'))
        # Por cada línea del cubo.
        for i, linea in enumerate(cont_cubo):
            # Por cada cuadro en la línea
            for j, cuadro in enumerate(linea):
                # Si el cuadro está en el diccionario, se actualiza la matriz
                # con el color correspondiente
                for color, contornos in contornos_colores.items():
                    for cont in contornos:
                        if np.array_equal(cuadro, cont):
                            matriz_cara[i, j] = color

        if not matriz_cara[1,1] == cara:
            warnings.warn(f'Cuadro central de "{cara}" detectado como '\
                          f'"{matriz_cara[1,1]}"; Revisar imágenes o '\
                           'calibrar colores.')

        if VERBOSE:
            print(f'\t matriz de cara {cara}:')
            print(matriz_cara)
        self.caras[cara] = matriz_cara

        # Si se activó la opcion de ver la cara, se muestra la imagen
        # procesada
        if ver:
            mascara  = np.zeros(orig_hsv.shape, dtype=np.uint8)
            for color, contornos in contornos_colores.items():
                for contorno in contornos:
                    x, y, w, h = cv2.boundingRect(contorno)
                    mascara = cv2.rectangle(img=mascara,
                                            rec=(x,y,w,h),
                                            color=self.colores['bgr'][color],
                                            thickness=-1)
                    mascara = cv2.putText(img=mascara, 
                                          text=f'{color}', 
                                          org=(x,y+50),
                                          fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                          fontScale=0.5, 
                                          color=(0, 0, 0),
                                          thickness=2)
            ver_imagen(**{cara:mascara, 'original': original})

    def escanear_caras(self, ver = False, *arg_imgs, **kwarg_imgs):
        if arg_imgs and kwarg_imgs:
            raise Exception('Sólo enviar args ó kwargs.')
        if arg_imgs:
            for cara, ruta_img in zip(self.caras, arg_imgs):
                self.escanear_cara(cara, ruta_img, ver)
        elif kwarg_imgs:
            for cara in self.caras:
                ruta_img = kwarg_imgs[cara]
                self.escanear_cara(cara, ruta_img, ver)
        else:
            raise Exception('Enviar rutas a las imágenes.')

colores.update(generar_colores(colores['bgr']))

Rubik = CuboRubik(colores)
Rubik.escanear_caras(ver=True,
                     amarillo = 'rubik_amarillo.jpg',
                     azul     = 'rubik_azul.jpg',
                     blanco   = 'rubik_blanco.jpg',
                     naranja  = 'rubik_naranja.jpg',
                     rojo     = 'rubik_rojo.jpg',
                     verde    = 'rubik_verde.jpg')
