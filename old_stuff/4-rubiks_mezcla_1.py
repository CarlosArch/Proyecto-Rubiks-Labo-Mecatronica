import warnings

import cv2
import numpy as np

from imutils import contours
from matplotlib import pyplot as plt

from Debugger import DebuggerObject

__author__ = 'Carlos Daniel Archundia Cejudo'

def ver_imagen(nombre=None, *a_imagenes, **kw_imagenes):
    if kw_imagenes:
        Debugger.print(f'Mostrando {", ".join([n for n in kw_imagenes])}.')

        for nom, imagen in kw_imagenes.items():
            cv2.imshow(nom, imagen)

    elif a_imagenes:
        Debugger.print(f'Mostrando {nombre}.')

        for i, imagen in enumerate(a_imagenes, 1):
            cv2.imshow(f'{nombre} ({i})', imagen)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def auto_canny(imagen, sigma = 0.33):
    '''
    Detecta los bordes en una imagen con el algoritmo 'Canny edge detector'

    Adaptado de código desarrollado por Adrian Rosebrock el 6/abril/2015
    https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    '''
    Debugger.print('Detectando bordes (auto canny)...')

    median = np.median(imagen)

    lower = int(max(0,   (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))

    bordeada = cv2.Canny(imagen, lower, upper)

    k = np.ones((3, 3), np.uint8)
    bordeada = cv2.dilate(bordeada, k, iterations=2)

    if Debugger.depth_in_range():
        ver_imagen(Auto_Canny=bordeada)
    return bordeada

def rgb_a_bgr(rgb):
    r, g, b = (x for x in rgb)
    bgr = (b, g, r)
    return bgr

def bgr_a_hsv(bgr):
    b, g, r = (float(x)/255.0 for x in bgr)

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

    hsv = tuple(x for x in (h, s, v))
    return hsv

def cambiar_rango(color, rango_original, rango_final):
    zipped = zip(color, rango_original, rango_final)
    final = tuple((col * fin / ori) for col, ori, fin in zipped)
    return final

def limites_color(color, rango, umbral):
    inf = list()
    sup = list()
    for col, ran, umb in zip(color, rango, umbral):
        inf.append(max(0  , col - (ran * umb)))
        sup.append(min(ran, col + (ran * umb)))
    inf = tuple(inf)
    sup = tuple(sup)
    return (inf, sup)

def espectros_colores(colores, umbral):
    Debugger.print('Generando espectros de colores.')

    rgb = colores['rgb']

    Debugger.print('Convirtiendo RGB a BGR.')
    bgr = dict()
    for col, val in rgb.items():
        bgr[col] = rgb_a_bgr(val)

    Debugger.print('Convirtiendo BGR a HSV.')
    hsv = dict()
    for col, val in bgr.items():
        hsv[col] = bgr_a_hsv(val)

    Debugger.print('Normalizando HSV.')
    hsv_norm = dict()
    for col, val in hsv.items():
        hsv_norm[col] = cambiar_rango(val,
                                      (360, 100, 100),
                                      (179, 255, 255))

    Debugger.print('Generando límites de HSV.')
    hsv_lims = dict()
    for col, val in hsv_norm.items():
        hsv_lims[col] = limites_color(val,
                                      (179, 255, 255),
                                      umbral)

    colores.update({
        'hsv'      : hsv,
        'hsv_norm' : hsv_norm,
        'hsv_lims' : hsv_lims,
    })

    if Debugger.depth_in_range():
        for tipo, cols in colores.items():
            Debugger.print(f'{tipo}:')
            for col, val in cols.items():
                Debugger.print(f'  {col} {tuple(x for x in val)}')

    return colores

class CuboRubik():
    colores = dict()
    caras = dict()
    rutas = dict()
    originales = dict()
    recortadas = dict()
    umbral = (0.05, 0.2, 0.25)
    tolerancia_cuadrada = 0.05
    def __init__(self, colores, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)
        # self.colores = espectros_colores(colores, self.umbral)
        self.colores = colores
        for nombre in colores['rgb']:
            self.caras[nombre] = np.full((3, 3), '---', np.dtype('U10'))

    def escanear_cara(self, cara, ruta):
        self.rutas[cara] = ruta
        original = cv2.imread(ruta)
        bil_filter = cv2.bilateralFilter(original.copy(), 10, 50, 50)
        ver_imagen(original=original, bil_filter=bil_filter)
        self.originales[cara] = original

        recortada = self.recortar_rubik(bil_filter, cara)
        self.recortadas[cara] = recortada

        recortada_hsv = cv2.cvtColor(recortada.copy(), cv2.COLOR_BGR2HSV)
        cerrar = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        abrir = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        if Debugger.depth_in_range():
            ver_imagen(recortada=recortada, hsv=recortada_hsv)
        mascaras = dict()
        for color, (inf, sup) in self.colores['hsv_lims'].items():
            masc = cv2.inRange(recortada_hsv,
                               np.array(inf, dtype=np.uint8),
                               np.array(sup, dtype=np.uint8))
            if Debugger.depth_in_range():
                masc = cv2.merge([masc, masc, masc])
                masc = cv2.cvtColor(masc, cv2.COLOR_BGR2GRAY)
                ver_imagen(**{
                    'recortada' : recortada,
                    color : masc
                })
            masc = cv2.morphologyEx(src=masc,
                                    op=cv2.MORPH_CLOSE,
                                    kernel=cerrar,
                                    iterations=1)
            masc = cv2.morphologyEx(src=masc,
                                    op=cv2.MORPH_OPEN,
                                    kernel=abrir,
                                    iterations=2)

            masc = cv2.merge([masc, masc, masc])
            masc = cv2.cvtColor(masc, cv2.COLOR_BGR2GRAY)
            if Debugger.depth_in_range():
                ver_imagen(**{
                    'recortada' : recortada,
                    color : masc
                })
            mascaras[color] = masc

        contornos = dict()
        todos_contornos = list()
        for color, mascara in mascaras.items():
            if mascara.any():
                conts = cv2.findContours(mascara,
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
                conts = conts[0] if len(conts) == 2 else conts[1]
                contornos[color] = conts
                todos_contornos += conts

        final = recortada.copy()
        for color, conts in contornos.items():
            for cont in conts:
                x, y, w, h = cv2.boundingRect(cont)
                final = cv2.rectangle(img=final,
                                      rec=(x,y,w,h),
                                      color=self.colores['gbr'][color],
                                      thickness=-1)
                final = cv2.putText(img=final, 
                                    text=f'{color}', 
                                    org=(x,y+50),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, 
                                    color=(0, 0, 0),
                                    thickness=2)
        if Debugger.depth_in_range():
            ver_imagen(recortada=recortada, final=final)

        # cubo = list()
        # linea = list()
        # for i, contorno in enumerate(todos_contornos, 1):
        #     linea.append(contorno)
        #     if (i % 3) == 0:
        #         linea, _ = contours.sort_contours(linea,
        #                                           method='left-to-right')
        #         cubo.append(linea)
        #         linea = list()

        # for i, linea in enumerate(cubo):
        #     if i >= 3:
        #         break
        #     for j, cuadro in enumerate(linea):
        #         if j >=3:
        #             break
        #         for color, conts in contornos.items():
        #             for cont in conts:
        #                 if np.array_equal(cuadro, cont):
        #                     self.caras[cara][i, j] = color

        # Debugger.print(self.caras[cara])

    def detectar_cuadrado(self, contorno):
        perimetro = cv2.arcLength(curve=contorno,
                                  closed=True)
        poligono = cv2.approxPolyDP(curve=contorno,
                                      epsilon=(.04*perimetro),
                                      closed=True)
        forma = 'otro'
        if len(poligono) == 4:
            (x, y, w, h) = cv2.boundingRect(poligono)
            ar = float(w) / float(h)
            tolerancia = self.tolerancia_cuadrada
            if ar >= 1.0 - tolerancia and ar <= 1.0 + tolerancia:
                forma = 'cuadrado'
        return poligono, forma

    def recortar_rubik(self, imagen, color):
        Debugger.print(f'Detectando bordes...')
        bordes = auto_canny(imagen)

        Debugger.print(f'Buscando contornos...')
        (contornos, jerarquia) = cv2.findContours(bordes.copy(),
                                              cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)

        poligonales = list()
        for cont, jer in zip(contornos, jerarquia[0]):
            poligono, forma = self.detectar_cuadrado(cont)
            poligonales.append(tuple((poligono, forma, jer)))

        if Debugger.depth_in_range():
            conts = imagen.copy()
            conts = cv2.drawContours(image=conts,
                                     contours=[c[0] for c in poligonales],
                                     contourIdx=-1,
                                     color=(255, 0, 0),
                                     thickness=2)
            ver_imagen(Poligonales=conts)

        Debugger.print(f'Seleccionando cuadrados...')
        cuadrados = list()
        descartados = list()
        for poligono, forma, jer in poligonales:
            if forma == 'cuadrado':
                cuadrados.append(tuple((poligono, jer)))
            else:
                descartados.append(tuple((poligono, jer)))

        if Debugger.depth_in_range():
            conts = imagen.copy()
            conts = cv2.drawContours(image=conts,
                                     contours=[c[0] for c in cuadrados],
                                     contourIdx=-1,
                                     color=(0, 255, 0),
                                     thickness=2)
            conts = cv2.drawContours(image=conts,
                                     contours=[c[0] for c in descartados],
                                     contourIdx=-1,
                                     color=(0, 0, 255),
                                     thickness=2)
            ver_imagen(Contornos=conts)
    
        Debugger.print(f'Buscando candidatos...')
        candidatos = list()
        for poligono, jer in cuadrados:
            if jer[3] == -1:
                candidatos += [poligono]
        if Debugger.depth_in_range():
            conts = imagen.copy()
            conts = cv2.drawContours(image=conts,
                                     contours=candidatos,
                                     contourIdx=-1,
                                     color=(0, 255, 0),
                                     thickness=2)
            ver_imagen(Candidatos=conts)

        if len(candidatos) > 1:
            candidatos = imagen.copy()
            candidatos = cv2.drawContours(image=candidatos,
                                          contours=candidatos,
                                          contourIdx=-1,
                                          color=(0, 255, 0),
                                          thickness=2)
            ver_imagen(Candidatos=candidatos)
            raise Exception('Se encontró más de 1 candidato para cara.')
        if len(candidatos) == 0:
            ver_imagen(Original=imagen, Bordes=bordes)
            warnings.warn(f'No se encontró la cara "{color}".')
            recortada = imagen
        else:
            cara = candidatos[0]
            x, y, w, h = cv2.boundingRect(cara)
            recortada = imagen.copy()[y:y+h, x:x+w]
        return recortada

if __name__ == '__main__':
    Debugger = DebuggerObject(max_depth=2, indent_string='  ')
    COLORES = {
        'rgb' : {
            'amarillo' : (176, 159,   8),
            'blanco'   : (200, 200, 200),
            'rojo'     : (150,   6,   0),
            'naranja'  : (194,  100,   5),
            'azul'     : (  2,  34, 111),
            'verde'    : (  0, 110,  39),
        },
        'gbr' : {
            'amarillo' : (  8, 159, 176),
            'blanco'   : (200, 200, 200),
            'rojo'     : (  0,   6, 150),
            'naranja'  : (  5, 100, 194),
            'azul'     : (111,  34,   2),
            'verde'    : ( 39, 110,   0),
        },
        'hsv_lims' : {
            'amarillo' : (( 20, 150,  50), ( 40, 255, 255)),
            'blanco'   : ((  0,   0, 125), (255,  50, 175)),
            'rojo'     : ((  0,  50,  50), (  5, 255, 255)),
            'naranja'  : ((  5, 100, 100), ( 15, 255, 255)),
            'azul'     : ((105, 100,  50), (115, 255, 255)),
            'verde'    : (( 65, 100,  50), ( 75, 255, 255)),
        }
    }
    UMBRALES_HSV = (0.05, 0.25, 0.5)

    Rubik = CuboRubik(colores=COLORES,
                      umbral=UMBRALES_HSV,
                      tolerancia_cuadrada=0.05)
    Rubik.escanear_cara('amarillo', 'rubik_amarillo.jpg')