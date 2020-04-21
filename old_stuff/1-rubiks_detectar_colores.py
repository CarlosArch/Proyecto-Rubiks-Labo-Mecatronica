import cv2
import numpy as np
from imutils import contours
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

COL_TRESHOLD = (0.1, 0.1, 0.1)

def mostrar(**imagenes):
    for nombre, imagen in imagenes.items():
        cv2.imshow(nombre,imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def bgr_a_hsv(bgr):

    b, g, r = bgr

    b, g, r = (x/255 for x in bgr)

    cmax = max(b, g, r)
    cmin = min(b, g, r)
    diff = cmax - cmin

    if cmax == cmin:
        h = 0
    elif cmax == r:
        h = (60 * ((g - b) / diff ) + 360) % 360
    elif cmax == g:
        h = (60 * ((b - r) / diff ) + 120) % 360
    elif cmax == b:
        h = (60 * ((r - g) / diff ) + 240) % 360

    s = 0 if cmax == 0 else (diff / cmax) * 100

    v = cmax * 100
    return (h, s, v)

def normalizar(colores, rango_original, rango_objetivo):
    normalizado = dict()
    for color, valores in colores.items():
        norm = [0,0,0]
        for i, valor in enumerate(valores):
            norm[i] = valor * rango_objetivo[i] / rango_original[i]
        normalizado[color] = tuple(norm)
    return normalizado

def ampliar_rango_col(colores, COL_TRESHOLD, rango):
    colores_minmax = dict()
    for color, valores in colores.items():
        minimo = [0,0,0]
        for i, valor in enumerate(valores):
            minimo[i] = valor - (COL_TRESHOLD[i] * rango[i])
            if minimo[i] < 0:
                minimo[i] = 0
        maximo = [0,0,0]
        for i, valor in enumerate(valores):
            maximo[i] = valor + (COL_TRESHOLD[i] * rango[i])
            if maximo[i] > rango[i]:
                maximo[i] = rango[i]
        colores_minmax[color] = (minimo, maximo)
    return colores_minmax


original = cv2.imread('rubik.jpg')
image = original.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask = np.zeros(image.shape, dtype=np.uint8)

colores_hsv = {
    'naranja'   : ( 15,  91,  80),
    'verde'     : (156,  69,  46),
    'blanco'    : ( 50,   3,  76),
    'amarillo'  : ( 45,  76,  85),
    'azul'      : (236,  60,  44),
}

colores_bgr = {
    'naranja'   : ( 20,  62, 205),
    'verde'     : ( 85, 117,  36),
    'blanco'    : (188, 193, 194),
    'amarillo'  : ( 52, 176, 216),
    'azul'      : (111,  49,  43),
}

colores_norm = normalizar(colores_hsv, (360, 100, 100), (179, 255, 255))

colores_minmax = ampliar_rango_col(colores_norm, COL_TRESHOLD, (179, 255, 255))

open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,50))
close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

mascaras_colores = {}
for color, (lower, upper) in colores_minmax.items():
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)
    color_mask = cv2.inRange(image, lower, upper)

    # Remover ruido
    color_mask = cv2.morphologyEx(src=color_mask,
                                  op=cv2.MORPH_OPEN,
                                  kernel=open_kernel,
                                  iterations=1)
    # Remover huecos
    color_mask = cv2.morphologyEx(src=color_mask,
                                  op=cv2.MORPH_CLOSE,
                                  kernel=close_kernel,
                                  iterations=5)

    color_mask = cv2.merge([color_mask, color_mask, color_mask])
    color_gris = cv2.cvtColor(color_mask, cv2.COLOR_BGR2GRAY)

    mascaras_colores[color] = color_gris

contornos_colores = {}
for color, mascara in mascaras_colores.items():
    contornos = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = contornos[0] if len(contornos) == 2 else contornos[1]
    (contornos, _) = contours.sort_contours(contornos, method="top-to-bottom")
    contornos_colores[color] = contornos

final = mask.copy()
for color, contornos in contornos_colores.items():
    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        final = cv2.rectangle(img=final,
                              rec=(x,y,w,h),
                              color=colores_bgr[color],
                              thickness=-1)
        final = cv2.putText(img=final, 
                            text=f'{color}', 
                            org=(x,y+50),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, 
                            color=(0, 0, 0),
                            thickness=2)

mostrar(original=original, final=final)


