import numpy as np
import cv2

if (VERBOSE := True):
    print()
    print('Corriendo en modo verbose.')


sigma = 0.33
ruta_img = 'rubik_verde.jpg'

def ver_imagen(**imagenes):
    for nombre, imagen in imagenes.items():
        cv2.imshow(nombre, imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if VERBOSE:
        print(f'Mostrando {", ".join([n for n in imagenes])}.')
        print()

def auto_canny(imagen, sigma = 0.33):
    '''
    Detecta los bordes en una imagen con el algoritmo 'Canny edge detector'

    Adaptado de código desarrollado por Adrian Rosebrock el 6/abril/2015
    https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    '''
    if VERBOSE: print('Corriendo auto_canny...')

    median = np.median(imagen)

    lower = int(max(0,   (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))

    bordeada = cv2.Canny(imagen, lower, upper)

    k = np.ones((3,3), np.uint8)
    bordeada = cv2.dilate(bordeada, k, iterations=2)

    if VERBOSE:
        ver_imagen(Auto_Canny=bordeada)
        print()

    return bordeada

def detectar_cuadrado(contorno, precision = 0.05):
    perimetro = cv2.arcLength(curve=contorno,
                              closed=True)
    aproximada = cv2.approxPolyDP(curve=contorno,
                                  epsilon=(0.04 * perimetro),
                                  closed=True)
    forma = [aproximada, 'otro']
    if len(aproximada) == 4:
        (x, y, w, h) = cv2.boundingRect(aproximada)
        ar = w / float(h)
        if ar >= 1.0 - precision and ar <= 1.0 + precision:
            forma[1] = 'cuadrado'

    if VERBOSE:
        pass

    return forma

def localizar_rubik(bordes, precision = 0.05):
    cara = list()
    (contornos, jerarquia) = cv2.findContours(bordes.copy(),
                                              cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)
    for i, contorno in enumerate(contornos):
        forma = detectar_cuadrado(contorno, precision)
        if  forma [1] == 'cuadrado' and jerarquia[0][i][3] == -1:
            cara = [forma[0]]
    return cara

def recortar_rubik(imagen, precision = 0.05):
    bordeada = auto_canny(original, .4)
    cara = localizar_rubik(bordeada, .15)
    final = original.copy()
    final = cv2.drawContours(image=final,
                            contours=cara,
                            contourIdx=-1,
                            color=(0,255,0),
                            thickness=2)
    x, y, w, h = cv2.boundingRect(cara[0])
    cropped = final[y:y+h, x:x+w]
    if VERBOSE:
        ver_imagen(Cara=final, cropped=cropped)
    return cropped

original = cv2.imread(ruta_img)

cropped = recortar_rubik(original, precision=.05)