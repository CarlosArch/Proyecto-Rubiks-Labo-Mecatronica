'''Objetos para rubiks'''
import copy
import numpy as np
import cv2
__author__ = 'Carlos Daniel Archundia Cejudo'

class Color:
    '''
    Clase sencilla para los colores.

    Atributos:
        nombre: Nombre del color
        numero: Representación numérica del color
        bgr: Colores BGR
        inf: límite inferior para hsv
        sup: límite superior para hsv
    '''

    def __init__(self, nombre: str, numero: int, rgb: tuple, umbral: tuple):
        self.nombre = nombre
        self.numero = numero
        r, g, b = rgb
        self.bgr = (b, g, r)
        self.inf = umbral[0]
        self.sup = umbral[1]

    def __str__(self):
        return f'Color {self.nombre} ({self.numero})'
    def __repr__(self):
        return f'Color(nombre="{self.nombre}", numero={self.numero})'

class Imagen:
    '''
    Clase básica de imagenes cv2

    Atributos:
        ruta: La ruta a la imagen original
        bgr: La imagen original, en bgr
        hsv: La imagen en espectro hsv
        bordes: Los bordes de la imagen

    Métodos:
        gen_bordes: consigue los bordes
        ver: Muestra una de las imágenes

    '''
    def __init__(self, ruta: str = None):
        self.ruta = ruta
        self.bgr = cv2.imread(ruta)
        self._hsv = None
        self.bordes = None

        if self.bgr is None:
            raise FileNotFoundError(f'No se encontró imagen en "{ruta}"')

    def __str__(self):
        return f'Imagen CV2 ({self.ruta})'
    def __repr__(self):
        if self.ruta is None:
            return f'Imagen()'
        else:
            return f'Imagen("{self.ruta}")'

    def recortar(self, rect, copia: bool = True):
        '''
        recorta la imagen a un rectángulo (x, y, width, height)
        '''
        if copia:
            obj = copy.deepcopy(self)
        else:
            obj = self

        x, y, w, h = rect
        if obj.bgr is not None:
            obj.bgr = obj.bgr[y:y+h, x:x+w]
        if obj._hsv is not None:
            obj._hsv = obj.hsv[y:y+h, x:x+w]
        if obj.bordes is not None:
            obj.bordes = obj.bordes[y:y+h, x:x+w]
        return obj if copia else None

    def escalar(self, alto: int, ancho: int, copia: bool = True):
        if copia:
            obj = copy.deepcopy(self)
        else:
            obj = self

        if obj.bgr is not None:
            obj.bgr = cv2.resize(obj.bgr, (ancho, alto))
        if obj._hsv is not None:
            obj._hsv = cv2.resize(obj._hsv, (ancho, alto))
        if obj.bordes is not None:
            obj.bordes = cv2.resize(obj.bordes, (ancho, alto))
        return obj if copia else None

    @property
    def hsv(self):
        '''
        Consigue la versión hsv de la imagen.
        '''
        if self._hsv is None:
            self._hsv = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2HSV)
        return self._hsv

    def gen_bordes(self, sigma=0.33):
        '''
        Detecta los bordes en una imagen con el algoritmo 'Canny edge detector'

        Adaptado de código desarrollado por Adrian Rosebrock el 6/abril/2015
        https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
        '''
        img = self.bgr

        median = np.median(img)
        lower = int(max(0,   (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))

        bordes = cv2.Canny(img, lower, upper)

        cerrar = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        bordes = cv2.morphologyEx(src=bordes,
                                  op=cv2.MORPH_CLOSE,
                                  kernel=cerrar,
                                  iterations=3)

        abrir = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bordes = cv2.morphologyEx(src=bordes,
                                  op=cv2.MORPH_OPEN,
                                  kernel=abrir,
                                  iterations=1)

        # erosionar = np.ones((3, 3), np.uint8)
        # bordes = cv2.erode(bordes, erosionar, iterations=2)

        # dilatar = np.ones((3, 3), np.uint8)
        # bordes = cv2.dilate(bordes, dilatar, iterations=1)

        self.bordes = bordes
        return bordes

    def ver(self, titulo='Imagen', atr='bgr'):
        if (img := getattr(self, atr)) is not None:
            cv2.imshow(titulo, img)
            cv2.waitKey(0)
        else:
            print(f'No existe {atr}.')

class CaraRubik:
    '''
    Clase de una de las caras del cubo Rubik

    Atributos:
        color: El color principal.
        matriz: Matriz representando los colores de cada cuadro. (con números)
        colores: El espacio de colores para buscar.
        ruta: La ruta a la imagen original
        imagenes: Las imagenes
            original: La imagen original
            recortada: La imagen recortada (después de encontrar_cara)
        candidatos: Listas de contornos candidatos para cuadrados
            caras: Candidatos a cara principal
            cuadrados: Candidatos a ser cuadrados de la cara
            descartados: Contornos descartados (No cuadrados)
        ancho_max: Máximos pixeles en ancho de imagen
        alto_max: Máximos pixeles en alto de imagen
        sigma: parámetro para la generación de bordes
        epsilon: parámetro para generación de polígonos (% de perímetro)
        tolerancia: parámetro para la tolerancia de forma de los cuadrados

    Métodos:
        encontrar_cara: recortar la imagen a sólo la cara principal
        ver_candidatos: ver los contornos candidatos
    '''
    def __init__(self,
                 color: str,
                 colores: dict,
                 ruta: str,
                 ancho_max=1000,
                 alto_max=1000,
                 sigma=0.33,
                 epsilon=0.07,
                 tolerancia=0.01):
        self.color = color
        self.matriz = np.full(shape=(3, 3), fill_value=0, dtype=np.uint8)

        self.colores = colores
        self.ruta = ruta
        self.imagenes = dict()
        self.imagenes['original'] = self.autoescalar(img=Imagen(ruta=ruta),
                                                     alto_max=alto_max,
                                                     ancho_max=ancho_max)
        self.candidatos = dict()

        self.sigma = sigma
        self.epsilon = epsilon
        self.tolerancia = tolerancia

    def __str__(self):
        return f'Cara {self.color}'
    def __repr__(self):
        return f'CaraRubik(color="{self.color}", ruta="{self.ruta}")'

    def encontrar_cara(self):
        original = self.imagenes['original']
        original.gen_bordes(sigma=self.sigma)
        conts, jers = cv2.findContours(original.bordes,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        jers = jers[0] # Por alguna razón tiene una dimensión extra.

        poligonos = list()
        for cont, jer in zip(conts, jers):
            pol, es_cuad = self._cuadratura(cont)
            poligonos += [(pol, jer, es_cuad)]

        cuadrados = list()
        descartados = list()
        for pol, jer, es_cuad in poligonos:
            if es_cuad:
                cuadrados += [(pol, jer)]
            else:
                descartados += [(pol, jer)]

        caras = list()
        for cuad, jer in cuadrados:
            # Si no tiene padres pero sí tiene hijos.
            if jer[3] == -1 and jer[2] != -1:
                caras += [(cuad, jer)]

        cara = None
        if len(caras) == 1:
            cara = caras[0]
            recortada = original.recortar(cv2.boundingRect(cara[0]))
        else:
            recortada = original

        self.candidatos['caras'] = caras
        self.candidatos['cuadrados'] = cuadrados
        self.candidatos['descartados'] = descartados
        self.imagenes['recortada'] = recortada

    def ver_candidatos(self, atr='cuadrados', color=(255, 0, 0), grosor=2):
        img = copy.deepcopy(self.imagenes['original'])
        conts = [cont for cont, jer in self.candidatos[atr]]
        img.bgr = cv2.drawContours(image=img.bgr,
                               contours=conts,
                               contourIdx=-1,
                               color=color,
                               thickness=grosor)
        img.ver(f'{atr}')

    def _cuadratura(self, contorno):
        '''
        Aproxima el contorno a un polígono y revisa la cuadratura.
        '''
        peri = cv2.arcLength(contorno, closed=True)
        poligono = cv2.approxPolyDP(contorno,
                                    epsilon=(self.epsilon * peri),
                                    closed=True)

        # Código viejo
        #   cambiado por haber encontrado una método mejor y porque
        #   causaba mucho falso positivo.

        # es_cuadrado = False
        # if len(poligono) == 4 and cv2.isContourConvex(poligono):
        #     (x, y, w, h) = cv2.boundingRect(poligono)
        #     razon = w / h
        #     if (1.0 - tolerancia) <= razon and razon <= (1.0 + tolerancia):
        #         es_cuadrado = True

        cuadrado_perfecto = np.array(
            [[[1, 0]],
             [[0, 0]],
             [[0, 1]],
             [[1, 1]]],
            dtype=np.int32)
        fit = cv2.matchShapes(poligono,
                              cuadrado_perfecto,
                              method=1,
                              parameter=1)
        es_cuadrado = (fit < self.tolerancia)
        return poligono, es_cuadrado

    def autoescalar(self, img: Imagen, alto_max: int, ancho_max: int):
        alto = img.bgr.shape[0]
        ancho = img.bgr.shape[1]

        dif_alto = alto_max / alto
        dif_ancho = ancho_max / ancho

        dif = min(dif_alto, dif_ancho)

        alto = int(alto * dif) 
        ancho = int(ancho * dif) 

        return img.escalar(alto, ancho)
class CuboRubik:
    '''
    Clase del cubo Rubik

    Atributos:
        colores: El espacio de colores
        caras: Las caras del cubo

    Métodos:
        resolver: Genera la solución al cubo.
    '''
    def __init__(self, **color_kwargs):
        self.colores = dict()
        self.caras = dict()
        for color in color_kwargs:
            kwargs = color_kwargs[color]['Color']
            self.colores[color] = Color(color, **kwargs)

        for color in color_kwargs:
            kwargs = color_kwargs[color]['Cara']
            self.caras[color] = CaraRubik(color, self.colores, **kwargs)

    def __str__(self):
        return f'Solucionador de Cubo Rubik'

    def resolver(self):
        for color, cara in self.caras.items():
            cara.encontrar_cara()
            cara.imagenes['original'].ver(f'Original de {color}', 'bgr')
            cara.imagenes['original'].ver(f'Bordes de {color}', 'bordes')
            cara.ver_candidatos()
            cara.ver_candidatos('descartados', color=(0, 0, 255))
            cara.imagenes['recortada'].ver(f'recortada {color}')
            cv2.destroyAllWindows()

if __name__ == '__main__':
    # folder = 'imgs/B&W'
    # folder = 'imgs/FullColor'
    folder = 'imgs/Mi_cubo'
    # La estructura más fea que he visto / hecho en mi vida
    configuracion = {
        'amarillo' : {
            'Color' : {
                'numero' : 1,
                'rgb' : (176, 159, 8),
                'umbral' : ((20, 150, 50), (40, 255, 255)),
                },
            'Cara' : {'ruta' : f'{folder}/rubik_amarillo.jpg'}
            },
        'blanco' : {
            'Color' : {
                'numero' : 2,
                'rgb' : (200, 200, 200),
                'umbral' : ((0, 0, 125), (255, 50, 175)),
                },
            'Cara' : {'ruta' : f'{folder}/rubik_blanco.jpg'}
            },
        'rojo' : {
            'Color' : {
                'numero' : 3,
                'rgb' : (150, 6, 0),
                'umbral' : ((0, 50, 50), (5, 255, 255)),
                },
            'Cara' : {'ruta' : f'{folder}/rubik_rojo.jpg'}
            },
        'naranja' : {
            'Color' : {
                'numero' : 4,
                'rgb' : (194, 100, 5),
                'umbral' : ((5, 100, 100), (15, 255, 255)),
                },
            'Cara' : {'ruta' : f'{folder}/rubik_naranja.jpg'}
            },
        'azul' : {
            'Color' : {
                'numero' : 5,
                'rgb' : (2, 34, 111),
                'umbral' : ((105, 100, 50), (115, 255, 255)),
                },
            'Cara' : {'ruta' : f'{folder}/rubik_azul.jpg'}
            },
        'verde' : {
            'Color' : {
                'numero' : 6,
                'rgb' : (0, 110, 39),
                'umbral' : ((65, 100, 50), (75, 255, 255)),
                },
            'Cara' : {'ruta' : f'{folder}/rubik_verde.jpg'}
            },
        }
    cubo = CuboRubik(**configuracion)

    cubo.resolver()

    # cara_amarilla = CaraRubik(ruta='rubik_amarillo.jpg')
    # cara_amarilla.imagen.encontrar_cara()
    # cara_amarilla.imagen.ver_contornos()
    # cara_amarilla.imagen.ver_contornos('descartados', color=(0, 0, 255))
    # cara_amarilla.imagen.original.ver('Original')
    # cara_amarilla.imagen.recortada.ver('Recortada')
    # print(cara_amarilla.imagen.cuadrados)
