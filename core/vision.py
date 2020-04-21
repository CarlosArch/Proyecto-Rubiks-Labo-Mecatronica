"""
Rubik's vision algorithms.
"""
import copy
import numpy as np
import cv2

from imutils import contours as cont_utils

DEBUG = False
COLOR_NAMES = ("R", "Y", "G", "B", "O", "W")
COLOR_RGBS = {
    "R" : (150, 6, 0),
    "Y" : (176, 159, 8),
    "G" : (0, 110, 39),
    "B" : (2, 34, 111),
    "O" : (194, 100, 5),
    "W" : (200, 200, 200),
}
COLOR_TRESHOLDS = {
    "R" : ((-5, 75, 75), (3, 255, 255)),
    "Y" : ((20, 75, 75), (45, 255, 255)),
    "G" : ((60, 75, 75), (80, 255, 255)),
    "B" : ((90, 75, 75), (110, 255, 255)),
    "O" : ((5, 75, 75), (20, 255, 255)),
    "W" : ((0, 0, 125), (255, 50, 255)),
}

def show_image(image, name=None):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_contours(image, contours, name=None, color=(255, 0, 0), thickness=2):
    img = image.copy()
    img = cv2.drawContours(
        image=img,
        contours=contours,
        contourIdx=-1,
        color=color,
        thickness=thickness)
    show_image(img, name)

class Color:
    '''
    Color class for use with cv2.

    Parameters:
    -----------
    name: str
        Color's name

    rgb: tuple
        The color's RGB values.

    treshold: tuple of tuples
        The color's HSV inferior and superor values.

    Attributes:
    -----------
    name: str
        Color's name

    bgr: tuple
        Color's BGR values.

    inf: tuple
        Color's inferior HSV values.

    sup: tuple
        Color's superior HSV values.
    '''

    def __init__(self, name: str, rgb: tuple, treshold: tuple):
        self.name = name
        r, g, b = rgb
        self.bgr = (b, g, r)
        self.inf = treshold[0]
        self.sup = treshold[1]

    def __str__(self):
        return f'Color {self.name}'
    def __repr__(self):
        return f'Color(name: "{self.name}", bgr: {self.bgr})'

class Image:
    '''
    Image class for use with cv2.

    Parameters:
    -----------
    path: str
        Path to image file.

    Attributes:
    -----------
    path: str
        Path to original image file.
    bgr: cv2 image
        Original cv2 image.

    hsv: cv2 image
        cv2 image in the HSV spectrum.

    borders:
        The image's borders.

    Methods:
    --------
    generate_borders(sigma)
        Generates the image's borders.

    crop_image(rect, make_copy)
        Cuts the image to a rectangle.

    rescale_image(height, width, make_copy)
        Rescales the image.

    show(title, name)
        Shows the image, using the title for the window.
    '''
    def __init__(self, path: str = None):
        self.path = path
        self.bgr = cv2.imread(path)
        self._hsv = None
        self.borders = None

        if self.bgr is None:
            raise FileNotFoundError(f'File not found on "{path}"')

    def __str__(self):
        return f'<Image on {self.path}>'
    def __repr__(self):
        if self.path is None:
            return f'Image()'
        else:
            return f'Image(path: "{self.path}")'

    def crop_image(self, rect, make_copy: bool = True):
        '''
        Crops the image to the rectangle, returns a copy if indicated.

        Parameters:
        -----------
        rect: 4 element array-like
            x, y, w, h

        make_copy: bool, optional (True)
            Whether to modify the original image or make a copy.

        Returns:
        -------
        None if make_copy is False.

        Image if make_copy is True.
        '''
        if make_copy:
            obj = copy.deepcopy(self)
        else:
            obj = self

        x, y, w, h = rect
        if obj.bgr is not None:
            obj.bgr = obj.bgr[y:y+h, x:x+w]
        if obj._hsv is not None:
            obj._hsv = obj.hsv[y:y+h, x:x+w]
        if obj.borders is not None:
            obj.borders = obj.borders[y:y+h, x:x+w]
        return obj if make_copy else None

    def rescale_image(self, height: int, width: int, make_copy: bool = True):
        """
        Rescales the image to a defined height and width.

        Parameters:
        -----------
        height: int
            Image height in pixels.

        width: int
            Image width in pixels.

        make_copy: bool
            Whether to make a copy of the image or not.

        """
        if make_copy:
            obj = copy.deepcopy(self)
        else:
            obj = self

        if obj.bgr is not None:
            obj.bgr = cv2.resize(obj.bgr, (width, height))
        if obj._hsv is not None:
            obj._hsv = cv2.resize(obj._hsv, (width, height))
        if obj.borders is not None:
            obj.borders = cv2.resize(obj.borders, (width, height))
        return obj if make_copy else None

    @property
    def hsv(self):
        '''
        Gets the hsv version of the image.
        '''
        if self._hsv is None:
            self._hsv = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2HSV)
        return self._hsv

    def generate_borders(self, sigma=0.33):
        '''
        Detects image borders using a Canny edge detector.

        Adapted from code made by Adrian Rosebrock on April 6th, 2015.
        https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/

        Parameters:
        -----------
        sigma: float
            A tweaking variable.
        '''
        img = self.bgr

        median = np.median(img)
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))

        borders = cv2.Canny(img, lower, upper)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        borders = cv2.morphologyEx(
            src=borders,
            op=cv2.MORPH_CLOSE,
            kernel=close_kernel,
            iterations=3)

        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        borders = cv2.morphologyEx(
            src=borders,
            op=cv2.MORPH_OPEN,
            kernel=open_kernel,
            iterations=1)

        # erode_kernel = np.ones((3, 3), np.uint8)
        # borders = cv2.erode(borders, erode_kernel, iterations=2)

        # dilate_kernel = np.ones((3, 3), np.uint8)
        # borders = cv2.dilate(borders, dilate_kernel, iterations=1)

        self.borders = borders
        return borders

    def show(self, title='Image', name='bgr'):
        """
        Shows the image.
        """
        if (img := getattr(self, name)) is not None:
            cv2.imshow(title, img)
            cv2.waitKey(0)
        else:
            print(f"{name} doesn't exist.")

class RubikFace:
    """
    A single one of the Rubik's cube faces.

    Parameters:
    -----------
    color: str
        Center color.

    colors: dict of Colors
        Color space.

    path: str
        Path to image file.

    max_width: int, optional (500)
        Maximum width to autoscale to.

    max_height: int, optional (500)
        Maximum height to autoscale to.

    sigma: float, optional (0.33)
        Parameter for border detection.

    epsilon: float, optional (0.07)
        Parameter for polygon generation (% of perimeter)

    tolerance: float, optional (0.01)
        Parameter for squareness tolerance

    Attributes:
    -----------
    color: str
        The center color.

    matrix: array
        The matrix representation of each facelet.

    colors: dict of Colors
        The colors used to search.

    path: str
        Path to the original image.

    images: dict of Images
        original: The original image.
        cropped: The image cropped to the cube face.

    candidates: dict of lists of contours
        faces: square candidates to the entire face.
        squares: all the squares seen on the image.
        discarded: all other shapes that weren't square enough.

    Methods:
    --------
        crop_to_face()
            Crop the image to only the face, if possible.

        show_candidates()
            Show the candidate contours.
    """
    def __init__(
            self,
            color: str,
            colors: dict,
            path: str,
            max_width=500,
            max_height=500,
            sigma=0.33,
            epsilon=0.07,
            tolerance=0.01
        ):
        self.color = color
        self._matrix = None

        self.colors = colors
        self.path = path
        self.images = dict()
        self.images['original'] = self.autoscale(
            img=Image(path=path),
            max_height=max_height,
            max_width=max_width)
        self.candidates = dict()

        self.sigma = sigma
        self.epsilon = epsilon
        self.tolerance = tolerance

    @property
    def matrix(self):
        """
        Gets the matrix representation of the facelets.
        """
        if self._matrix is None:
            self._matrix = self.get_face_matrix()
        return self._matrix

    def get_face_matrix(self):
        if not (image := self.images.get('cropped', None)):
            self.crop_to_face()
            return self.get_face_matrix()

        bil_filter = cv2.bilateralFilter(image.bgr.copy(), 10, 50, 50)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))

        masks = dict()
        for name, color in self.colors.items():
            if color.inf[0] < 0:
                sup1 = list(copy.deepcopy(color.sup))
                inf1 = list(copy.deepcopy(color.inf))
                sup1[0] = 180
                inf1[0] = 180 + color.inf[0]

                mask = cv2.inRange(
                    image.hsv,
                    np.array(inf1, dtype=np.uint8),
                    np.array(sup1, dtype=np.uint8)
                    )

                sup2 = list(copy.deepcopy(color.sup))
                inf2 = list(copy.deepcopy(color.inf))
                inf2[0] = 0

                mask += cv2.inRange(
                    image.hsv,
                    np.array(inf2, dtype=np.uint8),
                    np.array(sup2, dtype=np.uint8)
                    )
            else:
                mask = cv2.inRange(
                    image.hsv,
                    np.array(color.inf, dtype=np.uint8),
                    np.array(color.sup, dtype=np.uint8)
                    )


            if DEBUG: show_image(cv2.merge([mask]*3), name)

            mask = cv2.morphologyEx(
                src=mask,
                op=cv2.MORPH_CLOSE,
                kernel=close_kernel,
                iterations=1
            )

            if DEBUG: show_image(cv2.merge([mask]*3), name)

            mask = cv2.morphologyEx(
                src=mask,
                op=cv2.MORPH_OPEN,
                kernel=open_kernel,
                iterations=2
            )

            if DEBUG: show_image(cv2.merge([mask]*3), name)

            mask = cv2.merge([mask]*3)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            masks[name] = mask

        contours = dict()
        all_contours = list()
        for name, mask in masks.items():
            if mask.any():
                conts, _ = cv2.findContours(
                    mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                contours[name] = conts
                if DEBUG: show_contours(image.bgr, conts, name)
                all_contours += conts

        cube = list()
        line = list()
        all_contours, _ = cont_utils.sort_contours(
            all_contours,
            method="top-to-bottom"
        )
        all_contours = list(all_contours)
        for i, contour in enumerate(all_contours, 1):
            line += [contour]
            if (i % 3) == 0:
                line, _ = cont_utils.sort_contours(
                    line,
                    method="left-to-right"
                )
                cube += [line]
                line = list()

        matrix = np.full((3, 3), 'N', dtype="O")
        for i, line in enumerate(cube):
            if i >= 3:
                break
            for j, facelet in enumerate(line):
                if j >= 3:
                    break
                for color, conts, in contours.items():
                    for cont in conts:
                        if np.array_equal(facelet, cont):
                            matrix[i, j] = color

        return matrix

    def crop_to_face(self):
        """
        Crops the Rubik Face to only the face.
        """
        original = self.images['original']
        original.generate_borders(sigma=self.sigma)
        contours, hierarchy = cv2.findContours(
            original.borders,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0] # For some reason, there's an extra, useless dimension.

        polygons = list()
        for contour, h in zip(contours, hierarchy):
            polygon, is_square = self._squareness(contour)
            polygons += [(polygon, h, is_square)]

        squares = list()
        discarded = list()
        for p, h, is_square in polygons:
            if is_square:
                squares += [(p, h)]
            else:
                discarded += [(p, h)]

        faces = list()
        for s, h in squares:
            # If no parents but has children.
            if h[3] == -1 and h[2] != -1:
                faces += [(s, h)]

        face = None
        if len(faces) == 1:
            face = faces[0]
            rectangle = cv2.boundingRect(face[0])
            cropped = original.crop_image(rectangle)
        else:
            cropped = original

        self.candidates['faces'] = faces
        self.candidates['squares'] = squares
        self.candidates['discarded'] = discarded
        self.images['cropped'] = cropped

    def autoscale(self, img: Image, max_height: int, max_width: int):
        """
        Rescales the image to fit inside a maximum height and width.

        Parameters:
        -----------
        img: Image
            Image to autoscale.

        max_height: int
            Maximum amount of pixels in height.

        max_width
            Maximum amount of pixels in width.
        """
        height = img.bgr.shape[0]
        width = img.bgr.shape[1]

        diff_height = max_height / height
        diff_width = max_width / width

        diff = min(diff_height, diff_width)

        height = int(height * diff)
        width = int(width * diff)

        return img.rescale_image(height, width)

    def show_candidates(self, name='squares', color=(255, 0, 0), thickness=2):
        """
        Shows the candidate contours.

        Parameters:
        -----------
        name: str
            Candidate list to get.

        color: tuple
            Color to paint the contours

        thickness: int, optional (2)
            Thickness of contour lines.
            
        """
        img = copy.deepcopy(self.images['original'])
        contours = [c for c, h in self.candidates[name]]
        img.bgr = cv2.drawContours(
            image=img.bgr,
            contours=contours,
            contourIdx=-1,
            color=color,
            thickness=thickness)
        img.show(name)

    def _squareness(self, contour):
        '''
        Approximates a contour to a simpler polygon and checks for squareness.
        '''
        perimeter = cv2.arcLength(contour, closed=True)
        polygon = cv2.approxPolyDP(
            contour,
            epsilon=(self.epsilon * perimeter),
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

        ideal_square = np.array(
            [
                [[1, 0]],
                [[0, 0]],
                [[0, 1]],
                [[1, 1]]
            ],
            dtype=np.int32)
        fit = cv2.matchShapes(polygon, ideal_square, method=1, parameter=1)
        is_square = (fit < self.tolerance)
        return polygon, is_square

    def __str__(self):
        return f'<Face of rubiks "{self.color}">'
    def __repr__(self):
        return f'RubikFace(color: "{self.color}", path: "{self.path}")'

class RubiksCube:
    '''
    Rubik's Cube class for getting the colors on each facelet.

    Attributes:
    -----------
    colors: dict of Colors
        The color space.

    faces: dict of RubikFaces
        The different faces.

    Methods:
    --------
    get_all_faces()
        crops to all faces and generates matrixes.
    '''
    def __init__(self, paths):
        self.colors = dict()
        self.faces = dict()
        for name in COLOR_NAMES:
            rgb = COLOR_RGBS[name]
            treshold = COLOR_TRESHOLDS[name]
            self.colors[name] = Color(name, rgb, treshold)

        for name, path in paths.items():
            self.faces[name] = RubikFace(name, self.colors, path)

    def get_all_faces(self):
        for name, face in self.faces.items():
            face.crop_to_face()
            # cara.imagenes['original'].ver(f'Original de {color}', 'bgr')
            # cara.imagenes['original'].ver(f'Bordes de {color}', 'bordes')
            # cara.ver_candidatos()
            # cara.ver_candidatos('descartados', color=(0, 0, 255))
            # cara.imagenes['recortada'].ver(f'recortada {color}')
            # cv2.destroyAllWindows()

    def get_color_list(self):
        r = list(self.faces['G'].matrix.flatten())
        l = list(self.faces['B'].matrix.flatten())
        u = list(self.faces['Y'].matrix.flatten())
        d = list(self.faces['W'].matrix.flatten())
        f = list(self.faces['R'].matrix.flatten())
        b = list(self.faces['O'].matrix.flatten())
        return u \
            + l[0:3] + f[0:3] + r[0:3] + b[0:3] \
            + l[3:6] + f[3:6] + r[3:6] + b[3:6] \
            + l[6:9] + f[6:9] + r[6:9] + b[6:9] \
            + d

    def to_solve_string(self):
        r = list(self.faces['G'].matrix.flatten())
        l = list(self.faces['B'].matrix.flatten())
        u = list(self.faces['Y'].matrix.flatten())
        d = list(self.faces['W'].matrix.flatten())
        f = list(self.faces['R'].matrix.flatten())
        b = list(self.faces['O'].matrix.flatten())

        color_list = u+l+f+r+b+d

        return "".join(color_list).lower()

    def __str__(self):
        template = \
            '       {} {} {}\n'\
            '       {} {} {}\n'\
            '       {} {} {}\n\n'\
            '{} {} {}  {} {} {}  {} {} {}  {} {} {}\n'\
            '{} {} {}  {} {} {}  {} {} {}  {} {} {}\n'\
            '{} {} {}  {} {} {}  {} {} {}  {} {} {}\n\n'\
            '       {} {} {}\n'\
            '       {} {} {}\n'\
            '       {} {} {}'

        return template.format(*self.get_color_list())

if __name__ == '__main__':
    DEBUG = True
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
    cube = RubiksCube(paths)
    # print(cube)
    # for name in COLOR_NAMES:
    for name in ['R', 'Y', 'G']:
        print(f'------ {name} ------')
        print(cube.faces[name].matrix)
        print(f'---------------')

