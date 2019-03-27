from Inception import InceptionDR
from PIL import Image
import numpy


def predict(image_path: str, best_model='inception_v3_0.h5') -> float:
    """

    :param image_path: str: path to image to be evaluated
    :return: probability of having DR
    """
    pic = Image.open(image_path)
    pix = numpy.array(pic.getdata()).reshape(1, pic.size[0], pic.size[1], 3)

    pix = pix / 255
    # todo add preprocessing

    model = InceptionDR()
    model.load_best_model(best_model)
    y = model.model.predict(pix)
    return y[0]
