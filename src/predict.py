from PIL import Image
import numpy as np
from .preprocess import preprocess


# Usage:
# from .Inception import InceptionDR
# best_model_path='/home/ubuntu/EyeDiease_server/prediction/inception_v3_0.h5')
# best_model = InceptionDR("eval")
# best_model.load_best_model(best_model_path)
# ret = predict(os.getcwd() + result.Patient_Eye_Img.url, best_model)
def predict(image_path: str, best_model) -> float:
    """

    :param image_path: str: path to image to be evaluated
    :param best_model: str: path to best model weight file
    :return: probability of having DR
    """
    pic = Image.open(image_path)
    pix = np.array(pic.getdata()).reshape(pic.size[0], pic.size[1], 3)

    pix = preprocess(pix)/255   # not training from this preprocessing method...
    pix = np.expand_dims(pix, axis=0)

    y = best_model.model.predict(pix)
    return y[0][1]
