from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.vgg16 import preprocess_input
import numpy as np


def classif_pipeline(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x


if __name__ == '__main__':
    # output = 'top'
    # output = 'fc1'
    output = 'fc2'

    if output == 'top':
        model = VGG16(weights='imagenet', include_top=True)
    else:
        base_model = VGG16(weights='imagenet')
        model = Model(input=base_model.input, output=base_model.get_layer(output).output)

    img_path = '../data/elephant.jpg'
    x = classif_pipeline(model, img_path)

    features = model.predict(x)

    if output == 'top':
        # decode the results into a list of tuples (class, description, probability)
        print('Predicted:', decode_predictions(features, top=3)[0])
    else:
        print('Features shape: {}'.format(features.shape))