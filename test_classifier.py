import keras
import numpy as np
from cpc_model import get_custom_objects_cpc
from data_generator import NCEGenerator
from os.path import join


def load_model(path):
    model = keras.models.load_model(path,custom_objects=get_custom_objects_cpc())
    return model



def evaluate_model(model_path,input_dir,batch_size=64):
    model = load_model(model_path)
    test_data = NCEGenerator(
        x_path=join(input_dir, 'test_x.npy'),
        y_path=join(input_dir, 'test_y.npy'),
        batch_size=batch_size,
        n_classes=10,
        n_negatives=0,
        augment_image_fn=None,
        augment_crop_fn=None
    )
    model.summary()
    model.evaluate(test_data,steps=len(test_data))


if __name__ == '__main__':
    evaluate_model(model_path=join('.','resources','classifier_mode','checkpoint.h5'),input_dir=join('.','resources','data'))
