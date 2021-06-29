from sklearn.manifold import TSNE
from cpc_model import get_custom_objects_cpc
import keras
import numpy as np
from os.path import join
from data_generator import NCEGenerator
from tqdm import tqdm
from joblib import dump,load
import matplotlib.pyplot as plt

def main(encoder_path,input_dir,n_crops,crop_shape,code_size):

    encoder_model = keras.models.load_model(encoder_path, custom_objects=get_custom_objects_cpc())

    x_input = keras.layers.Input((n_crops, n_crops) + crop_shape)
    x = keras.layers.Reshape((n_crops * n_crops,) + crop_shape)(x_input)
    x = keras.layers.TimeDistributed(encoder_model)(x)
    x = keras.layers.Reshape((n_crops, n_crops, code_size))(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    # Tried to change the input to the input size of each image
    # Doesn't work let's see.

    # encoder_model.layers.pop(0)
    # encoder_input = keras.layers.Input((64,64,3))
    # encoder_output = encoder_model(encoder_input)
    # encoder_model = keras.models.Model(encoder_input, encoder_output, name='encoder')
    # encoder_model.compile()

    encoder_model = keras.models.Model(x_input, x, name='encoder')
    encoder_model.summary()

    training_data = NCEGenerator(
        x_path=join(input_dir, 'training_x.npy'),
        y_path=join(input_dir, 'training_y.npy'),
        batch_size=1000,
        n_classes=10,
        n_negatives=0,
        augment_image_fn=None,
        augment_crop_fn=None,
        random_sample=False
    )

    enc = encoder_model.predict(training_data,steps=len(training_data),use_multiprocessing=True)
    label = np.load(join(input_dir, 'test_y.npy'))

    tsne_model = TSNE(n_components=2)
    enc_tsne = tsne_model.fit_transform(enc)
    for i in set(label[:300]):
        plt.scatter(x=enc_tsne[:300,0][np.where(label[:300]==i)],y=enc_tsne[:300,1][np.where(label[:300]==i)],label=str(i))
    plt.legend(loc='best')
    plt.title('T-SNE Plot')
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    plt.savefig(join(input_dir,'tsne.png'))
if __name__ == '__main__':
    main(encoder_path=join('.', 'resources', 'cpc_model', 'encoder_model.h5'),input_dir=join('.', 'resources', 'data'),n_crops=7,crop_shape=(16, 16, 3),code_size=128)