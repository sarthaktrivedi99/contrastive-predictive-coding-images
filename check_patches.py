import numpy as np
from os.path import join
from data_generator import NCEGenerator
import matplotlib.pyplot as plt
from prepare_data import augment_images_mnist,augment_crops_mnist,augment_mnist

def main(input_dir,n_crops):
    training_data = NCEGenerator(
        x_path=join(input_dir, 'training_x.npy'),
        y_path=join(input_dir, 'training_y.npy'),
        batch_size=1,
        n_classes=10,
        n_negatives=0,
        augment_image_fn=None,
        augment_crop_fn=augment_crops_mnist,
        random_sample=False
    )
    x,_ = next(training_data)
    x = x[0,:,:,:,:,:]
    print(x)
    fig,ax = plt.subplots(n_crops,n_crops)
    for i in range(n_crops):
        for j in range(n_crops):
            ax[i,j].imshow(x[i,j,:,:,:])
            ax[i,j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
    # plt.tight_layout()
    plt.show()
    plt.savefig(join(input_dir,'sample_crops_without_aug.png'))
if __name__ == '__main__':
    main(input_dir=join('.', 'resources', 'data'),n_crops=7)