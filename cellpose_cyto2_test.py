from starfish import data, FieldOfView
from starfish.types import Axes
from cellpose import models
import matplotlib.pyplot as plt
from skimage import io

# Load the entire dataset
print('Loading experiment')
experiment = data.MERFISH(use_test_data=False)
imgs = experiment.fov().get_image(FieldOfView.PRIMARY_IMAGES)
cyto_img = imgs.reduce({Axes.ROUND, Axes.CH, Axes.ZPLANE}, func="max").xarray.data
nuclei = experiment.fov().get_image('nuclei').xarray.data
image_array = [cyto_img, nuclei]

# Cell segmentation using Cellpose
print('Segmenting cells', end='\r')
model = models.Cellpose(model_type='cyto2')
channels = [[0, 0]]

masks, flows, styles, diams = model.eval(image_array, diameter=200, channels=channels)
print('Segmenting cells - Done')

plt.imshow(masks[0])
plt.show()

io.imsave('cellpose_mask_cyto2.png', masks[0])
