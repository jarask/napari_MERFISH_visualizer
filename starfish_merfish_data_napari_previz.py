from starfish import data, FieldOfView
from starfish.image import Filter
from starfish.types import Levels, Features, Axes
from starfish.spots import DetectPixels, AssignTargets
import numpy as np
import pandas as pd
from copy import deepcopy
from cellpose import models
from starfish import BinaryMaskCollection
from skimage import io
import napari_merfish_starfish
from skimage.measure import regionprops
import warnings

if __name__ == '__main__':
    # Load the entire dataset
    print('Loading experiment')
    experiment = data.MERFISH(use_test_data=False)
    imgs = experiment.fov().get_image(FieldOfView.PRIMARY_IMAGES)

    # Filter and scale raw data before decoding
    print('Filtering')
    ghp = Filter.GaussianHighPass(sigma=3)
    high_passed = ghp.run(imgs, verbose=True, in_place=False)
    print('Deconvolving - May take some time...')
    dpsf = Filter.DeconvolvePSF(num_iter=15, sigma=2, level_method=Levels.SCALE_SATURATED_BY_CHUNK)
    deconvolved = dpsf.run(high_passed, verbose=True, in_place=False)
    glp = Filter.GaussianLowPass(sigma=1)
    low_passed = glp.run(deconvolved, in_place=False, verbose=True)

    print('Scaling data')
    scale_factors = {
        (t[Axes.ROUND], t[Axes.CH]): t['scale_factor']
        for t in experiment.extras['scale_factors']
    }
    # Starfish comment
    # this is a scaling method. It would be great to use image.apply here. It's possible, but we need to expose H & C to
    # at least we can do it with get_slice and set_slice right now.
    filtered_imgs = deepcopy(low_passed)
    for selector in imgs._iter_axes():
        data = filtered_imgs.get_slice(selector)[0]
        scaled = data / scale_factors[selector[Axes.ROUND.value], selector[Axes.CH.value]]
        filtered_imgs.set_slice(selector, scaled, [Axes.ZPLANE])

    # Decode spots via pixel decoding
    print('Decoding experiment')
    psd = DetectPixels.PixelSpotDecoder(
        codebook=experiment.codebook,
        metric='euclidean',  # distance metric to use for computing distance between a pixel vector and a codeword
        norm_order=2,        # the L_n norm is taken of each pixel vector and codeword before computing the distance
        distance_threshold=0.5176,
        # minimum distance between a pixel vector and a codeword for it to be called as a gene
        magnitude_threshold=1.77e-5,  # discard any pixel vectors below this magnitude
        min_area=2,  # do not call a 'spot' if it's area is below this threshold (measured in pixels)
        max_area=np.inf,  # do not call a 'spot' if it's area is above this threshold (measured in pixels)
    )

    initial_decoded, prop_results = psd.run(filtered_imgs)
    decoded = initial_decoded.loc[initial_decoded[Features.PASSES_THRESHOLDS]]
    decoded_filtered = decoded[decoded.target != 'nan']

    # At the moment, there is already a saved mask from cellpose in the directory
    # Cell segmentation using Cellpose
    # print('Segmenting cells')
    # model = models.Cellpose(model_type='cyto')
    # channels = [[0, 0]]
    # image = imgs.reduce({Axes.ROUND, Axes.CH, Axes.ZPLANE}, func="max")
    # image = image.xarray.data
    # nuclei = experiment.fov().get_image('nuclei')
    # masks, flows, styles, diams = model.eval(image, diameter=200, channels=channels)
    #
    # # Load in the cellpose mask to starfish
    # io.imsave('cellpose_mask.png', masks)
    # masks_starfish = BinaryMaskCollection.from_external_labeled_image('cellpose_mask.png', nuclei)

    image = imgs.reduce({Axes.ROUND, Axes.CH, Axes.ZPLANE}, func="max")
    image = image.xarray.data
    nuclei = experiment.fov().get_image('nuclei')
    masks_starfish = BinaryMaskCollection.from_external_labeled_image('cellpose_mask_cyto2.png', nuclei)

    # Assign spots to cells by labeling each spot with cell_id
    al = AssignTargets.Label()
    labeled = al.run(masks_starfish, decoded_filtered)

    """
    Workflow for outputting data to scanpy
    """
    """
    # Filter out spots that are not located in any cell mask
    labeled_filtered = labeled[labeled.cell_id != 'nan']
    # Transform to expression matrix
    mat = labeled_filtered.to_expression_matrix()
    # Add area (in pixels) of cell masks to expression matrix metadata
    mat[Features.AREA] = (Features.CELLS, [mask.data.sum() for _, mask in masks_starfish])
    # Add eccentricity of cell masks to expression matrix metadata
    mat['ecc'] = (Features.CELLS, [regionprops(mask.data.astype(int), coordinates='rc')[0].eccentricity
                                   for _, mask in masks_starfish])
    # Save as .h5ad file for loading in scanpy
    mat.save_anndata('expression_matrix.h5ad')
    """

    # Get the labeled data to a Pandas DataFrame
    spots_data = labeled[Features.AXIS].to_dataframe('labeled')
    # In this specific decoding, some low y-values are weird...Filter them out here
    spots_data = spots_data.loc[spots_data.y >= 10]
    spots_data.reset_index(inplace=True)

    expression_data = labeled.to_expression_matrix()
    exp_df = pd.DataFrame(expression_data.data)
    exp_df.drop(index=29, inplace=True)
    exp_df.columns = expression_data.genes
    exp_df.to_csv('expression_data_df.csv')

    # Save the spots_data to use it faster in the visualizer
    spots_data.to_csv('spots_data.csv')

    # This visualization part has been moved to another script...
    # # Test of custom napari_merfish visualizer
    # print('Starting visualizer - Be patient')
    # merfish_viz = napari_merfish_starfish.NapariMERFISH(labeled, image, nuclei.xarray.values, masks_starfish)
    # with warnings.catch_warnings():  # Ignore warnings here - Pycharm makes some weird stuff with pydev importing
    #     warnings.filterwarnings('ignore')
    #     merfish_viz.run()
