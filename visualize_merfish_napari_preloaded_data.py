import warnings
import pandas as pd
from starfish import data, FieldOfView, BinaryMaskCollection
from starfish.types import Axes
import napari_merfish_starfish

# Load the entire dataset
print("Loading experiment", end="\r")
experiment = data.MERFISH(use_test_data=False)
imgs = experiment.fov().get_image(FieldOfView.PRIMARY_IMAGES)
print("Loading experiment - Done")

# Load images from the data set
image = imgs.reduce({Axes.ROUND, Axes.CH, Axes.ZPLANE}, func="max")
image = image.xarray.data
nuclei = experiment.fov().get_image("nuclei")
# Load the previously created cellpose-mask
masks_starfish = BinaryMaskCollection.from_external_labeled_image(
    "cellpose_mask_cyto2.png", nuclei
)
# Get the spots data from a previously saved csv file
spots_data = pd.read_csv("spots_data.csv")
spots_data.cell_id = spots_data.cell_id.astype("Int64")
spots_data.cell_id = spots_data.cell_id.astype("str")
spots_data.cell_id.replace("<NA>", "nan", inplace=True)

# Load the expression data in
expr_data = pd.read_csv("expression_data_df.csv", index_col=0)

# Test of custom napari_merfish visualizer
# print('Starting visualizer - Be patient')
with warnings.catch_warnings():  # Ignore warnings here - Pycharm makes some weird stuff with pydev importing
    warnings.filterwarnings("ignore")
    # Initialize the 'big' viewer
    merfish_viz = napari_merfish_starfish.NapariMERFISH(spots_data, image, nuclei.xarray.values, masks_starfish,
                                                        expr_data)
    # Initialize the 'small' viewer with minimum requirements
    # merfish_viz = napari_merfish_starfish.NapariMERFISH(labeled_spots_data=spots_data, primary_image=image)
    merfish_viz.run()
