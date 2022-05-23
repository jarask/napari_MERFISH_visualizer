import math

import pandas as pd
import numpy as np
from PyQt5 import QtWebEngineWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QUrl

# from PyQt5.QtGui import QPalette, QColor
import napari
from matplotlib import pyplot as plt
from magicgui import magicgui, widgets

# import plotly.express as px
import plotly.offline
from plotly.graph_objects import *
import os
import sys
from plotly.subplots import make_subplots

# import tifffile
# from starfish.types import Levels, Features, Axes
from tqdm import tqdm


class PlotlyWidget(QtWebEngineWidgets.QWebEngineView):
    """
    Class to create a Plotly Widget in a QApplication instance
    This relies on creating a plot from a local HTML file, which is created in self.createPlot().
    """

    def __init__(self):
        # Create a QApplication instance or use the existing one if it exists
        self.app = (
            QApplication.instance()
            if QApplication.instance()
            else QApplication(sys.argv)
        )
        super().__init__()
        # Set the path to a local html file
        self.file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "Visualizer_files/temp.html")
        )
        # Initialize empty plot
        self.create_subplots()

    def create_subplots(self):
        """
        Method for creating two empty bar plots
        """
        # Old method where two subplots were made
        # fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'bar'}, {'type': 'bar'}]],
        #                     subplot_titles=("Selected points overview", "Place-holder plot"))
        # fig.add_bar(x=[], orientation='h', row=1, col=2)
        # fig.update_xaxes(title_text='None', row=1, col=2)
        # fig.update_xaxes(title_text='None', row=1, col=2)
        fig = make_subplots(
            rows=1,
            cols=1,
            specs=[[{"type": "bar"}]],
            subplot_titles=["Select something to plot!"],
        )
        fig.add_bar(x=[], orientation="h", row=1, col=1)
        fig.update_xaxes(title_text="None", row=1, col=1)
        fig.update_yaxes(title_text="None", row=1, col=1)
        fig.update_layout(
            margin=dict(l=10, r=10, t=20, b=0),
            template="plotly_dark",
            paper_bgcolor="#262930",
        )

        plotly.offline.plot(fig, filename=self.file_path, auto_open=False)
        self.load(QUrl.fromLocalFile(self.file_path))

    def update_subplots(self, x, y, title: str, x_label: str, y_label: str):
        """
        Method for updating two bar plot from a Pandas DataFrame
        :param x:
        :param y:
        :param title:
        :param x_label:
        :param y_label:
        """
        # Old method where two subplots were made
        # fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'bar'}, {'type': 'bar'}]],
        #                     subplot_titles=("Selected points overview", "Place-holder plot"))
        # fig.add_bar(x=[], y=[], orientation='h', row=1, col=2)
        # fig.update_xaxes(title_text='None', row=1, col=2)
        # fig.update_xaxes(title_text='None', row=1, col=2)

        fig = make_subplots(
            rows=1, cols=1, specs=[[{"type": "bar"}]], subplot_titles=[title]
        )
        fig.add_bar(x=x, y=y, orientation="h", row=1, col=1)
        fig.update_xaxes(title_text=x_label, row=1, col=1)
        fig.update_yaxes(title_text=y_label, row=1, col=1)
        fig.update_layout(
            margin=dict(l=10, r=10, t=20, b=0),
            template="plotly_dark",
            paper_bgcolor="#262930",
        )

        plotly.offline.plot(fig, filename=self.file_path, auto_open=False)
        self.load(QUrl.fromLocalFile(self.file_path))


def gene_text_reader(gene_text: str):
    if "\n" in gene_text:
        gene_text = gene_text.replace("\n", "")
    if "," in gene_text:
        temp_gene_list = list(gene_text.split(","))
    elif " " in gene_text:
        temp_gene_list = list(gene_text.split(" "))
    else:
        # Assume there's only one gene?
        temp_gene_list = [gene_text]
    # Clean up the list
    temp_gene_list = list(map(lambda x: x.replace(" ", ""), temp_gene_list))
    temp_gene_list = list(map(lambda x: x.upper(), temp_gene_list))
    while "" in temp_gene_list:
        temp_gene_list.remove("")

    return temp_gene_list


def gene_file_reader(filename: str):
    try:
        with open(filename, "r") as f:
            file_content = f.read()
            print("Read file " + filename)
        if not file_content:
            print("No data in file " + filename)
        else:
            return gene_text_reader(file_content)
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))
    except:  # Handle other exceptions such as attribute errors
        print("Unexpected error:", sys.exc_info()[0])


class NapariMERFISH:
    """
    Class for displaying MERFISH data in napari
    """

    def __init__(
        self,
        labeled_spots_data,
        primary_image,
        nuclei_image=None,
        starfish_masks=None,
        expression_data=None,
    ):
        self.spots_data = self._load_spots(labeled_data_starfish=labeled_spots_data)
        self.primary_image = primary_image

        if nuclei_image is not None:
            self.dapi_image = nuclei_image
        else:
            self.dapi_image = None
        if starfish_masks is not None:
            self.masks = starfish_masks.to_label_image().label_image
        else:
            self.masks = None
        if expression_data is not None:
            self.expr_data = expression_data
            # Find the mean for each column (gene) and divide this with
            for (index, _) in enumerate(self.expr_data.columns):
                self.expr_data.loc[:, self.expr_data.columns[index]] = (
                    self.expr_data.loc[:, self.expr_data.columns[index]]
                    / self.expr_data.loc[:, self.expr_data.columns[index]][
                        self.expr_data.loc[:, self.expr_data.columns[index]] != 0
                    ].mean()
                )
        else:
            self.expr_data = None

        # Napari-specific initializations for handling update of points and their properties
        self.lookup_spots_data = self.spots_data.copy()
        self.point_scale_factor = 2
        # global last_selected
        # last_selected = set()
        self.viewer = None
        self.primary_layer = None
        self.dapi_layer = None
        self.points_layer = None
        self.labels_layer = None
        self.coloring_method = None
        self.pbar = None
        self.plot_mode = ""
        self.last_selected = set()

    def _load_spots(self, labeled_data_starfish):
        # Read in the spots data
        spots_data = labeled_data_starfish
        # Assign IDs to each spot
        spots_data["id"] = range(len(spots_data))
        # Increment the cell_id for the purpose of relating the spots correctly to a cell
        for (index, cell_id) in enumerate(spots_data.cell_id):
            if cell_id != "nan":
                spots_data.cell_id.iloc[index] = str(int(int(cell_id) + 1))
        # Assign colors based on genes
        cats = spots_data.target.unique().tolist()
        colors = []
        for inst_index in range(len(cats)):
            if "Blank" in cats[inst_index]:
                colors.append((0.25, 0.25, 0.25))
            else:
                mod_index = inst_index % 60
                if mod_index < 20:
                    color_rgba = plt.cm.tab20(mod_index % 20)
                elif mod_index < 40:
                    color_rgba = plt.cm.tab20b(mod_index % 20)
                elif mod_index < 60:
                    color_rgba = plt.cm.tab20b(mod_index % 20)
                colors.append(color_rgba)
        cat_colors = dict(zip(cats, colors))
        temp_colors = []
        for gene in spots_data.target:
            for i in cat_colors:
                if gene == i:
                    temp_colors.append(cat_colors.get(i))
        spots_data["gene_colors"] = temp_colors

        # Assign colors based on cell id
        cats = spots_data.cell_id.unique().tolist()
        colors = []
        for inst_index in range(len(cats)):
            if "nan" in cats[inst_index]:
                colors.append((0.25, 0.25, 0.25))
            else:
                mod_index = inst_index % 60
                if mod_index < 20:
                    color_rgba = plt.cm.tab20(mod_index % 20)
                elif mod_index < 40:
                    color_rgba = plt.cm.tab20b(mod_index % 20)
                elif mod_index < 60:
                    color_rgba = plt.cm.tab20b(mod_index % 20)
                colors.append(color_rgba)
        cat_colors = dict(zip(cats, colors))
        temp_colors = []
        for cell_id in spots_data.cell_id:
            for i in cat_colors:
                if cell_id == i:
                    temp_colors.append(cat_colors.get(i))
        spots_data["cell_colors"] = temp_colors
        return spots_data

    def _init_widgets(self):
        # Initialize the Plotly widget and add it to Napari
        plt_viewer = PlotlyWidget()
        plt_viewer.setContentsMargins(1, 1, 1, 1)
        # plt_viewer.show()

        # Widget for toggling between the gene and cell coloring of the points
        colorMethod = widgets.RadioButtons(
            choices=["Gene", "Cell"], label=None, value="Gene"
        )
        colorMethod.max_height = 100
        label_color = widgets.Label(
            value="Select coloring method for spots:              "
        )
        colorContainer = widgets.Container(widgets=[label_color, colorMethod])
        colorContainer.max_height = 120
        widget_right1 = self.viewer.window.add_dock_widget(
            colorContainer, name="Color method", area="right"
        )

        # global coloring_method
        self.coloring_method = colorMethod.current_choice

        self.pbar.update(10)

        @colorMethod.changed.connect
        def change_colormethod():
            """
            Changes the coloring method based on the choice in the colorMethod widget.
            :return:
            """
            self.coloring_method = colorMethod.current_choice
            if self.masks is not None:
                if self.coloring_method == "Gene":
                    self.points_layer.face_color = (
                        self.lookup_spots_data.gene_colors.tolist()
                    )
                elif self.coloring_method == "Cell":
                    self.points_layer.face_color = (
                        self.lookup_spots_data.cell_colors.tolist()
                    )
            else:
                if self.coloring_method == "Gene":
                    self.points_layer.face_color = (
                        self.lookup_spots_data.gene_colors.tolist()
                    )
                elif self.coloring_method == "Cell":
                    napari.utils.notifications.show_info(
                        message="No cell-info available. Showing gene colors"
                    )
                    self.points_layer.face_color = (
                        self.lookup_spots_data.gene_colors.tolist()
                    )

        self.pbar.update(10)

        @self.points_layer.events.highlight.connect
        def handle_selection(event):
            """
            Handles the selected points and sends them to the plot_updater.
            """
            if self.points_layer.mode == "select":
                selected = self.points_layer.selected_data
                if selected != self.last_selected:
                    self.last_selected = selected
                    if selected != set():
                        if len(selected) > 1:
                            selection_widget.value = (
                                "%s spots selected            " % len(selected)
                            )
                            self.plot_updater(plot_widget=plt_viewer)
                        elif len(selected) == 1:
                            selection_widget.value = (
                                "%s spot selected             " % len(selected)
                            )

        self.pbar.update(10)

        if self.masks is not None:
            @self.labels_layer.events.selected_label.connect
            def cell_selection(event):
                """
                Handles the selection of points corresponding to the
                """
                if self.labels_layer.selected_label != 0:
                    cell_id_selected = self.labels_layer.selected_label
                    # Get the indices of the corresponding rows with the cell_id
                    cell_id_selected = str(cell_id_selected)
                    cell_spots_set = set(
                        self.lookup_spots_data[
                            self.lookup_spots_data.cell_id == cell_id_selected
                        ].index
                    )
                    # Set the current points selection to the set
                    self.points_layer.mode = "select"
                    self.points_layer.selected_data = cell_spots_set

        selection_widget = widgets.Label(value="No spots selected            ")
        # Prepare gene list to the container widget
        gene_list = list(self.spots_data.target.unique().tolist())
        gene_list.sort(key=str.lower)
        # Create widgets for the container
        showAllButton = widgets.PushButton(value=False, text="Show all genes")
        runButton = widgets.PushButton(value=False, text="Show selected genes")
        geneSelection = widgets.Select(choices=gene_list)
        geneText = widgets.TextEdit(
            value="", tooltip="Genes should be separated by comma or whitespace"
        )
        geneText.max_height = 100
        geneFile = widgets.FileEdit(
            value="",
            tooltip="File should be '.txt' and genes should be separated by comma or whitespace",
        )

        # Create labels for each widget in container
        label1 = widgets.Label(value="Select genes from a list:              ")
        label2 = widgets.Label(value="Type in genes manually:                ")
        label3 = widgets.Label(value="Upload file containing genes:          ")
        # Create the container and dock it to the viewer window
        container = widgets.Container(
            widgets=[
                selection_widget,
                showAllButton,
                runButton,
                label1,
                geneSelection,
                label2,
                geneText,
                label3,
                geneFile,
            ]
        )
        container.max_height = 500
        widget_right4 = self.viewer.window.add_dock_widget(
            container, name="Gene selection", area="right"
        )

        self.pbar.update(10)

        @showAllButton.clicked.connect
        def show_all_points_layer():
            """
            Updates the points layer to show all genes.
            :return: The temporary dataframe containing all genes again
            """
            temp_df = self.spots_data
            self.lookup_spots_data = temp_df.copy()
            self.points_layer.data = np.column_stack((temp_df.y, temp_df.x))
            if self.coloring_method == "Gene":
                self.points_layer.face_color = (
                    self.lookup_spots_data.gene_colors.tolist()
                )
            elif self.coloring_method == "Cell":
                self.points_layer.face_color = (
                    self.lookup_spots_data.cell_colors.tolist()
                )
            else:
                self.points_layer.face_color = (
                    self.lookup_spots_data.gene_colors.tolist()
                )

            self.points_layer.size = temp_df.radius * self.point_scale_factor

            self.points_layer.properties = {
                "Gene": np.array(self.lookup_spots_data.target),
                "Cell": np.array(self.lookup_spots_data.cell_id),
                "ID": np.array(self.lookup_spots_data.id),
                "x": np.array(self.lookup_spots_data.x),
                "y": np.array(self.lookup_spots_data.y),
            }
            # Deselect all points
            self.points_layer.mode = "select"
            self.points_layer.selected_data = set()

        plt_widget = self.viewer.window.add_dock_widget(
            plt_viewer, name="Plotting widget", area="right"
        )
        plt_widget.min_height = 400
        self.pbar.update(10)

        @runButton.clicked.connect
        def update_points_layer():
            """
            Updates the points layer based on the selected genes in geneSelection.
            :return: The temporary dataframe containing the selected genes
            """
            # global lookup_spots_data
            if geneSelection.current_choice:
                temp_list = list(geneSelection.current_choice)
            elif geneText.value:
                temp_list = gene_text_reader(geneText.value)
            elif geneFile.value.as_posix() != ".":
                temp_list = gene_file_reader(geneFile.value.as_posix())
            else:
                napari.utils.notifications.show_info(message="No genes selected!")
                # temp_list = list()
                return

            temp_df = pd.DataFrame(columns=self.spots_data.columns)
            if all(item in list(geneSelection.choices) for item in temp_list):
                print("All items you selected are actual genes!")
            elif any(item in list(geneSelection.choices) for item in temp_list):
                print("Not all provided genes were correct \n Showing correct genes")
                napari.utils.notifications.show_info(
                    message="Not all provided genes were correct. Showing correct genes"
                )
            else:
                print("None of your genes are correct...")
                napari.utils.notifications.show_info(
                    message="No provided genes are correct. Not updating layer"
                )
                return
            # Assign the genes to the layer
            for target_gene in temp_list:
                temp_df = temp_df.append(
                    self.spots_data.loc[self.spots_data["target"] == target_gene]
                )
            self.lookup_spots_data = temp_df.copy()
            self.lookup_spots_data.reset_index(inplace=True)

            self.points_layer.data = np.column_stack((temp_df.y, temp_df.x))
            if self.coloring_method == "Gene":
                self.points_layer.face_color = temp_df.gene_colors.tolist()
            elif self.coloring_method == "Cell":
                self.points_layer.face_color = temp_df.cell_colors.tolist()
            self.points_layer.size = temp_df.radius * self.point_scale_factor

            self.points_layer.properties = {
                "Gene": np.array(self.lookup_spots_data.target),
                "Cell": np.array(self.lookup_spots_data.cell_id),
                "ID": np.array(self.lookup_spots_data.id),
                "x": np.array(self.lookup_spots_data.x),
                "y": np.array(self.lookup_spots_data.y),
            }
            # Deselect all points
            self.points_layer.mode = "select"
            self.points_layer.selected_data = set()

        # Save widget
        fileEditWidget = widgets.FileEdit(value="", label="Output file", mode="w")
        saveSelection = widgets.RadioButtons(
            choices=["Selected", "Visible", "All"],
            label="Points to save",
            value="Selected",
        )
        saveButton = widgets.PushButton(value=False, text="Save points")
        saveContainer = widgets.Container(
            widgets=[fileEditWidget, saveSelection, saveButton]
        )
        saveContainer.max_height = 200
        save_widget = self.viewer.window.add_dock_widget(
            saveContainer, name="Saving points", area="left"
        )

        @saveButton.clicked.connect
        def save_selection():
            # Check if the path has csv at the end
            path = fileEditWidget.value.as_posix()
            if not path.endswith(".csv"):
                napari.utils.notifications.show_info(
                    message="Your path does not contain the -csv extension!"
                )
                return
            # Save the respective data
            elif saveSelection.value == "Selected":
                # Get the data from last_selected
                self.lookup_spots_data.iloc[list(self.last_selected)].to_csv(path)
                napari.utils.notifications.show_info(message="Points saved!")
            elif saveSelection.value == "Visible":
                self.lookup_spots_data.to_csv(path)
                napari.utils.notifications.show_info(message="Points saved!")
            elif saveSelection.value == "All":
                self.spots_data.to_csv(path)
                napari.utils.notifications.show_info(message="Points saved!")

        # Plot mode selection
        plotMode = widgets.Combobox(
            choices=["Value counts of selection", "Fold change in expression"],
            value="Value counts of selection",
            label="Plotting mode",
        )
        plotModeWidget = self.viewer.window.add_dock_widget(
            plotMode, name="Plotting mode", area="left"
        )
        plotModeWidget.min_height = 300
        # Set the plot mode to its default state
        self.plot_mode = plotMode.value

        @plotMode.changed.connect
        def set_plot_mode():
            # Set the plot mode to the current value of the plotMode widget
            self.plot_mode = plotMode.value

        self.pbar.update(10)

    def plot_updater(self, plot_widget):
        """
        Takes selected points, reformat them, and sends them to PlotlyWidget.update_plot()
        :param plotting_mode:
        :param plot_widget:
        """
        if self.plot_mode == "Value counts of selection":
            gene_list = list()
            for point in self.last_selected:
                gene_list.append(self.lookup_spots_data.target.iloc[point])
            df = pd.DataFrame(gene_list, columns=["gene"])
            sum_df = pd.DataFrame(df.gene.value_counts())
            plot_widget.update_subplots(
                x=sum_df.iloc[:, 0],
                y=sum_df.index,
                title="Selected points overview",
                x_label="Count",
                y_label="RNA",
            )
        elif self.plot_mode == "Fold change in expression":
            if self.masks is None:
                napari.utils.notifications.show_info(message="Unable to plot cell-info with no cell-masks")
            try:
                # Get the expression data corresponding to the given cell
                expr_df = self.expr_data.iloc[self.labels_layer.selected_label - 1].to_frame('fold_change')
                # Keep only the genes in lookup_spots_data
                expr_df = expr_df.iloc[expr_df.index.isin(self.lookup_spots_data.target.unique())]
                # Remove the genes, which are 0
                expr_df = expr_df.loc[expr_df.fold_change != 0]
                # Convert to log2 fold-change
                expr_df.fold_change = [math.log2(float(i)) for i in expr_df.fold_change]
                expr_df.sort_values(by="fold_change", ascending=False, inplace=True)
                plot_widget.update_subplots(
                    x=expr_df.iloc[:, 0],
                    y=expr_df.index,
                    title="Expression fold change",
                    x_label="log2 fold change",
                    y_label="RNA",
                )
            except IndexError:
                print("Cell ID: %s" % self.labels_layer.selected_label)
                print("Expr_data indices:")
                print(self.expr_data.index)

    def run(self):
        """
        Run the created napari instance
        """
        self.pbar = tqdm(desc="Starting Napari", total=100)
        points_properties = {
            "Gene": np.array(self.lookup_spots_data.target),
            "Cell": np.array(self.lookup_spots_data.cell_id),
            "ID": np.array(self.lookup_spots_data.id),
            "x": np.array(self.lookup_spots_data.x),
            "y": np.array(self.lookup_spots_data.y),
        }
        self.viewer = napari.Viewer()
        if self.primary_image is not None:
            self.primary_layer = self.viewer.add_image(
                self.primary_image, name="Primary image (max)", visible=False
            )
            self.pbar.update(10)
        if self.dapi_image is not None:
            self.dapi_layer = self.viewer.add_image(self.dapi_image, name="Nuclei")
            self.pbar.update(10)
        if self.masks is not None:
            self.labels_layer = self.viewer.add_labels(
                self.masks, name="Cells", opacity=0.25
            )
            self.pbar.update(10)
        self.points_layer = self.viewer.add_points(
            np.column_stack((self.lookup_spots_data.y, self.lookup_spots_data.x)),
            size=self.lookup_spots_data.radius * self.point_scale_factor,
            face_color=self.lookup_spots_data.gene_colors.tolist(),
            properties=points_properties,
            edge_width=0,
        )
        # Initialize widgets
        self._init_widgets()
        self.pbar.update(10)
        # Finally, run the napari-instance
        napari.run()
