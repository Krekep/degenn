import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QMainWindow, QFileDialog

import degann.expert.pipeline
import degann.networks
from degann.expert.tags import (
    map_value_to_equation_type,
    map_value_to_precision,
    map_value_to_predict,
    map_value_to_data_size,
)
from gui.export_window import ExportNNLayout
from gui.load_dataset_window import LoadDatasetLayout
from gui.train_window import SelectAndTrainLayout
from degann.networks.imodel import IModel
from degann.expert import selector, ExpertSystemTags


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setup_ui()

        self.parameters = selector.BaseParameters()

    def setup_ui(self) -> None:
        self.setObjectName("MainWindow")

        self.set_load_area()

        self.statusbar = QtWidgets.QStatusBar(parent=self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        QtCore.QMetaObject.connectSlotsByName(self)

    def setup_load_area(self) -> None:
        self.load_dataset_widget = QtWidgets.QWidget(parent=self)
        self.load_dataset_widget.setObjectName("load_dataset_widget")
        load_dataset_layout = LoadDatasetLayout(self.load_dataset_widget)
        load_dataset_button = self.load_dataset_widget.findChild(
            QtWidgets.QPushButton, "load_dataset_button"
        )
        load_dataset_button.clicked.connect(self.load_dataset)

    def set_load_area(self) -> None:
        self.setup_load_area()
        self.setCentralWidget(self.load_dataset_widget)
        # self.load_dataset_widget.show()
        # self.select_and_train_widget.hide()
        # self.export_widget.hide()
        self.resize(600, 400)

    def setup_select_area(self) -> None:
        self.select_and_train_widget = QtWidgets.QWidget(parent=self)
        self.select_and_train_widget.setObjectName("select_and_train_widget")
        select_layout = SelectAndTrainLayout(self.select_and_train_widget)
        back_button = self.select_and_train_widget.findChild(
            QtWidgets.QPushButton, "back_button"
        )
        back_button.clicked.connect(self.set_load_area)
        select_button = self.select_and_train_widget.findChild(
            QtWidgets.QPushButton, "select_button"
        )
        select_button.clicked.connect(self.select_parameters)
        train_button = self.select_and_train_widget.findChild(
            QtWidgets.QPushButton, "train_button"
        )
        train_button.clicked.connect(self.start_train)

    def set_select_area(self) -> None:
        self.setup_select_area()
        self.setCentralWidget(self.select_and_train_widget)
        # self.load_dataset_widget.hide()
        # self.select_and_train_widget.show()
        # self.export_widget.hide()
        self.resize(600, 400)

    def setup_export_area(self) -> None:
        self.export_widget = QtWidgets.QWidget(parent=self)
        self.export_widget.setObjectName("export_widget")
        export_layout = ExportNNLayout(self.export_widget)
        back_button = self.export_widget.findChild(QtWidgets.QPushButton, "back_button")
        back_button.clicked.connect(self.set_select_area)
        export_button = self.export_widget.findChild(
            QtWidgets.QPushButton, "export_button"
        )
        export_button.clicked.connect(self.export_to_file)
        export_cpp_button = self.export_widget.findChild(
            QtWidgets.QPushButton, "export_cpp_button"
        )
        export_cpp_button.clicked.connect(self.export_to_cpp)

    def set_export_area(self) -> None:
        self.setup_export_area()
        self.setCentralWidget(self.export_widget)
        # self.load_dataset_widget.hide()
        # self.select_and_train_widget.hide()
        # self.export_widget.show()
        self.resize(600, 400)

    def load_dataset(self) -> None:
        path_to_data, _ = QFileDialog.getOpenFileName(
            self, "Select file", "", "Table (*.csv)"
        )
        if path_to_data:
            self.x_dataset_size = int(
                self.load_dataset_widget.findChild(
                    QtWidgets.QPlainTextEdit, "x_dataset_size_plaintext"
                ).toPlainText()
            )
            self.y_dataset_size = int(
                self.load_dataset_widget.findChild(
                    QtWidgets.QPlainTextEdit, "y_dataset_size_plaintext"
                ).toPlainText()
            )
            self.train_data_x, self.train_data_y = np.genfromtxt(
                path_to_data,
                delimiter=",",
                unpack=True,
                usecols=list(range(self.x_dataset_size)),
            ), np.genfromtxt(
                path_to_data,
                delimiter=",",
                usecols=list(
                    range(
                        self.x_dataset_size, self.x_dataset_size + self.y_dataset_size
                    )
                ),
            )
            self.train_data_x = self.train_data_x.reshape((self.x_dataset_size, -1)).T
            self.train_data_y = self.train_data_y.reshape((self.y_dataset_size, -1)).T

            self.statusbar.showMessage("Success load file.")
            self.set_select_area()
        else:
            self.statusbar.showMessage(f"Fail to open file.")

    def select_parameters(self) -> None:
        tags = ExpertSystemTags()
        tags.equation_type = map_value_to_equation_type(
            self.select_and_train_widget.findChild(
                QtWidgets.QComboBox, "eq_type_combobox"
            ).currentText()
        )
        tags.model_precision = map_value_to_precision(
            self.select_and_train_widget.findChild(
                QtWidgets.QComboBox, "precision_combobox"
            ).currentText()
        )
        tags.predict_time = map_value_to_predict(
            self.select_and_train_widget.findChild(
                QtWidgets.QComboBox, "work_time_combobox"
            ).currentText()
        )
        tags.data_size = map_value_to_data_size(
            self.select_and_train_widget.findChild(
                QtWidgets.QComboBox, "data_size_combobox"
            ).currentText()
        )
        self.parameters = selector.suggest_parameters(
            (self.train_data_x, self.train_data_y), tags
        )

        self.select_and_train_widget.findChild(
            QtWidgets.QComboBox, "loss_func_combobox"
        ).setCurrentText(str(self.parameters.loss_function))
        self.select_and_train_widget.findChild(
            QtWidgets.QLineEdit, "metric_threshold_text"
        ).setText(str(self.parameters.metric_threshold))
        self.select_and_train_widget.findChild(
            QtWidgets.QComboBox, "optimizer_combobox"
        ).setCurrentText(str(self.parameters.optimizer))

        self.statusbar.showMessage("Success select parameters")

    def start_train(self) -> None:
        parameters = self.parameters

        parameters.loss_function = self.select_and_train_widget.findChild(
            QtWidgets.QComboBox, "loss_func_combobox"
        ).currentText()
        parameters.metric_threshold = float(
            self.select_and_train_widget.findChild(
                QtWidgets.QLineEdit, "metric_threshold_text"
            ).text()
        )
        parameters.optimizer = self.select_and_train_widget.findChild(
            QtWidgets.QComboBox, "optimizer_combobox"
        ).currentText()

        self.train_loss, self.trained_nn = degann.expert.pipeline.execute_pipeline(
            input_size=self.x_dataset_size,
            output_size=self.y_dataset_size,
            data=(self.train_data_x, self.train_data_y),
            parameters=parameters,
        )

        self.statusbar.showMessage(
            f"Neural network has been successfully trained. Final value of loss function = {self.train_loss}"
        )
        self.set_export_area()

    def export_to_file(self) -> None:
        export_path = self.export_widget.findChild(
            QtWidgets.QPlainTextEdit, "export_nn_plaintext"
        ).toPlainText()
        nn = IModel(
            input_size=self.x_dataset_size,
            block_size=[],
            output_size=self.y_dataset_size,
        )
        nn.from_dict(self.trained_nn)
        nn.export_to_file(export_path)
        self.statusbar.showMessage(f"Success export to {export_path + '.apg'}")

    def export_to_cpp(self) -> None:
        export_path = self.export_widget.findChild(
            QtWidgets.QPlainTextEdit, "export_nn_plaintext"
        ).toPlainText()
        nn = IModel(
            input_size=self.x_dataset_size,
            block_size=[],
            output_size=self.y_dataset_size,
        )
        nn.from_dict(self.trained_nn)
        nn.export_to_cpp(export_path)
        self.statusbar.showMessage(f"Success export to {export_path + '.cpp'}")
