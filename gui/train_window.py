import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QObject
from PyQt6.QtWidgets import QMainWindow, QLayout, QVBoxLayout

from degann import networks
from degann.networks import get_all_loss_functions
from degann.networks import get_all_optimizers
from degann.expert import selector
from gui.constants import (
    minimum_police_size,
    expand_minimum_police_size,
    font_12pt,
    font_6pt,
    fixed_police_size,
    expand_fixed_police_size,
    param_desc_phrases,
)


class SelectAndTrainLayout:
    def __init__(self, centralwidget: QtWidgets.QWidget):
        self.centralwidget = centralwidget
        self.setup_ui()

    def setup_ui(self) -> None:
        self.main_area = QtWidgets.QVBoxLayout(self.centralwidget)
        self.main_area.setObjectName("SelectAndTrainLayout")

        back_button = QtWidgets.QPushButton(parent=self.centralwidget)
        back_button.setSizePolicy(expand_fixed_police_size)
        back_button.setMinimumSize(QtCore.QSize(50, 40))
        back_button.setObjectName("back_button")
        back_button.setText("Back")
        self.main_area.addWidget(back_button)

        self.select_and_train_area = QtWidgets.QHBoxLayout()
        self.select_area = self.setup_tags_area()
        self.train_area = self.setup_train_area()
        self.select_and_train_area.addLayout(self.select_area)
        self.select_and_train_area.addLayout(self.train_area)

        self.main_area.addLayout(self.select_and_train_area)

    def setup_train_area(self) -> QtWidgets.QVBoxLayout:
        train_area = QtWidgets.QVBoxLayout()
        parameters_area = QtWidgets.QHBoxLayout()
        parameter_desc_area = self.setup_parameters_desc_area()
        parameter_value_area = self.setup_parameters_value_area()
        parameters_area.addLayout(parameter_desc_area)
        parameters_area.addLayout(parameter_value_area)
        train_area.addLayout(parameters_area)

        train_button = QtWidgets.QPushButton(parent=self.centralwidget)
        train_button.setObjectName("train_button")
        train_button.setText("train")
        train_area.addWidget(train_button)

        return train_area

    def setup_parameters_desc_area(self) -> QtWidgets.QVBoxLayout:
        param_desc_area = QtWidgets.QVBoxLayout()
        param_desc_area.setObjectName("param_desc_area")

        text_browser_count = len(param_desc_phrases)
        for i in range(text_browser_count):
            textBrowser = QtWidgets.QTextBrowser(parent=self.centralwidget)
            textBrowser.setSizePolicy(minimum_police_size)
            textBrowser.setMinimumSize(QtCore.QSize(0, 60))
            textBrowser.setObjectName(f"textBrowser_{i + 1}")
            textBrowser.setFontPointSize(12)
            textBrowser.setPlainText(param_desc_phrases[i])
            param_desc_area.addWidget(textBrowser)
        return param_desc_area

    def setup_parameters_value_area(self) -> QtWidgets.QVBoxLayout:
        param_value_area = QtWidgets.QVBoxLayout()
        param_value_area.setObjectName("param_value_area")

        loss_func_combobox = QtWidgets.QComboBox(parent=self.centralwidget)
        loss_func_combobox.setSizePolicy(minimum_police_size)
        loss_func_combobox.setMinimumSize(QtCore.QSize(0, 60))
        loss_func_combobox.setObjectName("loss_func_combobox")
        for loss_func in get_all_loss_functions().keys():
            loss_func_combobox.addItem(loss_func)
        loss_func_combobox.setCurrentText(str(selector.BaseParameters.loss_function))
        param_value_area.addWidget(loss_func_combobox)

        metric_threshold_text = QtWidgets.QLineEdit(parent=self.centralwidget)
        metric_threshold_text.setSizePolicy(minimum_police_size)
        metric_threshold_text.setMinimumSize(QtCore.QSize(0, 60))
        metric_threshold_text.setObjectName("metric_threshold_text")
        metric_threshold_text.setText(str(selector.BaseParameters.metric_threshold))
        param_value_area.addWidget(metric_threshold_text)

        optimizer_combobox = QtWidgets.QComboBox(parent=self.centralwidget)
        optimizer_combobox.setSizePolicy(minimum_police_size)
        optimizer_combobox.setMinimumSize(QtCore.QSize(0, 60))
        optimizer_combobox.setObjectName("optimizer_combobox")
        for optimizer in get_all_optimizers().keys():
            optimizer_combobox.addItem(optimizer)
        optimizer_combobox.setCurrentText(str(selector.BaseParameters.optimizer))
        param_value_area.addWidget(optimizer_combobox)

        return param_value_area

    def setup_tags_area(self) -> QVBoxLayout:
        tags_area = QtWidgets.QVBoxLayout()
        tags_area.setObjectName("tags_area")
        tags_value_area = QtWidgets.QFormLayout()
        # tags_value_area.setSizeConstraint(
        #     QtWidgets.QLayout.SizeConstraint.SetMinimumSize
        # )
        tags_value_area.setObjectName("tags_value_area")

        eq_type_text = QtWidgets.QTextEdit(parent=self.centralwidget)
        eq_type_text.setSizePolicy(minimum_police_size)
        eq_type_text.setMinimumSize(QtCore.QSize(10, 60))
        eq_type_text.setFontPointSize(12)
        eq_type_text.setPlainText("Type of equation")
        eq_type_text.setObjectName("eq_type_text")
        tags_value_area.setWidget(
            0, QtWidgets.QFormLayout.ItemRole.LabelRole, eq_type_text
        )

        eq_type_combobox = QtWidgets.QComboBox(parent=self.centralwidget)
        eq_type_combobox.setSizePolicy(minimum_police_size)
        eq_type_combobox.setMinimumSize(QtCore.QSize(0, 60))
        eq_type_combobox.setFont(font_12pt)
        eq_type_combobox.setObjectName("eq_type_combobox")
        for tag in selector.ExpertSystemTags.equation_type:
            eq_type_combobox.addItem(tag.value)
        eq_type_combobox.setCurrentText("unknown")
        tags_value_area.setWidget(
            0, QtWidgets.QFormLayout.ItemRole.FieldRole, eq_type_combobox
        )

        precision_text = QtWidgets.QTextEdit(parent=self.centralwidget)
        precision_text.setEnabled(True)
        precision_text.setSizePolicy(minimum_police_size)
        precision_text.setMinimumSize(QtCore.QSize(0, 60))
        precision_text.setFont(font_12pt)
        precision_text.setPlainText("Required level of accuracy")
        precision_text.setObjectName("precision_text")
        tags_value_area.setWidget(
            1, QtWidgets.QFormLayout.ItemRole.LabelRole, precision_text
        )

        precision_combobox = QtWidgets.QComboBox(parent=self.centralwidget)
        precision_combobox.setSizePolicy(minimum_police_size)
        precision_combobox.setMinimumSize(QtCore.QSize(0, 60))
        precision_combobox.setFont(font_12pt)
        precision_combobox.setObjectName("precision_combobox")
        for precision in selector.ExpertSystemTags.model_precision:
            precision_combobox.addItem(precision.value)
        precision_combobox.setCurrentText("median")
        tags_value_area.setWidget(
            1, QtWidgets.QFormLayout.ItemRole.FieldRole, precision_combobox
        )

        work_time_text = QtWidgets.QTextEdit(parent=self.centralwidget)
        work_time_text.setSizePolicy(minimum_police_size)
        work_time_text.setMinimumSize(QtCore.QSize(0, 60))
        work_time_text.setFont(font_12pt)
        work_time_text.setPlainText("Running time of a trained neural network")
        work_time_text.setObjectName("work_time_text")
        tags_value_area.setWidget(
            2, QtWidgets.QFormLayout.ItemRole.LabelRole, work_time_text
        )

        work_time_combobox = QtWidgets.QComboBox(parent=self.centralwidget)
        work_time_combobox.setSizePolicy(minimum_police_size)
        work_time_combobox.setMinimumSize(QtCore.QSize(0, 60))
        work_time_combobox.setFont(font_12pt)
        work_time_combobox.setObjectName("work_time_combobox")
        for work_time in selector.ExpertSystemTags.predict_time:
            work_time_combobox.addItem(work_time.value)
        work_time_combobox.setCurrentText("medium")
        tags_value_area.setWidget(
            2, QtWidgets.QFormLayout.ItemRole.FieldRole, work_time_combobox
        )

        data_size_text = QtWidgets.QTextEdit(parent=self.centralwidget)
        data_size_text.setSizePolicy(minimum_police_size)
        data_size_text.setMinimumSize(QtCore.QSize(10, 60))
        data_size_text.setFont(font_12pt)
        data_size_text.setObjectName("data_size_text")
        data_size_text.setPlainText("Number of samples in passed data")
        tags_value_area.setWidget(
            3, QtWidgets.QFormLayout.ItemRole.LabelRole, data_size_text
        )

        data_size_combobox = QtWidgets.QComboBox(parent=self.centralwidget)
        data_size_combobox.setSizePolicy(minimum_police_size)
        data_size_combobox.setMinimumSize(QtCore.QSize(0, 60))
        data_size_combobox.setFont(font_12pt)
        data_size_combobox.setObjectName("data_size_combobox")
        for data_size in selector.ExpertSystemTags.data_size:
            data_size_combobox.addItem(data_size.value)
        data_size_combobox.setCurrentText("auto")
        tags_value_area.setWidget(
            3, QtWidgets.QFormLayout.ItemRole.FieldRole, data_size_combobox
        )
        tags_area.addLayout(tags_value_area)

        selection_button = QtWidgets.QPushButton(parent=self.centralwidget)
        selection_button.setObjectName("select_button")
        selection_button.setText("select parameters")
        tags_area.addWidget(selection_button)

        return tags_area
