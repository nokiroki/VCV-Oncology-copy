import sys
from typing import Optional, Tuple

from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from .design import Ui_MainWindow
from .design_save import Ui_MainWindow as Ui_SaveWindow
from .design_test import Ui_MainWindow as Ui_TestWindow


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    filename_signal = pyqtSignal(str)
    test_window_signal = pyqtSignal()
    close_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.photo_btn.clicked.connect(self.btn_clicked)
        self.test_btn.clicked.connect(self.btn_clicked)
        self.off_btn.clicked.connect(self.btn_clicked)

    def btn_clicked(self) -> None:
        sender = self.sender()
        if sender == self.photo_btn:
            filename = QFileDialog.getOpenFileName(self, 'Open File', '.', 'Images (*.jpg *.png)')[0]
            if filename != '':
                self.filename_signal.emit(filename)
        elif sender == self.test_btn:
            self.test_window_signal.emit()
        else:
            self.close_signal.emit()

    def connect_slot_to_filename(self, slot: Optional[str]):
        self.filename_signal.connect(slot)

    def connect_slot_to_close(self, slot: Optional[str]):
        self.close_signal.connect(slot)

    def connect_slot_to_test(self, slot: Optional[str]):
        self.test_window_signal.connect(slot)


class SaveWindow(QtWidgets.QMainWindow, Ui_SaveWindow):
    snap_back = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.cancel_btn.clicked.connect(self.btn_clicked)
        self.save_btn.clicked.connect(self.btn_clicked)

    def setup_image(self, filename: str):
        pixmap = QPixmap(filename)
        self.image_label.setPixmap(pixmap)

    def btn_clicked(self) -> None:
        sender = self.sender()

        if sender == self.save_btn:
            is_local = self.get_saving_way()
            print(is_local)

        self.snap_back.emit()

    def connect_slot_to_snap_back(self, slot: Optional[str]):
        self.snap_back.connect(slot)

    def get_saving_way(self) -> bool:
        message_box = QMessageBox(self)
        message_box.setWindowTitle('Saving...')
        message_box.setText('Choose a way to save an image')

        btn_local = message_box.addButton('Locally', QMessageBox.YesRole)
        message_box.addButton('Cloud', QMessageBox.NoRole)

        message_box.exec_()

        if message_box.clickedButton() == btn_local:
            return True

        return False

    # TODO
    def save_locally(self): ...
    # TODO
    def save_cloud(self): ...


class TestWindow(QtWidgets.QMainWindow, Ui_TestWindow):

    snap_back = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.state_flag = False
        self.current_filename = None

        self.image_btn.clicked.connect(self._btn_clicked)
        self.get_score_btn.clicked.connect(self._btn_clicked)
        self.retry_btn.clicked.connect(self._btn_clicked)
        self.cancel_btn.clicked.connect(self._btn_clicked)

    def _change_state(self):
        self.state_flag = not self.state_flag

        self.image_btn.setEnabled(not self.state_flag)
        self.get_score_btn.setEnabled(self.state_flag)
        self.retry_btn.setEnabled(self.state_flag)

    def _btn_clicked(self):
        sender = self.sender()

        if sender == self.image_btn:
            filename = QFileDialog.getOpenFileName(self, 'Open File', '.', 'Images (*.jpg *.png)')[0]
            if filename != '':
                self.upload_image_on_start(filename)
        elif sender == self.get_score_btn:
            score = self._get_score()
            self.score_line.setText(str(score))
        elif sender == self.retry_btn:
            self.current_filename = None
            self._change_state()
            self.label.clear()
        else:
            self.snap_back.emit()

    # TODO (in different thread as inference can take a lot of time)
    def _get_score(self) -> float:
        return .86

    def upload_image_on_start(self, filename: str):
        if self.current_filename != filename:
            self.label.setPixmap(QPixmap(filename))
            self._change_state()
            self.current_filename = filename

    def connect_slot_to_snap_back(self, slot: Optional[str]):
        self.snap_back.connect(slot)


class MainStacked(QtWidgets.QStackedWidget):

    def __init__(self, windows_list: Tuple[MainWindow, SaveWindow, TestWindow]):
        super().__init__()
        self.current_image = None

        self.main_window, self.save_image_window, self.test_window = windows_list

        self.main_window.connect_slot_to_filename(self.filename_slot)
        self.main_window.connect_slot_to_close(self.close_manually)
        self.main_window.connect_slot_to_test(self.open_test_window)

        self.save_image_window.connect_slot_to_snap_back(self.snap_back_slot)

        self.test_window.connect_slot_to_snap_back(self.snap_back_slot)

        self.addWidget(self.main_window)
        self.addWidget(self.save_image_window)
        self.addWidget(self.test_window)

    @pyqtSlot(str)
    def filename_slot(self, filename: str):
        self.save_image_window.setup_image(filename)
        self.current_image = filename
        self.setCurrentIndex(1)

    @pyqtSlot()
    def snap_back_slot(self):
        self.setCurrentIndex(0)

    @pyqtSlot()
    def close_manually(self):
        self.close()

    @pyqtSlot()
    def open_test_window(self):
        self.setCurrentIndex(2)
        if self.current_image is not None:
            self.test_window.upload_image_on_start(self.current_image)

    @staticmethod
    def start_main(size: Optional[Tuple[int, int]] = None) -> None:
        app = QtWidgets.QApplication(sys.argv)
        window_list = (
            MainWindow(),
            SaveWindow(),
            TestWindow()
        )
        w = MainStacked(window_list)
        w.show()
        if size is not None:
            w.resize(size[0], size[1])
        app.exec_()

