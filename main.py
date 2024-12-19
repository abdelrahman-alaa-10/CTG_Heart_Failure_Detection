from PyQt5.QtWidgets import QMainWindow, QButtonGroup, QProgressBar ,QMessageBox, QApplication,QPushButton,QListWidget, QDoubleSpinBox ,QSpinBox, QWidget, QLabel ,  QSlider, QRadioButton, QComboBox, QTableWidget, QTableWidgetItem, QCheckBox,QMenu,QTextEdit, QDialog, QFileDialog, QInputDialog, QSizePolicy,QScrollArea,QVBoxLayout,QHBoxLayout,QFrame
from PyQt5.uic import loadUi
from PyQt5.QtCore import QTimer

import sys
import os


class main(QMainWindow):
    def __init__(self):
        super(main, self).__init__()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        loadUi("main.ui", self)

        self.original_graph=self.findChild(QFrame,"original_graph")
        self.filtered_graph=self.findChild(QFrame,"filtered_graph")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = main()
    window.show()
    window.showMaximized()
    sys.exit(app.exec_())