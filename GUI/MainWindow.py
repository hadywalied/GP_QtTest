import asyncio
import concurrent.futures
from PySide2.QtWidgets import (QMainWindow, QAction, QStackedWidget, QSplitter,
                               QPlainTextEdit, QMessageBox, QMenuBar, QPushButton, QVBoxLayout, QWidget, QLabel,
                               QHBoxLayout, QVBoxLayout, QCheckBox,
                               QProgressBar, QListWidget, QListWidgetItem, QListView,
                               QTextEdit, QRadioButton, QButtonGroup, QAbstractItemView)
from PySide2.QtGui import QColor, QIcon, QMouseEvent, QPixmap, QFont
from PySide2.QtCore import Qt, Signal

from Core.inference import InferenceClass


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Signals slots
        self.inference_class = InferenceClass()
        self.inference_class.outputSignal.connect(self.showOutput)
        self.inference_class.toggleProgressAndButton.connect(self.toggleViews)
        self.inference_class.loggingSignal.connect(self.printLogs)

        # components
        self.Title = QLabel('Hello\n Please, Choose a Model and Enter the text Passage to be summarized.')
        self.Title.setMargin(20)
        # self.Title.setPixmap(pixmap)
        self.Title.setAlignment(Qt.AlignHCenter)

        self.listWidget = QListWidget()

        item1 = QListWidgetItem('Pegasus-xsum  16-16')
        item2 = QListWidgetItem('Pegasus-xsum SF 16-12')
        item3 = QListWidgetItem('Pegasus-xsum SF 16-8')
        item4 = QListWidgetItem('Pegasus-xsum SF 16-4')
        item5 = QListWidgetItem('Pegasus-xsum PL 16-4')
        item6 = QListWidgetItem('Pegasus-xsum PL 12-6')
        item7 = QListWidgetItem('Pegasus-xsum PL 12-3')

        self.listWidget.addItem(item1)
        self.listWidget.addItem(item2)
        self.listWidget.addItem(item3)
        self.listWidget.addItem(item4)
        self.listWidget.addItem(item5)
        self.listWidget.addItem(item6)
        self.listWidget.addItem(item7)
        self.listWidget.setCurrentItem(item1)
        self.quantizedCheckbox = QCheckBox('Quantized')

        self.inputField = QTextEdit()
        t_input = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."
        self.inputField.setPlaceholderText(t_input)
        self.inputField.setFont(QFont('Ariel', 16, QFont.DemiBold))

        self.outputLabel = QLabel("Summary:")
        self.outputLabel.hide()
        self.outputLabel.setFont(QFont('Ariel', 12, QFont.ExtraBold))
        self.outputField = QTextEdit()
        # self.outputField.setDisabled(True)
        self.outputField.setDocumentTitle("Summary")
        self.outputField.hide()
        self.outputField.setFont(QFont('Ariel', 16, QFont.ExtraBold))

        self.button = QPushButton('Summarize')
        self.button.clicked.connect(lambda: self.onClicked())

        self.progress = QProgressBar()
        self.progress.hide()

        self.logLabel = QLabel("Log:")
        self.logLabel.hide()
        self.logLabel.setFont(QFont('Ariel', 10, QFont.ExtraBold))

        self.window = QWidget()
        self.window.resize(800, 800)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.Title)
        # self.layout.addLayout(self.studentGroup)

        self.modelLayout = QHBoxLayout()
        self.modelLayout.addWidget(self.listWidget)
        self.modelLayout.addWidget(self.quantizedCheckbox)

        self.layout.addLayout(self.modelLayout)

        self.layout.addWidget(self.inputField)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.progress)
        self.layout.addWidget(self.outputLabel)
        self.layout.addWidget(self.outputField)
        self.layout.addWidget(self.logLabel)

        self.window.setLayout(self.layout)

    def show(self):
        self.window.show()

    def onClicked(self):
        self.progress.show()
        self.progress.setMinimum(0)
        self.progress.setMaximum(0)
        self.button.setDisabled(True)
        text = self.inputField.toPlainText()
        text = self.inputField.placeholderText() if text == '' else text
        item = self.listWidget.currentItem()
        checked = self.quantizedCheckbox.isChecked()
        self.summarize(text, item, checked)

    def summarize(self, text, item, checked):
        asyncio.run(self.inference_class.infer(text=text, model=item, quantized=checked))

    def showOutput(self, output):
        self.outputField.show()
        self.outputLabel.show()
        self.outputField.setText(output if output != '' else 'please enter proper input')

    def toggleViews(self, flag):
        pass

    def printLogs(self, logMsg):
        pass
