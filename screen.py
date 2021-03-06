# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'screen.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(835, 537)
        MainWindow.setWindowIcon(QIcon("icons/transformImg.ico"))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_top = QtWidgets.QFrame(self.centralwidget)
        self.frame_top.setMaximumSize(QtCore.QSize(16777215, 50))
        self.frame_top.setStyleSheet("background-color: rgb(130, 130, 130);")
        self.frame_top.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_top.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_top.setObjectName("frame_top")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_top)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_title = QtWidgets.QLabel(self.frame_top)
        self.label_title.setMaximumSize(QtCore.QSize(16777215, 50))
        self.label_title.setStyleSheet("font: 14pt \"MS Shell Dlg 2\";\n"
"padding: 10px;")
        self.label_title.setObjectName("label_title")
        self.horizontalLayout_2.addWidget(self.label_title)
        self.pushButton_help = QtWidgets.QPushButton(self.frame_top)
        self.pushButton_help.setMinimumSize(QtCore.QSize(90, 40))
        self.pushButton_help.setMaximumSize(QtCore.QSize(90, 40))
        self.pushButton_help.setStyleSheet("QPushButton{\n"
"    margin-left: 10px;\n"
"    margin-right: 10px;\n"
"    \n"
"    background-color: rgb(150,150,150);\n"
"    border: 2px solid rgb(60,60,60);\n"
"    color: rgb(0,0,0);\n"
"    border-radius: 5px;\n"
"    font: 10pt \"MS Shell Dlg 2\";\n"
"}\n"
"QPushButton:hover{\n"
"    background-color: rgb(130,130,130);\n"
"    border: 2px solid rgb(70,70,70); \n"
"    color: rgb(255,255,255);\n"
"}\n"
"QPushButton:pressed{\n"
"    background-color: rgb(110,110,110);\n"
"    border: 2px solid rgb(0,0,0);\n"
"    color: rgb(255,255,255);\n"
"}")
        self.pushButton_help.setObjectName("pushButton_help")
        self.horizontalLayout_2.addWidget(self.pushButton_help)
        self.verticalLayout.addWidget(self.frame_top)
        self.frame_content = QtWidgets.QFrame(self.centralwidget)
        self.frame_content.setStyleSheet("")
        self.frame_content.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_content.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_content.setObjectName("frame_content")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_content)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.frame_images = QtWidgets.QFrame(self.frame_content)
        self.frame_images.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.frame_images.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_images.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_images.setObjectName("frame_images")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_images)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame_sidebar = QtWidgets.QFrame(self.frame_images)
        self.frame_sidebar.setMinimumSize(QtCore.QSize(260, 0))
        self.frame_sidebar.setMaximumSize(QtCore.QSize(260, 16777215))
        self.frame_sidebar.setStyleSheet("background-color: rgb(215, 215, 215);")
        self.frame_sidebar.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_sidebar.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_sidebar.setLineWidth(0)
        self.frame_sidebar.setObjectName("frame_sidebar")
        self.formLayout = QtWidgets.QFormLayout(self.frame_sidebar)
        self.formLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldsStayAtSizeHint)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setSpacing(0)
        self.formLayout.setObjectName("formLayout")
        self.pushButton_openImg = QtWidgets.QPushButton(self.frame_sidebar)
        self.pushButton_openImg.setMinimumSize(QtCore.QSize(250, 60))
        self.pushButton_openImg.setMaximumSize(QtCore.QSize(250, 60))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.pushButton_openImg.setFont(font)
        self.pushButton_openImg.setStyleSheet("QPushButton{\n"
"    margin-left: 10px;\n"
"    margin-right: 10px;\n"
"    margin-top: 10px;\n"
"    background-color: rgb(150,150,150);\n"
"    border: 2px solid rgb(60,60,60);\n"
"    color: rgb(0,0,0);\n"
"    border-radius: 5px;\n"
"    font: 10pt \"MS Shell Dlg 2\";\n"
"}\n"
"QPushButton:hover{\n"
"    background-color: rgb(130,130,130);\n"
"    border: 2px solid rgb(70,70,70); \n"
"    color: rgb(255,255,255);\n"
"}\n"
"QPushButton:pressed{\n"
"    background-color: rgb(110,110,110);\n"
"    border: 2px solid rgb(0,0,0);\n"
"    color: rgb(255,255,255);\n"
"}\n"
"")
        self.pushButton_openImg.setObjectName("pushButton_openImg")
        self.pushButton_openImg.setIcon(QIcon("icons/open.ico"))
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.pushButton_openImg)
        self.pushButton_convertImg = QtWidgets.QPushButton(self.frame_sidebar)
        self.pushButton_convertImg.setMinimumSize(QtCore.QSize(250, 60))
        self.pushButton_convertImg.setMaximumSize(QtCore.QSize(250, 60))
        self.pushButton_convertImg.setStyleSheet("QPushButton{\n"
"    margin-left: 10px;\n"
"    margin-right: 10px;\n"
"    margin-top: 10px;\n"
"    background-color: rgb(150,150,150);\n"
"    border: 2px solid rgb(60,60,60);\n"
"    color: rgb(0,0,0);\n"
"    border-radius: 5px;\n"
"    \n"
"    font: 10pt \"MS Shell Dlg 2\";\n"
"}\n"
"QPushButton:hover{\n"
"    background-color: rgb(130,130,130);\n"
"    border: 2px solid rgb(70,70,70); \n"
"    color: rgb(255,255,255);\n"
"}\n"
"QPushButton:pressed{\n"
"    background-color: rgb(110,110,110);\n"
"    border: 2px solid rgb(0,0,0);\n"
"    color: rgb(255,255,255);\n"
"}")
        self.pushButton_convertImg.setObjectName("pushButton_convertImg")
        self.pushButton_convertImg.setIcon(QIcon("icons/transformImg.ico"))
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.pushButton_convertImg)
        self.pushButton_downloadImg = QtWidgets.QPushButton(self.frame_sidebar)
        self.pushButton_downloadImg.setMinimumSize(QtCore.QSize(250, 60))
        self.pushButton_downloadImg.setMaximumSize(QtCore.QSize(250, 60))
        self.pushButton_downloadImg.setStyleSheet("QPushButton{\n"
"    margin-left: 10px;\n"
"    margin-right: 10px;\n"
"    margin-top: 10px;\n"
"    background-color: rgb(150,150,150);\n"
"    border: 2px solid rgb(60,60,60);\n"
"    color: rgb(0,0,0);\n"
"    border-radius: 5px;\n"
"    font: 10pt \"MS Shell Dlg 2\";\n"
"}\n"
"QPushButton:hover{\n"
"    background-color: rgb(130,130,130);\n"
"    border: 2px solid rgb(70,70,70); \n"
"    color: rgb(255,255,255);\n"
"}\n"
"QPushButton:pressed{\n"
"    background-color: rgb(110,110,110);\n"
"    border: 2px solid rgb(0,0,0);\n"
"    color: rgb(255,255,255);\n"
"}")
        self.pushButton_downloadImg.setObjectName("pushButton_downloadImg")
        self.pushButton_downloadImg.setIcon(QIcon("icons/download.ico"))
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.pushButton_downloadImg)
        self.pushButton_showGraphics = QtWidgets.QPushButton(self.frame_sidebar)
        self.pushButton_showGraphics.setMinimumSize(QtCore.QSize(250, 60))
        self.pushButton_showGraphics.setStyleSheet("QPushButton{\n"
"    margin-left: 10px;\n"
"    margin-right: 10px;\n"
"    margin-top: 10px;\n"
"    background-color: rgb(150,150,150);\n"
"    border: 2px solid rgb(60,60,60);\n"
"    color: rgb(0,0,0);\n"
"    border-radius: 5px;\n"
"    font: 10pt \"MS Shell Dlg 2\";\n"
"}\n"
"QPushButton:hover{\n"
"    background-color: rgb(130,130,130);\n"
"    border: 2px solid rgb(70,70,70); \n"
"    color: rgb(255,255,255);\n"
"}\n"
"QPushButton:pressed{\n"
"    background-color: rgb(110,110,110);\n"
"    border: 2px solid rgb(0,0,0);\n"
"    color: rgb(255,255,255);\n"
"}")
        self.pushButton_showGraphics.setObjectName("pushButton_showGraphics")
        self.pushButton_showGraphics.setIcon(QIcon("icons/graphic.ico"))
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.pushButton_showGraphics)
        self.horizontalLayout.addWidget(self.frame_sidebar)
        self.label_previousImg = QtWidgets.QLabel(self.frame_images)
        self.label_previousImg.setText("")
        self.label_previousImg.setObjectName("label_previousImg")
        self.horizontalLayout.addWidget(self.label_previousImg)
        self.verticalLayout_2.addWidget(self.frame_images)
        self.frame_credits = QtWidgets.QFrame(self.frame_content)
        self.frame_credits.setMaximumSize(QtCore.QSize(16777215, 20))
        self.frame_credits.setStyleSheet("background-color: rgb(215, 215, 215);")
        self.frame_credits.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_credits.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_credits.setObjectName("frame_credits")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_credits)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_credits = QtWidgets.QLabel(self.frame_credits)
        self.label_credits.setStyleSheet("margin-left: 5px;\n"
"color: rgb(100, 100, 100);")
        self.label_credits.setObjectName("label_credits")
        self.horizontalLayout_4.addWidget(self.label_credits)
        self.label_version = QtWidgets.QLabel(self.frame_credits)
        self.label_version.setStyleSheet("margin-right: 5px;\n"
"color: rgb(100, 100, 100);")
        self.label_version.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_version.setObjectName("label_version")
        self.horizontalLayout_4.addWidget(self.label_version)
        self.verticalLayout_2.addWidget(self.frame_credits)
        self.verticalLayout.addWidget(self.frame_content)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Conversor e equalizador de imagens"))
        self.label_title.setText(_translate("MainWindow", "Conversor e Equalizador de Imagens"))
        self.pushButton_help.setText(_translate("MainWindow", "Ajuda"))
        self.pushButton_openImg.setText(_translate("MainWindow", "Abrir Imagem"))
        self.pushButton_convertImg.setText(_translate("MainWindow", "Converter e Equalizar"))
        self.pushButton_downloadImg.setText(_translate("MainWindow", "Salvar"))
        self.pushButton_showGraphics.setText(_translate("MainWindow", "Mostrar gráficos"))
        self.label_credits.setText(_translate("MainWindow", "Desenvolvido por Orssatto, Camara, Veit"))
        self.label_version.setText(_translate("MainWindow", "v 1.0"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
