from screen import Ui_MainWindow
from about import about_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QIcon
import sys
import os
import copy
import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('Qt5')
np.set_printoptions(precision=3, suppress=True)

class aboutWindow(QtWidgets.QMainWindow, about_MainWindow):
    def __init__(self, parent=None):
        super(aboutWindow, self).__init__(parent)
        self.setupUi(self)


class program(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(program, self).__init__(parent)
        self.setupUi(self)

        self.imported_img = ''
        self.final_img = 0
        self.ycbcrimg = 0
        self.vec_hist_y = 0
        self.img_equali_y = 0

        about = aboutWindow(self)

        def getImage():
            # Abre janela de dialogo para selecionar imagem
            fname = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Image files (*.jpg *.bmp *png *tiff)")
            self.imported_img = os.path.normpath(fname[0])

            # Mostra imagem selecionada na tela
            pixmap = QPixmap(self.imported_img)
            self.label_previousImg.setPixmap(QPixmap(pixmap))
            return

        def convert():
            if self.imported_img == '': #Caso não tenha lido arquivo, mostra mensagem pro usuario
                alertMessage("Abra uma imagem para ser equalizada")
                return
            else:
                #Lê o arquivo
                img = cv2.imread(self.imported_img)

                #BGR para RGB
                img = img[:, :, ::-1]

                #Conversão pra Y'CbCr
                self.ycbcrimg = np.zeros(img.shape)
                matrix = np.array([[0.299, 0.587, 0.114],
                                   [-0.169, -0.331, 0.5],
                                   [0.5, -0.419, -0.081]])

                matrix2 = np.array([0, 128, 128])

                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        self.ycbcrimg[i, j,] = np.dot(matrix, img[i, j,])
                        self.ycbcrimg[i, j,] += matrix2

                #histograma Y
                self.vec_hist_y = np.zeros((256, 1))
                for i in range(self.ycbcrimg.shape[0]):
                    for j in range(self.ycbcrimg.shape[1]):
                        x = int(self.ycbcrimg[i, j, 0])
                        self.vec_hist_y[x] += 1

                self.vec_hist_y = self.vec_hist_y / (img.shape[0] * img.shape[1])

                for i in range(1, 256):
                    self.vec_hist_y[i] += self.vec_hist_y[i - 1]

                vec_cdf_y = copy.deepcopy(self.vec_hist_y)
                vec_cdf_y *= 255
                vec_cdf_y = np.floor(vec_cdf_y)

                self.img_equali_y = copy.deepcopy(self.ycbcrimg)
                self.img_equali_y[:, :, 0] = np.zeros((img.shape[0], img.shape[1]))

                for i in range(self.img_equali_y.shape[0]):
                    for j in range(self.img_equali_y.shape[1]):
                        x = int(self.ycbcrimg[i, j, 0])
                        self.img_equali_y[i, j, 0] = vec_cdf_y[x]

                new_rgb = np.zeros(img.shape)
                matrix_conv = np.array([[1, 0, 1.402],
                                        [1, -0.344, -0.714],
                                        [1, 1.772, 0]])

                matrix2_conv = np.array([0, -128, -128])

                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        new_rgb[i, j,] = self.img_equali_y[i, j,] + matrix2_conv
                        new_rgb[i, j,] = np.dot(matrix_conv, new_rgb[i, j,])
                        for k in range(img.shape[2]):
                            if new_rgb[i, j, k] < 0:
                                new_rgb[i, j, k] = 0
                            elif new_rgb[i, j, k] > 255:
                                new_rgb[i, j, k] = 255

                self.final_img = new_rgb[:, :, ::-1]
                self.final_img = self.final_img.astype('uint8')

                #Mostra a imagem equalizada
                altura, largura, canais = self.final_img.shape

                if altura > 1000 or largura > 2000:
                    self.final_img = cv2.resize(self.final_img, (int(largura / 1.5), int(altura / 1.5)))
                    cv2.imshow('image', self.final_img)
                    cv2.waitKey(0)
                else:
                    cv2.imshow('image', self.final_img)
                    cv2.waitKey(0)

                return

        def downloadImage():
            if self.imported_img == '':#Caso não tenha lido arquivo, mostra mensagem pro usuario
                alertMessage("Nenhuma imagem gerada")
                return
            else:
                #Baixa a imagem gerada na pasta do projeto
                cv2.imwrite("equalizedImg.jpg", self.final_img)
                return

        def showGraphics():
            if self.imported_img == '':#Caso não tenha lido arquivo, mostra mensagem pro usuario
                alertMessage("Nenhuma imagem foi equalizada para mostrar gráficos")
                return
            else:

                #Mostra os graficos na tela
                plt.figure(figsize=(18, 15))
                plt.subplots_adjust(hspace=0.25, wspace=0.2)

                a0 = plt.subplot(3, 3, 1)
                plt.hist(self.ycbcrimg[:, :, 0].ravel(), 256, [0, 256], color='red')
                a0.set_title("Y Pré Equalização")
                a0.plot()

                a1 = plt.subplot(3, 3, 3)
                plt.hist(self.img_equali_y[:, :, 0].ravel(), 256, [0, 256], color='green')
                a1.set_title("Y Após Equalização")

                a2 = plt.subplot(3, 3, 2)
                plt.plot(self.vec_hist_y)
                a2.set_title("Curva de Distribuição de Frequência de Y")

                a3 = plt.subplot(3, 3, 4)
                plt.hist(self.ycbcrimg[:, :, 0].ravel(), 256, [0, 256], color='red')  # verm
                a3.set_title("Vermelho Pré Equalização")

                a4 = plt.subplot(3, 3, 5)
                plt.hist(self.ycbcrimg[:, :, 1].ravel(), 256, [0, 256], color='green')  # verd
                a4.set_title("Verde Pré Equalização")

                a5 = plt.subplot(3, 3, 6)
                plt.hist(self.ycbcrimg[:, :, 2].ravel(), 256, [0, 256], color='blue')  # azul
                a5.set_title("Azul Pré Equalização")

                a6 = plt.subplot(3, 3, 7)
                plt.hist(self.final_img[:, :, 0].ravel(), 256, [0, 256], color='red')  # verm_new
                a6.set_title("Vermelho Pós Equalização")

                a7 = plt.subplot(3, 3, 8)
                plt.hist(self.final_img[:, :, 1].ravel(), 256, [0, 256], color='green')  # verd_new
                a7.set_title("Verde Pós Equalização")

                a8 = plt.subplot(3, 3, 9)
                plt.hist(self.final_img[:, :, 2].ravel(), 256, [0, 256], color='blue')  # azul_new
                a8.set_title("Azul Pós Equalização")

                plt.suptitle(
                    "Histogramas e Curva de Distribuição de Frequência Relativos à Equalização do Canal Y \n (Histogramas RGB Calculados Considerando Conversão YCbCr ↔ RGB)",
                    size=23)
                plt.show()
                return

        def openHelpScreen():
            #Mostra a tela de sobre/ajuda
            about.show()
            return

        def alertMessage(text):
            #Alertas
            msg = QMessageBox()
            msg.setWindowTitle("Ops!")
            msg.setWindowIcon(QIcon("icons/transformImg.ico"))
            msg.setText(text)
            msg.setIcon(QMessageBox.Warning)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return

        self.pushButton_openImg.clicked.connect(getImage)
        self.pushButton_convertImg.clicked.connect(convert)
        self.pushButton_downloadImg.clicked.connect(downloadImage)
        self.pushButton_showGraphics.clicked.connect(showGraphics)
        self.pushButton_help.clicked.connect(openHelpScreen)



App = QApplication(sys.argv)
window = program()
window.showMaximized()
sys.exit(App.exec())