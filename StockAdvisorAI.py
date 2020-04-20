import sys
import threading
import time
from functools import partial

from PyQt5 import QtCore,QtGui
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QMessageBox, QButtonGroup, QRadioButton, \
    QVBoxLayout, QDialog, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import requests
import base64
import math
import pandas as pd
import time
from sklearn.metrics import silhouette_score
from datetime import date

# Array of tuples. Tuple[0] => Question String Tuble[1] => Type(0 => HORIZON 1 => RISK 2 => Threshold)
question_array = [("Yatırımınızın süresi ne kadar olsun?", 0),
("Yatırımlarınızı yaklaşık ne sıklıkla kontrol edersiniz? ", 1),
("Mevduat faizlerini tatmin edici buluyor musunuz?", 1),
("Risk sözcüğü sizin için ne anlam ifade ediyor?",1),
("Kredi borcu, ipotek gibi ödemek zorunda olduğunuz belli bir mali yükümlülüğünüz var mı? Varsa toplam varlığınızın yüzde kaçını oluşturuyor?",1),
("Anaparadan kaybetmeyi göze alabilir misiniz?",1),
("Yatırım yapmak uykularınızı kaçırır mı?",1),
("Piyasadaki kötü gelişmelerden dolayı anaparadan %30 kaybettiniz. Bu durum yatırımlarınızı sonlandırmanıza sebep olur mu? ",2),
("Yaşınız?",1),
("Gelir ve gideriniz düzenli midir?",1),
("Yatırım bilgi seviyenizi nasıl tanımlarsınız?",1),
("Yatırım getirilerinin nasıl olmasını istersiniz?",1),
("Dövizdeki dalgalanma yatırımlarınızı sonlandırmanıza sebep olur mu?",1)]
# Array of tuples. Tuple=> Answers
answer_array = [("6 Ay", "9 Ay", "12 Ay","15 Ay", "21 Ay", "24 Ay"),
("Vade Sonuna Kadar Kontrol Etmem", "Ayda Bir Kontrol Ederim", "Haftada Bir Kontrol Ederim", "Her Gün Kontrol Ederim"),
("Evet", "Kısmen", "Hayır", "Mevduat Ne Demek?"),
("Kayıp","Belirsizlik","Fırsat","Heyecan"),
("Yok","0-%25","%26-%50","Yarısından Fazlası"),
("Asla","Düşük Bir Miktarda Kaybetmeyi Göze Alabilirim","Potansiyel Getiriye Göre Değişir","Evet"),
("Evet","Belki","Hayır"),
("Hayır","Evet"),
("18-30","31-50","51-65","65 ve üzeri"),
("Evet","Genelde","Hayır"),
("Bilgim Yok","Konuya Biraz Hakimim","İyi Olduğumu Düşünüyorum","Geniş Bir Bilgiye Sahibim"),
("Getirimin az ama düzenli olmasını tercih ederim.","Getirimin düzensiz de olsa maksimumda olmasını tercih ederim."),
("Evet","Hayır")]
vade_arr = [(180,270,360,450,630,720)]
beta_weights = [(0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0),(0,0),(0,0,0,0),(0,0,0),(0,0,0,0),(-0.6,0.6),(0.6,-0.6)]
std_weights = [(0,0,0),(0.5,0.2,-0.2,-0.5),(0,0,0,0),(-0.5,-0.2,0.2,0.5),(0,0,0,0),(0,0,0,0),(-0.5,0,0.5),(0,0),(0,0,0,0),(0.5,0,-0.5),(-0.5,-0.2,0.2,0.5),(0,0),(0,0)]
maxmin_weights = [(0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0),(0.5,-0.5),(0,0,0,0),(0,0,0),(-0.5,-0.2,0.2,0.5),(0,0),(0,0)]
vade_weights = [(0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(-0.5,-0.2,0.2,0.5),(-0.5,-0.2,0.2,0.5),(0,0,0),(0,0),(0,0,0,0),(0,0,0),(0,0,0,0),(0,0),(0,0)]

num_initial_questions = len(question_array)
# Number of questions that answered
question_number = 1
# Question index
question_index = 0
# Question index after VADE
start_index = 1
isStartable = False

isGraphable = False

isItterable = False

isResultReady = False

popupSignal = False

resultWaitSignal = False

LastSignal = False

vade = 0
user_beta = 0
user_std = 0
user_maxmin = 0
user_vade = 0
user_vade_threshold = 0

user_chg_beta = 0
user_chg_std = 0
user_chg_maxmin = 0
user_chg_vade = 0
beklenen_getiri = "0"

result_stock_name = []
result_values = []

class MyMplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100, label_ = None, value_ = None):
        fig = Figure(figsize=(width, height), dpi=dpi)

        self.axes = fig.add_subplot(111)

        self.compute_initial_figure(label_, value_)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class MyStaticMplCanvas(MyMplCanvas):

    def compute_initial_figure(self, label_, value_):
        self.axes.pie(value_,labels=label_,autopct="%1.1f%%")

def last_dialog():
    global isGraphable
    while(not isGraphable):
        time.sleep(0.0001)
    dialog = QDialog()
    dialog.setWindowTitle("Test Sonucunuz")
    dialog.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
    label = QLabel(dialog)
    label.setText("Size En Uygun Portföyümüz:")
    label2 = QLabel(dialog)
    label2.setText("Beklenen Getiriniz: " + beklenen_getiri)
    layout3 = QVBoxLayout()
    layout3.addWidget(label)
    layout3.addWidget(label2)
    layout2 = QVBoxLayout()
    layout2.addLayout(layout3)
    btn = QPushButton()
    btn.setText("Bitir")
    btn.clicked.connect(sys.exit)
    layout2.addWidget(btn)
    layout = QHBoxLayout()
    layout.addLayout(layout2)
    global result_stock_name,result_values
    print(result_stock_name)
    plot = MyStaticMplCanvas(dialog, width=7, height=4.5, dpi=100,label_=result_stock_name,value_=result_values)
    layout.addWidget(plot)
    dialog.setLayout(layout)
    print(user_vade_threshold, user_vade, user_std, user_maxmin, user_beta, vade)
    dialog.exec_()

def sleeper(sec, label, dialog):
    while (sec > 0):
        sec -= 1
        label.setText("Danışmanınız düşünüyor.Lütfen " + str(sec) + " saniye bekleyiniz...")
        time.sleep(0.17)
        label.setText("Danışmanınız düşünüyor.Lütfen " + str(sec) + " saniye bekleyiniz..")
        time.sleep(0.17)
        label.setText("Danışmanınız düşünüyor.Lütfen " + str(sec) + " saniye bekleyiniz.")
        time.sleep(0.17)
        label.setText("Danışmanınız düşünüyor.Lütfen " + str(sec) + " saniye bekleyiniz")
        time.sleep(0.17)
        label.setText("Danışmanınız düşünüyor.Lütfen " + str(sec) + " saniye bekleyiniz.")
        time.sleep(0.17)
        label.setText("Danışmanınız düşünüyor.Lütfen " + str(sec) + " saniye bekleyiniz..")
        time.sleep(0.17)
    dialog.reject()

def next_question(dialog,answersBG):
    global question_index
    answer_index = answersBG.checkedId()
    print(answer_index)
    global beta_weights, maxmin_weights, std_weights, vade_weights
    if question_array[question_index][1] is 0:
        global vade, vade_arr
        vade = vade_arr[question_index][answer_index]
    elif question_array[question_index][1] is 1 or question_array[question_index][1] is 2:
        global user_beta, beta_weights
        user_beta += beta_weights[question_index][answer_index]
        global user_maxmin, maxmin_weights
        user_maxmin += beta_weights[question_index][answer_index]
        global user_std, std_weights
        user_std += std_weights[question_index][answer_index]
        global user_vade, vade_weights
        user_vade += vade_weights[question_index][answer_index]
        if question_array[question_index][1] is 2:
            global user_vade_threshold
            if answer_index is 0:
                user_vade_threshold = -30
            elif answer_index is 1:
                user_vade_threshold = 0
    elif question_array[question_index][1] is 3:
        global user_chg_beta,user_chg_maxmin,user_chg_std,user_chg_vade
        user_chg_beta = beta_weights[question_index][answer_index]
        user_chg_maxmin = maxmin_weights[question_index][answer_index]
        user_chg_std = std_weights[question_index][answer_index]
        user_chg_vade = vade_weights[question_index][answer_index]
        global isItterable
        isItterable = True
    question_index += 1
    global question_number
    question_number += 1
    global resultWaitSignal
    global num_initial_questions
    global start_index
    global isStartable
    if start_index is question_index:
        isStartable = True
    check = True
    if question_index is num_initial_questions:
        isStartable = True
        check = False
    while((not resultWaitSignal) and (question_index >= num_initial_questions) and check):
        time.sleep(0.0001)
    resultWaitSignal = False
    global isResultReady
    if isResultReady:
        isStartable = True
        print("Buraya Girdi")
        mbox = QMessageBox()
        mbox.setWindowTitle("Teşekkürler!!")
        mbox.setText("Sorularımızı Tamamladınız.")
        mbox.setStandardButtons(QMessageBox.Ok)
        mbox.exec_()
        question_number = 1
        question_index = 0
        dialog.reject()
        wait_dialog = QDialog()
        wait_dialog.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
        wait_dialog.setWindowTitle("Bekleyiniz")
        sec = 5
        wait_layout = QHBoxLayout()
        wait_label = QLabel("Danışmanınız düşünüyor.Lütfen " + str(sec) + " saniye bekleyiniz..")
        wait_layout.addWidget(wait_label)
        wait_dialog.setLayout(wait_layout)
        t1 = threading.Thread(target=sleeper, args=(sec, wait_label, wait_dialog,))
        t1.start()
        wait_dialog.exec_()
        t1.join()
        last_dialog()
        return
    dialog.reject()
    if question_index < num_initial_questions:
        showdialog("İlerle")
    else:
        global popupSignal,LastSignal
        if(not LastSignal):
            wait_dialog2 = QDialog()
            wait_dialog2.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
            wait_dialog2.setWindowTitle("Bekleyiniz")
            sec = 15
            wait_layout2 = QHBoxLayout()
            wait_label2 = QLabel(wait_dialog2)
            wait_label2.setText("Danışmanınız.Lütfen " + str(sec) + " saniye bekleyiniz..")
            wait_layout2.addWidget(wait_label2)
            wait_dialog2.setLayout(wait_layout2)
            t2 = threading.Thread(target=sleeper, args=(sec, wait_label2, wait_dialog2,))
            t2.start()
            wait_dialog2.exec_()
            while(not LastSignal):
                time.sleep(1)
            t2.join()
        while (not popupSignal):
            time.sleep(0.0001)
        popupSignal = False
        showdialog("Bitir")

def close(dialog):
    global question_number
    question_number = 1
    global question_index
    question_index = 0
    dialog.reject()

def showdialog(state):
    dialog = QDialog()
    dialog.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
    dialog.setWindowTitle("Question " + str(question_number))
    dialog.setFixedSize(1000, 300)
    layout = QVBoxLayout()
    dialog.setLayout(layout)

    question_label = QLabel(dialog)
    question_label.setText(question_array[question_index][0])
    layout.addWidget(question_label)

    answersBG = QButtonGroup()
    answer_layout = QVBoxLayout()

    for index, answer in enumerate(answer_array[question_index]):
        temp_rb = QRadioButton(dialog)
        temp_rb.setText(answer)
        answer_layout.addWidget(temp_rb)
        answersBG.addButton(temp_rb, index)

    layout.addLayout(answer_layout)
    layout2 = QHBoxLayout()

    close_button = QPushButton(dialog)
    close_button.setText("Kapat")
    close_button.setFixedSize(100, 60)
    close_button.move(20, 250)
    close_button.clicked.connect(partial(close, dialog))
    layout2.addWidget(close_button)

    layout2.addSpacing(700)

    next_button = QPushButton(dialog)
    next_button.setText(state)
    next_button.setFixedSize(100, 60)
    next_button.clicked.connect(partial(next_question, dialog, answersBG))
    layout2.addWidget(next_button)

    layout.addLayout(layout2)
    dialog.exec_()

def gui():
    name_of_app = "InvestAI"

    app = QApplication(sys.argv)
    w = QWidget()
    w.setFixedSize(600, 600)
    w.setWindowTitle(name_of_app)

    icon_label = QLabel(w)
    pixmap = QtGui.QPixmap('icon.png')
    icon_label.setPixmap(pixmap)
    icon_label.move(10,10)
    icon_label.resize(pixmap.width(),pixmap.height())
    icon_label.show()

    label = QLabel(w)
    label.setText(name_of_app + "'a Hoşgeldiniz")
    label.move(230, 400)
    label.show()

    btn = QPushButton(w)
    btn.setText("Sizi Tanıyalım!")
    btn.resize(100, 60)
    btn.move(242, 450)
    btn.clicked.connect(partial(showdialog, "İlerle"))
    btn.show()

    w.show()
    sys.exit(app.exec_())

t1 = threading.Thread(target=gui, args=())
t1.start()
while (not isStartable):
    time.sleep(0.00001)
isStartable = False
print("Umut Kodu Başlangıç")
vade = int(vade * 5 / 7)

t1 = time.time()
def vade_worst_40(data, vade):
    if (vade >= len(data)):
        return -60 + np.random.normal(0, 0.01)
    data_return = []
    for i in range(len(data) - vade):
        data_return.append(((data[i] - data[i + vade]) / data[i + vade]) * 100)

    return np.mean(sorted(data_return)[round(len(data) * 0.2):round(len(data) * 0.5)])
def vade_worst_25(data, vade):
    if (vade >= len(data)):
        return -60 + np.random.normal(0, 0.01)
    data_return = []
    for i in range(len(data) - vade):
        data_return.append(((data[i] - data[i + vade]) / data[i + vade]) * 100)

    return np.mean(sorted(data_return)[:round(len(data) * 0.25)])
def vade_worst(data, vade):
    if (vade >= len(data)):
        return -60 + np.random.normal(0, 0.01)
    data_return = []
    for i in range(len(data) - vade):
        data_return.append(((data[i] - data[i + vade]) / data[i + vade]) * 100)

    return np.mean(sorted(data_return)[:round(len(data) * 0.05)])
def average_return(data, vade):
    data_return = []
    for i in range(len(data) - vade):
        data_return.append(((data[i + vade] - data[i]) / data[i]) * 100)

    return np.mean(data_return)
def beta_std_maxmin(bist100_data, data):
    bist100_return = []
    data_return = []
    for i in range(len(bist100_data) - 1):
        bist100_return.append(((bist100_data[i] - bist100_data[i + 1]) / bist100_data[i + 1]) * 100)
    for i in range(len(data) - 1):
        data_return.append(((data[i] - data[i + 1]) / data[i + 1]) * 100)

    if (len(bist100_data) != len(data)):
        return [1 + np.random.normal(0, 0.01), np.std(bist100_return) + np.random.normal(0, 0.01),
                ((bist100_data.max() - bist100_data.min()) / bist100_data.min()) + np.random.normal(0, 0.01)]
    return [np.cov(bist100_return, data_return)[0][1] / np.var(bist100_return), np.std(data_return),
            (data.max() - data.min()) / data.min()]
def read_(stocks_names,arr,index):
    temp_array = []
    for stock in stocks_names:
        data_request = requests.get(
            "https://api.matriksdata.com/dumrul/v1/tick/bar.gz?symbol=" + stock + "&period=1day&start=" + str(idx[0])[
                                                                                                          :10] + "&end=" + date_current,
            headers={"Authorization": "jwt " + myloginreq.text[myreq.text.find("token") + 8:-2]})
        temp_array.append(pd.DataFrame(data_request.json())["close"].values)
    arr[index]  = temp_array

myreq = requests.get("https://api.matriksdata.com/login",
                     headers={"Authorization": "Basic MjcyODA6MmtuYlV6MjE=", "X-Client-Type": "D"})
vade_multiplier = 5

if(vade > 200):
    vade_multiplier = 4

if(vade > 400):
    vade_multiplier = 3

vade_data = vade * vade_multiplier

date_current = date.today().strftime("%Y-%m-%d")

print("date of the day: ", date_current)

myloginreq = requests.get("https://api.matriksdata.com/login",
                          headers={"Authorization": "Basic MjcyODA6MmtuYlV6MjE=", "X-Client-Type": "D"})
print("status code of login request: ", myloginreq.status_code)

idx = pd.date_range(end=date_current, periods=vade_data, freq='B')

bist100_request = requests.get(
    "https://api.matriksdata.com/dumrul/v1/tick/bar.gz?symbol=XU100&period=1day&start=" + str(idx[0])[
                                                                                          :10] + "&end=" + date_current,
    headers={"Authorization": "jwt " + myloginreq.text[myreq.text.find("token") + 8:-2]})
bist100 = pd.DataFrame(bist100_request.json())["close"].values

stocks = [row.strip() for row in open("hisse_data/stock_names.txt", encoding="utf-8").read().split("\n")][:100]
stocks_names = [stock[stock.find("/") + 1: stock.find(" ")] for stock in stocks]

bist100_request = requests.get(
    "https://api.matriksdata.com/dumrul/v1/tick/bar.gz?symbol=XU100&period=1day&start=" + str(idx[0])[
                                                                                          :10] + "&end=" + date_current,
    headers={"Authorization": "jwt " + myloginreq.text[myreq.text.find("token") + 8:-2]})
bist100 = pd.DataFrame(bist100_request.json())["close"].values

print("bist 100 has been read: ", time.time() - t1)

stocks_values = []
temp_thrd_arr = [0,0,0,0]

t_1 =threading.Thread(target=read_,args=(stocks_names[:25],temp_thrd_arr,0))
t_2 =threading.Thread(target=read_,args=(stocks_names[25:50],temp_thrd_arr,1))
t_3 =threading.Thread(target=read_,args=(stocks_names[50:75],temp_thrd_arr,2))
t_4 =threading.Thread(target=read_,args=(stocks_names[75:],temp_thrd_arr,3))

t_1.start()
t_2.start()
t_3.start()
t_4.start()

t_1.join()
t_2.join()
t_3.join()
t_4.join()

print(len(temp_thrd_arr))

for x in temp_thrd_arr:
    for a in x:
        stocks_values.append(a)

print("time after request has finished: ", time.time() - t1)
stocks_metrics = []

for i in range(len(stocks_names)):
    stocks_metrics.append(beta_std_maxmin(bist100, stocks_values[i]))

for i in range(len(stocks_names)):
    stocks_metrics[i].append(vade_worst(stocks_values[i], vade))

stocks_beta = np.array([metric[0] for metric in stocks_metrics]).reshape(100, 1)
stocks_std = np.array([metric[1] for metric in stocks_metrics]).reshape(100, 1)
stocks_maxmin = np.array([metric[2] for metric in stocks_metrics]).reshape(100, 1)
stocks_vade = np.array([metric[3] for metric in stocks_metrics]).reshape(100, 1)

print("before kmeans", time.time() - t1)

sil = []

for k in range(5, 30):
    kmeans = KMeans(n_clusters=k, random_state=7).fit(stocks_beta)
    labels = kmeans.labels_
    sil.append((silhouette_score(stocks_beta, labels, metric='euclidean'), k))

kmeans_beta = KMeans(n_clusters=sorted(sil, reverse=True)[0][1], random_state=7).fit(stocks_beta)
print("best k for beta: ", sorted(sil, reverse=True)[0][1])

sil = []

for k in range(5, 30):
    kmeans = KMeans(n_clusters=k, random_state=7).fit(stocks_std)
    labels = kmeans.labels_
    sil.append((silhouette_score(stocks_std, labels, metric='euclidean'), k))

kmeans_std = KMeans(n_clusters=sorted(sil, reverse=True)[0][1], random_state=7).fit(stocks_std)

print("best k for std: ", sorted(sil, reverse=True)[0][1])

sil = []

for k in range(5, 30):
    kmeans = KMeans(n_clusters=k, random_state=7).fit(stocks_maxmin)
    labels = kmeans.labels_
    sil.append((silhouette_score(stocks_maxmin, labels, metric='euclidean'), k))

kmeans_maxmin = KMeans(n_clusters=sorted(sil, reverse=True)[0][1], random_state=7).fit(stocks_maxmin)

print("best k for maxmin: ", sorted(sil, reverse=True)[0][1])

sil = []

for k in range(5, 30):
    kmeans = KMeans(n_clusters=k, random_state=7).fit(stocks_vade)
    labels = kmeans.labels_
    sil.append((silhouette_score(stocks_vade, labels, metric='euclidean'), k))

kmeans_vade = KMeans(n_clusters=sorted(sil, reverse=True)[0][1], random_state=7).fit(stocks_vade)

print("best k for vade: ", sorted(sil, reverse=True)[0][1])
print("Son Soruya Geçebilirsin")
LastSignal = True
# 4 float 1 threshold user_beta    user_std     user_maxmin     user_vade ,user_vade_threshold
while(not isStartable):
    time.sleep(0.00001)

user_beta_new = stocks_beta.mean() + user_beta * stocks_beta.std()  # new values calculated by questions
user_std_new = stocks_std.mean() + user_std * stocks_std.std()
user_maxmin_new = stocks_maxmin.mean() + user_maxmin * stocks_maxmin.std()
user_vade_new = stocks_vade.mean() + user_vade * stocks_vade.std()

user_beta_group = kmeans_beta.predict([[user_beta_new]])[0]  # group label of user
user_std_group = kmeans_std.predict([[user_std_new]])[0]
user_maxmin_group = kmeans_maxmin.predict([[user_maxmin_new]])[0]
user_vade_group = kmeans_vade.predict([[user_vade_new]])[0]

user_beta_companies = [mytuple[1] for mytuple in list(zip(kmeans_beta.predict(stocks_beta), stocks_names)) if
                       mytuple[0] == user_beta_group]  # company names of that group for user
user_std_companies = [mytuple[1] for mytuple in list(zip(kmeans_std.predict(stocks_std), stocks_names)) if
                      mytuple[0] == user_std_group]
user_maxmin_companies = [mytuple[1] for mytuple in list(zip(kmeans_maxmin.predict(stocks_maxmin), stocks_names)) if
                         mytuple[0] == user_maxmin_group]
user_vade_companies = [mytuple[1] for mytuple in list(zip(kmeans_vade.predict(stocks_vade), stocks_names)) if
                       mytuple[0] == user_vade_group]

max_stock_per_group = 3

stock_name_to_id = dict(zip(stocks_names, range(100)))

beta_trials = []

for comp in user_beta_companies:
    beta_trials.append((average_return(stocks_values[stock_name_to_id[comp]], vade), comp))

beta_result_companies = [(possible[0], possible[1]) for possible in sorted(beta_trials, reverse=True)[:3] if
                         possible[0] > 1]

std_trials = []

for comp in user_std_companies:
    std_trials.append((average_return(stocks_values[stock_name_to_id[comp]], vade), comp))

std_result_companies = [(possible[0], possible[1]) for possible in sorted(std_trials, reverse=True)[:3] if
                        possible[0] > 1]

maxmin_trials = []

for comp in user_maxmin_companies:
    maxmin_trials.append((average_return(stocks_values[stock_name_to_id[comp]], vade), comp))

maxmin_result_companies = [(possible[0], possible[1]) for possible in sorted(maxmin_trials, reverse=True)[:3] if
                           possible[0] > 1]

vade_trials = []

for comp in user_vade_companies:
    vade_trials.append((average_return(stocks_values[stock_name_to_id[comp]], vade), comp))

vade_result_companies = [(possible[0], possible[1]) for possible in sorted(vade_trials, reverse=True)[:3] if
                         possible[0] > 1]

result_companies = set(beta_result_companies + std_result_companies + maxmin_result_companies + vade_result_companies)
result_companies_new = []
for comp in result_companies:

    if (comp[1] != "ICBCT" and comp[1] != "FLAP" and comp[1] != "ITTFH" and comp[1] != "IHLAS" and comp[1] != "KERVT"):
        result_companies_new.append((comp[0], comp[1]))

result_companies_new = set(result_companies_new)
first_denominator = 0
first_risk = 0
first_return = 0
first_risk_1 = 0
first_risk_2 = 0

for comp in result_companies_new:
    first_denominator += comp[0]

    first_risk += comp[0] * vade_worst(stocks_values[stock_name_to_id[comp[1]]], vade)
    first_risk_1 += comp[0] * vade_worst_25(stocks_values[stock_name_to_id[comp[1]]], vade)
    first_risk_2 += comp[0] * vade_worst_40(stocks_values[stock_name_to_id[comp[1]]], vade)

    first_return += comp[0] * average_return(stocks_values[stock_name_to_id[comp[1]]], vade)

print("risk of your portfolio: ", first_risk / first_denominator)
print("return of your portfolio: ", first_return / first_denominator)

while(True):
    question_array += [("Beklenen Kazancınız: " + "{0:.2f}".format(first_return / first_denominator) +"\nEn kötü durumda kaybedilecek: " + "{0:.2f}".format((-1)*(first_risk / first_denominator)) + "\nKötü durumda kaybedilecek: " + "{0:.2f}".format((-1)*(first_risk_1 / first_denominator)) + "\nOrtalamada kaybedilme olasılığı olan miktar: " + "{0:.2f}".format((-1)*first_risk_2 / first_denominator),3)]
    answer_array += [("Getiriyi Az Buldum","Uygundur","Riski Fazla Buldum")]
    beta_weights += [(-0.6,0,0,6)]
    std_weights += [(0.5,0,-0,5)]
    maxmin_weights +=  [(0.5,0,-0.5)]
    vade_weights += [(-0.6,0,0,6)]
    popupSignal = True
    while(not isItterable):
        time.sleep(0.00001)
    isItterable = False
    print("Changes", user_chg_vade,user_chg_std,user_chg_maxmin,user_chg_beta)
    if(user_chg_vade is 0 and user_chg_std is 0 and user_chg_maxmin is 0 and user_chg_beta is 0):
        isResultReady = True
        resultWaitSignal = True
        break
    print("Umut'un İterasyon")
    user_beta += user_chg_beta
    user_maxmin += user_chg_maxmin
    user_std += user_chg_std
    user_vade += user_chg_vade

    user_beta_new = stocks_beta.mean() + user_beta * stocks_beta.std()  # new values calculated by questions
    user_std_new = stocks_std.mean() + user_std * stocks_std.std()
    user_maxmin_new = stocks_maxmin.mean() + user_maxmin * stocks_maxmin.std()
    user_vade_new = stocks_vade.mean() + user_vade * stocks_vade.std()

    user_beta_group = kmeans_beta.predict([[user_beta_new]])[0]  # group label of user
    user_std_group = kmeans_std.predict([[user_std_new]])[0]
    user_maxmin_group = kmeans_maxmin.predict([[user_maxmin_new]])[0]
    user_vade_group = kmeans_vade.predict([[user_vade_new]])[0]

    user_beta_companies = [mytuple[1] for mytuple in list(zip(kmeans_beta.predict(stocks_beta), stocks_names)) if
                           mytuple[0] == user_beta_group]  # company names of that group for user
    user_std_companies = [mytuple[1] for mytuple in list(zip(kmeans_std.predict(stocks_std), stocks_names)) if
                          mytuple[0] == user_std_group]
    user_maxmin_companies = [mytuple[1] for mytuple in list(zip(kmeans_maxmin.predict(stocks_maxmin), stocks_names)) if
                             mytuple[0] == user_maxmin_group]
    user_vade_companies = [mytuple[1] for mytuple in list(zip(kmeans_vade.predict(stocks_vade), stocks_names)) if
                           mytuple[0] == user_vade_group]

    max_stock_per_group = 3

    stock_name_to_id = dict(zip(stocks_names, range(100)))

    beta_trials = []

    for comp in user_beta_companies:
        beta_trials.append((average_return(stocks_values[stock_name_to_id[comp]], vade), comp))

    beta_result_companies = [(possible[0], possible[1]) for possible in sorted(beta_trials, reverse=True)[:3] if
                             possible[0] > 1]

    std_trials = []

    for comp in user_std_companies:
        std_trials.append((average_return(stocks_values[stock_name_to_id[comp]], vade), comp))

    std_result_companies = [(possible[0], possible[1]) for possible in sorted(std_trials, reverse=True)[:3] if
                            possible[0] > 1]

    maxmin_trials = []

    for comp in user_maxmin_companies:
        maxmin_trials.append((average_return(stocks_values[stock_name_to_id[comp]], vade), comp))

    maxmin_result_companies = [(possible[0], possible[1]) for possible in sorted(maxmin_trials, reverse=True)[:3] if
                               possible[0] > 1]

    vade_trials = []

    for comp in user_vade_companies:
        vade_trials.append((average_return(stocks_values[stock_name_to_id[comp]], vade), comp))

    vade_result_companies = [(possible[0], possible[1]) for possible in sorted(vade_trials, reverse=True)[:3] if
                             possible[0] > 1]

    result_companies = set(
        beta_result_companies + std_result_companies + maxmin_result_companies + vade_result_companies)

    # result_companies = beta_result_companies + std_result_companies + maxmin_result_companies + vade_result_companies

    result_companies_new = []
    for comp in result_companies:

        if (comp[1] != "ICBCT" and comp[1] != "FLAP" and comp[1] != "ITTFH" and comp[1] != "IHLAS" and comp[
            1] != "KERVT"):
            result_companies_new.append((comp[0], comp[1]))

    result_companies_new = set(result_companies_new)
    first_denominator = 0
    first_risk = 0
    first_return = 0
    first_risk_1 = 0
    first_risk_2 = 0

    for comp in result_companies_new:
        first_denominator += comp[0]

        first_risk += comp[0] * vade_worst(stocks_values[stock_name_to_id[comp[1]]], vade)
        first_risk_1 += comp[0] * vade_worst_25(stocks_values[stock_name_to_id[comp[1]]], vade)
        first_risk_2 += comp[0] * vade_worst_40(stocks_values[stock_name_to_id[comp[1]]], vade)

        first_return += comp[0] * average_return(stocks_values[stock_name_to_id[comp[1]]], vade)

    resultWaitSignal = True
    print(resultWaitSignal)

beklenen_getiri = "{0:.2f}".format(first_return / first_denominator)
print(isResultReady)

for value, name in result_companies_new:
    result_stock_name += [name]
    result_values += [value]
isGraphable = True

print(result_companies)