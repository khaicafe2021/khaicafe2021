import pickle
import random
import re
import shutil
import sqlite3
import threading
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
#from keras.utils import to_categorical
import matplotlib
import matplotlib.pyplot as plt
import joblib
from sklearn import preprocessing

import pygame
import gym
import numpy as np
import pytz
import pywintypes
import win32api
import win32con
import win32gui
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


a = 30
b = 20


def msg():
    # a = a + b
    print(str(a + b))


def changeme(mylist):
    # "Thay doi list da truyen cho ham nay"
    print(r"Cac gia tri ben ngoai ham la: ", mylist)
    mylist = [1, 2, 3, 4]  # Lenh nay gan mot tham chieu moi cho mylist
    print("Cac gia tri ben trong ham la: ", mylist)
    return  # Bay gio ban co the goi ham changeme


# hàm regex
def regex_code():
    sentence = '"Start" áa sentence and then bring it to an end'
    pattern = re.compile(r'a', re.I)
    matches = pattern.findall(sentence)
    # đếm danh sach
    print(len(matches))


def openfile():
    with open(r"source.txt", encoding='UTF-8') as f:
        content = f.read()
    content = content.replace('\n', '')
    pattern = re.compile('<div class="table--(.*?)<div class="footerControls--')
    Dsban = pattern.findall(content)
    print(len(Dsban))
    # iter_ = (x for x in range(Dsban))
    tab = 0
    Ds = list(x for x in range(40))
    for ban in Dsban:
        # print(ban + '\n'*4)
        tab += 1
        TenBan = re.findall(r'data-role="table-name">(.*?)</span>', ban)[0]
        TongVanofBan = re.findall(r'<div class="roadContainer--(.*?)<div class="roadContainer--', ban)

        print(TenBan)
        try:
            van = re.findall(r'name="(.*?)"', TongVanofBan[0])

            Dss = ''
            for kq in van:
                if kq == 'Player' or kq == 'Player Player' or kq == 'Player Banker' or kq == 'Player BankerPlayer' or kq == 'Player PlayerBanker':
                    Dss = Dss + 'P'
                    # print(Ds[a])
                elif kq == "Banker" or kq == "Banker Banker" or kq == "Banker Player" or kq == "Banker BankerPlayer" or kq == "Banker PlayerBanker":
                    Dss = Dss + 'B'
                    # print(Ds[a])
                elif kq == "Player Tie" or kq == "Player TiePlayer" or kq == "Player TieBanker" or kq == "Player TieBankerPlayer" or kq == "Player TiePlayerBanker":
                    Dss = Dss + 'PT'
                    # print(Ds[a])
                elif kq == "Banker Tie" or kq == "Banker TiePlayer" or kq == "Banker TieBanker" or kq == "Banker TieBankerPlayer" or kq == "Banker TiePlayerBanker":
                    Dss = Dss + 'BT'
                    # print(Ds[a])
            Ds[tab] = Dss
        except:
            print('khong co')
        print(len(Ds[tab]))
        print(Ds[tab])
        con = sqlite3.connect('Baccarat.db')  # (":memory:")
        cur = con.cursor()
        print(tab)
        cur.execute("""UPDATE BCRTotal SET tenban=:ban, vanofban=:van, B=:B, P=:P, T=:T WHERE tabid=:Id""",
                    {"ban": TenBan, "van": Dss, "B": Dss.count('B'), "P": Dss.count('P'), "T": Dss.count('T'),
                     "Id": tab})
        # cur.execute("insert into BCRTotal values (?, ?, ?)", (tab, TenBan[0], Dss))
        con.commit()
        cur.close()
        print('B ->', Dss.count('B'))
        print('P ->', Dss.count('P'))
        print('T ->', Dss.count('T'))

    # else:
    # print(Dss)


def sqlitetest():
    con = sqlite3.connect('Baccarat.db')  # (":memory:")
    cur = con.cursor()
    # cur.execute("insert into BCRTotal values (null, ?, ?)", ("Baccarat Tốc độ A", 'BPBP'))
    # con.commit()

    # update ban ghi
    # cur.execute("update BCRTotal set tenban = 'Baccarat Tốc Độ B' where id = 2")
    # cur.execute('UPDATE BCRTotal SET tenban = ? WHERE tenban = ?', (1, 1))
    # con.commit()

    # chon het from
    # cur.execute("select * from BCRTotal")
    # cur.execute("select COUNT (tenban) from BCRTotal")
    # cur.execute("SELECT * FROM BCRTotal WHERE tabid=:Id", {"Id": 2})
    cur.execute("SELECT * FROM BCRTotal order by tabid")  # =:Id", {"Id": 'Baccarat Tốc Độ P'})

    # chon 1 truong trong from
    # cur.execute("select * from BCRTotal where tenban = 'C'")

    # lay tat ca data from
    getdata = cur.fetchall()
    print(getdata)
    print(getdata[0])

    BCR = {
        "tabid": getdata[0][0],
        "tenban": getdata[0][1],
        "vanofban": getdata[0][2]
    }

    print('lay data tabid[0][0] ->', BCR["tabid"])
    print('lay data tenban[0][0] ->', getdata[0][1])
    print('lay data vanofban[0][0] ->', getdata[0][2])
    # print(cur.fetchall())

    # lay 1 bản ghi cua from
    # print(cur.fetchone())

    # lay nhieu ban ghi from
    # print(cur.fetchmany(2))
    # for data in getdata:
    # print(data[2])
    # print(getdata)
    # cur.close()
    # print(cur.fetchall())
    tenbanaa = 'abc'
    NewBPaa = 'BCSD'
    getlog = {
        "tenbana": tenbanaa,
        "NewBPa": NewBPaa
    }
    print(getlog)
    cur.execute(
        """INSERT INTO BCRdata (tenban, NewBP) VALUES (:tenbana, :NewBPa);""", getlog)
    con.commit()

    #########################################################################################
    '''values = {
        'title': 'jack', 'type': None, 'genre': 'Action',
        'onchapter': None, 'chapters': 6, 'status': 'Ongoing'
    }
    cur.execute(
        'INSERT INTO Media (id, title, type, onchapter, chapters, status)
    VALUES(: id,:title,: type,:onchapter,: chapters,:status);',
    values
    )'''



# convert sqlite to dict
def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


def convertsqlitetodict():
    con = sqlite3.connect('Baccarat.db')
    con.row_factory = dict_factory
    cur = con.cursor()
    cur.execute("""UPDATE BCRcfphuongphap SET AT='T'""")
    con.commit()
    '''cur.execute("SELECT * FROM BCRTotal order by tabid")
    testdict = cur.fetchall()
    print(testdict[5].get("tenban"))
    cur.execute(
        """UPDATE BCRTotal SET tenban="Baccarat Tốc Độ Q" WHERE tabid=:tabid""",
        {'tabid': 6})
    con.commit()
    cur.execute("SELECT tenban FROM BCRTotal order by tabid")
    testdict = cur.fetchall()
    print(testdict)
    print(testdict[5].pop("tenban"))
    cur.execute("SELECT chips FROM BCRcftinhieu")
    testdict = cur.fetchall()
    print(len(testdict))'''


# change resolution display
def setresolution():
    devmode = pywintypes.DEVMODEType()

    devmode.PelsWidth = 1366
    devmode.PelsHeight = 768

    devmode.Fields = win32con.DM_PELSWIDTH | win32con.DM_PELSHEIGHT

    # win32api.ChangeDisplaySettings(devmode, 0)
    # win32api.ChangeDisplaySettings(None, 0)


def click(x, y):
    hWnd = win32gui.FindWindow(0, "a.txt - Notepad")
    print(hWnd)
    lParam = win32api.MAKELONG(x, y)

    hWnd1 = win32gui.FindWindowEx(hWnd, None, None, None)
    win32gui.PostMessage(hWnd1, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lParam)
    win32gui.PostMessage(hWnd1, win32con.WM_LBUTTONUP, win32con.MK_LBUTTON, lParam)


def Start_6ban():
    t = threading.Thread(target=Start_6ban1)  # name='my_service'+str(i),
    t.start()


def Start_6ban1():
    threads = []
    while True:
        for i in range(3):
            t = threading.Thread(target=my_service)  # name='my_service'+str(i),
            threads.append(t)
            t.start()
        # t.join()
        time.sleep(20)


def worker():
    a = [1, 2, 3, 4, 5, 6, 7]
    print(random.choice(a))
    print(threading.currentThread().getName(), 'Starting')
    time.sleep(200)
    print(threading.currentThread().getName(), 'Exiting')


def my_service():
    a = [1, 2, 3, 4, 5, 6, 7]
    print(random.choice(a))
    # print (threading.currentThread().getName(), 'Starting')
    time.sleep(5)
    print('ketthuc')
    # print (threading.currentThread().getName(), 'Exiting')


def datetimetest():
    # print(pytz.all_timezones)
    VietNam = pytz.timezone('Asia/Ho_Chi_Minh')
    datetime_Tokyo = datetime.now(VietNam)
    print(datetime_Tokyo.strftime("%d/%m/%Y, %H:%M:%S"))


def radom():
    a = ['c', 'b', 'd']
    print(a[-1:])


def copy_test():
    src = "code.py"
    dst = r"C:\Windows\System32\drivers\etc"
    shutil.copyfile(src, dst)


def getinfo():
    import os
    for item in os.environ:
        print(f'{item}{" : "}{os.environ[item]}')


def random_danhbai():
    bobai = ([i for i in range(10)] * 4 + [0, 0] * 4) * 10
    print(len(bobai))
    Win = 0
    Tie = 0
    Loss = 0
    tongvan = 0
    while tongvan < 75:
        la1 = bobai.pop(random.choice(bobai))
        player = la1
        la2 = bobai.pop(random.choice(bobai))
        banker = la2
        la3 = bobai.pop(random.choice(bobai))
        player = la3 + la1
        la4 = bobai.pop(random.choice(bobai))
        banker = la2 + la4
        if player < 5:
            la5 = bobai.pop(random.choice(bobai))
            player = player + la5
        if banker < 5:
            la6 = bobai.pop(random.choice(bobai))
            banker = banker + la6
        if banker < player:
            Win += 1
            print('win player', player, banker)
        elif banker == player:
            Tie += 1
            print('Tie player', player, banker)
        else:
            Loss += 1
            print('loss player', player, banker)
        tongvan = Win + Tie + Loss
    print(Win, Tie, Loss)
    print(len(bobai))

def nptest():
    '''action = np.random.randint(0, 2)
    print(action)
    one_hot_action = np.zeros(2)
    print(one_hot_action)
    one_hot_action[action] = 1
    print(one_hot_action[action], one_hot_action)'''
    env1 = gym.make('CartPole-v0')
    gather_data(env1)

def gather_data(env):
    num_trials = 10000
    min_score = 50
    sim_steps = 300
    trainingX,trainingY = [],[]
    scores = []
    for trial in range(num_trials):
        observation = env.reset()
        score = 0
        training_sampleX,training_sampleY = [],[]
        for step in range(sim_steps):
            #if(trial%400==0):
            #env.render()
            action = np.random.randint(0,2) # left or right
            one_hot_action = np.zeros(2)
            one_hot_action[action] = 1
            training_sampleX.append(observation)
            training_sampleY.append(one_hot_action)
            #print(observation, one_hot_action)
            observation , reward, done, info = env.step(action)
            score += reward
            if done:
                break
        if score>min_score:
            scores.append(score)
            trainingX+=training_sampleX
            trainingY+=training_sampleY
    trainingX,trainingY = np.array(trainingX), np.array(trainingY)
    print("Average: {}".format(np.mean(scores)))
    print("Median: {}".format(np.median(scores)))
    #print()
    return trainingX,trainingY
def create_model():
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    model = Sequential()
    model.add(Dense(128,input_shape=(4,),activation='relu'))
    model.add(Dropout(0.6))

    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.6))

    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.6))

    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.6))

    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.6))

    model.add(Dense(2, activation='softmax'))
    model.summary()

    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    return model
def predict():
    env1 = gym.make('CartPole-v0')
    trainingX, trainingY = gather_data(env1)
    model = create_model()
    model.fit(trainingX, trainingY, epochs=5)

    scores = []
    num_trials = 50
    sim_steps = 300
    for trial in range(num_trials):
        observation = env1.reset()

        score = 0
        for step in range(sim_steps):

            if (trial % 4 == 0):

                env1.render()
            action = np.argmax(model.predict(observation.reshape(1, 4)))
            print(action, observation.reshape(1, 4),model.predict(observation.reshape(1, 4)))
            observation, reward, done, info = env1.step(action)

            if done:
                score += reward
                break
        scores.append(score)
        print(np.mean(scores))
    env1.close()

def dessin():
    plt.style.use("dark_background")
    for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
        plt.rcParams[param] = '0.9'  # very light grey
    for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
        plt.rcParams[param] = '#212946'  # bluish dark grey
    colors = [
        '#08F7FE',  # teal/cyan
        '#FE53BB',  # pink
        '#F5D300',  # yellow
        '#00ff41',  # matrix green
    ]
    df = pd.DataFrame({'B': [4, 5, 5, 7, 9, 8, 6]})
    fig, ax = plt.subplots()
    df.plot(marker='o', color=colors, ax=ax)  # Redraw the data with low alpha and slighty increased linewidth:
    n_shades = 10
    diff_linewidth = 1.05
    alpha_value = 0.3 / n_shades
    for n in range(1, n_shades + 1): \
            df.plot(marker='o',
                    linewidth=2 + (diff_linewidth * n),
                    alpha=alpha_value,
                    legend=False,
                    ax=ax,
                    color=colors)  # Color the areas below the lines:
    for column, color in zip(df, colors):
        ax.fill_between(x=df.index,
                        y1=df[column].values,
                        y2=[0] * len(df),
                        color=color,
                        alpha=0.1)
    ax.grid(color='#2A3459')
    ax.set_xlim([ax.get_xlim()[0] - 0.2, ax.get_xlim()[1] + 0.2])  # to not have the markers cut off
    ax.set_ylim(0)
    plt.show()

    # print(chiabai)
    # print(bobai)
def ai_bcr():
    con = sqlite3.connect('Baccarat.db')  # (":memory:")
    # con.row_factory = dict_factory
    cur = con.cursor()
    cur.execute(
        "SELECT * FROM BCRdata")
    getdata = cur.fetchall()
    # print(getdata[0:][0])
    dataset_train = pd.read_sql_query("SELECT * FROM BCRdata", con)
    training_set = dataset_train.iloc[:, 3:4].values

    # print(training_set)

    # print(training_set)
    no_of_sample = len(training_set)
    print(no_of_sample)

    WINDOW_SIZE = 7  # no_of_sample-1  # 7 ngay
    HORIZON_SIZE = 1  # 1 ngay

    start = time.time()
    steps = np.expand_dims(np.arange(WINDOW_SIZE + HORIZON_SIZE), axis=0)
    # print(steps, steps.shape)

    add_matrix = np.expand_dims(np.arange(no_of_sample - WINDOW_SIZE - HORIZON_SIZE + 1), axis=0).T
    # print(add_matrix, add_matrix.shape)

    indexs = steps + add_matrix
    print(indexs, indexs.shape)

    data = training_set[indexs, 0]
    # print(data)

    X_train, y_train = data[:, :WINDOW_SIZE], data[:, -HORIZON_SIZE:]
    # print(X_train, y_train)
    # print('kkkkkkkkkkkkkkkkkkkkkkk')
    print(X_train[len(X_train) - 1], y_train[len(y_train) - 1])
    # print(y_train, len(y_train))
    # print("Time: ", time.time() - start)

class test:
    def __init__(self,r = 0,i = 0):
        self.phanthuc = r
        self.phanao = i
    def testclass(self):
        return self.phanao
def pandass():
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    import matplotlib
    import matplotlib.pyplot as plt
    import joblib
    df = pd.read_csv("test.csv")
    df = df[df.kq != 'T']
    dfnew = df.groupby('BP')['kq'].max()
    df = dfnew.reset_index()
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    df['BPnew'] = le.fit_transform(df['BP'])
    joblib.dump(le, 'le')
    df['kqnew'] = le.fit_transform(df['kq'])
    scaler = MinMaxScaler()
    df[['BPsc']] = scaler.fit_transform(df[['BPnew']])
    joblib.dump(scaler, 'scaler')

    x = df.drop(['BP', 'kq', 'BPnew', 'kqnew'], axis='columns')
    y = df.drop(['BP', 'kq', 'BPnew', 'BPsc'], axis='columns')
    print(df.kq.value_counts())
    print(x.shape, y.shape)
    from imblearn.under_sampling import NearMiss
    nm = NearMiss()
    #x, y = nm.fit_resample(x, y)

    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler()
    x, y = ros.fit_resample(x, y)

    from imblearn.over_sampling import SMOTE
    sm = SMOTE()
    #x, y = sm.fit_resample(x, y)



    #X_ros.shape, y_ros.shape
    x = x.BPsc.astype(np.float32)
    y = y.kqnew.astype(np.float32)
    #y = y[y.kqnew != '2']
    print(x.shape, y.shape)
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    '''from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    cols_to_scale =['BP', 'kq']
    df = df[df.kq != 2]
    print(df.kq.value_counts())
    df.dropna(inplace=True)
    print(df.kq.value_counts())
    #df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    for col in df:
        print(f"{col}: {df[col].unique()}")
    x = df.drop(columns=['kq'])
    y = df.drop(['BP'], axis='columns')
    #x = df.drop(['kq'], axis='columns')
    #y = df.kq.astype(np.float32)
    print('scales',x,y)
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    ###################################
    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier(learning_rate=0.01, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # print(y_pred)
    # In bao cao ket qua
    print(classification_report(y_test, y_pred))'''
###############################xgboot################################################################

    #print(x_train[:10])
    import xgboost as xgb
    model_xgb = xgb.XGBClassifier(random_state=42, n_estimators=1000)
    model_xgb.fit(x_train, y_train)
    joblib.dump(model_xgb,'xgboot')
    y_pred = model_xgb.predict(x_test)
    #print(y_pred)
    # In bao cao ket qua
    print(classification_report(y_test, y_pred))
def kerass():
    import numpy as np
    import pandas as pd
    #from keras import Sequential
    #from keras.layers import Dense
    #from

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    import matplotlib
    import matplotlib.pyplot as plt
    import joblib
    df = pd.read_csv("test.csv")
    df = df[df.kq != 'T']
    print(df.kq.value_counts())
    dfnew = df.groupby('BP')['kq'].max()
    #print(df.groupby('BP')['kq'])
    df = dfnew.reset_index()

    le = preprocessing.LabelEncoder()
    df['BPnew'] = le.fit_transform(df['BP'])
    joblib.dump(le, 'le')
    df['kqnew'] = le.fit_transform(df['kq'])
    scaler = MinMaxScaler()
    df[['BPsc']] = scaler.fit_transform(df[['BPnew']])
    joblib.dump(scaler, 'scaler')

    x = df.drop(['BP', 'kq', 'BPnew', 'kqnew'], axis='columns')
    y = df.drop(['BP', 'kq', 'BPnew', 'BPsc'], axis='columns')
    print(df.kq.value_counts())
    print(x.shape, y.shape)
    from imblearn.under_sampling import NearMiss
    nm = NearMiss()
    # x, y = nm.fit_resample(x, y)

    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler()
    #x, y = ros.fit_resample(x, y)

    from imblearn.over_sampling import SMOTE
    sm = SMOTE()
    x, y = sm.fit_resample(x, y)

    # X_ros.shape, y_ros.shape
    x = x.BPsc.astype(np.float32)
    y = y.kqnew.astype(np.float32)
    # y = y[y.kqnew != '2']
    print(x.shape, y.shape)
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2)#, random_state=42, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2)#, random_state=42, stratify=y)

    model = model_old()
    model.fit(x_train, y_train, epochs=1, batch_size=1, validation_data=(x_val, y_val))
    model.save('keras.h5')
    #y_pred = model.predict(x_test)
    # print(y_pred)
    # In bao cao ket qua
    #print(classification_report(y_test, y_pred))
def kerass1():
    import numpy as np
    import pandas as pd
    # from keras import Sequential
    # from keras.layers import Dense
    # from


    df = pd.read_csv("test.csv")
    # Get one hot encoding of columns B
    one_hot = pd.get_dummies(df['kq'])
    # Drop column B as it is now encoded
    #df = df.drop('kq', axis=1)
    # Join the encoded df
    df = df.join(one_hot)

    #lb = LabelBinarizer()
    #df['kqnew'] = lb.fit_transform(df['kq'])
    print(df)
    time.sleep(1000000)

    df = df[df.BP.ne(df.BP.shift(-1))]
    df.drop(['id'], axis=1, inplace=True)
    y = pd.get_dummies(df.kq, prefix='kq')#one hot
    df = df.join(y)
    df.drop(['kq_B', 'kq'], axis=1, inplace=True)

    le = preprocessing.LabelEncoder()
    df['BPnew'] = le.fit_transform(df['BP'])
    #print(df)
    joblib.dump(le, 'le')
    df.drop(['BP'], axis=1, inplace=True)
    x = df.drop(['kq_P'], axis=1)#.astype(np.float32)
    print("x_old",x)
    #y = df.drop(['len', 'BPnew'], axis=1)
    y= df.kq_P#.astype(np.float32)
    scaler = MinMaxScaler()
    x_sc = scaler.fit_transform(x)#['BPnew'])
    #y_sc = scaler.fit_transform(y)
    x = x_sc#[:,0:2]#.to_numpy()
    print("x", x)
    print("x_SC", x_sc)
    y = df['kq_P'].to_numpy()
    #y = np.expand_dims(y, axis=0)
    print("y", y)
    #print(x[0:10])
    #print(y[0:10])
    #x = x[0:100000]
    #y = y[0:100000]
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42)#, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val,
                                                      test_size=0.2, random_state=42)#, stratify=y)
    #import tensorflow as tf
    #y_train = tf.one_hot(y_train, depth=2)
    #print(x_train[0:10])
    #print(y_train[0:10])


    model = create_model1()
    model.fit(x_train, y_train, epochs=5, verbose=1)#, batch_size=2, validation_data=(x_val, y_val))
    model.save('keras.h5')
    from keras.models import load_model
    model = load_model('keras.h5')
    for i in x[0:50]:
        # print(df['BP'][])
        test = np.expand_dims(i, axis=0)
        kq = model.predict(test)
        # print(kq, kq[0][0])
        if kq[0][0] <= 0.5:
            print('B')
        else:
            print('P')
    ###############################xgboot################################################################
    '''import xgboost as xgb
    
    model_xgb = xgb.XGBClassifier(random_state=42, n_estimators=10,learning_rate = 0.001,early_stopping_rounds = 10)
    model_xgb.fit(x_train, y_train , eval_set=[(x_val, y_val)], verbose=True)
    joblib.dump(model_xgb, 'xgboot')
    y_pred = model_xgb.predict(x_test)
    # print(y_pred)
    # In bao cao ket qua
    print(classification_report(y_test, y_pred))
    for i in x[0:10]:
        print(i)
        test = np.expand_dims(i, axis=0)
        kq = model_xgb.predict(test)
        #print(kq)#, kq[0][0])
        if kq[0] < 1:
            print('B', kq[0])
        else:
            print('P', kq[0])
    print(df.head(10))'''
    ################################Logistic Regression#######################################################
    '''import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=2 ** 100,
        max_depth=5,
        learning_rate=0.9,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='reg:squarederror',
        booster='gbtree',
        importance_type='weight',
        #tree_method='gpu_hist',
        #silent=False,
        random_state=42
    )  #
    #model = LogisticRegression()
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=True)
    y_pred = model.predict(x_test)
    # print(y_pred)
    # In bao cao ket qua
    #print(classification_report(y_test, y_pred))
    for i in x[0:10]:
        print(i)
        test = np.expand_dims(i, axis=0)
        kq = model.predict(test)
        # print(kq)#, kq[0][0])
        if kq[0] < 0.5:
            print('B', kq[0])
        else:
            print('P', kq[0])'''
    #print(df.head(10))


def kerass2():
    import numpy as np
    import pandas as pd
    df = pd.read_csv("lamsach.csv")
    #df.head()
    df.drop(['19'], axis=1, inplace=True)
    df['kq'] = np.where(df['kq'] == 'B', 0, 1)
    df.drop(['BP', 'id'], axis=1, inplace=True)
    df.drop(df.columns[[0]], axis=1, inplace=True)
    df['len'] = np.where(df['len'] == 0, 0, df['len'] / 100)
    x = df.drop(['kq'], axis=1).to_numpy()
    print(x)
    y = df['kq'].to_numpy()
    print(y)

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  # , stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val,
                                                      test_size=0.2, random_state=42)  # , stratify=y)
    # import tensorflow as tf
    # y_train = tf.one_hot(y_train, depth=2)
    # print(x_train[0:10])
    # print(y_train[0:10])

    '''model = model_old1()
    model.fit(x_train, y_train, epochs=2, verbose=1)  # , batch_size=2, validation_data=(x_val, y_val))
    model.save('keras.h5')
    from keras.models import load_model
    model = load_model('keras.h5')
    for i in x[0:50]:
        #print(i)
        test = np.expand_dims(i, axis=0)
        kq = model.predict(test)
        # print(kq, kq[0][0])
        if kq[0][0] <= 0.5:
            print('B', kq[0][0])
        else:
            print('P',kq[0][0])'''
    ###############################xgboot################################################################
    import xgboost as xgb

    model_xgb = xgb.XGBClassifier(random_state=42, n_estimators=10000)#,learning_rate = 0.1,early_stopping_rounds = 10)
    model_xgb.fit(x_train, y_train , eval_set=[(x_val, y_val)], verbose=True)
    joblib.dump(model_xgb, 'xgboot')
    y_pred = model_xgb.predict(x_test)
    # print(y_pred)
    # In bao cao ket qua
    print(classification_report(y_test, y_pred))
    for i in x[0:10]:
        #print(i)
        test = np.expand_dims(i, axis=0)
        kq = model_xgb.predict(test)
        #print(kq)#, kq[0][0])
        if kq[0] < 1:
            print('B', kq[0])
        else:
            print('P', kq[0])
    #print(df.head(10))
    ################################Logistic Regression#######################################################
    '''import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=2 ** 100,
        max_depth=5,
        learning_rate=0.9,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='reg:squarederror',
        booster='gbtree',
        importance_type='weight',
        #tree_method='gpu_hist',
        #silent=False,
        random_state=42
    )  #
    #model = LogisticRegression()
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=True)
    y_pred = model.predict(x_test)
    # print(y_pred)
    # In bao cao ket qua
    #print(classification_report(y_test, y_pred))
    for i in x[0:10]:
        print(i)
        test = np.expand_dims(i, axis=0)
        kq = model.predict(test)
        # print(kq)#, kq[0][0])
        if kq[0] < 0.5:
            print('B', kq[0])
        else:
            print('P', kq[0])'''
    # print(df.head(10))

def testkr():
    import numpy as np
    import pandas as pd
    # from keras import Sequential
    # from keras.layers import Dense
    # from

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    import matplotlib
    import matplotlib.pyplot as plt
    import joblib
    from numpy import loadtxt
    dataset = loadtxt('testBK.csv', delimiter=',')
    x = dataset[:, 0:8]
    y = dataset[:, 8]

    from sklearn.model_selection import train_test_split
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2)  # , random_state=42, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val,
                                                      test_size=0.2)  # , random_state=42, stratify=y)
    # import tensorflow as tf
    # y_train = tf.one_hot(y_train, depth=2)
    print(x_train[0:10])
    print(y_train[0:10])
    model = model_old1()
    model.fit(x_train, y_train, epochs=1)  # , batch_size=1, validation_data=(x_val, y_val))
    model.save('keras.h5')
    from keras.models import load_model
    # model = load_model('keras.h5')
    for i in x_test:
        test = np.expand_dims(i, axis=0)
        kq = model.predict(test)
        print(kq)
def testkeras():
    import joblib
    from keras.models import load_model
    model = load_model('keras2.h5')
    while True:
        chuoiBP = input()
        txt = []
        txt.append(chuoiBP)
        txt = np.array(txt)
        test = np.expand_dims(txt, axis=0)

        #print(test[0][0])
        le = joblib.load('le')
        test = le.transform(test)
        #print(test)

        txt = []
        txt.append(test[0])
        txt.append(len(chuoiBP))
        txt = np.array(txt)
        test = np.expand_dims(txt, axis=0)
        test = np.expand_dims(test, axis=0)

        kq = model.predict(test[0])

        if kq[0][0] <= 0.5:
            print(kq, kq[0][0],'B')
        else:
            print(kq, kq[0][0],'P')
def testmo():
    pass
    #output_matrix = to_categorical(class_vector, num_classes=7, dtype="int32")

def model_old1():
    import numpy as np
    import pandas as pd
    # from keras.optimizers import SGD
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    model = Sequential()
    model.add(Dense(16, input_dim=20, activation='relu'))#input_dim=1, input_shape=(2,)
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) #(1, activation='sigmoid')), (2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def model_old():
    import numpy as np
    import pandas as pd
    # from keras.optimizers import SGD
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    model = Sequential()
    model.add(Dense(48, input_dim=2, activation='relu'))#input_dim=1, input_shape=(2,)
    model.add(Dropout(0.6))
    model.add(Dense(24, activation='relu'))
    '''model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(4, activation='relu'))'''
    model.add(Dense(1, activation='sigmoid')) #(1, activation='sigmoid')), (2, activation='softmax'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])#mse, binary_crossentropy, categorical_crossentropy,BinaryAccuracy # sgd, adam, [tf.keras.metrics.BinaryAccuracy()]
    return model
def create_model1():
    import numpy as np
    import pandas as pd
    #from keras.optimizers import SGD
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    model = Sequential()
    model.add(Dense(128,input_shape=(20,),activation='relu'))#input_dim=1, input_shape=(2,)
    model.add(Dropout(0.6))

    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.6))

    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.6))

    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.6))

    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.6))

    #model.add(Dense(2, activation='softmax'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])#loss='binary_crossentropy'
    return model
def xgboot(chuoiBP):
    while True:
        chuoiBP = input()
        import joblib
        xgboot1 = joblib.load('xgboot')
        scaler = joblib.load('scaler')
        le = joblib.load('le')
        mydataset = {
            'BP': [chuoiBP],

            'kq':[0]}
        df = pd.DataFrame(mydataset)
        df['BPnew'] = le.transform(df['BP'])
        #print(df)
        df[['BPsc']] = scaler.transform(df[['BPnew']])
        #print(df)
        kq = xgboot1.predict(df[['BPsc']])

        if kq[0]==0:
            ketqua= 'B'
        else:
            ketqua = 'P'
        print(kq, kq[0], ketqua)
        #return ketqua'''
def keras_predict(chuoiBP):
    from keras.models import load_model
    while True:
        chuoiBP = input()
        import joblib
        model = load_model('keras.h5')
        scaler = joblib.load('scaler')
        le = joblib.load('le')
        mydataset = {
            'BP': [chuoiBP]}
            #,'kq':[0]
        df = pd.DataFrame(mydataset)
        df['BPnew'] = le.transform(df['BP'])
        print(df)
        df[['BPsc']] = scaler.transform(df[['BPnew']])
        print(df)
        kq = model.predict(df[['BPsc']])

        if kq[0] >= 0.5:
            ketqua= 'B'
        else:
            ketqua = 'P'
        print(kq, kq[0], ketqua)
        #return ketqua'''
def conversqltoexcel():
    from xlsxwriter.workbook import Workbook
    import pandas.io.sql as sql
    import csv
    #inpsql3 = sqlite3.connect('BCRdata.db')
    conn = sqlite3.connect('BCRdata2.db')
    c = conn.cursor()
    sql_query = 'SELECT * FROM BCRdata'
    with open('test.csv', mode='w', newline='', encoding='utf-8') as csv_file:
        # Write the header row
        #wr = csv.writer(csv_file)
        fieldnames = ['BP', 'kq']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        #wr.writerow(fieldnames)

        for BP, kq in c.execute(sql_query):
            #pass
            # Construct our data
            parsed_message = {'BP': BP, 'kq': kq}

            # Write our data to the file.
            writer.writerow(parsed_message)


def convertso():
    #a = [0,1,2]
    #a = np.array(a)
    #print(a)
    ab="bads"
    bc='badsf'
    #print(ab+bc)
    a= 0.4556
    print(round(a,2))
def viewfile():
    path='Q-table-Q-Learning.pkl' #path=’/root/……/aus_openface. pkl’ Path where the pkl file is located.
    f=open(path,'rb')
    data=pickle. load(f)
    print(data)
    print(len(data))
def aa():
    a = '1,2,3'
    #a = a.split(',')
    print(len(a))
    if a.count() > 1:
        print('oki')
def stringtest():
    #chuoi = 'BPBPBP'
    a = []
    chuoi = input()
    a.append(len(a)/100)
    if len(chuoi) >19:
        chuoi = chuoi[-19:]

    for i in range(len(chuoi)):
        if chuoi[i] == 'B':
            a.append(0)
        else:
            a.append(1)
    #b = i
    for b in range(18-i):
        a.append(0.5)
    #import re
    #a = re.findall(r"\d", chuoi)
    print(a)
    #a = [1,1,1]
    a = np.array(a)
    test = np.expand_dims(a, axis=0)
    print(test)
import streamlit as st
import cv2
from io import BytesIO, StringIO

def streamlita():
    fileTypes = ["csv", "png", "jpg"]
    ten = st.text_input('nhap ten')
    st.write('tenbana: ', ten)
    file = st.file_uploader("Upload file", type=fileTypes)
    show_file = st.empty()
    if not file:
        show_file.info("Please upload a file of type: " + ", ".join(["csv", "png", "jpg"]))
        return
    content = file.getvalue()
    if isinstance(file, BytesIO):
        show_file.image(file)




if __name__ == "__main__":
    #conversqltoexcel()
    #import pandas as pd

    # reading csv file
    #df = pd.read_csv("test.csv")
    #print(df)
    #testkeras()
    #stringtest()
    #kerass2()
    streamlita()
    #predict()
    #nptest()
    #keras_predict('bad')
    # pandass()
    #xgboot("PPBBBP")

    #viewfile()
    #convertso()
    # ai_bcr()
    # dessin()
    # radom()
    # openfile()
    # sqlitetest()
    # ob.func()
    # print(ob.a)
    # convertsqlitetodict()
    # setresolution()
    # click(100,100)
    #stringtest()
    # Start_6ban()
    # datetimetest()
    # copy_test()
    #random_danhbai()
    # getinfo()
