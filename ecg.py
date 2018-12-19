import pickle
import matplotlib.pyplot as plt
import wfdb
from wfdb import processing
import numpy as np
from IPython.display import display
from biosppy import storage
from biosppy.signals import ecg

SAMPTO = 10000

class QRS:
    def __init__(self, px, py, qx, qy, rx, ry, sx, sy, tx, ty):
        self.px = px
        self.py = py
        self.qx = qx
        self.qy = qy
        self.rx = rx
        self.ry = ry
        self.sx = sx
        self.sy = sy
        self.tx = tx
        self.ty = ty
    def printconsole(self):
        print("px:", self.px)
        print("py:", self.py)
        print("qx:", self.qx)
        print("qy:", self.qy)
        print("rx:", self.rx)
        print("ry:", self.ry)
        print("sx:", self.sx)
        print("sy:", self.sy)
        print("tx:", self.tx)
        print("ty:", self.ty)

def indexOfMaxElement(array):
    sortarray = sorted([(index, value) for (index, value) in enumerate(array)], key=lambda x: x[1])
    i = -1
    retvalue = sortarray[i][0]
    return retvalue

def indexOfMinElement(array):
    sortarray = sorted([(index, value) for (index, value) in enumerate(array)], key=lambda x: x[1])
    i = 0
    retvalue = sortarray[i][0]
    return retvalue

def get_ecg_db():
    data = []
    max_R = 2.0
    min_S = -1.3
    for k in range(1, 91):
        if k in [5, 6, 11, 14, 20, 21, 22, 40, 48, 49, 71, 74, 79, 84, 88, 90]:
            continue
        #Получение данных с сайта
        if k < 10:
            record  = wfdb.rdrecord('rec_1', pb_dir='ecgiddb/Person_0{0}/'.format(k), sampto=SAMPTO, channels=[0])
            record2 = wfdb.rdrecord('rec_2', pb_dir='ecgiddb/Person_0{0}/'.format(k), sampto=SAMPTO, channels=[0])
        else:
            record =  wfdb.rdrecord('rec_1', pb_dir='ecgiddb/Person_{0}/'.format(k), sampto=SAMPTO, channels=[0])
            record2 = wfdb.rdrecord('rec_2', pb_dir='ecgiddb/Person_{0}/'.format(k), sampto=SAMPTO, channels=[0])
        #Преобразование данных к списку
        sig =  [record.p_signal[i][0] for i in range(len(record.p_signal))]
        sig2 = [record2.p_signal[i][0] for i in range(len(record2.p_signal))]
        #обработка сигнала
        out =  ecg.ecg(signal=sig, sampling_rate=500., show=False)
        out2 = ecg.ecg(signal=sig2, sampling_rate=500., show=False)
        #Обработанный сигнал
        filtered = list(out[1])
        filtered.extend(list(out2[1]))
        i = 0
        P = []
        Q = []
        R = list(out[2])
        R.extend([list(out2[2])[i] + 10000 for i in range(len(list(out2[2])))])
        S = []
        T = []
        '''
        plt.plot([i for i in range(20000)], filtered)
        plt.plot(R, [filtered[i] for i in R], 'ro')
        plt.show()
        '''
        for r in R:
            if i == len(R) - 1:
                break
            dx = (R[i+1] - r) / 2
            min_x = int(r - dx) if (r-dx > 0) else 0
            max_x = int(r + dx)
            q_ = indexOfMinElement(filtered[min_x:r]) + min_x
            Q.append(q_)
            s_ = indexOfMinElement(filtered[r:max_x]) + r
            S.append(s_)
            P.append(indexOfMaxElement(filtered[min_x:q_]) + min_x)
            T.append(indexOfMaxElement(filtered[s_:max_x]) + s_)
            i += 1
        del R[-1]
        max_R = max([filtered[j] for j in R]) if max([filtered[j] for j in R]) > max_R else max_R
        min_S = min([filtered[j] for j in S]) if min([filtered[j] for j in S]) < min_S else min_S
        print("max R -> {0:.4f}, min_S -> {1:.4f}".format(max_R, min_S))
        person_data = QRS(P, [filtered[j] for j in P],
                          Q, [filtered[j] for j in Q], 
                          R, [filtered[j] for j in R], 
                          S, [filtered[j] for j in S],
                          T, [filtered[j] for j in T])
        data.append(person_data)
        print("Info with QRS of {0} person added".format(k))
    return data

data = get_ecg_db()

for person in data:
    person.printconsole()

#save to file
with open('ecgiddb_qrs.dat', 'wb') as f:
    pickle.dump(data, f)
    print("Data dumped")
'''
#export from file
with open('data.pickle', 'rb') as f:
    data_new = pickle.load(f)
'''
