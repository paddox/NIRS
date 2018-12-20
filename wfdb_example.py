import wfdb
from wfdb import processing
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from biosppy import storage
from biosppy.signals import ecg
import biosppy
import plot
import pylab
from matplotlib import mlab

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

#SAMPTO = секунды * частота дискретизации
SAMPTO = 20 * 500


# Demo 1 - Read a wfdb record using the 'rdrecord' function into a wfdb.Record object.
# Plot the signals, and show the data.
#Получение сигнала из Physionet с помощью библиотеки wfdb
record = wfdb.rdrecord('rec_1', pb_dir='ecgiddb/Person_01/', channels=[0], sampto=SAMPTO)#, channels=[1])
#wfdb.plot_wfdb(record=record, title='The ECG-ID Database')

#display(record.__dict__)

#Получение сигнала из record для обработки
sig = [record.p_signal[i][0] for i in range(len(record.p_signal))]

#Чтение сигнала Wmodul
'''
sig = []
with open("Wmodul.txt", "r") as f:
    for row in f:
        a = row.replace(",", ".")
        sig.append(float(a[:-2]))
print(sig)
'''

# load raw ECG signal
#signal, mdata = storage.load_txt('ecg.txt')
#print(mdata)
#Обработка сигнала с помощью библиотеки biosppy
out = ecg.ecg(signal=sig, sampling_rate=500., show=True)
print(out.keys())
print(list(out[1]))


#Построение графиков исходного (sig) и отфильтрованного (out[1] или out['filtered']) сигнала
plt.subplot(211)
plt.plot([i / 1000 for i in range(int(SAMPTO))],sig[:int(SAMPTO)])
plt.title('Raw signal')
plt.xlabel('time (s)') 
plt.ylabel('Raw amplitude') 
plt.grid(True)
#subplot 2
plt.subplot(212)
plt.plot([i / 1000 for i in range(int(SAMPTO))],out['filtered'][:int(SAMPTO)])
plt.grid(True)
plt.xlabel('time (s)') 
plt.ylabel('Filtered amplitude') 
plt.title('Filtered signal')
plt.show()



#Далее идёт поиск зубцов PQRST
filtered = list(out['filtered'])
i = 0
Q = []
R = list(out[2])
S = []
P = []
T = []
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
print(R,Q,S)


#Далее на графике ЭКГ покажем найденные точки PQRST 
xlist1 = [i for i in range(SAMPTO)]
# Вычислим значение функции в заданных точках
ylist1 = out[1]
ylistR = [filtered[i] for i in R]
ylistQ = [filtered[i] for i in Q]
ylistS = [filtered[i] for i in S]
ylistP = [filtered[i] for i in P]
ylistT = [filtered[i] for i in T]
# !!! Нарисуем одномерные графики
plt.plot(list(map(lambda x: x / 500.0, xlist1)), ylist1)
xlistR = out[2]

plt.plot(list(map(lambda x: x / 500.0, R)), ylistR, 'ro')
plt.plot(list(map(lambda x: x / 500.0, Q)), ylistQ, 'go')
plt.plot(list(map(lambda x: x / 500.0, S)), ylistS, 'bo')
plt.plot(list(map(lambda x: x / 500.0, P)), ylistP, 'co')
plt.plot(list(map(lambda x: x / 500.0, T)), ylistT, 'mo')
plt.title('PQRST Extraction')
plt.xlabel('time (s)') 
plt.ylabel('Aamplitude (mV)') 
plt.grid(True)

plt.show()


#Далее покажем найденные точки PQRST относительно циклов серцебиения
R_cent_y = ylistR
N = len(R_cent_y)
R_cent_x = [0 for i in range(N)]
P_cent_x = [P[i] - R[i] for i in range(N)]
Q_cent_x = [Q[i] - R[i] for i in range(N)]
S_cent_x = [S[i] - R[i] for i in range(N)]
T_cent_x = [T[i] - R[i] for i in range(N)]

plt.plot(P_cent_x, ylistP, 'co')
plt.plot(Q_cent_x, ylistQ, 'go')
plt.plot(R_cent_x, R_cent_y, 'ro')
plt.plot(S_cent_x, ylistS, 'bo')
plt.plot(T_cent_x, ylistT, 'mo')

plt.show()
