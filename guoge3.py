# -*- coding: utf-8 -*-
import numpy as np
import os
from math import *
import csv
import pylab as pl
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool
from time import time



# define some function
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    from math import factorial
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    # except ValueError, msg:
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')

# -------Path of data fold ---------------------------------------------
path = r"C:\Xiangtuo\Dropbox\tuotuo\data"
# -------Path of results fold --------------------------------------------
savepath = r"C:\Xiangtuo\Dropbox\tuotuo"
###################################################################################
# define some globe variables


# ---------file to record results
#fout = open(savepath + "/OUT.txt", "wt")
#fplot = open(savepath + "/PLOT.txt", "wt")
# --------------------------parameters---------------------------------


def finalfunction(temp):

    #print("i = ", i)
    tg = temp[0]
    #tg = A[0]
    tstart = temp[1]
    alfat = temp[2]
    tmax = 1600
    denp = 7700
    cp = 0.65
    ug = 39.948
    gama = 1.667
    #tg = 300.0 + 273
    #tg = 573
    #tstart = 2946
    delta_t = 1.0
    wlenth = 694
    sigmag = 1.31

    #print('density is', denp)
    #print('specific heat is', cp)
    #print('alfa T is', alfat)
    #print('sigmag, deviation', sigmag)
    #print('gas temperature', tg)
    #print('wavelength used', wlenth)
    #print('initial temperature', tstart)

    ###########Input############
    #print '-------------------------READDATA------------------------------------'
    files = os.listdir(path)  # Get the name of all files in the fold
    folder_waves = []
    csvfile_name = []
    tt = 0
    for file in files:
        if not os.path.isdir(file):  # check if it is file, open it if it is file
            f = open(path + "/" + file)
            reader = csv.reader(f)
            file_waves = []
            # print '#######################READ', tt, 'th file',file,
            t = 0
            for line in reader:
                if t > 31:
                    string = line[3]  # line1 correspond to 694
                    waves = -float(string)
                    file_waves.append(waves)
                t = t + 1
            f.close()
            # -------------              part: test bad data
            maxsvalue = max(file_waves)
            location = file_waves.index(max(file_waves))
            # -------------  part: cheack bad data and remove file

            # print>>fout, 'check bad data !!!'
            if maxsvalue <= 0.3 or location < 80:
                print 'BAD DATA!!!!!!!!!!!!! and you should delete', tt, 'th file', file, '!!!!!!!!!!!!!!!!!!!!!!!!!'
                # remove must be after f.close()
                os.remove(path + "/" + file)
            else:
                folder_waves.append(file_waves)
                csvfile_name.append(file)
            ########################################################################################

        tt = tt + 1
    #print '-----------------------------------------'
    #print 'Number of orignal files ', tt
    #print('Number of files ', len(csvfile_name), len(folder_waves), 'left')
    #print('--------------------------------------')
    #              part: cooling down data and cut tail
    #--------------------------------
    # ------smooth if
    folder_smooth = []
    for i1 in folder_waves:
        file_smooth = []
        arr_waves = np.array(i1)
        arr_wavessmooth = savitzky_golay(arr_waves, 11, 2)
        file_smooth = arr_wavessmooth.tolist()
        folder_smooth.append(file_smooth)
    folder_waves = folder_smooth
    # -----------------Normalization-------------------------------------------------
    fold_wavenorm = []
    for i1 in folder_waves:
        maxsnormn = i1.index(max(i1))
        maxsnorm = max(i1)
        i1 = [x / maxsnorm for x in i1]
        fold_wavenorm.append(i1)

    folder_waves = fold_wavenorm
    # ---------------------------------------------------------------------------
    # -----------------Cut tail------------------------------------------------
    tmax = 1600
    folder_wavestmax = []
    for i1 in folder_waves:
        maxstmaxn = i1.index(max(i1))
        maxstmax = max(i1)
        file_stmax = []
        tt = 0
        while tt < tmax:
            stmax = i1[maxstmaxn + tt]
            file_stmax.append(stmax)
            tt = tt + 1
        # print 'aaaaaaaaa',file_stmax
        folder_wavestmax.append(file_stmax)

    folder_waves = folder_wavestmax

    #-------------------CALCULATION-------------------------'
    #######################################################################################
    #print '#################model################'
    def func(x, a):
        def lognormalfun(x, cmd2):
            logexpup = (log(x) - log(cmd2)) ** 2
            logexpdown = 2 * log(sigmag) * log(sigmag)
            logexpfront = 1.0 / (sqrt(2 * pi) * x * log(sigmag))
            return logexpfront * exp(-logexpup / logexpdown)

        def Runge_Kutta_step(y, t, dh, ft):
            k1 = ft(y, t)
            k2 = ft(y + 0.5 * k1 * dh, t + dh / 2)
            k3 = ft(y + 0.5 * k2 * dh, t + dh / 2)
            k4 = ft(y + k3 * dh, t + dh)
            return y + (k1 + (2 * k2) + (2 * k3) + k4) * (dh / 6)

        # ---------------------------------------------------------------------------
        def ft(y, t):
            condpart = -3039.75 * alfat * (gama + 1) * sqrt(13.233 * tg / ug) * (y / tg - 1) / (
                        (gama - 1) * dia * denp * cp)
            radpart = -3.4e-10 * (y ** 4 - tg ** 4) / (dia * denp * cp)
            return condpart + radpart

        # ----------------------------------------------------------------------------
        def scalculator(tempy, dia3):
            expup11 = 1.44e7 / (wlenth * tempy)
            expup22 = 1.44e7 / (wlenth * tg)
            exptemp11 = exp(expup11) - 1
            exptemp22 = exp(expup22) - 1
            return dia3 * dia3 * dia3 / exptemp11 - dia3 * dia3 * dia3 / exptemp22
            # print a,x

        cmd = a
        l_l_s = []
        dia = 1
        l_dia = np.arange(1, 100, 2.0)
        for dia in l_dia:
            # -----loop dia-----------
            prob = lognormalfun(dia, cmd)
            # print 'prob',prob
            y = tstart
            l_s = []
            t3 = 0
            while t3 < tmax:
                y = Runge_Kutta_step(y, t3, delta_t, ft)
                s = scalculator(y, dia)
                l_s.append(s)
                t3 = t3 + delta_t
            l_s = [x * prob for x in l_s]
            l_l_s.append(l_s)

        # -------------loop dia--------------------
        ##sumation of s1,s2,s3,.... to s_total
        l_totals = map(sum, zip(*l_l_s))
        # normaliztion
        l_totals = [j / max(l_totals) for j in l_totals]
        arr_totals = np.array(l_totals)
        return arr_totals
    nvariable = 1
    ###########################################################

    #fold_cmd = []
    t = 0
    for i1, i2 in zip(csvfile_name, folder_waves):
        l_t = np.linspace(0, tmax - 1, tmax)
        # print l_t
        popt, pcov = curve_fit(func, l_t, i2, bounds=([3.0], [80.0]))
        arrayfit = func(l_t, *popt)
        print popt[0]
        #fold_cmd.append(popt[0])
        #cmdstring = str(popt[0])

        #pl.close('all')
        #pl.plot(l_t, i2, label="Measured signal")
        #pl.plot(l_t, arrayfit, label='mean d$_m$=' + cmdstring + ' nm')
        #pl.axis([0, tmax, 0, 1.2])
        #pl.legend(), pl.show()
        #pl.savefig(savepath + "/" + i1 + '.png')
        t = t + 1
    #print fold_cmd
    return popt[0]
#######################################################

if __name__ == '__main__':
    N = 10
    tg_0 = 573
    np.random.seed(180428)
    tg_1 = np.random.uniform(low=tg_0 * 0.95, high=tg_0 * 1.05, size=N)
    tstart_0 = 2946
    tstart_1 = np.random.uniform(low=tstart_0 * 0.95, high=tstart_0 * 1.05, size=N)
    alfat_0 = 0.05
    alfat_1 = np.random.uniform(low=0.07, high=0.17, size=N)

    tg_2 = np.random.uniform(low=tg_0 * 0.95, high=tg_0 * 1.05, size=N)
    tstart_2 = np.random.uniform(low=tstart_0 * 0.95, high=tstart_0 * 1.05, size=N)
    alfat_2 = np.random.uniform(low=0.07, high=0.17, size=N)

    # A has N colones and 3 rows
    A = np.vstack((tg_1, tstart_1, alfat_1))
    B = np.vstack((tg_2, tstart_2, alfat_2))
    AB = np.hstack((A, B))
    for j in range(3):
        temp = A
        temp[j, :] = B[j, :]
        AB = np.hstack((AB, temp))

    inputs = range(N*(3+2))
    num_cores = multiprocessing.cpu_count()
    print(num_cores)
    start = time()
    results = Parallel(n_jobs=num_cores-1)(delayed(finalfunction)(AB[:, i]) for i in inputs)
    stop = time()
    print(str(stop - start) + " seconds")
    np.savetxt('y.csv', results, delimiter=',')
    y = np.reshape(results, (-1, N))

    ya = y[0, :]
    np.savetxt('ya.csv', ya, delimiter=',')
    yb = y[1, :]
    np.savetxt('yb.csv', yb, delimiter=',')
    ynormal = y[2:, :]
    np.savetxt('ynormal.csv', ynormal, delimiter=',')

    Vtotal = np.var(results)
    firstorder = []
    Stotal = []

    for i in range(3):
        first = np.mean((ynormal[i, :]-ya)*yb)/Vtotal
        firstorder.append(first)
    print(firstorder)
    np.savetxt('D:/Dropbox/Sfirstorder.csv', firstorder, delimiter=',')

    for i in range(3):
        total = 0.5*np.mean((ya-ynormal[i, :])**2)/Vtotal
        Stotal.append(total)
    print(Stotal)
    np.savetxt('D:/Dropbox/Stotal.csv', Stotal, delimiter=',')



    #Parallel(n_jobs=2, backend="threading")(delayed(finalfunction)(A[:, i]) for i in range(10))

    #with Pool(3) as p:
    #    print(p.map(finalfunction, [A[:, 1], A[:, 2], A[:, 3]]))