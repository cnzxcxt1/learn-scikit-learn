# -*- coding: utf-8 -*-
import scipy
import numpy as np
import sys
import os
from math import *
import csv
import pylab as pl
from scipy.optimize import curve_fit


# define some function
# -------------function to smooth-------------------------------------------
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


###################################################################################
# define some globe variables
tmax = 1600

#######################################################################################
#######################################################################################
alfat = 0.29
#########################Input########################################################
#########################Input#########################################################
# -------Path of data fold ---------------------------------------------
path = r"C:\Xiangtuo\Dropbox\tuotuo\data"
# -------Path of results fold --------------------------------------------
savepath = r"C:\Xiangtuo\Dropbox\tuotuo"
# ---------file to record results
fout = open(savepath + "/OUT.txt", "wt")
fplot = open(savepath + "/PLOT.txt", "wt")
# --------------------------parameters---------------------------------
# denp is the density of np, for C np  is 7700, for Fe is 1860, kg/m3
denp = 7700
# cp is the specific heat of np, for C np is 1.9, for Fe is 0.65, J/gK
cp = 0.65
# alfat is thermal accommodation coefficent, for Fe in Ar around 0.1, in He 0.01, for C 0.23-0.1  in Ar                                                                                                                                                  ##################################
# alfat=0.11
# ug is average molecular weight of gas arond
# ug is 39.948 for Ar , ug=28.74  for Air , g/mol
ug = 39.948
# gama is heat capacity ratio of gas arond,cp/cv
#  gama is 1.667 for Ar, ,1.3-1.4 for Air
gama = 1.667
# tgs=80.0
tg = 300.0 + 273
# tgs is temperature of gas arond
# tstart is the peak temperture measured by two-color method, K
tstart = 2946
# ---------------------------------------------------------
# delta_t is time step length,  1 ns,  5th line in the file
delta_t = 1.0
# --------------------------------------------------
# wlenth is the wavelength used, 694 nm, or 492 nm
wlenth = 694
# sigmag is the deviation of NP.
# sigmag=1.25
sigmag = 1.31
# 1.31 for 850 , log(sigmag)=0.27
# 1.246  for 750, log(sigmag)=0.22
# --------------------------------------------------
# ------------------------------------------------------------------------------
#################################Record 
#########################Input########################################################
#########################Input
files = os.listdir(path)  # Get the name of all files in the fold
folder_waves = []
csvfile_name = []
tt = 0
for file in files:
    if not os.path.isdir(file):  # check if it is file, open it if it is file
        f = open(path + "/" + file);
        reader = csv.reader(f)
        file_waves = []
        # print '#######################READ', tt, 'th file',file,'#########################'
        # print>>fout, '#################READ', tt, 'th file',file,'#########################'
        t = 0
        for line in reader:
            # if t==5:
            # print line
            # print >>fout,line
            # print 'Please check this line carefully'
            # print >>fout,'Please check this line carefully'
            if t > 31:
                string = line[3]  # line1 correspond to 694
                waves = -float(string)
                file_waves.append(waves)
            t = t + 1
        f.close()
        # --------------------------------------------------------------------------------------------
        # -------------              part: test bad data   -----------------------------------------------
        # -----------------------------------------------------------------------------------------------
        maxsvalue = max(file_waves)
        location = file_waves.index(max(file_waves))
        # print 'max value:', maxsvalue
        # print >> fout, 'max value:', maxsvalue
        # print 'location of max value:', location
        # print >> fout,'location of max value:', location
        # print 'length :', len(file_waves)
        # print >> fout,'length :', len(file_waves)
        # --------------------------------------------------------------------------------------------
        # -------------              part: cheack bad data and remove file   -----------------------------------------------
        # -----------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------
        # print 'check bad data !!!'
        # print>>fout, 'check bad data !!!'
        if maxsvalue <= 0.3 or location < 80:

            # remove must be after f.close()
            os.remove(path + "/" + file)
        else:
            folder_waves.append(file_waves)
            csvfile_name.append(file)
        ########################################################################################


# ----------------------------------------------------------------------------------------
#              part: cooling down data and cut tail
# -------------------------------------------------------------------------------
# ---------------------smooth if need-------------------------------------------------------
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
# -------------------------------------------------------------------
# ---------------------Information for each files---------------------
# print '-------------------------Information for each file-------------------------'
# print >> fout, '-------------------------Information for each file------------------------'
# print 'number of files in the folder filename','filename',len(csvfile_name),'scal',len(folder_waves)
# print>>fout, 'number of files in the folder filename','filename',len(csvfile_name),'scal',len(folder_waves)
# t=0
# for i1,i2 in zip(csvfile_name,folder_waves):
#        print t,'th','file:', i1
#       print >> fout,t,'th','file:', i1
#      print 'max  value:', max(i2)
#     print >> fout, 'max  value:', max(i2)
#    print 'location of max value:', i2.index(max(i2))
#   print >> fout,'location of max value:',  i2.index(max(i2))
#  print 'length :', len(i2)
# print >> fout,'length :', len(i2)
# t=t+1
# ---------------------plot each files ---------------------------------------
# for i1,i2 in zip(csvfile_name,folder_waves):
#   l_t=np.linspace(0, tmax-1, tmax)
#  pl.close('all')
#   pl.plot(l_t,i2,label="Measured signal" )
#  pl.axis([0,tmax,0,1.2])
#  pl.legend(),pl.show()
#   pl.savefig(savepath+"/"+i1+'MEASURE.png')
# fout.close()
# sys.exit()
##-------------------------------------------------------------------------------------
# ---------------------------part:LII calculation---------------------------------------
#######################################################################################

# print 'number of files in the folder filename','filename',len(csvfile_name),'scal',len(folder_waves)
# print>>fout, 'number of files in the folder filename','filename',len(csvfile_name),'scal',len(folder_waves)


def func(x, a):
    # ---------------------------------------------------------------------------------------
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
    # arr_s=[]
    l_totals = []
    l_l_s = []
    dia = 1
    l_dia = np.arange(1, 100, 2.0)
    for dia in l_dia:
        # -----------------------loop dia--------------------------------------------
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

    # -----------------------loop dia--------------------------------------------
    ##sumation of s1,s2,s3,.... to s_total
    l_totals = map(sum, zip(*l_l_s))
    # normaliztion
    l_totals = [j / max(l_totals) for j in l_totals]
    arr_totals = np.array(l_totals)
    return arr_totals


nvariable = 1
###########################################################

fold_cmd = []
t = 0
for i1, i2 in zip(csvfile_name, folder_waves):
    l_t = np.linspace(0, tmax - 1, tmax)
    # print l_t
    popt, pcov = curve_fit(func, l_t, i2, bounds=([3.0], [50.0]))
    arrayfit = func(l_t, popt[0])
    print(popt[0])
    fold_cmd.append(popt[0])
    cmdstring = str(popt[0])
    # print 't111',fit1stemp
    # fit1s.append(fit1stemp)

    # fitbar = np.sum(wave1sfit)/len(wave1sfit)  # average
    # ssr=sum([ (ifit1s - fitbar)**2 for ifit1s in fit1s])
    # mse=sum([(ifit1s-iwave1s)**2 for ifit1s,iwave1s in zip(fit1s,wave1sfit)])/len(wave1sfit)
    # sst=sum([ (iwave1s - fitbar)**2 for iwave1s in wave1sfit])

    # rsquare=ssr/sst
    # rsquare2=1-sse/sst
    # adjrsquare=1-(1-rsquare)*(len(wave1sfit)-1)/(len(wave1sfit)-nvariable)
    # print '100*MSE',100*mse,'adj.R-square',adjrsquare
    # print >>fout,'100*MSE',100*mse,'adj.R-square',adjrsquare
    # for idecayfit1s,ilt in zip(fit1s,l_t):
    # if idecayfit1s<=0.2:
    # decay_fold.append(ilt)
    # print t,'th','file:', filenamefit,'decay time',ilt,'adj.R-square',adjrsquare
    # print >>fout,t,'th','file:', filenamefit,'decay time',ilt,'adj.R-square',adjrsquare
    # break
    #    print sst,ssr,sse,ssr+sse
    # print len(l_t),len(arrayfit)
    pl.close('all')
    pl.plot(l_t, i2, label="Measured signal")
    pl.plot(l_t, arrayfit, label='mean d$_m$=' + cmdstring + ' nm')
    pl.axis([0, tmax, 0, 1.2])
    pl.legend(), pl.show()
    pl.savefig(savepath + "/" + i1 + '.png')
    t = t + 1
#print
#fold_cmd
#print >> fout, fold_cmd
# for decaytimes in decay_fold:
#    print  decaytimes
#    print>>fout,decaytimes
#######################################################
fout.close()
sys.exit()