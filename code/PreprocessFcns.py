#Helper fcns for Data Preprocessing
import numpy as np
import pandas as pd
import pathlib
import pickle #to save files
from itertools import product
from scipy.stats import skew, kurtosis
from scipy.signal import butter, welch, filtfilt
import nolds


#extract clips for accelerometer and gyro data (allows selecting start and end fraction)
#lentol is the % of the intended clipsize below which clip is not used
def gen_clips(act_dict,task,location,clipsize=5000,overlap=0,verbose=False,startTS=0,endTS=1,len_tol=0.8,resample=False):

    clip_data = {} #the dictionary with clips

    for trial in act_dict[task].keys():
        clip_data[trial] = {}

        for s in ['accel','gyro']:

            if verbose:
                print(task,' sensortype = %s - trial %d'%(s,trial))
            #create clips and store in a list
            rawdata = act_dict[task][trial][location][s]
            if rawdata.empty is True: #skip if no data for current sensor
                continue
            #reindex time (relative to start)
            idx = rawdata.index
            idx = idx-idx[0]
            rawdata.index = idx
            #choose to create clips only on a fraction of the data (0<[startTS,endTS]<1)
            if (startTS > 0) | (endTS < 1):
                rawdata = rawdata.iloc[round(startTS*len(rawdata)):round(endTS*len(rawdata)),:]
                #reindex time (relative to start)
                idx = rawdata.index
                idx = idx-idx[0]
                rawdata.index = idx
            #create clips data
            deltat = np.median(np.diff(rawdata.index))
            clips = []
            #use entire recording
            if clipsize == 0:
                clips.append(rawdata)
            #take clips
            else:
                idx = np.arange(0,rawdata.index[-1],clipsize*(1-overlap))
                for i in idx:
                    c = rawdata[(rawdata.index>=i) & (rawdata.index<i+clipsize)]
                    if len(c) > len_tol*int(clipsize/deltat): #discard clips whose length is less than len_tol% of the window size
                        clips.append(c)

            #store clip length
            clip_len = [clips[c].index[-1]-clips[c].index[0] for c in range(len(clips))] #store the length of each clip
            #assemble in dict
            clip_data[trial][s] = {'data':clips, 'clip_len':clip_len}

    return clip_data


#store clips from all locations in a dataframe, indexed by the visit - NEED TO REFINE, do NOT USE
def gen_clips_alllocs(act_dict,task,clipsize=5000,overlap=0,verbose=False,len_tol=0.95):

    clip_data = pd.DataFrame()

    for visit in act_dict[task].keys():

        clip_data_visit=pd.DataFrame()

        #loop through locations
        for location in act_dict[task][visit].keys():

            for s in ['accel','gyro']:

                if verbose:
                    print(task,' sensortype = %s - visit %d'%(s,visit))
                #create clips and store in a list
                rawdata = act_dict[task][visit][location][s]
                if rawdata.empty is True: #skip if no data for current sensor
                    continue
                #reindex time (relative to start)
                idx = rawdata.index
                idx = idx-idx[0]
                rawdata.index = idx
                #create clips data
                deltat = np.median(np.diff(rawdata.index))
                clips = pd.DataFrame()
                #take clips
                idx = np.arange(0,rawdata.index[-1],clipsize*(1-overlap))
                for i in idx:
                    c = rawdata[(rawdata.index>=i) & (rawdata.index<i+clipsize)]
                    if len(c) > len_tol*int(clipsize/deltat): #discard clips whose length is less than len_tol% of the window size
                        df = pd.DataFrame({location+'_'+s:[c.values]},index=[visit])
                        clips=pd.concat((clips,df))

                clip_data_visit=pd.concat((clip_data_visit,clips),axis=1) #all clips from all locs for current visit

        clip_data = pd.concat((clip_data_visit,clip_data)) #contains clips from all visits (index) and sensors (cols)

    return clip_data



#PSD on magnitude using Welch method
def power_spectra_welch(rawdata,fm,fM):
    #compute PSD on signal magnitude
    x = rawdata.iloc[:,-1]
    n = len(x) #number of samples in clip
    Fs = np.mean(1/(np.diff(x.index)/1000)) #sampling rate in clip
    f,Pxx_den = welch(x,Fs,nperseg=min(256,n))
    #return PSD in desired interval of freq
    inds = (f<=fM)&(f>=fm)
    f=f[inds]
    Pxx_den=Pxx_den[inds]
    Pxxdf = pd.DataFrame(data=Pxx_den,index=f,columns=['PSD_magnitude'])

    return Pxxdf



#extract features from both sensors (accel and gyro) for current clips and trials
#input: dictionary of clips from each subject
#output: feature matrix from all clips from given subject and scores for each clip
def feature_extraction(clip_data):

    features_list = ['RMSX','RMSY','RMSZ','rangeX','rangeY','rangeZ','meanX','meanY','meanZ','varX','varY','varZ',
                    'skewX','skewY','skewZ','kurtX','kurtY','kurtZ','xcor_peakXY','xcorr_peakXZ','xcorr_peakYZ',
                    'xcorr_lagXY','xcorr_lagXZ','xcorr_lagYZ','Dom_freq','Pdom_rel','PSD_mean','PSD_std','PSD_skew',
                    'PSD_kur','jerk_mean','jerk_std','jerk_skew','jerk_kur','Sen_X','Sen_Y','Sen_Z']

    for trial in clip_data.keys():

        for sensor in clip_data[trial].keys():

            #cycle through all clips for current trial and save dataframe of features for current trial and sensor
            features = []
            for c in range(len(clip_data[trial][sensor]['data'])):
                rawdata = clip_data[trial][sensor]['data'][c]
                #acceleration magnitude
                rawdata_wmag = rawdata.copy()
                rawdata_wmag['Accel_Mag']=np.sqrt((rawdata**2).sum(axis=1))

                #extract features on current clip

                #Root mean square of signal on each axis
                N = len(rawdata)
                RMS = 1/N*np.sqrt(np.asarray(np.sum(rawdata**2,axis=0)))

                #range on each axis
                min_xyz = np.min(rawdata,axis=0)
                max_xyz = np.max(rawdata,axis=0)
                r = np.asarray(max_xyz-min_xyz)

                #Moments on each axis
                mean = np.asarray(np.mean(rawdata,axis=0))
                var = np.asarray(np.std(rawdata,axis=0))
                sk = skew(rawdata)
                kurt = kurtosis(rawdata)

                #Cross-correlation between axes pairs
                xcorr_xy = np.correlate(rawdata.iloc[:,0],rawdata.iloc[:,1],mode='same')
                # xcorr_xy = xcorr_xy/np.abs(np.sum(xcorr_xy)) #normalize values
                xcorr_peak_xy = np.max(xcorr_xy)
                xcorr_lag_xy = (np.argmax(xcorr_xy))/len(xcorr_xy) #normalized lag

                xcorr_xz = np.correlate(rawdata.iloc[:,0],rawdata.iloc[:,2],mode='same')
                # xcorr_xz = xcorr_xz/np.abs(np.sum(xcorr_xz)) #normalize values
                xcorr_peak_xz = np.max(xcorr_xz)
                xcorr_lag_xz = (np.argmax(xcorr_xz))/len(xcorr_xz)

                xcorr_yz = np.correlate(rawdata.iloc[:,1],rawdata.iloc[:,2],mode='same')
                # xcorr_yz = xcorr_yz/np.abs(np.sum(xcorr_yz)) #normalize values
                xcorr_peak_yz = np.max(xcorr_yz)
                xcorr_lag_yz = (np.argmax(xcorr_yz))/len(xcorr_yz)

                #pack xcorr features
                xcorr_peak = np.array([xcorr_peak_xy,xcorr_peak_xz,xcorr_peak_yz])
                xcorr_lag = np.array([xcorr_lag_xy,xcorr_lag_xz,xcorr_lag_yz])

                #Dominant freq and relative magnitude (on acc magnitude)
                Pxx = power_spectra_welch(rawdata_wmag,fm=0,fM=10)
                domfreq = np.asarray([Pxx.iloc[:,-1].argmax()])
                Pdom_rel = Pxx.loc[domfreq].iloc[:,-1].values/Pxx.iloc[:,-1].sum() #power at dominant freq rel to total

                #moments of PSD
                Pxx_moments = np.array([np.nanmean(Pxx.values),np.nanstd(Pxx.values),skew(Pxx.values),kurtosis(Pxx.values)])

                #moments of jerk magnitude
                jerk = rawdata.iloc[:,-1].diff().values
                jerk_moments = np.array([np.nanmean(jerk),np.nanstd(jerk),skew(jerk[~np.isnan(jerk)]),kurtosis(jerk[~np.isnan(jerk)])])

                #sample entropy raw data (magnitude) and FFT
                sH_raw = []; sH_fft = []

                for a in range(3):
                    x = rawdata.iloc[:,a]
                    n = len(x) #number of samples in clip
                    Fs = np.mean(1/(np.diff(x.index)/1000)) #sampling rate in clip
                    sH_raw.append(nolds.sampen(x)) #samp entr raw data
                    #for now disable SH on fft
                    # f,Pxx_den = welch(x,Fs,nperseg=min(256,n/4))
                    # sH_fft.append(nolds.sampen(Pxx_den)) #samp entr fft

                #Assemble features in array
                X = np.concatenate((RMS,r,mean,var,sk,kurt,xcorr_peak,xcorr_lag,domfreq,Pdom_rel,Pxx_moments,jerk_moments,sH_raw))
                features.append(X)

            F = np.asarray(features) #feature matrix for all clips from current trial
            clip_data[trial][sensor]['features'] = pd.DataFrame(data=F,columns=features_list,dtype='float32')

#     return clip_data #not necessary


def HPfilter(act_dict,task,loc,cutoff=0.75,ftype='highpass'):
#highpass (or lowpass) filter data. HP to remove gravity (offset - limb orientation) from accelerometer data from each visit (trial)
#input: Activity dictionary, cutoff freq [Hz], task, sensor location and type of filter (highpass or lowpass).

    sensor = 'accel'
    for trial in act_dict[task].keys():
        rawdata = act_dict[task][trial][loc][sensor]
        if rawdata.empty is True: #skip if no data for current sensor
            continue
        idx = rawdata.index
        idx = idx-idx[0]
        rawdata.index = idx
        x = rawdata.values
        Fs = np.mean(1/(np.diff(rawdata.index)/1000)) #sampling rate
        #filter design
        cutoff_norm = cutoff/(0.5*Fs)
        b,a = butter(4,cutoff_norm,btype=ftype,analog=False)
        #filter data
        xfilt = filtfilt(b,a,x,axis=0)
        rawdatafilt = pd.DataFrame(data=xfilt,index=rawdata.index,columns=rawdata.columns)
        act_dict[task][trial][loc][sensor] = rawdatafilt


#bandpass filter data (analysis of Tremor)
#input: Activity dictionary, min,max freq [Hz], task and sensor location to filter
def BPfilter(act_dict,task,loc,cutoff_low=3,cutoff_high=8,order=4):

    sensor = 'accel'
    for trial in act_dict[task].keys():
        rawdata = act_dict[task][trial][loc][sensor]
        idx = rawdata.index
        idx = idx-idx[0]
        rawdata.index = idx
        x = rawdata.values
        Fs = np.mean(1/(np.diff(rawdata.index)/1000)) #sampling rate
        #filter design
        cutoff_low_norm = cutoff_low/(0.5*Fs)
        cutoff_high_norm = cutoff_high/(0.5*Fs)
        b,a = butter(order,[cutoff_low_norm,cutoff_high_norm],btype='bandpass',analog=False)
        #filter data
        xfilt = filtfilt(b,a,x,axis=0)
        rawdatafilt = pd.DataFrame(data=xfilt,index=rawdata.index,columns=rawdata.columns)
        act_dict[task][trial][loc][sensor] = rawdatafilt

#filters data in all recordings (visits)
def filterdata(act_dict,task,loc,trial,sensor='accel',ftype='highpass',cutoff=0.5,cutoff_bp=[3,8],order=4):

    rawdata = act_dict[task][trial][loc][sensor]
    if not rawdata.empty:
        idx = rawdata.index
        idx = idx-idx[0]
        rawdata.index = idx
        x = rawdata.values
        Fs = np.mean(1/(np.diff(rawdata.index)/1000)) #sampling rate
        if ftype != 'bandpass':
            #filter design
            cutoff_norm = cutoff/(0.5*Fs)
            b,a = butter(4,cutoff_norm,btype=ftype,analog=False)
        else:
            #filter design
            cutoff_low_norm = cutoff_bp[0]/(0.5*Fs)
            cutoff_high_norm = cutoff_bp[1]/(0.5*Fs)
            b,a = butter(order,[cutoff_low_norm,cutoff_high_norm],btype='bandpass',analog=False)

        #filter data
        xfilt = filtfilt(b,a,x,axis=0)
        rawdatafilt = pd.DataFrame(data=xfilt,index=rawdata.index,columns=rawdata.columns)
        act_dict[task][trial][loc][sensor] = rawdatafilt



## CURRENTLY deprecated
#returns power spectra of the signal over each channel between min and max freq at given resolution (nbins)
#returns the labels for each bin
#if binavg is True it averages the PSD within bins to reduce PSD noise
def powerspectra(x,fm,fM,nbins=10,relative=False,binavg=True):

    #feature labels
    labels=[]
    s = np.linspace(fm,fM,nbins)
    lax = ['X','Y','Z']
    for l in lax:
        for i in s:
            labels.append('fft'+l+str(int(i)))

    #signal features
    n = len(x) #number of samples in clip
    Fs = np.mean(1/(np.diff(x.index)/1000)) #sampling rate in clip
    timestep = 1/Fs
    freq = np.fft.fftfreq(n,d=timestep) #frequency bins

    #run FFT on each channel
    Xf = x.apply(np.fft.fft)
    Xf.index = np.round(freq,decimals=1) #reindex w frequency bin
    Pxx = Xf.apply(np.abs)
    Pxx = Pxx**2 #power spectra
    if relative:
        Pxx = Pxx/np.sum(Pxx,axis=0) #power relative to total

    #power spectra between fm-fM Hz
    bin1 = int(timestep*n*fm)
    bin2 = int(timestep*n*fM)
    bins = np.linspace(bin1,bin2,nbins,dtype=int)
#     print(bins/(round(timestep*n)))

    #average power spectra within bins
    if binavg:
        deltab = int(0.5*np.diff(bins)[0]) #half the size of a bin (in samples)
        Pxxm = []
        for i in bins:
            start = int(max(i-deltab,bins[0]))
            end = int(min(i+deltab,bins[-1]))
            Pxxm.append(np.mean(Pxx.iloc[start:end,:].values,axis=0))
        Pxxm = np.asarray(Pxxm)
        Pxx = pd.DataFrame(data=Pxxm,index=Pxx.index[bins],columns=Pxx.columns)
        return Pxx, labels

    else:
        return Pxx.iloc[bins,:], labels
