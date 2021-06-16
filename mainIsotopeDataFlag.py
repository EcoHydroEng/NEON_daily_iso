# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:55:36 2021

@author: libon
"""


import matplotlib.pyplot as plt
from functools import reduce
import datetime
import numpy as np
import pandas as pd
import copy
import os


'''This script read in the keelinged output files and 
    generate the flags, filled values is -9999.
    Users should filter these values on the istope dataset and the flag dataset before
    qc the dataset
    
    xxxx_npts_qc : xxxx is the four letter abbreviation of NEON sites, 
                   0 [AKA.Good number for keeling] represent number of 
                   points that are used to keeling the isotope is greater or equal to 5,
                   1 is # points smaller than 5, 2 is the filled values 
    xxxx_rsq_qc  : xxxx is the four letter abbreviation of NEON sites, 
                   0 [AKA.Good quality keeling] represent the r2 score of 
                   the Miller-Tans is greater or equal to 0.9,
                   1 is r2 smaller than 0.9, 2 is the filled values 
    xxxx_IQR_qc  : xxxx is the four letter abbreviation of NEON sites, 
                   0 [AKA.Good quality overall] represent if the isotope value is within the
                   interquantile range after xxxx_npts_qc == 0 and xxxx_rsq_qc == 0,
                   1 is not within the range, 2 is the filled values 
'''




def getTimeSeriesPlot(site,name,df):
    cols = df.columns[1:]
    fig, ax = plt.subplots(3,figsize= (70, 60))
    for i in np.arange(len(cols)):
        x = np.arange(len(df))
        ax[i].set_xlim([0,len(df)])
        ax[i].set(xticks=x, xticklabels = df['date'])
        ax[i].scatter(x, df[cols[i]], c = 'r',
                      s = 300, alpha = 0.8)
        for n, label in enumerate(ax[i].xaxis.get_ticklabels()):
            if n % 200 != 0:
                label.set_visible(False)
        ax[i].tick_params("both", labelsize = 70)
        ax[i].set_ylabel(cols[i],fontsize = 80)
    fig.suptitle(name + " (" + site + ')', x = 0.5, y = 0.9, fontsize= 80)

       
def getCleanedIsotope(rawIsoPath, whichIsotope,
                      flagPath = None, 
                      outputIsoPath = None,
                      isTimeSeries = False): 
                                                         
    '''     rawIsoPath  : the absolute path of istope .csv files
              flagPath  : the absolute path where the user would like to store the isotope flags
              dataPath  : the absolute path where the user would like to store the processed isotope datasets
          whichIsotope  : the isotope you need please either specify H2, O18 or C13 
          isTimeSeries  : if the user need the time series plot for the isotope 
    '''
    ##read in the raw isotope dataset 
    assert whichIsotope in ['C13', 'O18', 'H2'], "Please name whichIsotope to be one of [C13, O18, H2]"
    rawIsotope = pd.read_csv(rawIsoPath, index_col=0) 
    # isotopeSite = list(rawIsotope['site'].unique())
    listOfDataDF = []
    listOfFlagDf = []
    isotopeSites = ["BONA","CLBJ","CPER","GUAN","HARV","KONZ",
                    "NIWO","ONAQ","ORNL","OSBS","PUUM","SCBI",
                    "SJER","SRER","TALL","TOOL","UNDE","WOOD",
                    "WREF","YELL","ABBY","BARR","BART","BLAN",
                    "DELA","DSNY","GRSM","HEAL","JERC","JORN",
                    "KONA","LAJA","LENO","MLBS","MOAB","NOGP",
                    "OAES","RMNP","SERC","SOAP","STEI","STER",
                    "TEAK","TREE","UKFS","DCFS","DEJU"]
    isotopeSites.sort()
    print(isotopeSites)
    for site in isotopeSites:
        
        iso = rawIsotope[rawIsotope['site'] == site]
        if iso.empty:
            raw = pd.DataFrame(pd.date_range('2017-01-01', '2020-01-01'), columns=['date'])
            raw['date'] = raw['date'].dt.date
            raw[site] = raw.shape[0]*[np.nan]
            raw[site + '_npts_qc'] = raw.shape[0]*[np.nan]
            raw[site + '_rsq_qc'] = raw.shape[0]*[np.nan]
            raw[site + '_IQR_qc'] = raw.shape[0]*[np.nan]
            listOfDataDF.append(raw[['date', site]])
            listOfFlagDf.append(raw[['date',
                                     site + '_npts_qc',
                                     site + '_rsq_qc',
                                     site + '_IQR_qc']])
            continue
          
            
        ########raw isotope dataset ########
        isotopeFrame = copy.deepcopy(iso) 
        isotopeFrame['date'] = pd.to_datetime(isotopeFrame['date'], format = '%Y-%m-%d %H:%M:%S')
        isotopeFrame['date'] = isotopeFrame['date'].dt.date
        #######################################
        isoRawData = copy.deepcopy(isotopeFrame)
        isoRawData.rename(columns = {'flux':site}, inplace=True)        
        listOfDataDF.append(isoRawData[['date', site]])
        
        #######Generating isotope flags######
        isoFlagData = copy.deepcopy(isotopeFrame)
        ##number of points to generate keeling plot nptsReg >= 5 is 0  1 otherwise
        isoFlagData[site +'_npts_qc'] = np.where(isoFlagData['nptsReg'] >= 5, 0, 1) 
        ##rsq > 0.9 assign flag as 0 otherwise as 1
        isoFlagData[site + '_rsq_qc'] = np.where(isoFlagData['rsq'] >= 0.9, 0, 1)
        ##IQR flag to check if values are within IQR after the r2 >= 0.9 and nptsReg >= 5
        isoFlagData.loc[(isoFlagData[site +'_npts_qc'] == 1) |
                     (isoFlagData[site + '_rsq_qc'] == 1), 'flux'] = np.nan   
        p75, p25 = np.nanpercentile(isoFlagData['flux'], [75, 25]) 
        IQR = p75 - p25
        
        isoFlagData[site + '_IQR_qc'] = isoFlagData['flux'].apply(lambda x: 0 if (x >= p25 - 1.5*IQR and x <= p75 + 1.5*IQR) else 1)
        
        if isTimeSeries:
            rawSeries = isoRawData[['date', site]].copy()
            rawSeries.rename(columns = {site:'raw'}, inplace = True)

            afterqcFlag = isoFlagData[['date', 'flux']].copy()
            afterqcFlag.rename(columns = {'flux':'flag(npt,r2)'}, inplace = True)

            afterAllFlag = isoFlagData.loc[isoFlagData[site + '_IQR_qc'] == 0, ['date','flux']].copy()
            afterAllFlag.rename(columns = {'flux':'flag(npt,r2,IQR)'}, inplace = True)
            
            merged_ = reduce(lambda x, y: pd.merge(x, y, on = 'date', how = 'outer'), 
                              [rawSeries, afterqcFlag, afterAllFlag])  
            getTimeSeriesPlot(site,whichIsotope,merged_)

    ##Overall time series starts and ends        
        listOfFlagDf.append(isoFlagData[['date', site + '_npts_qc',
                                              site + '_rsq_qc',
                                              site + '_IQR_qc']])

      ################flag datafrane  
    flags = reduce(lambda x, y: pd.merge(x, y, on = 'date', how = 'outer'), 
                              listOfFlagDf)  
    Data =  reduce(lambda x, y: pd.merge(x, y, on = 'date', how = 'outer'), 
                                listOfDataDF)
    
    assert min(flags['date']) == min(Data['date']), "start time of flags and data should be the same"
    assert max(flags['date']) == max(Data['date']), "end time of flags and data should be the same"
    
    startDate = min(flags['date'])
    endDate = max(flags['date'])
    
    dts = pd.DataFrame({'date':pd.date_range(startDate, endDate)})
    dts['date'] = dts['date'].dt.date
    
    flags = flags.merge(dts, on = 'date', how = 'outer')

    flags = flags.fillna(2)
    flags.sort_values(by=['date'], ignore_index = True, inplace=True)
    ################isotope dataframe
   
    Data = Data.fillna(-9999)
    Data.sort_values(by=['date'], ignore_index = True, inplace=True)
    
    if (flagPath is not None) and (outputIsoPath is not None):
        Data.to_csv(outputIsoPath + '_qc.csv')
        flags.to_csv(flagPath + whichIsotope + '_data.csv')
    else:
        return Data, flags
       

if __name__ == '__main__':
  dt = getCleanedIsotope('C:/Users/libon/Box/neon_extrac_data/Scientific Data Code/keeling isotopes/et_O18_iso.csv', 'H2',
                         isTimeSeries=False)
                         
    
    
    


    
    
    
    