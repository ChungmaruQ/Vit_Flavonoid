#!/usr/bin/env python
# coding: utf-8

# In[8]:


from pyteomics import mgf, mzxml, auxiliary
import numpy as np
import pandas as pd


# In[30]:


def input_from_mgf(path):
    # Load the MGF file into a Pyteomics "MGF" object
    mgf_file = mgf.MGF(path)
    # Create an empty list to store the MS2 spectra
    ms2_spectrum = {}

    # Loop over each spectrum in the MGF file
    for spectrum in mgf_file:
        # Check if the spectrum is an MS2 spectrum
        # Extract the relevant data from the spectrum
        mzs, intensities = spectrum['m/z array'], spectrum['intensity array']
        precursor_mz = spectrum['params']['pepmass'][0]
        precursor_rt = float(spectrum['params']['rtinseconds'])
        df = pd.DataFrame(np.stack((mzs,intensities),axis=-1))
        mz = df[0] < precursor_mz+1
        df = df[mz]

        df[0] = df[0].round(4)
        df = df.sort_values(1,ascending=False)

        df[1] = df[1]/max(df[1])
        #ion_list = df[1] 
        #df = df[ion_list]
        ion = np.array(df)

        # Store the data in a dictionary
        ms2_spectrum[spectrum['params']['feature_id']] = {'Precursor': precursor_mz,
            'Peaks':ion ,
            'precursor_rt': precursor_rt/60
        }
    return ms2_spectrum


# In[31]:


def image_converter(ID, ms2_spectrum):       
    X = np.zeros((1,2000,1),float) #channel1: defect, channel2: peak channel3: parentMS\
    gnps_peak = ms2_spectrum[ID]['Peaks']

    precursor = ms2_spectrum[ID]['Precursor']
   
        
    for gnps_x,gnps_y in reversed(gnps_peak):
        X[0][int(gnps_x)][0] = gnps_y #intensity      
        
    X[0][int(precursor)+1:] = 0        
    X[0][int(precursor)][0] = 1    
    
    return X


# In[32]:


def input_generator(ID_list, ms2_spectrum):
    batch_size = len(ID_list)
    X = np.zeros((batch_size, 1,2000,1),float)
    # Generate data

    for i, ID in enumerate(ID_list):
        #smiles = gnps[ID]['SMILES']
        #gnps_adduct = gnps[ID]['Adduct']
        X[i], = image_converter(ID,ms2_spectrum)

    return X


# In[35]:


def import_mgf(path):
    ms2_spectrum = input_from_mgf(path)
    spectrum_keys = list(ms2_spectrum.keys())
    inputs = input_generator(spectrum_keys, ms2_spectrum)
    return inputs
    


# In[ ]:




