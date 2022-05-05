#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" May 4th
@author: yujun, austin
Read .bit and rf files
3 Main goals
A: Peaks matching/comparison between bitstream transitions and rf energy data
B: Re-engineer JTAG freqeuncy from rf data
C: Envelop matching/comparison between chosen bitstream transition and rf energy data

"""

import csv
from fileinput import filename
import matplotlib.pyplot as plt
import chardet
import pandas as pd
import numpy as np
from matplotlib.pyplot import figure
import math
import os
import glob
import scipy
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
from numpy import array, sign, zeros
from scipy.interpolate import interp1d
from matplotlib.pyplot import plot,show,grid
import os

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

"""*************** 0. Data Structure definitions ***************"""
class bitstream:
    """
    Stores bistream bits and data
    """
    def __init__(self, filename, bins):
        self.bits, self.zeros, self.ones, self.bins, self.test_flips = process_bistream(filename, bins)
        self.peaks, self.bound = get_peaks(self.test_flips)
        

class allData:
    """
    Stores rf data and its analytics
    """
    def __init__(self, data, energy, ifft):
        self.data = data
        self.energy = energy
        self.ifft = ifft
        self.num_files = len(data)
        self.peaks, self.bound = get_peaks(self.energy)

class rfData:
    """
    Stores individual freqeuncy spectrum data directly read from CSV
    """
    def __init__(self, freq, mag):
        self.freq = freq
        self.mag = mag


"""*************** 1. Function definitions ***************"""

def read_values(file_name):
    """
    Reads in the frqeuency spectrum information of a csv with file_name
    Returns two lists - frequency, magnitude
    Assumes data starts at row 22
    """

    file = open(file_name)
    data = file.readlines()

    freq = []   
    mag = []
    for row_count in range(len(data)-1):
        if row_count >= 22: 
            f = data[row_count].split(',')
            if len(f) == 2:   # ignore /n row
                freq.append(float(f[0]))
                mag.append(10**(float(f[1])/20))

    file.close()
    
    return freq, mag

def read_binary_file(fileName):
    """
    Reads in binary bitstream files
    Returns input file as a list of bytes
    """
    with open(fileName, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
    return fileContent

def inverse_transform_helper(data):
    """
    Reads in a specific instance of data from rf at JTAG during loading
    Returns its corresponding time-domain data
    """
    ifft = []
    for i, item in enumerate(data):
        ifft.append(inverse_transform(item.mag))

    return ifft
        

def inverse_transform(freq_domain):
    """
    Reads in list of freqeuncy domain data
    Calculates the inverse real fast fourier transform from the Numpy library
    on the given freqeuncy domain data
    Returns time-domain data
    """

    a = np.fft.irfft(freq_domain)
    return a

def compute_energy(freq, mag):
    """
    Reads in a specific spectrum snapshot
    Calculates the signal energy, integrating amplitude^2 over frequency
    Returns the energy quantity
    """
    num_samples = len(freq)   # same as length of magnitude list
    energy = 0
    for idx in range(num_samples-1):
        delta_f = freq[idx+1] - freq[idx]
        delta_E = delta_f*(mag[idx]**2)
        energy += delta_E
    
    return energy


def process_signal(option):
    """
    Reads in a folder name with rf data
    Performs various analysis
    Returns data structure storing all data and analytics
    """

    dir_list = './' + option + '/'

    # Get list of all files in a given directory sorted by name
    list_of_files = sorted(filter(os.path.isfile, glob.glob(dir_list + '*')))

    list_of_data = []
    list_of_energies = []
    list_of_iffts = []

    for i, item in enumerate(list_of_files):

        this_freq, this_mag = read_values(item) # Reads in frequency and magnitude of given slice
        this_energy = compute_energy(this_freq, this_mag) # Calculates energy
        
        if this_energy < 800:
            # Filters out data when JTAG has no bitstream input action
            pass
        else:
            this_csv = rfData(this_freq, this_mag)
            this_ifft = inverse_transform(this_mag) # IRFFT

            list_of_data.append(this_csv)
            list_of_energies.append(this_energy)
            list_of_iffts.append(this_ifft)

    
    result = allData(list_of_data, list_of_energies, list_of_iffts)
    print("RF files read for folder " + option)
    return result

def generate_file_name(start, end, option):
    """
    Helper function to generate file name, based on csv naming convention
    e.g. 9  --> 'tek0009.csv'
         20 --> 'tek0020.csv'
    """
    filenames = []
    temp = ""
    for number in range(start, end + 1):
        if number in range(0,10):
            temp = './' + option + '/tek000' + str(number) + 'ALL.csv'   # concatenate string
        elif number in range(10, 100):
            temp = './' + option + 'tek00' + str(number) + 'ALL.csv' 
        elif number in range(100, 1000):
            temp = './' + option + 'tek0' + str(number) + 'ALL.csv' 
        filenames.append(temp)
    return filenames
        


def flip_counter(full_list, sampling):
    """
    Reads in bitstream data and sampling rate
    Function to down-sample full_list (high frequency signal of 0 and 1s)
    into chunks of width sampling, and counts the number of 0->1 or 1->0
    transitions within each chunk. 
    Returns two lists of "time" and flip count values (for ease of plotting)  
    """
    total_flips = []   # initialize history list
    
    work_list = full_list[0:(len(full_list) // sampling)*sampling]
    # discard last few bits
    length = len(work_list)
    
    for idx in range(len(work_list)):
        flip = 0
        if np.mod(idx, sampling)  == 0:     
            for idx2 in range(idx, idx+sampling-1):
                if work_list[idx2+1] != work_list[idx2]:
                    flip += 1
            total_flips.append(flip)
     
    return total_flips

def process_bistream(filename, bins):
    """
    Reads in filename of bitstream and number of bins for bitstream to be broken down into
    Calculates number of bit transitions, 0->1 / 1->0
    Returns bitstream read-in and its analytics
    """

    bits = read_binary_file(filename)

    # compute number of 1s and 0s
    zeros = 0
    ones = 0
    rejects = 0
    byteValues = []
    bitValues = []        # history of each 1, 0 bit
    length = len(bits)

    print("Reading bitstream file " + filename)

    for x, i in enumerate(bits): # Iterates over each byte

        # make binary string e.g. '0b110' gut the 0b part
        i_bin = bin(i)[2:] 

        # fit to e.g. 00000110 format
        if len(i_bin) != 8:
            i_bin = '0' * (8 - len(i_bin)) + i_bin

        for idx in range(len(i_bin)):
            if i_bin[idx] == '1':
                ones += 1
                bitValues.append(1)
            else:
                zeros += 1
                bitValues.append(0)
    
    # print(filename)
    # print("Number of bytes: " + str(len(bits)) + "\nNumber of zeros: " + str(zeros) + "\nNumber of ones: " + str(ones))

    # Counts how many transitions within chunks of this size
    sampling = len(bitValues) // bins 

    # print("To achieve", bins, "bins, sample at", sampling, "size.")
    print("Counting 0/1 transitions in bitstream file " + filename)
    test_flips = flip_counter(bitValues, sampling) # Counts number of flips
    print("Bitstream imported for file " + filename)
    return bits, zeros, ones, bins, test_flips

def remove_extra_signal(signal):
    """
    Reads in signal
    Returns signal with non-active files removed
    *** Deprecated for newest version of code
    """
    energies = signal.energy

    for i, item in enumerate(energies):
        if item <= 500:
            signal.energy.pop(i)
            signal.data.pop(i)
            signal.num_files = signal.num_files - 1
        else:
            pass
    signal.ifft = inverse_transform_helper(signal.data)
    signal.peaks, signal.bound = get_peaks(signal.energy)
    return signal
    
def get_peaks(test_list):
    """
    Reads in values of a signal in a list
    Calculates mean, s.d., and find peaks based on highest x percentile
    Return peaks as a list of indices
    """
    mean = sum(test_list) / len(test_list)
    variance = sum([((x - mean) ** 2) for x in test_list]) / len(test_list)
    res = variance ** 0.5

    bound = mean + 1.5 * res # 1.5 standard deviation above mean, 87% percentile of data treated as peaks

    peaks, _ = find_peaks(test_list, height = bound)
    return peaks, bound

def print_peaks(all_data, all_bitstreams):
    """
    Reads in all data collected
    Visualizes all rf energy and bitstream transistion datasets side-by-side, with peaks
    """

    # Creates a subplot
    fig, axes = plt.subplots(nrows=len(all_data), ncols=2, figsize=(12,8))
    i = 0

    if len(all_data) == 1:
        axes[0].plot(range(0, len(all_data[i].energy)), all_data[i].energy) # Plot the data
        res_list = [all_data[i].energy[j] for j in all_data[i].peaks] # Finds peaks of data
        axes[0].plot(all_data[i].peaks, res_list, "x") # Plots the peaks
        axes[0].set_title('Energy Graph for RF Dataset Number ' + str(i))
        axes[0].set_xlabel('Sample Count', fontsize = 8)

        axes[1].plot(range(0, len(all_bitstreams[i].test_flips)), all_bitstreams[i].test_flips)
        res_list = [all_bitstreams[i].test_flips[j] for j in all_bitstreams[i].peaks]
        axes[1].plot(all_bitstreams[i].peaks, res_list, "x")
        axes[1].set_title('0/1 Transitions in Bitstream for Bitstream Number ' + str(i))
        axes[1].set_xlabel('Sample Count', fontsize = 8)

    else:
        for i in range(0, len(all_data)):
            axes[i, 0].plot(range(0, len(all_data[i].energy)), all_data[i].energy)
            res_list = [all_data[i].energy[j] for j in all_data[i].peaks]
            axes[i, 0].plot(all_data[i].peaks, res_list, "x")
            axes[i, 0].set_title('Energy Graph for RF Dataset Number ' + str(i))
            axes[i, 0].set_xlabel('Sample Count', fontsize = 8)

            axes[i, 1].plot(range(0, len(all_bitstreams[i].test_flips)), all_bitstreams[i].test_flips)
            res_list = [all_bitstreams[i].test_flips[j] for j in all_bitstreams[i].peaks]
            axes[i, 1].plot(all_bitstreams[i].peaks, res_list, "x")
            axes[i, 1].set_title('0/1 Transitions in Bitstream for Bitstream Number ' + str(i))
            axes[i, 1].set_xlabel('Sample Count', fontsize = 8)
    fig.tight_layout()
    plt.xticks(fontsize=6)
    plt.show()

    return

def get_one_envelop(input):
    """
    Reads in values of a signal as a list
    Referenced https://stackoverflow.com/questions/34235530/how-to-get-high-and-low-envelope-of-a-signal
    Calculates an upper envelope of given signal
    Returns signal and envelope signal
    """

    s = array(input) # Vector of values

    q_u = zeros(s.shape)
    q_l = zeros(s.shape)

    #Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.
    u_x = [0,]
    u_y = [s[0],]

    l_x = [0,]
    l_y = [s[0],]

    #Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.
    for k in range(1,len(s)-1):
        if (sign(s[k]-s[k-1])==1) and (sign(s[k]-s[k+1])==1):
            u_x.append(k)
            u_y.append(s[k])

        if (sign(s[k]-s[k-1])==-1) and ((sign(s[k]-s[k+1]))==-1):
            l_x.append(k)
            l_y.append(s[k])

    #Append the last value of (s) to the interpolating values. This forces the model to use the same ending point for both the upper and lower envelope models.
    u_x.append(len(s)-1)
    u_y.append(s[-1])

    l_x.append(len(s)-1)
    l_y.append(s[-1])

    #Fit suitable models to the data. Here I am using cubic splines, similarly to the MATLAB example given in the question.
    u_p = interp1d(u_x,u_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
    l_p = interp1d(l_x,l_y,kind = 'cubic',bounds_error = False, fill_value=0.0)

    #Evaluate each model over the domain of (s)
    for k in range(0,len(s)):
        q_u[k] = u_p(k)
        q_l[k] = l_p(k)

    # plot(s);hold(True);plot(q_u,'r');plot(q_l,'g');grid(True);show()
    # plot(s);plot(q_u,'r');plot(q_l,'g');grid(True);show()    
    # plot(s, alpha = 0.3); plot(q_u,'r'); grid(True);show()

    # Implements min and max caps to prevent non-converging tail endings
    max_cap = max(s) * 1.2
    for i in range(0, len(q_u)):
        if q_u[i] < 0:
            q_u[i] = 0
        elif q_u[i] > max_cap:
            q_u[i] = max_cap

    return s, q_u

    return

# """************ 2. Main Section of User Interaction ************"""

all_data = []
all_bitstreams = []

input_num = input("Number of datasets to categorize: ")
i = 0

# Reads in rf and bitstream data
for i in range (1, int(input_num) + 1):
    all_data.append(process_signal(str(i)))
    all_bitstreams.append(bitstream(str(i) + '.bit', all_data[i - 1].num_files))

# Deprecated signal filtering code
# for i in range(0, len(all_data)):
#     all_data[i] = remove_extra_signal(all_data[i])

print("Number of datasets: " + str(len(all_data)))

# Main UI loop
input_choice = ""
while(input_choice != 'q'):
    input_choice = input("Import complete, choose between a(peaks), b(JTAG freq.), c(envelop), q(quit): ")

    if input_choice == 'a':
        print_peaks(all_data, all_bitstreams)
        
    elif input_choice == 'b':
        for i in range(0, len(all_data)):
            length = len(all_data[i].data)
            jtag_freq = (length - 310.83) / (-5.75)
            print("JTAG Frequency Estimate for Sample " + str(i+1) + " : " + str(jtag_freq))        
            
    elif input_choice == 'c':
        rf_choice = input("Pick a rf energy signal for comparison (0-3, when 4 datasets were orginally imported): ")
        bit_choice = input("Pick a bitstream for comparison (0-3, when 4 datasets were orginally imported): ")

        s1, q_u1 = get_one_envelop(all_data[int(rf_choice)].energy)
        s2, q_u2 = get_one_envelop(all_bitstreams[int(bit_choice)].test_flips)

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20,8))

        axes[0].plot(s1, alpha = 0.3); axes[0].plot(q_u1,'r'); axes[0].grid(True)
        axes[0].set_title('Energy Graph for RF Dataset Number ' + str(rf_choice))
        axes[0].set_xlabel('Sample Count', fontsize = 8)

        axes[1].plot(s2, alpha = 0.3); axes[1].plot(q_u2,'r'); axes[1].grid(True)
        axes[1].set_title('0/1 Transitions in Bitstream for Bitstream Number ' + str(bit_choice))
        axes[1].set_xlabel('Sample Count', fontsize = 8)
        
        fig.tight_layout()
        show()

    elif input_choice == 'q':
        print("Program ends")
    else:
        print("Invalid input")
