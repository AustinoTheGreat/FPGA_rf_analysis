#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 18 April      v2.1 "Prowler"
@author: yujun, austin
This file computes the frequency-domain signal energy plot over time
for each bitsteam upload intercept set of csv files. We hope to see
certain patterns between different sets of data. 

To use this file, copy all the relevant CSV files to the Computations folder
and run. The energies computed are exported to a new CSV. """
# path name: cd /Users/yujun/Desktop/EENG428_FPGA/Computations

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

class bitstream:
    def __init__(self, filename):
        self.bits, self.zeros, self.ones, self.bins, self.test_counter, self.test_flips = process_bistream(filename)

class allData:
    def __init__(self, data, energy, ifft):
        self.data = data
        self.energy = energy
        self.ifft = ifft
        self.num_files = len(data)

class rfData:
    def __init__(self, freq, mag):
        self.freq = freq
        self.mag = mag


"""*************** 1. Function definitions ***************"""


def read_values(file_name):
    """
    Reads in the frqeuency spectrum information of a csv with file_name
    Returns two lists - frequency, magnitude
    Assumes data starts at row 22
    Subtracts from noise_file
    """
    # # get noise
    # file_n = open(noise_file)
    # data_n = file_n.readlines()
    # noise_mag = []
    # noise = data_n[-1].split(',')
    # for row_count_n in range(len(data_n)):
    #     if row_count_n >= 22: 
    #         f_n = data_n[row_count_n].split(',')
    #         if len(f_n) == 2:   # ignore /n row
    #             noise_mag.append(10**(float(f_n[1])/20))
    #             #print("Noise:", round(10**(float(f_n[1])/20), 4))
    # file_n.close()
    
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
                #if float(f[1]) > -63:                
                #    mag.append(10**(float(f[1])/20))
                #else:
                #    mag.append(0)
    file.close()
    
    # for idx in range(len(mag)):
    #     #print("Before:",mag[idx], " after: ",mag[idx] - noise_mag[idx])
    #     mag[idx] = mag[idx] - noise_mag[idx]    # subtract linearized version
    
    return freq, mag

def read_binary_file(fileName):
    """
    Read binary bitstream files
    """
    with open(fileName, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
    return fileContent

def inverse_transform(freq_domain):
    """
    Calculates the inverse real fast fourier transform from the Numpy library
    on the given freqeuncy domain data
    """
    a = np.fft.irfft(freq_domain)
    return a

def compute_energy(freq, mag):
    """
    Calculates the signal energy, integrating amplitude^2 over frequency
    Input is spectrum dictionary and outputs the energy quantity
    """
    num_samples = len(freq)   # same as length of magnitude list
    energy = 0
    for idx in range(num_samples-1):
        delta_f = freq[idx+1] - freq[idx]
        delta_E = delta_f*(mag[idx]**2)
        energy += delta_E
    
    return energy


def process_signal(option):
    dir_list = os.listdir('./' + option)

    # Get list of all files in a given directory sorted by name
    list_of_files = sorted(filter(os.path.isfile, glob.glob(dir_list + '*')))

    # filenames = generate_file_name()
    list_of_data = []
    list_of_energies = []
    list_of_iffts = []

    for i, item in enumerate(list_of_files):
        this_freq, this_mag = read_values('./' + option + '/' + item)
        this_csv = rfData(this_freq, this_mag)

        this_energy = compute_energy(this_freq, this_mag)

        this_ifft = inverse_transform(this_mag)

        list_of_data.append(this_csv)
        list_of_energies.append(this_energy)
        list_of_iffts.append(this_ifft)

    
    result = allData(list_of_data, list_of_energies, list_of_iffts)

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
    Function to down-sample full_list (high frequency signal of 0 and 1s)
    into chunks of width sampling, and counts the number of 0->1 or 1->0
    transitions within each chunk. 
    Returns two lists of "time" and flip count values (for ease of plotting)  
    """
    total_flips = []   # initialize history list
    
    work_list = full_list[0:(len(full_list) // sampling)*sampling]
    # discard last few bits
    
    for idx in range(len(work_list)):
        flip = 0
        if np.mod(idx, sampling)  == 0:     
            for idx2 in range(idx, idx+sampling-1):
                if work_list[idx2+1] != work_list[idx2]:
                    flip += 1
            total_flips.append(flip)
     
    return total_flips

def process_bistream(filename, bins):
    bits = read_binary_file(filename)


    # compute number of 1s and 0s
    zeros = 0
    ones = 0
    rejects = 0
    byteValues = []
    bitValues = []        # history of each 1, 0 bit

    for i in bits:           # iterate over each byte

        i_bin = bin(i)[2:]    # make binary string e.g. '0b110' gut the 0b part
        if len(i_bin) != 8:
            i_bin = '0' * (8 - len(i_bin)) + i_bin   # fit to e.g. 00000110 format

        for idx in range(len(i_bin)):
            if i_bin[idx] == '1':
                ones += 1
                bitValues.append(1)
            else:
                zeros += 1
                bitValues.append(0)
    
    # print(filename)
    # print("Number of bytes: " + str(len(bits)) + "\nNumber of zeros: " + str(zeros) + "\nNumber of ones: " + str(ones))

    # binned plot of theoretical information signature using bins

    # count how many transitions within chunks of this size
    sampling = len(bitValues) // bins 
    # print("To achieve", bins, "bins, sample at", sampling, "size.")
    
    test_flips = flip_counter(bitValues, sampling)

    return bits, zeros, ones, bins, test_flips






# """************ 2. Working with csv files (oscilloscope) ************"""

all_data = []
all_bitstreams = []

input_num = input("Number of datasets to categorize: ")

for i in range (0, input_num):
    all_data.append(process_signal(input_num))
    all_bitstreams.append(bitstream(input_num + '.bit'), all_data[i].num_files)


# """*************** 3. Working with bitsteam file ***************"""


# # read binary file and show its values
# binaryFileName = "ro_11percentLUTs.bit"
# #ac701.bit   kc705.bit    vcu118.bit

# x = read_binary_file(binaryFileName)
# print(type(x))






# test_counter, test_flips = flip_counter(bitValues, sampling)
# print("test_flips:", test_flips)


# """*************** 4. Plot and save analysis***************"""


# # make a binned version, to match with our energy signature
# fig_tile = "Binary File Transition Values"
# ax1 = plt.subplot()
# l1, = ax1.plot(normalize(energies), color='red', alpha = 0.5)
# ax2 = ax1.twinx()
# l2, = ax2.plot(test_flips, 'bo') # 'ro'
# plt.legend([l1,l2],["Normalized","Bitstream"])
# plt.xlabel("Time")
# plt.ylabel("Transitions")
# plt.title(fig_tile)
# plt.savefig(fig_tile+".png")
# plt.show()


# fig_tile = "Time Domain Data From IRFFT"
# plt.plot(range(0, len(timeDomain)), timeDomain)
# plt.xlabel("Sample count")
# plt.ylabel("Magnitude")
# plt.title(fig_tile)
# plt.savefig(fig_tile+".png")
# plt.show()

# fig_tile = "Energy signature (original)"
# plt.plot(times, energies)
# plt.xlabel("Sample count")
# plt.ylabel("Frequency-domain signal energy")
# plt.title(fig_tile)
# plt.savefig(fig_tile+".png")
# plt.show()

# fig_tile = "Energy signature (normalized)"
# plt.plot(times, normalize(energies))
# plt.xlabel("Sample count")
# plt.ylabel("Frequency-domain signal energy")
# plt.title(fig_tile)
# plt.savefig(fig_tile+".png")
# plt.show()

# fig_tile = "Energy signature (log)"
# plt.plot(times, get_log(energies))
# plt.xlabel("Sample count")
# plt.ylabel("Frequency-domain signal energy")
# plt.title(fig_tile)
# plt.savefig(fig_tile+".png")
# plt.show()

# # open the file in the write mode
# f_out = open('energy0.csv', 'w')
# writer = csv.writer(f_out)
# for value in energies:
#     # write a row to the csv file
#     writer.writerow([float(value)])
# # close the file
# f_out.close()

# print("All computations done, energies exported")

