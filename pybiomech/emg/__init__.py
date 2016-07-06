"""
 PyBiomech -> EMG
 Author: Jeff M. Barrett, M.Sc. Candidate
                          University of Waterloo

This is a package for processing EMG and calculating


"""

import numpy as np
import scipy.signal as sig


'''
==============================================================================================================
                    ELECTROMYOGRAPHY
==============================================================================================================
Basic tools for electromyography
'''



def remove_bias(emg):
    """
    Removes the bias in the provided EMG array
    :param emg:         NxM (N is the number of frames, and M is the number of channels) EMG data
    :return:
    """
    return emg - np.mean(emg, axis = 0)


def full_wave_rectify(emg):
    """
    Purpose: Computes the full-wave rectification of the provided EMG data
             Note: This will assume that the EMG dataset has zero mean
    :param emg:         NxM (N is the number of frames and M is the number of channels)
    :return:
    """
    return np.abs(emg)


def linear_envelope(emg, fc, fs):
    """
    Purpose: Linearly envelopes the provided EMG data using the cutoff frequency specified in fs for the low-pass filter
             The procedure follows that as described in Winter's 1990 textbook:
                1. remove the bias of the EMG
                2. full-wave rectify the EMG
                3. apply a dual-pas
    :param emg:             the emg array   (NxM) where N is the number of frames and M is the number of channels
                            Note: this emg matrix could, in principle, be (Nx(M1xM2)) or any subsize, as long as the
                                  first dimension is the time-dimension, this function will work.
    :param fc:              the cutoff frequency
    :param fs:              the sampling rate of the EMG
    :return:                returns linearly enveloped EMG data
    """
    b, a = sig.butter(2, fc/(2*fs))
    return sig.lfilter(b, a, full_wave_rectify(remove_bias(emg)), axis = 0)











