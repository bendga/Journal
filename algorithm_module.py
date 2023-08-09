import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import os
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import h5py

# to use with amc_algorithm.ipynb

def start_amc():
    # get the data preprocessed
    dataset = init_setup()
    dataset_prep = prep_data(dataset)
    easy_mods = [
        "OOK",
        "4ASK",
        "BPSK",
        "QPSK",
        "8PSK",
        "16QAM",
        "AM-SSB-SC",
        "AM-DSB-SC",
        "FM",
        "GMSK",
        "OQPSK",
    ]
    easy_mask = np.isin(dataset_prep['label'], easy_mods)
    labels = dataset_prep['label'][easy_mask]
    snr = dataset_prep['snr'][easy_mask]
    samples = dataset_prep['data'][easy_mask]

    return samples,labels,snr,dataset_prep['features']

def import_data(filepath):
    with h5py.File(filepath, "r") as f:
        data = f["X"]
        modulation_onehot = f["Y"]
        snr = f["Z"]
        (
            samples,
            samples_dx,
            sample_real,
            sample_imag,
            sample_amp,
            sample_phs,
            labels,
            snr,
        ) = process_data(data, modulation_onehot, snr)
    # Save the cumulants, labels, and SNRs to disk
    print("Data generation complete")
    print("Saving.....")
    idx = 1
    save_and_load_list(snr, "snr_vec", idx)
    save_and_load_list(labels, "labels", idx)
    save_and_load_list(samples, "cumulants_vec", idx)
    save_and_load_list(sample_real, "cum_real_vec", idx)
    save_and_load_list(sample_imag, "cum_imag_vec", idx)
    save_and_load_list(samples_dx, "cum_dx_vec", idx)
    save_and_load_list(sample_amp, "cum_amp_vec", idx)
    save_and_load_list(sample_phs, "cum_phs_vec", idx)
    
    dataset = {'samples': samples, 'dx': samples_dx, 'real': sample_real, 'imag': sample_imag, 
                    'amplitude': sample_amp, 'phase': sample_phs, 'label': labels, 'snr':snr}   
    # save_and_load_list(dataset, "dataset", idx)
    print("save is done")
    return dataset

def process_data(data, modulation_onehot, snr):
    mods_total = [
        "OOK",
        "4ASK",
        "8ASK",
        "BPSK",
        "QPSK",
        "8PSK",
        "16PSK",
        "32PSK",
        "16APSK",
        "32APSK",
        "64APSK",
        "128APSK",
        "16QAM",
        "32QAM",
        "64QAM",
        "128QAM",
        "256QAM",
        "AM-SSB-WC",
        "AM-SSB-SC",
        "AM-DSB-WC",
        "AM-DSB-SC",
        "FM",
        "GMSK",
        "OQPSK",
    ]  # all labels
    mod_target = mods_total
    # init vectors
    cumulants_vec = []
    cum_dx_vec = []
    cum_real_vec = []
    cum_imag_vec = []
    cum_amp_vec = []
    cum_phs_vec = []
    labels = []
    snr_vec = []

    # Loop through all the samples in the dataset
    for idx in range(snr.size):
        # Get the modulation type for the current sample
        modulation_idx = np.nonzero(modulation_onehot[idx])[0][0]
        current_modulation = mods_total[modulation_idx]
        # Skip the sample if it's below the threshold SNR or if the modulation type is not in the target list
        # if snr[idx][0] < -10 or current_modulation not in mod_target:
        #     continue
        if idx % 10000 == 0:
            print(
                "we have reached ",
                idx,
                " current modulation is ",
                current_modulation,
                " SNR=",
                snr[idx][0],
                "dB",
            )
        signal = data[idx]
        real = signal[:, 0]
        imag = signal[:, 1]
        phase = np.angle(real+1j*imag)
        amplitude = np.abs(real+1j*imag)
        # Calculate the derivative of the signal
        real_dx = np.diff(real)
        imag_dx = np.diff(imag)
        dx = np.transpose(np.array((real_dx, imag_dx)))
        # Calculate the cumulants of the signal and its derivative
        cumulant = cumulant_generation_complex(signal)
        cum_dx = cumulant_generation_complex(dx)
        cum_real = cumulant_generation_real(real)
        cum_imag = cumulant_generation_real(imag)
        cum_amp = cumulant_generation_real(amplitude)
        cum_phs = cumulant_generation_real(phase)
        # Append the cumulants, labels, and SNRs to their respective lists
        cumulants_vec.append(cumulant)
        cum_dx_vec.append(cum_dx)
        cum_real_vec.append(cum_real)
        cum_imag_vec.append(cum_imag)
        cum_amp_vec.append(cum_amp)
        cum_phs_vec.append(cum_phs)
        
        snr_vec.append(snr[idx][0])
        labels.append(current_modulation)
        # var_vec.append(variance)
        # var_vec_dx.append(var_dx)

    return cumulants_vec, cum_dx_vec, cum_real_vec, cum_imag_vec, cum_amp_vec,cum_phs_vec, labels, snr_vec

def cumulant_generation_real(signal):
    # initiate all the needed signals
    signal_squared = pow(signal, 2)
    signal_cubed = pow(signal, 3)
    # get the moments
    M_20 = np.mean(signal_squared, 0)
    M_40 = np.mean(pow(signal, 4), 0)
    M_60 = np.mean(pow(signal, 6), 0)
    M_80 = np.mean(pow(signal, 8), 0)

    # get the cumulants
    cumulant = np.empty(4)
    cumulant[0] = M_20  # c_20
    cumulant[1] = M_40 - 3 * pow(M_20, 2)  # c_40
    cumulant[2] = M_60 - 15 * M_40 * M_20 + 30 * pow(M_20, 3)  # c_60
    cumulant[3] = (
        M_80
        - 28 * M_60 * M_20
        - 35 * pow(M_40, 2)
        + 420 * M_40 * pow(M_20, 2)
        - 630 * pow(M_20, 4)
    )  # c_80

    return cumulant

def cumulant_generation_complex(input):
    # initiate all the needed signals
    real = input[:, 0]
    imag = input[:, 1]
    signal = real + imag * 1j
    signal_conjugate = np.conjugate(signal)
    signal_squared = pow(signal, 2)
    signal_cubed = pow(signal, 3)
    # get the moments
    M_20 = np.mean(signal_squared, 0)
    M_21 = np.mean(signal * signal_conjugate, 0)
    M_40 = np.mean(pow(signal, 4), 0)
    M_41 = np.mean(signal_cubed * signal_conjugate, 0)
    M_42 = np.mean(signal_squared * pow(signal_conjugate, 2), 0)
    M_60 = np.mean(pow(signal, 6), 0)
    M_63 = np.mean(signal_cubed * pow(signal_conjugate, 3), 0)
    M_80 = np.mean(pow(signal, 8), 0)

    # get the cumulants
    cumulant = np.empty(8, dtype=np.complex128)
    cumulant[0] = M_20  # c_20
    cumulant[1] = M_21  # c_21
    cumulant[2] = M_40 - 3 * pow(M_20, 2)  # c_40
    cumulant[3] = M_41 - 3 * M_21 * M_20  # c_41
    cumulant[4] = M_42 - pow(abs(M_20), 2) - 2 * pow(M_21, 2)  # c_42
    cumulant[5] = M_60 - 15 * M_40 * M_20 + 30 * pow(M_20, 3)  # c_60
    cumulant[6] = M_63 - 9 * cumulant[2] * M_21 - 6 * pow(M_21, 3)  # c_63
    cumulant[7] = (
        M_80
        - 28 * M_60 * M_20
        - 35 * pow(M_40, 2)
        + 420 * M_40 * pow(M_20, 2)
        - 630 * pow(M_20, 4)
    )  # c_80

    return cumulant

def init_setup():
    # initiate the dataset for the algorithm
    if not os.path.exists("data"):
        # Create the folder if it doesn't exist
        os.makedirs("data")
    if not os.listdir("data"):
        filepath = "/home/bendegani/AMC_v2/GOLD_XYZ_OSC.0001_1024.hdf5"
        import_data(filepath)
    idx = 2
    print("loading data")
    # load the data
    samples = save_and_load_list(2, "cumulants_vec", idx)
    samples_dx = save_and_load_list(2, "cum_dx_vec", idx)
    sample_real = save_and_load_list(2, "cum_real_vec", idx)
    sample_imag = save_and_load_list(2, "cum_imag_vec", idx)
    # sample_amp = save_and_load_list(2, "cum_amp_vec", idx)
    sample_phs = save_and_load_list(2, "cum_phs_vec", idx)
    labels = save_and_load_list(2, "labels", idx)
    snr = save_and_load_list(2, "snr_vec", idx)
    # dataset = save_and_load_list(2, "dataset", idx)
    dataset = {'samples': samples, 'dx': samples_dx, 'real': sample_real, 'imag': sample_imag, 
                'phase': sample_phs, 'label': labels, 'snr':snr}
    # dataset = save_and_load_list(2, "dataset", idx)
    print('finished')
    return dataset

def save_and_load_list(variable, name, case):
    # save and load using pickle based on what to do (case)
    # case 1 for save my_list
    # case 2 for load file
    path = "data/"
    if case == 1:
        with open(path + name + ".pkl", "wb") as f:
            pickle.dump(variable, f)
    if case == 2:
        with open(path + name + ".pkl", "rb") as f:
            load_variable = pickle.load(f)
        return load_variable
    
def prep_data(dataset):
    # get the data ready for classification
    dataset_prep = {}
    dataset_prep['label'] = np.array(dataset['label'])
    dataset_prep['snr'] = np.array(dataset['snr'])
    samples,feat_labels =cumulant_fix(dataset['samples'],dataset['dx'],dataset['real'],dataset['imag'],dataset['phase'])
    dataset_prep['data'] = samples
    dataset_prep['features'] = feat_labels
    return dataset_prep

def cumulant_fix(cum,dx,I,Q,phase):
    features_vec = []
    for ii in range(len(cum)):
        cum_abs = abs(cum[ii])
        dx_abs = abs(dx[ii])
        abs_I = abs(I[ii])
        abs_Q = abs(Q[ii])
        cum_ph = abs(phase[ii])
        feats = []
        for jj in range(len(cum_abs)):
            # feats.append(np.log10(cum_abs[jj]/cum_abs[1]))
            feats.append(cum_abs[jj])
            
        for jj in range(2,len(dx_abs)):
            # feats.append(np.log10(dx_abs[jj]))
            feats.append(dx_abs[jj])
            
        for jj in range(len(abs_I)):
            # feats.append(np.log10(abs_I[jj]*0.5 + abs_Q[jj]*0.5))
            feats.append(abs_I[jj])
            feats.append(abs_Q[jj])
            
        for jj in range(1,len(cum_ph)):
            # feats.append(np.log10(cum_ph[jj]))
            feats.append(cum_ph[jj])
        features_vec.append(feats) 
    feat_labels = ['C20','C21','C40','C41','C42','C60','C63','C80','Cd40','Cd41','Cd42','Cd60','Cd63','Cd80'
                ,'Ci2','Cq2','Ci4','Cq4','Ci6','Cq6','Ci8','Cq8','Cp4','C6','Cp8']
    return np.array(features_vec),feat_labels