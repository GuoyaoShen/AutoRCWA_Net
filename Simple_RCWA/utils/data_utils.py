import numpy as np
import scipy.io as sio

def load_property_txt(PATH, line_start=-1, line_end=-1):
    '''
    Load .txt material permittivity file.
    return:
    eps_property: permittivity from the file, np array, shape [N_freq, 3].
                          dim1 (3): [freq, real-part, imaginary-part].
    '''
    eps_property = np.loadtxt(PATH)
    if line_start==-1 & line_end==-1:
        pass
    else:
        eps_property = eps_property[line_start:line_end,...]

    return eps_property

def load_property_mat(PATH):
    eps_property = sio.loadmat(PATH)

    return eps_property

def truncate_freq(eps_property, freq_range=None, freq_step=1):
    '''
    Truncate the material property file to focus on a certain freq band.
    eps_property: material property read from the file, a numpy array of shape [N_freq, 3],
                  dim1: [freq, real-part, imaginary-part].
    freq_range: (THz) the range of truncated freq, a list.
                default: None. No truncation.
                if truncate on both sides, pass in [freq_low, freq_high];
                if truncate on high freq side only, pass in [-1, freq_high];
                if truncate on low freq side only, pass in [freq_low, -1]
    freq_step: the freq step to give out final property. An integer.
               default: 1, i.e. step size=1.
    return:
    freq: (Hz) final truncated freq with the indicated freq_step. np array, shape [N_freq_truncated,].
    eps: final truncated permittivity. np array, shape [N_freq_truncated,].
    '''

    if freq_range == None:  # no truncation
        eps_absorber_file = eps_property
    elif freq_range[0] != -1 and freq_range[1] != -1:  # [freq_low, freq_high]
        if freq_range[0] > eps_property[0, 0] and freq_range[1] < eps_property[-1, 0]:
            print('Freq truncated.')
            N_freq_start = np.argmax(eps_property[:, 0] > freq_range[0])
            N_freq_stop = np.argmax(eps_property[:, 0] > freq_range[1])
            eps_absorber_file = eps_property[N_freq_start: N_freq_stop]
        else:
            raise ValueError('Indicated freq range is not available')
    elif freq_range[0] != -1 and freq_range[1] == -1:  # [freq_low, -1]
        if freq_range[0] > eps_property[0, 0]:
            print('Freq truncated.')
            N_freq_start = np.argmax(eps_property[:, 0] > freq_range[0])
            eps_absorber_file = eps_property[N_freq_start:]
        else:
            raise ValueError('Indicated freq range is not available')
    elif freq_range[0] == -1 and freq_range[1] != -1:  # [-1, freq_high]
        if freq_range[1] < eps_property[-1, 0]:
            print('Freq truncated.')
            N_freq_stop = np.argmax(eps_property[:, 0] > freq_range[1])
            eps_absorber_file = eps_property[: N_freq_stop]
        else:
            raise ValueError('Indicated freq range is not available')
    else:
        raise ValueError('Indicated freq range is not available')

    eps_absorber_file = eps_absorber_file[::freq_step]  # larger step size to save calculation time
    eps = eps_absorber_file[:, 1] + eps_absorber_file[:, 2] * 1j

    freq = eps_absorber_file[:, 0] * 1e12

    return freq, eps