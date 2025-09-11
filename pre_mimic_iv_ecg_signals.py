### FOR THE NEW FILE    

## Documentacion:
## (el presente archivo va a estar ubicado en el directorio  )
## Ejecutar el presente archivo desde la terminal de comandos de Anaconda prompt, habiendo activado previamente el entorno en cuestion

## python pre_mimic_iv_ecg_signals.py RUTA_DE_ARCHIVO.erg


## Ejemplo de invocacion:

## python pre_mimic_iv_ecg_signals.py C:/Users/tomas/Downloads/fairseq-signals/scripts/preprocess/MIMIC-IV-ECG/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/files/p1000/p10000032/s40689238/20250717074237.erg


import wfdb
import pandas as pd
import numpy as np
import os, sys

# Define functions to Interpolate values (interpolate_means) and load ECG data (load_and_duplicate_ecg)
def interpolate_means(ecg: np.ndarray) -> np.ndarray:
    """
    Insert interpolated points between every pair of time steps
    in an ECG signal by taking their mean.
    
    Input shape:  (T, C)
    Output shape: (2T - 1, C)
    """
    count = 0
    for k in ecg.shape:
        count +=1

    if count == 2:
        
        T, C = ecg.shape
        interpolated = np.zeros((2 * T - 1, C), dtype=ecg.dtype)
    elif count == 1:
        T,  = ecg.shape
        interpolated = np.zeros((2 * T - 1, ), dtype=ecg.dtype)
    
    # Original values go into even indices
    interpolated[0::2] = ecg

    # Insert means between consecutive points
    interpolated[1::2] = (ecg[:-1] + ecg[1:]) / 2

    interpolated = np.append(interpolated, [interpolated[-1]], axis=0) 

    win_size = 50
    # Takes it all      & computes the median (window=500)
    interpolated_mediana = interpolated #pd.Series( interpolated ).rolling(window=win_size, center=True, min_periods=1).median()    


    return interpolated_mediana


def zscore_segment(segment):
    """
    Apply z-score normalization to each channel (row) of the segment.
    """
    mean = np.mean(segment, keepdims=True)
    std  = np.std( segment, keepdims=True)

    return (segment - mean) / (std + 1e-8)  # small epsilon avoids divide by zero



### PIPELINE ###

## Define & Initialize
col_width = 5
num_cols = 12
E = {}
E['I']   = []
E['II']  = []
E['III'] = []
E['aVR'] = []
E['aVL'] = []
E['aVF'] = []
E['V1']  = []
E['V2']  = []
E['V3']  = []
E['V4']  = []
E['V5']  = []
E['V6']  = []


## Cargar archivo .erg (input = ECG 12-leads)

#ruta = 'C:/Users/tomas/Downloads/fairseq-signals/scripts/preprocess/MIMIC-IV-ECG/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/files/p1000/p10000032/s40689238/20250717074237.erg'
ruta = sys.argv[1:][0]
print(ruta)
print( type(ruta) )
print( len(ruta) )
with open(ruta, 'r') as file:
    for line in file:
        line = line.rstrip('\r\n')
        columns = [int(line[i:i+col_width]) for i in range(0, col_width * num_cols, col_width)]
        E['I'].append( columns[0] )
        E['II'].append( columns[1] )
        E['III'].append( columns[2] )
        E['aVR'].append( columns[3] )
        E['aVL'].append( columns[4] )
        E['aVF'].append( columns[5] )
        E['V1'].append( columns[6] )
        E['V2'].append( columns[7] )
        E['V3'].append( columns[8] )
        E['V4'].append( columns[9] )
        E['V5'].append( columns[10] )
        E['V6'].append( columns[11] )

E_casted = {}
E_casted['I']   = np.array(E['I'], dtype=np.float16)
E_casted['II']  = np.array(E['II'], dtype=np.float16)
E_casted['III'] = np.array(E['III'], dtype=np.float16)
E_casted['aVR'] = np.array(E['aVR'], dtype=np.float16)
E_casted['aVL'] = np.array(E['aVL'], dtype=np.float16)
E_casted['aVF'] = np.array(E['aVF'], dtype=np.float16)
E_casted['V1']  = np.array(E['V1'], dtype=np.float16)
E_casted['V2']  = np.array(E['V2'], dtype=np.float16)
E_casted['V3']  = np.array(E['V3'], dtype=np.float16)
E_casted['V4']  = np.array(E['V4'], dtype=np.float16)
E_casted['V5']  = np.array(E['V5'], dtype=np.float16)
E_casted['V6']  = np.array(E['V6'], dtype=np.float16)

df = pd.DataFrame( E_casted )

# Convertir el DataFrame a NumPy array
signals = df.to_numpy()

# # Definir metadata para el WFDB record
initial_record_fs = 250
initial_record_units     = ['mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV']  # Set units for the first lead
initial_record_sig_name  = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']  # Set signal name for the first lead
record_fmt       = ['16', '16', '16', '16', '16', '16', '16', '16', '16', '16', '16', '16']  # Set format for the first lead (16-bit integer
record = wfdb.Record(  record_name = "40689238",   
                    fs = initial_record_fs,
                    units = initial_record_units,
                    sig_name = initial_record_sig_name,
                    p_signal = signals.astype(float),
                    fmt = record_fmt
                )
record.adc_gain  = [200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0]  # Set gain for all leads
record.adc_res   = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]     # Set baseline for all leads
record.adc_zero  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]     # Set baseline for all leads
record.baseline  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]     # Set baseline for all leads
record.init_value= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]     # Set baseline for all leads

record.samp_rate = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500]  # Set sampling rate
record.samps_per_frame = [15815, 15815, 15815, 15815, 15815, 15815, 15815, 15815, 15815, 15815, 15815, 15815]
record.n_sig     = 12
record.sig_name  = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']  # Set signal name for the first lead
record.units     = ['mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV']  # Set units for the first lead
record.fmt       = ['16', '16', '16', '16', '16', '16', '16', '16', '16', '16', '16', '16']  # Set format for the first lead (16-bit integer)

record.fs = 500
record.byte_offset = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
record.skew        = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
record.sig_len     = [len(E['I']), len(E['II']), len(E['III']), len(E['aVR']), len(E['aVL']), len(E['aVF']), len(E['V1']), len(E['V2']), len(E['V3']), len(E['V4']), len(E['V5']), len(E['V6'])]
record.checksum    = [57052, 57052, 57052, 57052, 57052, 57052, 57052, 57052, 57052, 57052, 57052, 57052]
record.file_name   = ['40689238.dat', '40689238.dat', '40689238.dat', '40689238.dat', '40689238.dat', '40689238.dat', '40689238.dat', '40689238.dat', '40689238.dat', '40689238.dat', '40689238.dat', '40689238.dat']


## z-Normalize & Interpolate:

# Agregar valores intercalados (promedios)
lead_I_  = interpolate_means( zscore_segment(E['I'])   )
lead_II_ = interpolate_means( zscore_segment(E['II'])  )
lead_III_= interpolate_means( zscore_segment(E['III']) )
aVR_     = interpolate_means( zscore_segment(E['aVR']) )
aVL_     = interpolate_means( zscore_segment(E['aVL']) )
aVF_     = interpolate_means( zscore_segment(E['aVF']) )

V1_      = interpolate_means( zscore_segment(E['V1']) )
V2_      = interpolate_means( zscore_segment(E['V2']) )
V3_      = interpolate_means( zscore_segment(E['V3']) )
V4_      = interpolate_means( zscore_segment(E['V4']) )
V5_      = interpolate_means( zscore_segment(E['V5']) )
V6_      = interpolate_means( zscore_segment(E['V6']) )


# Stack all for convenience
partial_1 = np.column_stack((lead_I_, lead_II_))
partial_2 = np.column_stack((lead_III_, aVR_))
partial_3 = np.column_stack((aVL_, aVF_))
partial_4 = np.column_stack((V1_, V2_))
partial_5 = np.column_stack((V3_, V4_))
partial_6 = np.column_stack((V5_, V6_))
all_leads = np.column_stack((partial_1, partial_2, partial_3, partial_4, partial_5, partial_6))

record.p_signal = all_leads
record.adc(inplace=True)
A, B = all_leads.shape
record.sig_len = A
record.block_size = [ int(A), int(A), int(A), int(A), int(A), int(A), int(A), int(A), int(A), int(A), int(A), int(A)]




### Save Interpolated Signal
## Para generar el path de salida (output) automaticamente:
# Toma el primer argumento de la invocacion del presente archivo.py
current_file_path   = os.path.abspath( sys.argv[0] )
local_dir           = os.path.dirname(current_file_path)
reversed_path       = local_dir[::-1]
index_from_end_path = reversed_path.index('\\')
base_path           = reversed_path[index_from_end_path:][::-1]
sub_directorio      = "MIMIC-IV-ECG/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/files/p1000/p10000032/s40689238/"
output_file_path     = base_path + sub_directorio

#output_file_path2   = "C:/Users/tomas/Downloads/fairseq-signals/scripts/preprocess/MIMIC-IV-ECG/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/files/p1000/p10000032/s40689238/"
record.wrsamp( write_dir = output_file_path )