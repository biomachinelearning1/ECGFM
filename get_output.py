### PARA EL ARCHIVO DE OUTPUT 
import sys
import pandas as pd
import torch
from fairseq_signals.utils.store import MemmapReader
import os
import glob


def get_last_created_folder(directory):
    # Get full paths of all subfolders
    folders = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, f))
    ]

    if not folders:
        return None

    # Get the newest one by creation time
    last_folder = max(folders, key=os.path.getctime)
    return last_folder



def get_last_created_file(directory, extension):
    # Build the search pattern (e.g., "*.csv")
    pattern = os.path.join(directory, f"*.{extension.lstrip('.')}")
    files = glob.glob(pattern)

    if not files:
        return None  # No files found

    # Get the file with the newest creation time
    last_file = max(files, key=os.path.getctime)

    return last_file



def get_output_file( ruta_del_output_file ):

    logits = MemmapReader.from_header(
        ruta_del_output_file
    )[:]

    mimic_iv_label_names = ['Poor data quality', 'Sinus rhythm', 'Premature ventricular contraction', 'Tachycardia', 'Ventricular tachycardia', 'Supraventricular tachycardia with aberrancy', 'Atrial fibrillation', 'Atrial flutter', 'Bradycardia', 'Accessory pathway conduction', 'Atrioventricular block', '1st degree atrioventricular block', 'Bifascicular block', 'Right bundle branch block', 'Left bundle branch block', 'Infarction', 'Electronic pacemaker']
    predic = pd.DataFrame(
        torch.sigmoid(torch.tensor(logits)).numpy(),
        columns=mimic_iv_label_names,
    )
    # Join in sample information
    new_df = predic.transpose().reset_index()

    print( new_df )

    return new_df



def retornar_output():
    directory = sys.argv[1:][0]
    
    #directory = 'C:/Users/tomas/outputs/'
    # Buscar la carpeta con la fecha
    last_directory_fecha = get_last_created_folder(directory)

    # Buscar la carpeta con la hora
    last_directory_hora = get_last_created_folder(last_directory_fecha)

    # Buscar el archivo correspondiente
    last_npy_filepath = get_last_created_file(last_directory_hora, "npy")
    if last_npy_filepath is None:
        print("No output files found in the specified directory.")
        return None

    return get_output_file( last_npy_filepath )



if __name__ == "__main__":

    retornar_output()
    