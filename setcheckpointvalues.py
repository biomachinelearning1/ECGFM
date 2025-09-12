### Funcion alteradora del .pt (union de diccionarios)

import torch
import sys
import pickle 
import torch
import pickle
def load_dict_from_pkl(pkl_file):
    
    with open(pkl_file, "rb") as f:
        d = pickle.load(f)
    if not isinstance(d, dict):
        raise ValueError(f"El archivo .pkl  {pkl_file} no contiene un diccionario.")
    return d



def merge_dicts(first_py_path, second_pt_path):
    # Load dict1 from Python file
    dict1 = load_dict_from_pkl(first_py_path)

    # Load dict2 from .pt file
    dict2 = torch.load(second_pt_path, map_location=torch.device('cpu'))

    if not isinstance(dict2, dict):
        raise ValueError("El archivo .pt debe contener un diccionario de Python.")


    # Merge dict1 into dict2 
    for k, v in dict1.items():
        if k in dict2:

            if type(v) == 'dict':
                for k2, v2 in v.items():
                    if k2 in dict2[k]:

                        if type(v2) == 'dict':
                            for k3, v3 in v2.items():
                                if k3 in dict2[k][k2]:

                                    if dict2[k][k2][k3] != v3:
                                        print(f"Rewr {k}{k2}.{k3}")
                                        print("Antes: ", dict2[k][k2][k3])
                                        dict2[k][k2][k3] = v3
                                        print("Despues: ", dict2[k][k2][k3])
                                else:
                                    print(f"Add  {k}.{k2}{k3}")
                                    dict2[k][k2][k3] = v3
                                    print("Despues: ", dict2[k][k2][k3])
                                print(" ")

                        else:
                            if dict2[k][k2] != v2:
                                print(f"Rewr {k}.{k2}")
                                print("Antes: ", dict2[k][k2])
                                dict2[k][k2] = v2
                                print("Despues: ", dict2[k][k2])
                    else:
                        print(f"Add  {k}.{k2}")
                        dict2[k][k2] = v2
                        print("Despues: ", dict2[k][k2])
                    print(" ")

            else:
                if dict2[k] != v:
                    print(f"Rewr {k}")
                    print("Antes: ", dict2[k])
                    dict2[k] = v
                    print("Despues: ", dict2[k])

        else:
            print(f"Add  {k}")
            dict2[k] = v
            print("Despues: ", dict2[k])
        
        print(" ")

    # Save result
    output_pt_path_mod = second_pt_path #.replace('.pt', '_merged.pt')
    torch.save(dict2, output_pt_path_mod)
    print(f"\n Diccionario unido y guardado, pisando el archivo original {output_pt_path_mod}")


if __name__ == "__main__":

    dict1_file = sys.argv[1] #'exported_checkpoint_no_model.pkl'#
    pt2_file   = sys.argv[2] #'mimic_iv_ecg_physionet_pretrained.pt'#


    merge_dicts(dict1_file, pt2_file)
