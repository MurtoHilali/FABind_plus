import torch
from torch.multiprocessing import Pool, set_start_method
from tqdm import tqdm
import os
import argparse
import esm  # Assuming esm is properly installed
from utils.inference_pdb_utils import extract_protein_structure
import h5py
import tempfile
import shutil

parser = argparse.ArgumentParser(description='Preprocess protein using multiple GPUs.')
parser.add_argument("--pdb_file_dir", type=str, default="../inference_examples/pdb_files",
                    help="Specify the pdb data path.")
parser.add_argument("--save_pt_dir", type=str, default="../inference_examples",
                    help="Specify where to save the processed pt.")
parser.add_argument("--batch_size", type=int, default=4,
                    help="Number of proteins to process per batch.")
args = parser.parse_args()

def extract_esm_feature(proteins, device):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.to(device)
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    data = [(f"protein{i}", protein['seq']) for i, protein in enumerate(proteins)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])

    token_representations = results["representations"][33]
    esm_features = []
    for i, protein in enumerate(proteins):
        seq_len = len(protein['seq'])
        esm_features.append(token_representations[i, 1:seq_len + 1].cpu())

    return esm_features

def process_files_on_gpu(pdb_files, gpu_id, temp_dir):
    device = torch.device(f"cuda:{gpu_id}")
    temp_file = os.path.join(temp_dir, f"temp_gpu_{gpu_id}.h5")

    with h5py.File(temp_file, 'w') as f:
        batch_size = args.batch_size
        for batch_start in tqdm(range(0, len(pdb_files), batch_size), desc=f"Processing on GPU {gpu_id}"):
            batch_files = pdb_files[batch_start:batch_start + batch_size]
            proteins = []
            pdb_names = []

            for pdb_file in batch_files:
                pdb = pdb_file.split(".")[0]
                pdb_filepath = os.path.join(args.pdb_file_dir, pdb_file)
                protein_structure = extract_protein_structure(pdb_filepath)
                protein_structure['name'] = pdb
                proteins.append(protein_structure)
                pdb_names.append(pdb)

            esm_features = extract_esm_feature(proteins, device=device)

            for pdb, esm_feature, protein_structure in zip(pdb_names, esm_features, proteins):
                f.create_dataset(f"esm_features/{pdb}", data=esm_feature.numpy())
                grp = f.create_group(f"protein_metadata/{pdb}")
                for key, value in protein_structure.items():
                    if isinstance(value, str):
                        grp.attrs[key] = value

def merge_temp_files(temp_dir, output_file):
    with h5py.File(output_file, 'w') as out_f:
        for temp_file in os.listdir(temp_dir):
            if not temp_file.endswith('.h5'):
                continue
            temp_path = os.path.join(temp_dir, temp_file)
            with h5py.File(temp_path, 'r') as in_f:
                for key in in_f:
                    # Ensure unique keys by prefixing with temp file name
                    if key in out_f:
                        new_key = f"{temp_file}_{key}"
                    else:
                        new_key = key
                    in_f.copy(key, out_f, name=new_key)

def main():
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    pdb_files = os.listdir(args.pdb_file_dir)
    os.makedirs(args.save_pt_dir, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available. This script requires at least one GPU.")

    chunks = [pdb_files[i::num_gpus] for i in range(num_gpus)]
    temp_dir = tempfile.mkdtemp()

    try:
        with Pool(processes=num_gpus) as pool:
            pool.starmap(process_files_on_gpu, [(chunk, gpu_id, temp_dir) for gpu_id, chunk in enumerate(chunks)])

        h5_file = os.path.join(args.save_pt_dir, 'processed_protein.h5')
        merge_temp_files(temp_dir, h5_file)

        with h5py.File(h5_file, 'r') as f:
            esm2_dict = {key: torch.tensor(f[f"esm_features/{key}"][:]) for key in f["esm_features"]}
            protein_dict = {key: dict(f[f"protein_metadata/{key}"].attrs) for key in f["protein_metadata"]}

        torch.save([esm2_dict, protein_dict], os.path.join(args.save_pt_dir, 'processed_protein.pt'))
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()