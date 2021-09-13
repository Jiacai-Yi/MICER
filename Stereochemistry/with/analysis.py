import pandas as pd
import Levenshtein
from numpy import mean
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem
import json
import os

gen_result_filename = 'result/data.txt'

results = pd.read_table(gen_result_filename, header=None)[0].tolist()
targets = pd.read_csv('data/label/test_label.csv')['smiles'].tolist()

metrics_filename = os.path.splitext(gen_result_filename)[0].split('/')[-1]

distances = []
currect_cnt = 0
rdkitfp_tanimotos = []
morganfp_tanimotos = []
rdkitfp_tanimotos_one = 0
morganfp_tanimotos_one = 0

for _, result in (enumerate(results)):
    result = result.strip()
    targets[_] = targets[_].strip()

    distances.append(Levenshtein.distance(result, targets[_]))

    x = Chem.MolFromSmiles(result)
    y = Chem.MolFromSmiles(targets[_])
    if x and y:
        fps1 = FingerprintMols.FingerprintMol(x)
        fps2 = FingerprintMols.FingerprintMol(y)
        tani = DataStructs.TanimotoSimilarity(fps1, fps2)
        rdkitfp_tanimotos.append(tani)
        if tani == 1:
            rdkitfp_tanimotos_one += 1
    else:
        rdkitfp_tanimotos.append(0)
    if x and y:
        morganfps1 = AllChem.GetMorganFingerprint(x, 2)
        morganfps2 = AllChem.GetMorganFingerprint(y, 2)
        morgan_tani = DataStructs.DiceSimilarity(morganfps1, morganfps2)
        morganfp_tanimotos.append(morgan_tani)
        if morgan_tani == 1:
            morganfp_tanimotos_one += 1
    else:
        morganfp_tanimotos.append(0)

    if result == targets[_]:
        currect_cnt += 1

metrics = dict()
metrics['Average Levenshtein Distance'] = mean(distances)
metrics['Sequence Accuracy'] = currect_cnt / len(results)
metrics['Average RDkit Fingerprint Tanimoto Similarity'] = mean(rdkitfp_tanimotos)
metrics['Average Morgan Fingerprint Tanimoto Similarity'] = mean(morganfp_tanimotos)
metrics['RDkit Fingerprint Tanimoto 1.0'] = rdkitfp_tanimotos_one / len(results)
metrics['Morgan Fingerprint Tanimoto 1.0'] = morganfp_tanimotos_one / len(results)

with open('metric_files/' + metrics_filename + '.json', 'w') as f:
    load_dict = json.dump(metrics, f)
