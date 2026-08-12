from scipy.io import loadmat
import pandas as pd

def mat2csv(mat_path):
    data = loadmat(mat_path)
    hpc_df = pd.DataFrame(data['hpcfull'])
    rsc_df = pd.DataFrame(data['rscfull'])
    hpc_df.to_csv('hpcfull.csv', index=False)
    rsc_df.to_csv('rscfull.csv', index=False)
    return hpc_df, rsc_df