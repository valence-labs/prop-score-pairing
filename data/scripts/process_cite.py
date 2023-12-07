import os
import sys
os.chdir("..")
raw_path = "raw/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad"
save_base_dir = "datasets/neurips_2021_bm/"

if not os.path.exists(save_base_dir):
    os.makedirs(save_base_dir)     

if os.path.exists(save_base_dir + "gex_pca_200.parquet") and os.path.exists(save_base_dir + "adt.parquet"):
    print("Processed files already exist, exiting program")
    sys.exit()

import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc # import scanpy to handle our AnnData 


adata = ad.read_h5ad(raw_path)
gex = adata[:,adata.var["feature_types"] == "GEX"]
adt = adata[:,adata.var["feature_types"] == "ADT"]

if not os.path.exists(save_base_dir + "adt.parquet"):
    adt_md = adt.obs[["cell_type", "batch", "is_train"]].reset_index(drop = True)
    adt_md["CT_id"] = adt_md.cell_type.cat.codes
    adt_md["split"] = np.where(adt_md["is_train"] == "train", "train", np.where(adt_md["is_train"] == "test", "val", "test"))
    adt_df = pd.concat([pd.DataFrame(adt.X.toarray()), adt_md], axis = 1)
    adt_df.columns = adt_df.columns.astype(str)
    adt_df.to_parquet(save_base_dir + "adt.parquet")

    print(f"Protein ADT data processed and saved to {save_base_dir}.")


if not os.path.exists(save_base_dir + "gex_pca_200.parquet"):
    print("scaling")

    sc.pp.scale(gex, max_value=10)

    print("scaling complete")

    sc.tl.pca(gex, n_comps=200, svd_solver="arpack")

    gex_md = gex.obs[["cell_type", "batch", "is_train"]].reset_index(drop = True)
    gex_md["CT_id"] = gex_md.cell_type.cat.codes
    gex_md["split"] = np.where(gex_md["is_train"] == "train", "train", np.where(gex_md["is_train"] == "test", "val", "test"))

    gex_df = pd.concat([pd.DataFrame(gex.obsm["X_pca"]), gex_md], axis = 1)
    gex_df.columns = gex_df.columns.astype(str)
    gex_df.to_parquet(save_base_dir + "gex_pca_200.parquet")

    print(f"200-d PCA Reduction saved to {save_base_dir}.")


