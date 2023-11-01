import os
import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc # import scanpy to handle our AnnData 

adata = ad.read_h5ad("matching/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad")
gex = adata[:,adata.var["feature_types"] == "GEX"]
adt = adata[:,adata.var["feature_types"] == "ADT"]

print("scaling")

sc.pp.scale(gex, max_value=10)

print("scaling complete")

sc.tl.pca(gex, n_comps=200, svd_solver="arpack")

gex_md = gex.obs[["cell_type", "batch", "is_train"]].reset_index(drop = True)
gex_md["CT_id"] = gex_md.cell_type.cat.codes
gex_md["split"] = np.where(gex_md["is_train"] == "train", "train", np.where(gex_md["is_train"] == "test", "val", "test"))

gex_df = pd.concat([pd.DataFrame(gex.obsm["X_pca"]), gex_md], axis = 1)

print(gex_df.columns)

gex_df.to_parquet("data/gex_pca_200.parquet")

adt_md = adt.obs[["cell_type", "batch", "is_train"]].reset_index(drop = True)
adt_md["CT_id"] = adt_md.cell_type.cat.codes
adt_md["split"] = np.where(adt_md["is_train"] == "train", "train", np.where(adt_md["is_train"] == "test", "val", "test"))
adt_df = pd.concat([pd.DataFrame(adt.X.toarray()), adt_md], axis = 1)

adt_df.to_parquet("data/adt.parquet")