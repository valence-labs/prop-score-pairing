import scanpy as sc
import pandas as pd
import numpy as np
import networkx as nx
import scglue
import anndata as ad
import os
import sys
sys.path.insert(0, os.path.abspath(".."))
from matching.utils import snn_matching, eot_matching, calc_domainAveraged_FOSCTTM
import warnings
warnings.filterwarnings('ignore')

cite = sc.read("/mnt/ps/home/CORP/johnny.xi/sandbox/matching/data/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad")

adt_ad = cite[:,cite.var.feature_types == "ADT"]
gex_ad = cite[:,cite.var.feature_types == "GEX"]

gex_ad.X = gex_ad.layers["counts"].copy()
sc.pp.normalize_total(gex_ad)
sc.pp.log1p(gex_ad)
sc.pp.scale(gex_ad)
sc.tl.pca(gex_ad, n_comps=100, svd_solver="auto")

adt_ad.X = adt_ad.layers["counts"].copy()
sc.pp.normalize_total(adt_ad)
sc.pp.log1p(adt_ad)
sc.pp.scale(adt_ad)
sc.tl.pca(adt_ad, n_comps=100, svd_solver="auto")

p = np.array(adt_ad.var_names)
r = np.array(gex_ad.var_names)
# mask entries are set to 1 where protein name is the same as gene name
mask = np.repeat(p.reshape(-1, 1), r.shape[0], axis=1) == r
mask = np.array(mask)

rna_vars = [v + "_rna" for v in gex_ad.var_names]
prot_vars = [v + "_prot" for v in adt_ad.var_names]
gex_ad.var_names = rna_vars
adt_ad.var_names = prot_vars

adj = pd.DataFrame(mask, index=prot_vars, columns=rna_vars)
diag_edges = adj[adj > 0].stack().index.tolist()
diag_edges = [(n1, n2, {"weight": 1.0, "sign": 1}) for n1, n2 in diag_edges]
self_loop_rna = [(g, g, {"weight": 1.0, "sign": 1}) for g in rna_vars]
self_loop_prot = [(g, g, {"weight": 1.0, "sign": 1}) for g in prot_vars]

graph = nx.Graph()
graph.add_nodes_from(rna_vars)
graph.add_nodes_from(prot_vars)
graph.add_edges_from(diag_edges)
graph.add_edges_from(self_loop_prot)
graph.add_edges_from(self_loop_rna)

scglue.models.configure_dataset(
    gex_ad,
    "NB",
    use_highly_variable=False,
    use_batch="Samplename",
    use_layer="counts",
    use_rep="X_pca",
)

scglue.models.configure_dataset(
    adt_ad,
    "Normal",
    use_highly_variable=False,
    use_batch="Samplename",
    use_rep="X_pca",
)

glue = scglue.models.fit_SCGLUE(
    {"rna": gex_ad, "adt": adt_ad},
    graph,
)

rna_encodings = glue.encode_data("rna", gex_ad)
adt_encodings = glue.encode_data("adt", adt_ad)

cell_types = np.unique(gex_ad.obs.cell_type)

knn_trace_avg, eot_trace_avg, knn_foscttm_avg, eot_foscttm_avg = 0, 0, 0, 0

for ct in cell_types:
    idx = np.where(gex_ad.obs.cell_type == ct)
    rna_sub, adt_sub = gex_ad.obsm["X_pca"][idx], adt_ad.obsm["X_pca"][idx]
    rna_match_sub, adt_match_sub = rna_encodings[idx], adt_encodings[idx]
    print(f"Cell type: {ct}, Number of samples: {rna_sub.shape[0]}")
    snn_sub = snn_matching(rna_match_sub, adt_match_sub)
    print(f"Cell type: {ct}, kNN trace: {np.trace(snn_sub)}")
    eot_sub = eot_matching(rna_match_sub, adt_match_sub, use_sinkhorn_log = True, verbose = False, max_iter = 200)
    print(f"Cell type: {ct}, EOT trace: {np.trace(eot_sub)}")
    snn_match = snn_sub @ adt_sub
    eot_match = eot_sub @ adt_sub
    knn_foscttm = np.array(calc_domainAveraged_FOSCTTM(adt_sub, snn_match)).mean()
    eot_foscttm = np.array(calc_domainAveraged_FOSCTTM(adt_sub, eot_match)).mean()
    print(f"Cell type: {ct}, kNN FOSCTTM: {knn_foscttm}") 
    print(f"Cell type: {ct}, EOT FOSCTTM: {eot_foscttm}") 

    knn_trace_avg += np.trace(snn_sub)/len(cell_types)
    eot_trace_avg += np.trace(eot_sub)/len(cell_types)
    knn_foscttm_avg += knn_foscttm/len(cell_types)
    eot_foscttm_avg += eot_foscttm/len(cell_types)

print(f"Average kNN Trace {knn_trace_avg}")
print(f"Average EOT Trace {eot_trace_avg}")
print(f"Average kNN FOSCTTM {knn_foscttm_avg}")
print(f"Average EOT FOSCTTM {eot_foscttm_avg}")
