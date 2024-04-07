#!/usr/bin/env python
# coding: utf-8

# # Applying mAP to Plate 4 data at Single Cell level.

# In[10]:


import pathlib
import sys

import copairs.map as map
import pandas as pd
from pycytominer.cyto_utils import infer_cp_features

sys.path.append("../")  # noqa
from src.utils import shuffle_features  # noqa

# In[11]:


# setting up paths
plate4_path = pathlib.Path("../data/Plate_4_sc_normalized.parquet")

# output paths
results_path = pathlib.Path("./results").resolve()
results_path.mkdir(exist_ok=True)


# In[13]:


# loading in plate 4 normalized profile
plate4_df = pd.read_parquet(plate4_path)

# replacing None with "No Constructs"
plate4_df["Metadata_siRNA"].fillna("No Construct", inplace=True)
plate4_df.dropna(inplace=True)

# display dataframe with
print("shape:", plate4_df.shape)
plate4_df.head()


# In[14]:


# splitting feature space
meta_features = infer_cp_features(plate4_df, metadata=True)
cp_features = infer_cp_features(plate4_df)

# extract siRNA perturbations
siRNAs = plate4_df["Metadata_siRNA"].unique().tolist()

# display
print("aggregated profile shape", plate4_df.shape)
print("siRNA types", plate4_df["Metadata_siRNA"].unique().tolist())
print(
    f"Number of '{siRNAs[0]}' wells: ",
    plate4_df.loc[plate4_df["Metadata_siRNA"] == siRNAs[0]].shape[0],
)
print(
    f"Number of '{siRNAs[1]}' wells: ",
    plate4_df.loc[plate4_df["Metadata_siRNA"] == siRNAs[1]].shape[0],
)
print(
    f"Number of '{siRNAs[2]}' wells: ",
    plate4_df.loc[plate4_df["Metadata_siRNA"] == siRNAs[2]].shape[0],
)
print(
    f"Number of '{siRNAs[3]}' wells: ",
    plate4_df.loc[plate4_df["Metadata_siRNA"] == siRNAs[3]].shape[0],
)
plate4_df.head()


# ## Applying mAP analysis with Well Level Profile
#
# Parameter docs:
# - **pos_samby**: Dictating comparison within the siRNA Group
# - **pos_diffby**: Dictating differences of entries (in this case wells) comparing different wells within the same wells
# - **neg_sameby**: kept blank
# - **neg_diffby**: Establishing which groups to compare with each other (control vs treatment)
# - **null_size**:
# - **batch_size**: Amount of calculations done per thread

# In[15]:


# setting parameters for mAP
seed = 0
ref_siRNA = "No Construct"
pos_sameby = ["Metadata_siRNA"]
pos_diffby = ["Metadata_Well"]
neg_sameby = []
neg_diffby = ["Metadata_siRNA"]
null_size = plate4_df.loc[plate4_df["Metadata_siRNA"] == ref_siRNA].shape[0] * 100
batch_size = 100

# generate a ref siRNA, this dataframe will be used to be compared across all siRNAs
ref_siRNA_df = plate4_df.loc[plate4_df["Metadata_siRNA"] == ref_siRNA]


# ### Running mAP with original dataset

# In[16]:


# storing all mAP scores
map_results = []

for siRNA in siRNAs:
    # skipping ref to ref comparison
    if siRNA == ref_siRNA:
        continue

    # selecting 1 siRNA treatment
    siRNA_df = plate4_df.loc[plate4_df["Metadata_siRNA"] == siRNA]

    # concat ref with selected siRNA wells
    concat_df = pd.concat([ref_siRNA_df, siRNA_df])

    # execute mAP, comparing the reference siRNA and selected siRNA
    # store into a list
    map_result = map.run_pipeline(
        meta=concat_df[meta_features],
        feats=concat_df[cp_features].values,
        pos_sameby=pos_sameby,
        pos_diffby=pos_diffby,
        neg_sameby=neg_sameby,
        neg_diffby=neg_diffby,
        batch_size=batch_size,
        null_size=null_size,
    )

    # adding shuffled column
    map_result.insert(0, "shuffled", "Not Shuffled")

    # store to list
    map_results.append(map_result)

# convert mAP results to a dataframe
map_results = pd.concat(map_results)
map_results.to_csv("./results/sc_AP_scores.csv", index=False)

map_results


# In[7]:


# aggregate values based on siRNA
agg_map_results = map.aggregate(map_results, sameby="Metadata_siRNA", threshold=0.05)
agg_map_results.to_csv("./results/sc_mAP_scores.csv", index=False)
agg_map_results


# ### Running mAP with shuffled feature space dataset

# In[8]:


# storing all mAP scores
shuffled_map_results = []

for siRNA in siRNAs:
    # skipping ref to ref comparison
    if siRNA == ref_siRNA:
        continue

    # selecting 1 siRNA treatment
    siRNA_df = plate4_df.loc[plate4_df["Metadata_siRNA"] == siRNA]

    # concat ref with selected siRNA wells
    concat_df = pd.concat([ref_siRNA_df, siRNA_df])

    shuffled_concat_vales = shuffle_features(concat_df[cp_features].values, seed=0)

    # execute mAP, comparing the reference siRNA and selected siRNA
    # store into a list
    map_result = map.run_pipeline(
        meta=concat_df[meta_features],
        feats=shuffled_concat_vales,
        pos_sameby=pos_sameby,
        pos_diffby=pos_diffby,
        neg_sameby=neg_sameby,
        neg_diffby=neg_diffby,
        batch_size=batch_size,
        null_size=null_size,
    )

    # adding shuffled column
    map_result.insert(0, "shuffled", "Features Shuffled")

    # store to list
    shuffled_map_results.append(map_result)

# convert mAP results to a dataframe
shuffled_map_results = pd.concat(shuffled_map_results)
shuffled_map_results.to_csv("./results/sc_well_AP_scores.csv", index=False)


# In[9]:


shuffled_agg_map = map.aggregate(
    shuffled_map_results, sameby="Metadata_siRNA", threshold=0.05
)
shuffled_agg_map.to_csv("./results/sc_well_mAP_scores.csv")
