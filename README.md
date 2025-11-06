# *LCK* and *HOMER1* gene expression-based classifier distinguish early mycosis fungoides from eczema and psoriasis
This is the repository for the paper: *LCK* and *HOMER1* gene expression-based classifier distinguish early mycosis fungoides from eczema and psoriasis from Meinel et al.


### Dependencies
The necessary dependencies can be installed using conda with the given environment.yaml file.
``` conda env create -f environment.yaml ```

# Code
All the scripts are setup with hydra with the corresponding configuration files in the conf directory.

### Feature selection with SAFARI 
The feature selection method SAFARI can be executed by running Feature_selection_run.py and defining the following arguments in the feature_selection.yaml config:
- counts: summarizedExperiment file from R containing:
  - counts: raw count matrix with genes x samples
  - rowData: containing hugo gene name annotation for ensemble gene names
  - colData: specifying sample by ```sampleID``` and disease with ```diag``` 
- data_split.xlsx with the sheets train and test containing the columns ```sampleID``` and ```diag``` specifying which samples are used for the feature selection and which ones are not used for unbiased testing later on. All samples can also be used for feature selection.
- d1_nl_path: specifying .rds file with the differential gene expression results comparing condition1 from non-lesional. The indexes are the genes using ensembl_ids as the count matrix and the ```padj``` values have to present in one column. 
- d2_nl_path: analogous as for d1_nl_path just for the 2nd condition.

The model within SAFARI can be chosen between by setting ```ffs_model```
- logistic
- SVM
- xgboost

