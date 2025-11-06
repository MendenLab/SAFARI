from pyutils import createDataFromFiles
import pandas as pd
from Model import Model
import logging
import hydra
from  omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="evaluation")
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    run_config = getattr(cfg, cfg.run_type)
    logger.info(fr"Run config for: {run_config}")
    mf_path = run_config.d1_nl_path
    eczema_path = run_config.d2_nl_path
    split_path = run_config.data_split
    count_path = run_config.counts
    dataset = createDataFromFiles(eczema_path=eczema_path, mf_path=mf_path, count_matrix_path=count_path,
                                  split_path=split_path, fdr=0.05)
    significant_features = []
    logger.info("Read in feature file")
    file = pd.read_excel(cfg.featureFile, sheet_name=None)
    for key in file.keys():
        significant_features.extend(
            file[key].loc[file[key]["Corrected_p_values"] < 0.01, "Gene_name"].values.tolist())
    significant_features = list(set(significant_features))
    print(f"Significant features: {significant_features}")
    model = Model(dataset, hugo_gene_list=significant_features)
    logger.info(f"Evaluation starts with {len(significant_features)} features")
    print(cfg.scorer)
    print(cfg.basePath)
    model.estimatePerformance(model_name=cfg.model, kernel=cfg.svm_kernel, scorer=cfg.scorer, base_path=cfg.basePath, debug=cfg.debug)

if __name__ == "__main__":
    main()
