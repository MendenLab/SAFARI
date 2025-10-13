import logging
import hydra
from omegaconf import DictConfig
import numpy as np
from tqdm import tqdm
import os
from R_helpers import robustRankAggregation
from pyutils import createDataFromFiles, addSignInformation


@hydra.main(version_base=None, config_path="conf", config_name="feature_selection")
def main(cfg: DictConfig) -> None:
    # All the important paths
    logger = logging.getLogger(__name__)
    run_config = getattr(cfg, cfg.run_type)
    logger.info(fr"Run config for: {run_config}")
    mf_path = run_config.d1_nl_path
    eczema_path = run_config.d2_nl_path
    split_path = run_config.data_split
    count_path = run_config.counts
    bootstrap_iterations = 200
    training_sets = []
    bootstrap_lists = []
    dataset = createDataFromFiles(eczema_path=eczema_path, mf_path=mf_path, count_matrix_path=count_path,
                                  split_path=split_path, fdr=0.05)
    feature_lists = []
    for i in tqdm(range(bootstrap_iterations)):
        bootstrap, ts = dataset.samplePatients(seed=i)
        bootstrap.selectOverallSignificantGenes(fdr=0.05, paired_significant_genes=dataset.significant_genes)
        ts.filterSignificantGenes(bootstrap.significantGenes)
        # Check whether there are genes left after filtering for significant ones
        if ts.normalizedCounts.shape[0] == 0:
            continue
        ts.train(method=cfg.model, ffs_model=cfg.ffs_model)
        bootstrap_lists.append(bootstrap)
        # used only for debugging
        training_sets.append(ts)
        feature_lists.append(ts.trainingResult.features)
    logger.info("Aggregate results")
    ranked_features = robustRankAggregation(feature_lists=feature_lists)
    # add signed information for gene names
    ranked_features = addSignInformation(bootstrapping_list=bootstrap_lists, ranked_features=ranked_features,
                                         rowData=dataset.rowData)
    significant_features = ranked_features.loc[ranked_features["Corrected_p_values"] < 0.05, :]
    # Save results here
    significant_features.to_excel(run_config.output_file, index=False)


if __name__ == '__main__':
    main()
