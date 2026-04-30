function run_ppi_sign_top5_standard()
%RUN_PPI_SIGN_TOP5_STANDARD  Verify top-5 PPI SIGN configs (by F1) under all-node perturbation.

    here = fileparts(mfilename('fullpath'));
    sign_base = fullfile(here, '..', 'artifacts', 'ppi_sign_pst');

    % Top 5 by F1 that we don't already have: (4,4,3), (4,2,2), (4,3,4), (3,3,3), (4,4,4)
    % (4,4,3) already done as p4_s4_t3 — keep it. Adding other 4.
    configs = {'p4_s2_t2', 'p4_s3_t4', 'p3_s3_t3', 'p4_s4_t4'};

    for i = 1:numel(configs)
        tag = configs{i};
        fprintf('\n=== PPI standard pert: SIGN %s ===\n', tag);

        cfg = default_sign_verification_config();
        cfg.model_json = fullfile(sign_base, tag, 'model_export.json');
        cfg.data_json  = fullfile(sign_base, tag, 'data_export.json');
        cfg.run_name = ['ppi_std_', tag];
        cfg.model_name = tag;
        cfg.dataset = 'PPI';
        cfg.max_samples = 200;
        cfg.eps_list = [0.001, 0.005, 0.01, 0.05];
        cfg.num_generators = 3000;
        cfg.task_level = 'multilabel';
        cfg.threshold = 0.0;

        [~, ri] = verify_node_batch(cfg);
        fprintf('[%s] V=%d F=%d U=%d E=%d\n', tag, ...
            ri.status_counts.VERIFIED, ri.status_counts.FALSIFIED, ...
            ri.status_counts.UNKNOWN, ri.status_counts.ERROR);
    end
    fprintf('\nDone.\n');
end
