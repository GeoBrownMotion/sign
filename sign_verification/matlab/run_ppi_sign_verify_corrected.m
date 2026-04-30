function run_ppi_sign_verify_corrected()
%RUN_PPI_SIGN_VERIFY_CORRECTED  Verify PPI SIGN models with projection composition.

    here = fileparts(mfilename('fullpath'));
    base = fullfile(here, '..', 'artifacts', 'ppi_sign_pst');

    configs = {'p1_s0_t0', 'p2_s0_t0', 'p0_s2_t2', 'p4_s2_t2', 'p4_s4_t3'};

    for i = 1:numel(configs)
        tag = configs{i};
        fprintf('\n=== PPI SIGN %s ===\n', tag);

        cfg = default_sign_verification_config();
        cfg.model_json = fullfile(base, tag, 'model_export.json');
        cfg.data_json  = fullfile(base, tag, 'data_export.json');
        cfg.run_name = ['ppi_sign_corrected_', tag];
        cfg.model_name = ['ppi_sign_', tag];
        cfg.dataset = 'PPI';
        cfg.max_samples = 500;
        cfg.eps_list = [0.001, 0.005, 0.01, 0.05];
        cfg.num_generators = 5000;
        cfg.task_level = 'multilabel';
        cfg.threshold = 0.0;

        [~, ri] = verify_node_batch(cfg);
        fprintf('[%s] V=%d F=%d U=%d E=%d (total=%d)\n', tag, ...
            ri.status_counts.VERIFIED, ri.status_counts.FALSIFIED, ...
            ri.status_counts.UNKNOWN, ri.status_counts.ERROR, ri.num_rows);
    end
    fprintf('\nDone.\n');
end
