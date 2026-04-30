%RUN_PPI_FULL  Verify ALL correct nodes for all PPI MLP models.

    here = fileparts(mfilename('fullpath'));
    models = {'mlp', 'mlp_fair_k1', 'mlp_fair_k2', 'mlp_fair_k3', ...
              'mlp_fair_k4', 'mlp_kd'};

    for i = 1:numel(models)
        name = models{i};
        fprintf('\n=== PPI %s (full) ===\n', name);

        cfg = default_sign_verification_config();
        cfg.model_json = fullfile(here, '..', '..', 'results', 'ppi', name, 'model_export.json');
        cfg.data_json  = fullfile(here, '..', '..', 'results', 'ppi', name, 'data_export.json');
        cfg.run_name = ['ppi_full_', name];
        cfg.model_name = ['ppi_', name];
        cfg.dataset = 'PPI';
        cfg.max_samples = 500;  % more than enough for all correct nodes
        cfg.eps_list = [0.001, 0.005, 0.01];
        cfg.num_generators = 5000;
        cfg.task_level = 'multilabel';
        cfg.threshold = 0.0;

        [~, ri] = verify_node_batch(cfg);
        fprintf('[%s] V=%d F=%d U=%d E=%d (total=%d)\n', name, ...
            ri.status_counts.VERIFIED, ri.status_counts.FALSIFIED, ...
            ri.status_counts.UNKNOWN, ri.status_counts.ERROR, ri.num_rows);
    end
    fprintf('\nAll PPI full runs done.\n');
