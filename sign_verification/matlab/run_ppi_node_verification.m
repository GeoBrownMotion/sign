%RUN_PPI_NODE_VERIFICATION  Verify PPI MLP models (node-level, multi-label).

    here = fileparts(mfilename('fullpath'));
    models = {'mlp', 'mlp_kd'};

    for i = 1:numel(models)
        name = models{i};
        fprintf('\n=== PPI %s ===\n', name);

        cfg = default_sign_verification_config();
        cfg.model_json = fullfile(here, '..', '..', 'results', 'ppi', name, 'model_export.json');
        cfg.data_json  = fullfile(here, '..', '..', 'results', 'ppi', name, 'data_export.json');
        cfg.run_name = ['ppi_', name];
        cfg.model_name = ['ppi_', name];
        cfg.dataset = 'PPI';
        cfg.max_samples = 20;
        cfg.eps_list = [0.001, 0.005, 0.01];
        cfg.num_generators = 5000;
        cfg.task_level = 'multilabel';
        cfg.threshold = 0.0;

        [~, ri] = verify_node_batch(cfg);
        fprintf('[%s] V=%d F=%d U=%d E=%d\n', name, ...
            ri.status_counts.VERIFIED, ri.status_counts.FALSIFIED, ...
            ri.status_counts.UNKNOWN, ri.status_counts.ERROR);
    end
    fprintf('\nDone.\n');
