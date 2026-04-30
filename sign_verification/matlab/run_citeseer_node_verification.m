%RUN_CITESEER_NODE_VERIFICATION  Verify CiteSeer MLP models (node-level).

    here = fileparts(mfilename('fullpath'));

    models = {'mlp', 'mlp_fair', 'mlp_fair_k1', 'mlp_fair_k2', ...
              'mlp_fair_k3', 'mlp_fair_k4', 'mlp_kd'};

    for i = 1:numel(models)
        name = models{i};
        fprintf('\n=== CiteSeer %s ===\n', name);

        cfg = default_sign_verification_config();
        cfg.model_json = fullfile(here, '..', '..', 'results', 'citeseer', name, 'model_export.json');
        cfg.data_json  = fullfile(here, '..', '..', 'results', 'citeseer', name, 'data_export.json');
        cfg.run_name = ['citeseer_', name];
        cfg.model_name = ['citeseer_', name];
        cfg.dataset = 'CiteSeer';
        cfg.max_samples = 50;
        cfg.eps_list = [0.001, 0.005, 0.01];
        cfg.num_generators = 5000;

        [rows, run_info] = verify_node_batch(cfg);
        fprintf('[%s] V=%d F=%d U=%d E=%d\n', name, ...
            run_info.status_counts.VERIFIED, run_info.status_counts.FALSIFIED, ...
            run_info.status_counts.UNKNOWN, run_info.status_counts.ERROR);
    end

    fprintf('\n=== All CiteSeer models done ===\n');
