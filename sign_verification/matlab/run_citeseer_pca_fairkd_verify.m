function run_citeseer_pca_fairkd_verify()
%RUN_CITESEER_PCA_FAIRKD_VERIFY  Verify teammate's CiteSeer PCA MLP-fair/KD models.
% Run both target-node and all-node perturbation.

    here = fileparts(mfilename('fullpath'));
    tm = fullfile(here, '..', '..', 'results', 'citeseer_pca');

    configs = {'mlp_fair_k1', 'mlp_fair_k2', 'mlp_fair_k3', 'mlp_fair_k4', 'mlp_kd'};

    for mode = {'target_node', 'all'}
        mode = mode{1};
        for i = 1:numel(configs)
            tag = configs{i};
            fprintf('\n=== CiteSeer PCA %s: %s ===\n', mode, tag);

            cfg = default_sign_verification_config();
            cfg.model_json = fullfile(tm, tag, 'model_export.json');
            cfg.data_json  = fullfile(tm, tag, 'data_export.json');
            cfg.run_name = ['cs_pca_', mode, '_', tag];
            cfg.model_name = tag;
            cfg.dataset = 'CiteSeer_PCA32';
            cfg.max_samples = 500;
            cfg.eps_list = [0.001, 0.005, 0.01, 0.05];
            cfg.num_generators = 5000;
            cfg.perturb_scope = mode;

            [~, ri] = verify_node_batch(cfg);
            fprintf('[%s %s] V=%d F=%d U=%d E=%d\n', mode, tag, ...
                ri.status_counts.VERIFIED, ri.status_counts.FALSIFIED, ...
                ri.status_counts.UNKNOWN, ri.status_counts.ERROR);
        end
    end
    fprintf('\nDone.\n');
end
