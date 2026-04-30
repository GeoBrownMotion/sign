function run_ppi_fair_k123_standard()
%RUN_PPI_FAIR_K123_STANDARD  PPI MLP-fair k=1,2,3 under all-node perturbation
%   (equivalent to target-node for MLP, since input is just 50 target features).

    here = fileparts(mfilename('fullpath'));
    teammate = fullfile(here, '..', '..', 'results', 'ppi');

    configs = {'mlp_fair_k1', 'mlp_fair_k2', 'mlp_fair_k3'};

    for i = 1:numel(configs)
        tag = configs{i};
        fprintf('\n=== PPI standard pert: %s ===\n', tag);

        cfg = default_sign_verification_config();
        cfg.model_json = fullfile(teammate, tag, 'model_export.json');
        cfg.data_json  = fullfile(teammate, tag, 'data_export.json');
        cfg.run_name = ['ppi_std_', tag];
        cfg.model_name = tag;
        cfg.dataset = 'PPI';
        cfg.max_samples = 200;
        cfg.eps_list = [0.001, 0.005, 0.01, 0.05];
        cfg.num_generators = 3000;
        cfg.task_level = 'multilabel';
        cfg.threshold = 0.0;
        % default perturb_scope='all' (equivalent to target-node for MLP)

        [~, ri] = verify_node_batch(cfg);
        fprintf('[%s] V=%d F=%d U=%d E=%d\n', tag, ...
            ri.status_counts.VERIFIED, ri.status_counts.FALSIFIED, ...
            ri.status_counts.UNKNOWN, ri.status_counts.ERROR);
    end
    fprintf('\nDone.\n');
end
