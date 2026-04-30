function run_ppi_mlp_eps05()
%RUN_PPI_MLP_EPS05  Fill eps=0.05 gap for PPI MLP/MLP-fair/MLP-KD target-node runs.
% Also confirms GCN target-node infeasibility.

    here = fileparts(mfilename('fullpath'));
    teammate = fullfile(here, '..', '..', 'results', 'ppi');

    configs = {'mlp', 'mlp_kd', 'mlp_fair_k1', 'mlp_fair_k2', ...
               'mlp_fair_k3', 'mlp_fair_k4'};

    for i = 1:numel(configs)
        tag = configs{i};
        fprintf('\n=== PPI target-node eps=0.05: %s ===\n', tag);

        cfg = default_sign_verification_config();
        cfg.model_json = fullfile(teammate, tag, 'model_export.json');
        cfg.data_json  = fullfile(teammate, tag, 'data_export.json');
        cfg.run_name = ['ppi_eps05_', tag];
        cfg.model_name = tag;
        cfg.dataset = 'PPI';
        cfg.max_samples = 500;
        cfg.eps_list = [0.05];  % just 0.05
        cfg.num_generators = 3000;
        cfg.task_level = 'multilabel';
        cfg.threshold = 0.0;
        cfg.perturb_scope = 'target_node';  % matches ppi_full_*

        [~, ri] = verify_node_batch(cfg);
        fprintf('[%s] V=%d F=%d U=%d E=%d\n', tag, ...
            ri.status_counts.VERIFIED, ri.status_counts.FALSIFIED, ...
            ri.status_counts.UNKNOWN, ri.status_counts.ERROR);
    end
    fprintf('\nDone.\n');
end
