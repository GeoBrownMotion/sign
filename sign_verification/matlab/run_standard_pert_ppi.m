function run_standard_pert_ppi()
%RUN_STANDARD_PERT_PPI  PPI with standard all-node perturbation.

    here = fileparts(mfilename('fullpath'));

    sign_base = fullfile(here, '..', 'artifacts', 'ppi_sign_pst');
    teammate_base = fullfile(here, '..', '..', 'results', 'ppi');

    models = {
        'mlp',           teammate_base, 'mlp';
        'mlp_fair_k4',   teammate_base, 'mlp';
        'mlp_kd',        teammate_base, 'mlp';
        'p1_s0_t0',      sign_base,     'sign';
        'p0_s2_t2',      sign_base,     'sign';
        'p4_s4_t3',      sign_base,     'sign';
    };

    for i = 1:size(models, 1)
        tag = models{i, 1};
        base = models{i, 2};
        kind = models{i, 3};
        fprintf('\n=== PPI standard pert: %s ===\n', tag);

        cfg = default_sign_verification_config();
        cfg.model_json = fullfile(base, tag, 'model_export.json');
        cfg.data_json  = fullfile(base, tag, 'data_export.json');
        cfg.run_name = ['ppi_std_', tag];
        cfg.model_name = tag;
        cfg.dataset = 'PPI';
        cfg.max_samples = 200;  % PPI is larger, reduce for speed
        cfg.eps_list = [0.001, 0.005, 0.01, 0.05];
        cfg.num_generators = 3000;
        cfg.task_level = 'multilabel';
        cfg.threshold = 0.0;
        % use default perturb_scope='all'

        [~, ri] = verify_node_batch(cfg);
        fprintf('[%s] V=%d F=%d U=%d E=%d (total=%d)\n', tag, ...
            ri.status_counts.VERIFIED, ri.status_counts.FALSIFIED, ...
            ri.status_counts.UNKNOWN, ri.status_counts.ERROR, ri.num_rows);
    end
    fprintf('\nDone.\n');
end
