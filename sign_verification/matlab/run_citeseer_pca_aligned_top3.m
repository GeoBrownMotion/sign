function run_citeseer_pca_aligned_top3()
%RUN_CITESEER_PCA_ALIGNED_TOP3  Verify aligned-PCA MLP + SIGN top-3 under both threat models.
%   Replaces legacy-PCA rows in CiteSeer PCA T3 so all rows share the same PCA basis
%   as T2/T4 (PCA fit on the 120 training nodes only).

    here = fileparts(mfilename('fullpath'));
    base = fullfile(here, '..', 'artifacts', 'citeseer_pca_aligned', 'd32');

    configs = {'mlp', 'sign_p3_s4_t1', 'sign_p2_s3_t3', 'sign_p3_s3_t2'};

    scopes = {'target_node', 'all'};
    scope_tags = {'tn', 'all'};

    for si = 1:numel(scopes)
        scope = scopes{si};
        scope_tag = scope_tags{si};
        for i = 1:numel(configs)
            tag = configs{i};
            fprintf('\n=== aligned %s scope=%s ===\n', tag, scope);

            cfg = default_sign_verification_config();
            cfg.model_json = fullfile(base, tag, 'model_export.json');
            cfg.data_json  = fullfile(base, tag, 'data_export.json');
            cfg.run_name   = ['cs_pca_top3_', scope_tag, '_', tag];
            cfg.model_name = tag;
            cfg.dataset    = 'CiteSeer_PCA32_aligned';
            cfg.max_samples = 500;
            cfg.eps_list = [0.001, 0.005, 0.01, 0.05];
            cfg.num_generators = 5000;
            cfg.perturb_scope = scope;

            try
                [~, ri] = verify_node_batch(cfg);
                fprintf('[%s/%s] V=%d F=%d U=%d E=%d (total=%d)\n', ...
                    scope_tag, tag, ...
                    ri.status_counts.VERIFIED, ri.status_counts.FALSIFIED, ...
                    ri.status_counts.UNKNOWN, ri.status_counts.ERROR, ri.num_rows);
            catch ME
                fprintf('[%s/%s] FAILED: %s\n', scope_tag, tag, ME.message);
            end
        end
    end
    fprintf('\nDone.\n');
end
