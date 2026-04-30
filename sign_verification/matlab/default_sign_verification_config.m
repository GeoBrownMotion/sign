function cfg = default_sign_verification_config()
%DEFAULT_SIGN_VERIFICATION_CONFIG Default CORA verification configuration.

    here = fileparts(mfilename('fullpath'));
    repo_root = fullfile(here, '..', '..');

    cfg = struct;
    cfg.backend = 'cora';
    cfg.cora_root = fullfile(repo_root, 'cora');
    cfg.model_json = fullfile(repo_root, 'results', 'mlp', 'model_export.json');
    cfg.data_json = fullfile(repo_root, 'results', 'mlp', 'data_export.json');
    cfg.results_dir = fullfile(here, '..', 'results');
    cfg.run_name = 'teammate_mlp';
    cfg.dataset = 'ENZYMES';
    cfg.model_name = 'teammate_mlp';
    cfg.max_samples = 120;
    cfg.eps_list = [0.01, 0.02, 0.05];
    cfg.spec_margin = 0;
    cfg.falsifier_samples = 64;
    cfg.num_generators = 10000;
    cfg.nominal_tol = 1e-8;
end
