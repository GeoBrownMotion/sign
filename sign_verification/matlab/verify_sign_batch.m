function [rows, run_info] = verify_sign_batch(varargin)
%VERIFY_SIGN_BATCH  Backend dispatch wrapper (currently routes to verify_cora_batch).

    cfg = aux_parse_cfg(varargin{:});
    switch lower(cfg.backend)
        case 'cora'
            [rows, run_info] = verify_cora_batch(cfg);
        otherwise
            error('Unsupported verification backend: %s', cfg.backend);
    end
end


function cfg = aux_parse_cfg(varargin)
    if nargin == 1 && isstruct(varargin{1})
        cfg = varargin{1};
    else
        if nargin < 3
            error(['verify_sign_batch expects either a config struct or ', ...
                '(model_json_path, data_json_path, out_csv_path, ..., max_samples).']);
        end
        [out_dir, run_name, ~] = fileparts(varargin{3});
        if isempty(out_dir)
            out_dir = '.';
        end
        if isempty(run_name)
            run_name = 'verification_run';
        end
        cfg = default_sign_verification_config();
        cfg.model_json = varargin{1};
        cfg.data_json = varargin{2};
        cfg.results_dir = out_dir;
        cfg.run_name = run_name;
        if nargin >= 5 && ~isempty(varargin{5})
            cfg.max_samples = varargin{5};
        end
    end

    defaults = default_sign_verification_config();
    fields = fieldnames(defaults);
    for i = 1:numel(fields)
        if ~isfield(cfg, fields{i}) || isempty(cfg.(fields{i}))
            cfg.(fields{i}) = defaults.(fields{i});
        end
    end
end
