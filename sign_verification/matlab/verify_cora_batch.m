function [rows, run_info] = verify_cora_batch(cfg)
%VERIFY_CORA_BATCH Run CORA-based verification over a batch of graph samples.

    setup_cora(cfg.cora_root);
    [net, net_info] = load_cora_network_from_json(cfg.model_json);
    data = load_gnn_data_export(cfg.data_json);

    fprintf('[cora] Imported network: %s (%s, task=%s)\n', ...
        cfg.model_json, net_info.model_family, net_info.task_level);
    fprintf('[cora] Data source: %s (%d samples)\n', cfg.data_json, height(data));

    if strcmp(net_info.task_level, 'node')
        rows = aux_make_task_level_unsupported_rows(cfg, data, net_info);
        witnesses = struct('sample_id', {}, 'eps', {}, 'x_counterexample', {});
        aux_write_outputs(cfg, rows, witnesses, net_info);
        run_info = aux_build_run_info(cfg, rows, witnesses, net_info);
        return;
    end

    num_samples = min(cfg.max_samples, height(data));
    row_idx = 1;
    rows = repmat(aux_row_template(cfg, net_info), num_samples * numel(cfg.eps_list), 1);
    witnesses = struct('sample_id', {}, 'eps', {}, 'x_counterexample', {});

    for i = 1:num_samples
        sample = aux_extract_sample(data, i);

        if sample.unsupported_edge_features
            for eps_value = cfg.eps_list
                rows(row_idx) = aux_make_row(cfg, net_info, sample, eps_value, ...
                    'UNSUPPORTED', sample.unsupported_reason, NaN, 0, 0, NaN);
                row_idx = row_idx + 1;
            end
            continue;
        end

        [graph_info, eval_net, options] = aux_prepare_sample_context(cfg, net, net_info, sample);

        eval_net.reset();
        clean_scores = double(eval_net.evaluate(sample.x0_vec, options));
        clean_scores = clean_scores(:);
        [~, clean_idx] = max(clean_scores);
        y_clean_pred = clean_idx - 1;
        clean_correct = (y_clean_pred == sample.y_true);

        if ~isempty(sample.y_logits)
            nominal_match = aux_compare_vectors(clean_scores, sample.y_logits, cfg.nominal_tol);
            fprintf('[cora] sample=%d nominal_match=%d num_nodes=%d num_edges=%d self_loops=%d\n', ...
                sample.sample_id, nominal_match, graph_info.num_nodes, ...
                graph_info.num_edges, graph_info.self_loops_added);
        else
            fprintf('[cora] sample=%d num_nodes=%d num_edges=%d self_loops=%d\n', ...
                sample.sample_id, graph_info.num_nodes, graph_info.num_edges, ...
                graph_info.self_loops_added);
        end

        for eps_value = cfg.eps_list
            options.nn.feature_eps = eps_value;
            [~, X_set, input_info] = build_cora_feature_set(sample.X_nodes, eps_value);
            spec = build_cora_classification_spec(numel(clean_scores), sample.y_true, cfg.spec_margin);

            fprintf('[cora] sample=%d eps=%.6g spec=%s\n', ...
                sample.sample_id, eps_value, spec.type);
            eval_net.reset();
            case_result = run_cora_verification_case( ...
                eval_net, sample.x0_vec, X_set, options, spec, clean_scores, ...
                clean_correct, cfg.falsifier_samples);

            rows(row_idx) = aux_make_row(cfg, net_info, sample, eps_value, ...
                case_result.status, case_result.status_reason, ...
                case_result.runtime_sec, clean_correct, ...
                strcmp(case_result.status, 'VERIFIED') && clean_correct, ...
                case_result.margin_lb);
            rows(row_idx).y_clean_pred = y_clean_pred;
            rows(row_idx).num_nodes = graph_info.num_nodes;
            rows(row_idx).num_edges = graph_info.num_edges;
            rows(row_idx).self_loops_added = graph_info.self_loops_added;
            rows(row_idx).input_set_type = input_info.input_set_type;
            rows(row_idx).graph_type = graph_info.graph_type;
            rows(row_idx).counterexample_available = case_result.counterexample_available;
            if case_result.counterexample_available
                witnesses(end + 1) = struct( ... %#ok<AGROW>
                    'sample_id', sample.sample_id, ...
                    'eps', eps_value, ...
                    'x_counterexample', case_result.counterexample);
            end

            fprintf('[cora] sample=%d eps=%.6g status=%s runtime=%.4f\n', ...
                sample.sample_id, eps_value, case_result.status, case_result.runtime_sec);
            row_idx = row_idx + 1;
        end
    end

    rows = rows(1:row_idx - 1);
    aux_write_outputs(cfg, rows, witnesses, net_info);
    run_info = aux_build_run_info(cfg, rows, witnesses, net_info);
end


function [graph_info, eval_net, options] = aux_prepare_sample_context(cfg, net, net_info, sample)
    [~, graph_info] = build_cora_graph_from_sample(sample.num_nodes, sample.edge_index);

    options = struct;
    options.nn = struct;
    options.nn.num_generators = cfg.num_generators;
    % Reusing activation bounds across different graphs/samples can create
    % inconsistent intervals in this CORA version.
    options.nn.reuse_bounds = false;

    if strcmp(net_info.model_family, 'sign') || strcmp(net_info.input_mode, 'sign_pooled_vector')
        [eval_net, ~] = build_cora_sign_network_for_sample(net_info, sample.X_nodes, sample.edge_index);
        return;
    end

    [G, ~] = build_cora_graph_from_sample(sample.num_nodes, sample.edge_index);
    options.nn.graph = G;
    options.nn.idx_pert_edges = [];
    options.nn.invsqrt_order = 2;
    eval_net = net;
end


function rows = aux_make_task_level_unsupported_rows(cfg, data, net_info)
    num_samples = min(cfg.max_samples, height(data));
    row_idx = 1;
    rows = repmat(aux_row_template(cfg, net_info), num_samples * numel(cfg.eps_list), 1);
    for i = 1:num_samples
        X_nodes = double(aux_get_value(data, i, 'input'));
        sample = struct;
        sample.sample_id = i - 1;
        sample.num_nodes = size(X_nodes, 1);
        sample.y_true = NaN;
        sample.y_ref = NaN;
        for eps_value = cfg.eps_list
            rows(row_idx) = aux_make_row(cfg, net_info, sample, eps_value, ...
                'UNSUPPORTED', ...
                'Node-classification batch verification is not wired into the default runner; use extract_cora_node_problem.m.', ...
                NaN, 0, 0, NaN);
            row_idx = row_idx + 1;
        end
    end
    rows = rows(1:row_idx - 1);
end


function row = aux_row_template(cfg, net_info)
    sample = struct;
    sample.sample_id = NaN;
    sample.num_nodes = NaN;
    sample.y_true = NaN;
    sample.y_ref = NaN;
    row = aux_make_row(cfg, net_info, sample, NaN, 'ERROR', '', NaN, 0, 0, NaN);
end


function sample = aux_extract_sample(data, idx)
    sample = struct;
    sample.sample_id = idx - 1;
    sample.X_nodes = double(aux_get_value(data, idx, 'input'));
    sample.edge_index = double(aux_get_value(data, idx, 'edge_index'));
    sample.num_nodes = size(sample.X_nodes, 1);
    sample.x0_vec = reshape(sample.X_nodes, [], 1);
    sample.y_true = aux_field_or_fallback(data, idx, 'target_label', 'output_label');
    sample.y_ref = aux_field_or_fallback(data, idx, 'output_label', 'target_label');
    sample.y_logits = aux_optional_logits(data, idx);
    [sample.unsupported_edge_features, sample.unsupported_reason] = ...
        aux_detect_unsupported_edge_features(data, idx);
end


function value = aux_field_or_fallback(data, idx, primary, fallback)
    if ismember(primary, data.Properties.VariableNames)
        raw = aux_get_value(data, idx, primary);
        if ~isempty(raw)
            value = double(raw);
            return;
        end
    end
    if ismember(fallback, data.Properties.VariableNames)
        raw = aux_get_value(data, idx, fallback);
        if ~isempty(raw)
            value = double(raw);
            return;
        end
    end
    error('Missing label fields ''%s'' and ''%s''.', primary, fallback);
end


function raw = aux_get_value(data, idx, field_name)
    raw = data{idx, field_name};
    if iscell(raw)
        raw = raw{1};
    end
end


function logits = aux_optional_logits(data, idx)
    logits = [];
    if ismember('output', data.Properties.VariableNames)
        raw = aux_get_value(data, idx, 'output');
        if ~isempty(raw)
            logits = double(raw(:));
        end
    end
end


function [unsupported, reason] = aux_detect_unsupported_edge_features(data, idx)
    unsupported = false;
    reason = '';
    for field_name = {'edge_attr', 'edge_features'}
        if ismember(field_name{1}, data.Properties.VariableNames)
            raw = aux_get_value(data, idx, field_name{1});
            if ~isempty(raw)
                unsupported = true;
                reason = sprintf('Unsupported edge feature field ''%s'' is present.', field_name{1});
                return;
            end
        end
    end

    if ismember('edge_weight', data.Properties.VariableNames)
        raw = aux_get_value(data, idx, 'edge_weight');
        if ~isempty(raw)
            unsupported = true;
            reason = 'Non-empty edge weights are treated as unsupported edge features.';
        end
    end
end


function same = aux_compare_vectors(a, b, tol)
    a = double(a(:));
    b = double(b(:));
    same = numel(a) == numel(b) && all(abs(a - b) <= tol);
end


function row = aux_make_row(cfg, net_info, sample, eps_value, status, status_reason, runtime_sec, clean_correct, certified_correct, margin_lb)
    row = struct;
    row.backend = 'cora';
    row.run_name = cfg.run_name;
    row.dataset = cfg.dataset;
    row.model_name = cfg.model_name;
    row.model_family = net_info.model_family;
    row.task_level = net_info.task_level;
    row.sample_id = sample.sample_id;
    row.eps = eps_value;
    row.status = status;
    row.status_reason = status_reason;
    row.y_true = sample.y_true;
    row.y_ref = sample.y_ref;
    row.y_clean_pred = NaN;
    row.clean_correct = clean_correct;
    row.certified_correct = certified_correct;
    row.counterexample_available = 0;
    row.runtime_sec = runtime_sec;
    row.margin_lb = margin_lb;
    row.num_nodes = sample.num_nodes;
    row.num_edges = NaN;
    row.self_loops_added = NaN;
    row.spec_type = 'argmax_polytope';
    row.input_set_type = '';
    row.graph_type = '';
end


function aux_write_outputs(cfg, rows, witnesses, net_info)
    if ~exist(cfg.results_dir, 'dir')
        mkdir(cfg.results_dir);
    end

    csv_path = fullfile(cfg.results_dir, [cfg.run_name, '.csv']);
    json_path = fullfile(cfg.results_dir, [cfg.run_name, '.json']);
    witness_path = fullfile(cfg.results_dir, [cfg.run_name, '_witnesses.mat']);

    T = struct2table(rows, 'AsArray', true);
    writetable(T, csv_path);

    payload = aux_build_run_info(cfg, rows, witnesses, net_info);
    fid = fopen(json_path, 'w');
    fwrite(fid, jsonencode(payload, 'PrettyPrint', true));
    fclose(fid);

    save(witness_path, 'witnesses');
    fprintf('[cora] Wrote CSV: %s\n', csv_path);
    fprintf('[cora] Wrote JSON: %s\n', json_path);
    fprintf('[cora] Wrote witnesses: %s\n', witness_path);
end


function info = aux_build_run_info(cfg, rows, witnesses, net_info)
    info = struct;
    info.backend = 'cora';
    info.run_name = cfg.run_name;
    info.dataset = cfg.dataset;
    info.model_name = cfg.model_name;
    info.model_family = net_info.model_family;
    info.task_level = net_info.task_level;
    info.model_json = cfg.model_json;
    info.data_json = cfg.data_json;
    info.max_samples = cfg.max_samples;
    info.eps_list = cfg.eps_list;
    info.spec_margin = cfg.spec_margin;
    info.falsifier_samples = cfg.falsifier_samples;
    info.status_counts = aux_status_counts(rows);
    info.num_rows = numel(rows);
    info.num_witnesses = numel(witnesses);
end


function counts = aux_status_counts(rows)
    counts = struct('VERIFIED', 0, 'FALSIFIED', 0, 'UNKNOWN', 0, 'UNSUPPORTED', 0, 'ERROR', 0);
    for i = 1:numel(rows)
        counts.(rows(i).status) = counts.(rows(i).status) + 1;
    end
end
