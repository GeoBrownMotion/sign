function [rows, run_info] = verify_node_batch(cfg)
%VERIFY_NODE_BATCH  Node-level CORA verification for MLP / SIGN / GCN.
%   GCN uses reduceGNNForNode on a k-hop subgraph; MLP/SIGN run as flat MLPs
%   (SIGN with the per-node projection composed into the first layer).
%   cfg extras: .task_level ('node'|'multilabel'), .threshold, .perturb_scope
%   ('all'|'target_node'), .node_indices.

    setup_cora(cfg.cora_root);

    % Read metadata first to decide which loader to use
    raw_meta = fileread(cfg.model_json);
    meta_check = jsondecode(raw_meta);
    has_gcn = false;
    for k = 1:numel(meta_check.layers)
        if strcmp(meta_check.layers(k).type, 'gcn')
            has_gcn = true;
            break;
        end
    end

    if has_gcn
        % GCN: use CORA's reader (nnGCNLayer + nnGNNLinearLayer)
        nn = neuralNetwork.readGNNetwork(cfg.model_json);
        net_info = struct('sign_config', struct('p',0,'s',0,'t',0));
    else
        % MLP/SIGN: use our reader (nnLinearLayer, works with interval input)
        [nn, net_info] = load_cora_network_from_json(cfg.model_json);
    end
    num_mp_steps = nn.getNumMessagePassingSteps();

    % Read metadata to determine model family and task
    raw = fileread(cfg.model_json);
    meta_obj = jsondecode(raw);
    if isfield(meta_obj, 'metadata')
        model_family = meta_obj.metadata.model_family;
        task_level = meta_obj.metadata.task_level;
    else
        if num_mp_steps > 0
            model_family = 'gcn';
        else
            model_family = 'mlp';
        end
        task_level = 'node';
    end

    fprintf('[node] Loaded network: family=%s, task=%s, MP_steps=%d\n', ...
        model_family, task_level, num_mp_steps);

    % Load data
    data = load_gnn_data_export(cfg.data_json);
    fprintf('[node] Data: %d samples\n', height(data));

    % For node-level: data has one row per graph (transductive = 1 graph)
    % Extract the single graph's features, edges, labels
    sample = aux_extract_node_data(data, 1);
    num_nodes_total = sample.num_nodes;
    feat_dim = sample.feat_dim;
    num_outputs = size(sample.Y_ref, 2);

    fprintf('[node] Graph: %d nodes, %d features, %d outputs\n', ...
        num_nodes_total, feat_dim, num_outputs);

    % Build graph
    [G, graph_info] = build_cora_graph_from_sample(num_nodes_total, sample.edge_index);
    fprintf('[node] Graph edges: %d (with self-loops)\n', graph_info.num_edges);

    % Determine which nodes to verify
    if isfield(cfg, 'node_indices') && ~isempty(cfg.node_indices)
        node_list = cfg.node_indices;  % 0-based
    else
        % Find correctly classified nodes
        correct_nodes = find(sample.correct_mask) - 1;  % 0-based
        max_nodes = min(cfg.max_samples, numel(correct_nodes));
        node_list = correct_nodes(1:max_nodes);
    end
    num_nodes = numel(node_list);
    fprintf('[node] Verifying %d nodes\n\n', num_nodes);

    % Determine spec type
    is_multilabel = strcmp(task_level, 'multilabel') || ...
                    (isfield(cfg, 'task_level') && strcmp(cfg.task_level, 'multilabel'));
    threshold = 0.0;
    if isfield(cfg, 'threshold')
        threshold = cfg.threshold;
    end

    % Preallocate results
    row_idx = 1;
    rows = repmat(aux_node_row_template(cfg, model_family, task_level), ...
        num_nodes * numel(cfg.eps_list), 1);

    options = struct;
    options.nn = struct;
    options.nn.num_generators = cfg.num_generators;
    options.nn.reuse_bounds = false;

    for ni = 1:num_nodes
        n0 = node_list(ni);          % 0-based
        n0_matlab = n0 + 1;         % 1-based for MATLAB

        % Get node features and reference output
        x_node = sample.X_nodes(n0_matlab, :);
        y_ref = sample.Y_ref(n0_matlab, :)';  % column vector

        if is_multilabel
            target_labels = double(sample.target_labels(n0_matlab, :));
        else
            target_label = sample.target_labels(n0_matlab);
        end

        % Prepare network for this node
        try
        is_sign = strcmp(model_family, 'sign') || ...
                  (isfield(net_info, 'sign_config') && ...
                   (net_info.sign_config.p + net_info.sign_config.s + net_info.sign_config.t) > 0);

        if is_sign && num_mp_steps == 0
            % SIGN node-level: compose per-node projection into first layer.
            % Perturb ALL raw node features (not precomputed SIGN features).
            % The projection P_i maps flatten(X_all_nodes) -> sign_features(node_i).
            % We only perturb node i's raw features (first feat_dim dims after reorder).
            [eval_net, ~] = build_node_sign_network( ...
                net_info, sample.X_nodes, sample.edge_index, n0_matlab);
            x_vec = reshape(sample.X_nodes, [], 1);  % all nodes' raw features
        elseif num_mp_steps == 0
            % Plain MLP: just evaluate on node features
            eval_net = nn;
            x_vec = x_node(:);
        else
            % GCN: extract k-hop subgraph, then reduce network.
            [khop_neighbors, ~] = nearest(G, n0_matlab, num_mp_steps + 1);
            sub_nodes = [n0_matlab; khop_neighbors];

            G_sub = subgraph(G, sub_nodes);
            x_vec = reshape(sample.X_nodes(sub_nodes, :), [], 1);

            eval_net = nn.reduceGNNForNode(1, G_sub);

            options.nn.graph = G_sub;
            options.nn.idx_pert_edges = [];
            options.nn.invsqrt_order = 2;
        end

        % Nominal forward pass
        eval_net.reset();
        clean_logits = double(eval_net.evaluate(x_vec, options));
        clean_logits = clean_logits(:);
        catch ME
            fprintf('[node] n=%d  SKIPPED (subgraph error: %s)\n', n0, ME.message);
            for eps_value = cfg.eps_list
                row = aux_node_row_template(cfg, model_family, task_level);
                row.sample_id = n0;
                row.eps = eps_value;
                row.status = 'ERROR';
                row.status_reason = sprintf('Subgraph: %s', ME.message);
                rows(row_idx) = row;
                row_idx = row_idx + 1;
            end
            continue;
        end

        if is_multilabel
            clean_pred = double(clean_logits > threshold)';
            clean_correct = isequal(clean_pred, target_labels);
        else
            [~, clean_idx] = max(clean_logits);
            clean_pred = clean_idx - 1;
            clean_correct = (clean_pred == target_label);
        end

        fprintf('[node] n=%d  clean_correct=%d  logits=[%s]\n', ...
            n0, clean_correct, num2str(clean_logits(1:min(6,end))', '%+.3f '));

        % Verify at each epsilon
        for eps_value = cfg.eps_list
            options.nn.feature_eps = eps_value;

            % Build perturbation set.
            % DEFAULT: perturb ALL features in x_vec (matches Ladner et al. 2024
            % threat model used in CORA GNN verification). x_vec semantics:
            % - MLP: target node's d features (d gens, = target-node-only)
            % - GCN: all k-hop subgraph features (K*d gens)
            % - SIGN: all N raw node features (N*d gens) — perturbation
            %         propagates through composed SIGN projection P
            % Set cfg.perturb_scope = 'target_node' for the weaker threat model.
            perturb_scope = 'all';
            if isfield(cfg, 'perturb_scope')
                perturb_scope = cfg.perturb_scope;
            end

            if eps_value == 0
                X_set = polyZonotope(x_vec);
            elseif strcmp(perturb_scope, 'target_node') && is_sign && num_mp_steps == 0
                num_n = size(sample.X_nodes, 1);
                generators = zeros(numel(x_vec), feat_dim);
                for jj = 1:feat_dim
                    generators((jj-1)*num_n + n0_matlab, jj) = eps_value;
                end
                X_set = compact(polyZonotope(x_vec, generators));
            elseif strcmp(perturb_scope, 'target_node') && num_mp_steps > 0
                generators = zeros(numel(x_vec), feat_dim);
                generators(1:feat_dim, :) = eps_value * eye(feat_dim);
                X_set = compact(polyZonotope(x_vec, generators));
            else
                % perturb_scope == 'all': perturb every entry of x_vec.
                % For large inputs (>5000), use interval (O(h) memory)
                % because polyZonotope's eye(h) exponent matrix is infeasible.
                h = numel(x_vec);
                if h > 5000
                    X_set = interval(x_vec - eps_value, x_vec + eps_value);
                else
                    generators = eps_value * eye(h);
                    X_set = compact(polyZonotope(x_vec, generators));
                end
            end

            % Build spec
            if is_multilabel
                spec = build_cora_multilabel_spec(num_outputs, target_labels, threshold);
            else
                spec = build_cora_classification_spec(num_outputs, target_label, cfg.spec_margin);
            end

            % Perturbed indices for falsifier sampling — match perturb_scope
            if strcmp(perturb_scope, 'target_node') && is_sign && num_mp_steps == 0
                num_n = size(sample.X_nodes, 1);
                pert_indices = zeros(feat_dim, 1);
                for jj = 1:feat_dim
                    pert_indices(jj) = (jj-1)*num_n + n0_matlab;
                end
            elseif strcmp(perturb_scope, 'target_node') && num_mp_steps > 0
                pert_indices = (1:feat_dim)';
            else
                pert_indices = (1:numel(x_vec))';
            end

            % Run verification
            eval_net.reset();
            case_result = aux_run_node_verification( ...
                eval_net, x_vec, X_set, options, spec, ...
                clean_logits, clean_correct, cfg.falsifier_samples, pert_indices);

            % Store result
            row = aux_node_row_template(cfg, model_family, task_level);
            row.sample_id = n0;
            row.eps = eps_value;
            row.status = case_result.status;
            row.status_reason = case_result.status_reason;
            row.runtime_sec = case_result.runtime_sec;
            row.margin_lb = case_result.margin_lb;
            row.clean_correct = clean_correct;
            row.certified_correct = strcmp(case_result.status, 'VERIFIED') && clean_correct;
            row.num_nodes = numel(x_vec) / feat_dim;
            row.counterexample_available = case_result.counterexample_available;
            if is_multilabel
                row.spec_type = 'multilabel_threshold';
            end
            rows(row_idx) = row;
            row_idx = row_idx + 1;

            fprintf('[node] n=%d eps=%.4f status=%-10s margin=%+.4f rt=%.3fs\n', ...
                n0, eps_value, case_result.status, case_result.margin_lb, case_result.runtime_sec);
        end
    end

    rows = rows(1:row_idx - 1);

    % Write outputs
    if ~exist(cfg.results_dir, 'dir')
        mkdir(cfg.results_dir);
    end
    T = struct2table(rows, 'AsArray', true);
    csv_path = fullfile(cfg.results_dir, [cfg.run_name, '.csv']);
    writetable(T, csv_path);

    run_info = struct;
    run_info.backend = 'cora';
    run_info.run_name = cfg.run_name;
    run_info.num_rows = numel(rows);
    run_info.status_counts = aux_status_counts(rows);
    fprintf('\n[node] Wrote %s (%d rows)\n', csv_path, numel(rows));
    fprintf('[node] V=%d F=%d U=%d E=%d\n', ...
        run_info.status_counts.VERIFIED, run_info.status_counts.FALSIFIED, ...
        run_info.status_counts.UNKNOWN, run_info.status_counts.ERROR);
end


% ═══════════════════════════════════════════════════════════════════════════

function sample = aux_extract_node_data(data, idx)
    sample = struct;

    raw_input = data{idx, 'input'};
    if iscell(raw_input); raw_input = raw_input{1}; end
    sample.X_nodes = double(raw_input);  % [num_nodes, feat_dim]
    sample.num_nodes = size(sample.X_nodes, 1);
    sample.feat_dim = size(sample.X_nodes, 2);

    raw_ei = data{idx, 'edge_index'};
    if iscell(raw_ei); raw_ei = raw_ei{1}; end
    sample.edge_index = double(raw_ei);

    % Output logits: [num_nodes, num_outputs]
    raw_output = data{idx, 'output'};
    if iscell(raw_output); raw_output = raw_output{1}; end
    sample.Y_ref = double(raw_output);

    % Target labels: could be [num_nodes,1] (single-label) or [num_nodes, num_labels] (multi-label)
    raw_target = data{idx, 'target_label'};
    if iscell(raw_target); raw_target = raw_target{1}; end
    sample.target_labels = double(raw_target);

    % Determine correct mask
    if size(sample.target_labels, 2) == 1
        % Single-label: argmax match
        [~, pred_idx] = max(sample.Y_ref, [], 2);
        sample.correct_mask = (pred_idx - 1) == sample.target_labels;
    else
        % Multi-label: all labels match at threshold 0
        pred = double(sample.Y_ref > 0);
        sample.correct_mask = all(pred == sample.target_labels, 2);
    end
end


function result = aux_run_node_verification(net, x0_vec, X_set, options, spec, clean_logits, clean_correct, num_falsifier_samples, pert_indices)
    result = struct;
    result.status = 'ERROR';
    result.status_reason = '';
    result.runtime_sec = NaN;
    result.margin_lb = NaN;
    result.counterexample_available = 0;

    try
        tic;
        Y = net.evaluate(X_set, options);
        result.runtime_sec = toc;

        if isa(Y, 'contSet') && representsa(Y, 'emptySet')
            result.status = 'UNKNOWN';
            result.status_reason = 'CORA returned empty set.';
            return;
        end

        I = interval(spec.A * Y);
        spec_slack = I.sup - spec.b;
        result.margin_lb = -max(spec_slack);

        if all(spec_slack <= 0)
            result.status = 'VERIFIED';
            result.status_reason = 'Spec verified.';
            return;
        end

        % Nominal check
        nominal_slack = spec.A * clean_logits(:) - spec.b;
        if any(nominal_slack > 0)
            result.status = 'FALSIFIED';
            result.status_reason = 'Nominal violates spec.';
            result.margin_lb = -max(nominal_slack);
            result.counterexample_available = 1;
            return;
        end

        % Sampled falsification (perturb only pert_indices)
        dim = numel(x0_vec);
        lb = x0_vec;
        ub = x0_vec;
        lb(pert_indices) = x0_vec(pert_indices) - options.nn.feature_eps;
        ub(pert_indices) = x0_vec(pert_indices) + options.nn.feature_eps;
        for k = 1:num_falsifier_samples
            candidate = lb + rand(dim, 1) .* (ub - lb);
            scores = double(net.evaluate(candidate, options));
            slack = spec.A * scores(:) - spec.b;
            if any(slack > 0)
                result.status = 'FALSIFIED';
                result.status_reason = 'Sampled witness violates spec.';
                result.margin_lb = -max(slack);
                result.counterexample_available = 1;
                return;
            end
        end

        result.status = 'UNKNOWN';
        result.status_reason = 'CORA could not prove spec; no counterexample found.';

    catch ME
        result.status = 'ERROR';
        result.status_reason = sprintf('%s: %s', ME.identifier, ME.message);
        fprintf('[verif_error] %s: %s\n', ME.identifier, ME.message);
    end
end


function row = aux_node_row_template(cfg, model_family, task_level)
    row = struct;
    row.backend = 'cora';
    row.run_name = cfg.run_name;
    row.dataset = cfg.dataset;
    row.model_name = cfg.model_name;
    row.model_family = model_family;
    row.task_level = task_level;
    row.sample_id = NaN;
    row.eps = NaN;
    row.status = 'ERROR';
    row.status_reason = '';
    row.clean_correct = 0;
    row.certified_correct = 0;
    row.counterexample_available = 0;
    row.runtime_sec = NaN;
    row.margin_lb = NaN;
    row.num_nodes = NaN;
    row.spec_type = 'argmax_polytope';
end


function counts = aux_status_counts(rows)
    counts = struct('VERIFIED', 0, 'FALSIFIED', 0, 'UNKNOWN', 0, 'UNSUPPORTED', 0, 'ERROR', 0);
    for i = 1:numel(rows)
        s = rows(i).status;
        if isfield(counts, s)
            counts.(s) = counts.(s) + 1;
        end
    end
end
