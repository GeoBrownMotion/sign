function [rows, run_info] = verify_node_edge_pert(cfg)
%VERIFY_NODE_EDGE_PERT  Edge perturbation verification for GCN node models
%   (delta=0, MST preserved). cfg.edge_frac_list, cfg.max_samples.

    setup_cora(cfg.cora_root);

    % Load network (GCN only)
    nn = neuralNetwork.readGNNetwork(cfg.model_json);
    num_mp_steps = nn.getNumMessagePassingSteps();

    if num_mp_steps == 0
        error('Edge perturbation only applies to GCN models (need message passing layers).');
    end

    % Read metadata
    raw = fileread(cfg.model_json);
    meta_obj = jsondecode(raw);
    model_family = 'gcn';
    task_level = 'node';

    fprintf('[edge_pert] Loaded GCN: MP_steps=%d\n', num_mp_steps);

    % Load data
    data = load_gnn_data_export(cfg.data_json);
    sample = aux_extract_node_data(data, 1);
    num_nodes_total = sample.num_nodes;
    feat_dim = sample.feat_dim;
    num_outputs = size(sample.Y_ref, 2);

    fprintf('[edge_pert] Graph: %d nodes, %d features, %d outputs\n', ...
        num_nodes_total, feat_dim, num_outputs);

    % Build full graph
    [G, graph_info] = build_cora_graph_from_sample(num_nodes_total, sample.edge_index);
    fprintf('[edge_pert] Graph edges: %d (with self-loops)\n', graph_info.num_edges);

    % Determine which nodes to verify
    correct_nodes = find(sample.correct_mask) - 1;  % 0-based
    max_nodes = min(cfg.max_samples, numel(correct_nodes));
    node_list = correct_nodes(1:max_nodes);
    num_nodes = numel(node_list);
    fprintf('[edge_pert] Verifying %d nodes\n\n', num_nodes);

    % Determine spec type
    is_multilabel = isfield(cfg, 'task_level') && strcmp(cfg.task_level, 'multilabel');
    threshold = 0.0;
    if isfield(cfg, 'threshold')
        threshold = cfg.threshold;
    end

    edge_frac_list = cfg.edge_frac_list;

    % Preallocate
    row_idx = 1;
    rows = repmat(aux_edge_row_template(cfg, model_family, task_level), ...
        num_nodes * numel(edge_frac_list), 1);

    options = struct;
    options.nn = struct;
    options.nn.num_generators = cfg.num_generators;
    options.nn.reuse_bounds = false;

    for ni = 1:num_nodes
        n0 = node_list(ni);
        n0_matlab = n0 + 1;

        % Extract k-hop subgraph
        try
            [khop_neighbors, ~] = nearest(G, n0_matlab, num_mp_steps + 1);
            sub_nodes = [n0_matlab; khop_neighbors];
            G_sub = subgraph(G, sub_nodes);
            x_vec = reshape(sample.X_nodes(sub_nodes, :), [], 1);

            eval_net = nn.reduceGNNForNode(1, G_sub);
            options.nn.graph = G_sub;
            options.nn.invsqrt_order = 2;

            % Nominal forward pass (no edge perturbation)
            options.nn.idx_pert_edges = [];
            eval_net.reset();
            clean_logits = double(eval_net.evaluate(x_vec, options));
            clean_logits = clean_logits(:);
        catch ME
            fprintf('[edge_pert] n=%d  SKIPPED (subgraph error: %s)\n', n0, ME.message);
            for fi = 1:numel(edge_frac_list)
                row = aux_edge_row_template(cfg, model_family, task_level);
                row.sample_id = n0;
                row.edge_frac = edge_frac_list(fi);
                row.status = 'ERROR';
                row.status_reason = sprintf('Subgraph: %s', ME.message);
                rows(row_idx) = row;
                row_idx = row_idx + 1;
            end
            continue;
        end

        if is_multilabel
            target_labels = double(sample.target_labels(n0_matlab, :));
            clean_pred = double(clean_logits > threshold)';
            clean_correct = isequal(clean_pred, target_labels);
        else
            target_label = sample.target_labels(n0_matlab);
            [~, clean_idx] = max(clean_logits);
            clean_pred = clean_idx - 1;
            clean_correct = (clean_pred == target_label);
        end

        fprintf('[edge_pert] n=%d  clean_correct=%d  subgraph_nodes=%d  subgraph_edges=%d\n', ...
            n0, clean_correct, G_sub.numnodes, G_sub.numedges);

        % Verify at each edge perturbation fraction
        for edge_frac = edge_frac_list
            % Select edges to perturb (following CORA paper approach)
            [~, idxPertEdges] = aux_perturb_graph(G_sub, edge_frac);

            options.nn.idx_pert_edges = idxPertEdges;
            options.nn.feature_eps = 0;  % no feature perturbation

            % No feature perturbation - use exact point
            X_set = polyZonotope(x_vec);

            % Build spec
            if is_multilabel
                spec = build_cora_multilabel_spec(num_outputs, target_labels, threshold);
            else
                spec = build_cora_classification_spec(num_outputs, target_label, cfg.spec_margin);
            end

            % Run verification
            eval_net.reset();
            result = aux_run_edge_verification(eval_net, x_vec, X_set, options, spec, ...
                clean_logits, clean_correct);

            % Store result
            row = aux_edge_row_template(cfg, model_family, task_level);
            row.sample_id = n0;
            row.edge_frac = edge_frac;
            row.num_pert_edges = numel(idxPertEdges);
            row.status = result.status;
            row.status_reason = result.status_reason;
            row.runtime_sec = result.runtime_sec;
            row.margin_lb = result.margin_lb;
            row.clean_correct = clean_correct;
            row.certified_correct = strcmp(result.status, 'VERIFIED') && clean_correct;
            row.num_nodes = G_sub.numnodes;
            rows(row_idx) = row;
            row_idx = row_idx + 1;

            fprintf('[edge_pert] n=%d frac=%.4f pert_edges=%d status=%-10s rt=%.3fs\n', ...
                n0, edge_frac, numel(idxPertEdges), result.status, result.runtime_sec);
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
    fprintf('\n[edge_pert] Wrote %s (%d rows)\n', csv_path, numel(rows));
    fprintf('[edge_pert] V=%d F=%d U=%d E=%d\n', ...
        run_info.status_counts.VERIFIED, run_info.status_counts.FALSIFIED, ...
        run_info.status_counts.UNKNOWN, run_info.status_counts.ERROR);
end


% ═══════════════════════════════════════════════════════════════════════════

function [numPert, idxPertEdges] = aux_perturb_graph(G, frac)
    % Select edges to perturb, following CORA paper approach:
    % - Keep minimal spanning tree (preserve connectivity)
    % - Don't perturb self-loops
    % - Randomly select from remaining edges

    if frac == 0
        numPert = 0;
        idxPertEdges = [];
        return;
    end

    numPertEdges = max(1, ceil(G.numedges * frac));

    % Keep minimal spanning tree
    [~, n0] = max(degree(G));
    G_msp = G.minspantree('Root', n0, 'Type', 'forest');
    msp_edges = G_msp.Edges.EndNodes;
    G_pert = G.rmedge(msp_edges(:,1), msp_edges(:,2));

    % Don't perturb self-loops
    G_pert = G_pert.rmedge(1:G.numnodes, 1:G.numnodes);

    if G_pert.numedges == 0
        numPert = 0;
        idxPertEdges = [];
        return;
    end

    % Randomly select
    numPert = min(numPertEdges, G_pert.numedges);
    idxPertEdges_pert = randsample(G_pert.numedges, numPert);
    pertEdges = G_pert.Edges.EndNodes(idxPertEdges_pert, :);

    % Map back to original graph indices
    idxPertEdges = G.findedge(pertEdges(:,1), pertEdges(:,2));
end


function sample = aux_extract_node_data(data, idx)
    sample = struct;
    raw_input = data{idx, 'input'};
    if iscell(raw_input); raw_input = raw_input{1}; end
    sample.X_nodes = double(raw_input);
    sample.num_nodes = size(sample.X_nodes, 1);
    sample.feat_dim = size(sample.X_nodes, 2);

    raw_ei = data{idx, 'edge_index'};
    if iscell(raw_ei); raw_ei = raw_ei{1}; end
    sample.edge_index = double(raw_ei);

    raw_output = data{idx, 'output'};
    if iscell(raw_output); raw_output = raw_output{1}; end
    sample.Y_ref = double(raw_output);

    raw_target = data{idx, 'target_label'};
    if iscell(raw_target); raw_target = raw_target{1}; end
    sample.target_labels = double(raw_target);

    if size(sample.target_labels, 2) == 1
        [~, pred_idx] = max(sample.Y_ref, [], 2);
        sample.correct_mask = (pred_idx - 1) == sample.target_labels;
    else
        pred = double(sample.Y_ref > 0);
        sample.correct_mask = all(pred == sample.target_labels, 2);
    end
end


function result = aux_run_edge_verification(net, x0_vec, X_set, options, spec, clean_logits, clean_correct)
    result = struct;
    result.status = 'ERROR';
    result.status_reason = '';
    result.runtime_sec = NaN;
    result.margin_lb = NaN;

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
            result.status_reason = 'Edge perturbation spec verified.';
            return;
        end

        % Nominal check
        nominal_slack = spec.A * clean_logits(:) - spec.b;
        if any(nominal_slack > 0)
            result.status = 'FALSIFIED';
            result.status_reason = 'Nominal violates spec.';
            result.margin_lb = -max(nominal_slack);
            return;
        end

        result.status = 'UNKNOWN';
        result.status_reason = 'CORA could not prove spec under edge perturbation.';

    catch ME
        result.status = 'ERROR';
        result.status_reason = sprintf('%s: %s', ME.identifier, ME.message);
        fprintf('[edge_verif_error] %s: %s\n', ME.identifier, ME.message);
    end
end


function row = aux_edge_row_template(cfg, model_family, task_level)
    row = struct;
    row.backend = 'cora';
    row.run_name = cfg.run_name;
    row.dataset = cfg.dataset;
    row.model_name = cfg.model_name;
    row.model_family = model_family;
    row.task_level = task_level;
    row.sample_id = NaN;
    row.edge_frac = NaN;
    row.num_pert_edges = 0;
    row.status = 'ERROR';
    row.status_reason = '';
    row.clean_correct = 0;
    row.certified_correct = 0;
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
