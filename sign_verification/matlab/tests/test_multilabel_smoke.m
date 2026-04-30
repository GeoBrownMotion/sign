function test_multilabel_smoke()
%TEST_MULTILABEL_SMOKE  End-to-end smoke test for multi-label verification
%   on tiny MLP/GCN/SIGN fixtures. Run from sign_verification/matlab/ with
%   tests/ on path. Generate fixtures first with:
%     python ../python/make_multilabel_smoke_fixture.py

    fprintf('\n============================================================\n');
    fprintf('  MULTI-LABEL VERIFICATION SMOKE TEST  (MLP + GCN + SIGN)\n');
    fprintf('============================================================\n');

    here = fileparts(mfilename('fullpath'));
    project_root = fullfile(here, '..', '..');
    cora_root = fullfile(project_root, '..', 'cora');
    setup_cora(cora_root);

    model_types = {'mlp', 'gcn', 'sign'};
    for m = 1:numel(model_types)
        run_one_model(project_root, model_types{m});
    end

    fprintf('\n============================================================\n');
    fprintf('  ALL SMOKE TESTS COMPLETE\n');
    fprintf('============================================================\n');
end


function run_one_model(project_root, model_type)
    fprintf('\n┌──────────────────────────────────────────────────────────┐\n');
    fprintf('│  Model type: %-44s│\n', upper(model_type));
    fprintf('└──────────────────────────────────────────────────────────┘\n');

    fixture_dir = fullfile(project_root, 'artifacts', 'multilabel_smoke', model_type);
    model_json = fullfile(fixture_dir, 'model_export.json');
    data_json  = fullfile(fixture_dir, 'data_export.json');

    assert(isfile(model_json), ...
        sprintf('Fixture not found: %s. Run make_multilabel_smoke_fixture.py first.', model_json));

    % ── load network ─────────────────────────────────────────────────────
    [net, net_info] = load_cora_network_from_json(model_json);
    fprintf('[%s] Loaded: family=%s, task=%s\n', ...
        model_type, net_info.model_family, net_info.task_level);

    % ── load data ────────────────────────────────────────────────────────
    data = load_gnn_data_export(data_json);
    num_samples = height(data);
    fprintf('[%s] %d samples\n\n', model_type, num_samples);

    % ── settings ─────────────────────────────────────────────────────────
    eps_list = [0.0, 0.01, 0.05, 0.10];
    threshold = 0.0;
    num_falsifier_samples = 32;
    num_generators = 5000;

    % ── verify each sample ───────────────────────────────────────────────
    for i = 1:num_samples
        sample = aux_extract_multilabel_sample(data, i);

        fprintf('────────────────────────────────────────\n');
        fprintf('Sample %d  (%d nodes, %d features)\n', ...
            sample.sample_id, sample.num_nodes, sample.feat_dim);
        fprintf('  Target labels: [%s]\n', num2str(sample.target_labels, '%d '));

        % ── prepare network and options for this sample ──────────────────
        [eval_net, options] = aux_prepare_eval_context( ...
            net, net_info, sample, num_generators);

        % ── nominal forward pass ─────────────────────────────────────────
        eval_net.reset();
        clean_logits = double(eval_net.evaluate(sample.x0_vec, options));
        clean_logits = clean_logits(:);
        clean_pred = double(clean_logits > threshold)';
        num_outputs = numel(clean_logits);

        fprintf('  Nominal logits: [%s]\n', num2str(clean_logits', '%+.4f '));
        fprintf('  Nominal pred:   [%s]\n', num2str(clean_pred, '%d '));
        fprintf('  Clean correct:  %d\n', isequal(clean_pred, sample.target_labels));

        % ── build full multi-label spec ──────────────────────────────────
        spec = build_cora_multilabel_spec(num_outputs, sample.target_labels, threshold);
        fprintf('  Spec: %d pos + %d neg constraints\n', spec.num_pos, spec.num_neg);

        % ── verify at each epsilon ───────────────────────────────────────
        for eps_value = eps_list
            fprintf('\n  eps = %.4f\n', eps_value);
            options.nn.feature_eps = eps_value;
            [~, X_set, ~] = build_cora_feature_set(sample.X_nodes, eps_value);

            % Full label-set verification
            eval_net.reset();
            full_result = aux_run_multilabel_verification( ...
                eval_net, sample.x0_vec, X_set, options, spec, ...
                clean_logits, num_falsifier_samples);

            fprintf('    FULL SET:  status=%-10s  margin_lb=%+.4f', ...
                full_result.status, full_result.margin_lb);
            if ~isnan(full_result.runtime_sec)
                fprintf('  runtime=%.4fs', full_result.runtime_sec);
            end
            fprintf('\n');

            % Per-label verification
            for k = 1:num_outputs
                single_spec = aux_build_single_label_spec( ...
                    num_outputs, k, sample.target_labels(k), threshold);

                eval_net.reset();
                per_result = aux_run_multilabel_verification( ...
                    eval_net, sample.x0_vec, X_set, options, single_spec, ...
                    clean_logits, num_falsifier_samples);

                if sample.target_labels(k) == 1
                    sense = 'pos';
                else
                    sense = 'neg';
                end
                fprintf('    label %d (%s): status=%-10s  margin_lb=%+.4f\n', ...
                    k-1, sense, per_result.status, per_result.margin_lb);
            end
        end
        fprintf('\n');
    end
end


% ═══════════════════════════════════════════════════════════════════════════
%  HELPER FUNCTIONS
% ═══════════════════════════════════════════════════════════════════════════

function sample = aux_extract_multilabel_sample(data, idx)
%AUX_EXTRACT_MULTILABEL_SAMPLE Extract a multi-label sample from the table.
    sample = struct;
    sample.sample_id = idx - 1;

    raw_input = data{idx, 'input'};
    if iscell(raw_input); raw_input = raw_input{1}; end
    X = double(raw_input);
    if isvector(X)
        % MLP: 1-D input treated as single-node with feat_dim features
        X = X(:)';  % (1, feat_dim)
    end
    sample.X_nodes = X;
    sample.num_nodes = size(X, 1);
    sample.feat_dim = size(X, 2);
    sample.x0_vec = reshape(X, [], 1);

    raw_labels = data{idx, 'target_labels'};
    if iscell(raw_labels); raw_labels = raw_labels{1}; end
    sample.target_labels = double(raw_labels(:))';

    raw_ei = data{idx, 'edge_index'};
    if iscell(raw_ei); raw_ei = raw_ei{1}; end
    sample.edge_index = double(raw_ei);
end


function [eval_net, options] = aux_prepare_eval_context(net, net_info, sample, num_generators)
%AUX_PREPARE_EVAL_CONTEXT Set up network and options for evaluation.
%
% For GCN:  pass the graph object in options.nn.graph.
% For SIGN: compose the SIGN projection into the first layer.
% For MLP:  use the network directly.

    options = struct;
    options.nn = struct;
    options.nn.num_generators = num_generators;
    options.nn.reuse_bounds = false;
    options.nn.feature_eps = 0;

    if strcmp(net_info.model_family, 'sign') || ...
            strcmp(net_info.input_mode, 'sign_pooled_vector')
        % SIGN: compose linear projection into first layer
        [eval_net, ~] = build_cora_sign_network_for_sample( ...
            net_info, sample.X_nodes, sample.edge_index);
        return;
    end

    if strcmp(net_info.model_family, 'gcn')
        % GCN: build graph and pass in options
        [G, ~] = build_cora_graph_from_sample(sample.num_nodes, sample.edge_index);
        options.nn.graph = G;
        options.nn.idx_pert_edges = [];
        options.nn.invsqrt_order = 2;
        eval_net = net;
        return;
    end

    % MLP: use directly
    eval_net = net;
end


function result = aux_run_multilabel_verification(net, x0_vec, X_set, options, spec, clean_logits, num_falsifier_samples)
%AUX_RUN_MULTILABEL_VERIFICATION Run CORA verification with a multi-label spec.

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

        % Compute constraint satisfaction: spec.A * Y <= spec.b
        I = interval(spec.A * Y);
        spec_slack = I.sup - spec.b;
        result.margin_lb = -max(spec_slack);

        if all(spec_slack <= 0)
            result.status = 'VERIFIED';
            result.status_reason = 'All threshold constraints verified.';
            return;
        end

        % Check if nominal input already violates
        nominal_slack = spec.A * clean_logits(:) - spec.b;
        if any(nominal_slack > 0)
            result.status = 'FALSIFIED';
            result.status_reason = 'Nominal input violates threshold spec.';
            result.margin_lb = -max(nominal_slack);
            result.counterexample_available = 1;
            return;
        end

        % Sampled falsification
        lb = x0_vec - options.nn.feature_eps;
        ub = x0_vec + options.nn.feature_eps;
        dim = numel(x0_vec);
        for k = 1:num_falsifier_samples
            alpha = rand(dim, 1);
            candidate = lb + alpha .* (ub - lb);
            scores = double(net.evaluate(candidate, options));
            slack = spec.A * scores(:) - spec.b;
            if any(slack > 0)
                result.status = 'FALSIFIED';
                result.status_reason = 'Sampled witness violates threshold spec.';
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
    end
end


function spec = aux_build_single_label_spec(num_outputs, label_idx, label_value, threshold)
%AUX_BUILD_SINGLE_LABEL_SPEC Build a spec for verifying a single output label.

    row = zeros(1, num_outputs);
    if label_value == 1
        row(label_idx) = -1;
        bound = -threshold;
        sense = 'pos';
    else
        row(label_idx) = 1;
        bound = threshold;
        sense = 'neg';
    end

    spec = struct;
    spec.type = 'multilabel_threshold';
    spec.A = row;
    spec.b = bound;
    spec.polytope = polytope(row, bound);
    spec.target_labels = zeros(1, num_outputs);
    spec.target_labels(label_idx) = label_value;
    spec.threshold = threshold;
    spec.num_pos = double(label_value == 1);
    spec.num_neg = double(label_value == 0);
    spec.constraint_labels = label_idx;
    spec.constraint_sense = {sense};
end
