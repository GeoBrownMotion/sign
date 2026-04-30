function run_gcn_fair_comparison()
%RUN_GCN_FAIR_COMPARISON  GCN verification under three threat models:
%   target-node L_inf, all-node L_inf, and edge perturbation (delta=0).

    here = fileparts(mfilename('fullpath'));
    setup_cora(fullfile(here, '..', '..', 'cora'));

    nn = neuralNetwork.readGNNetwork( ...
        fullfile(here, '..', '..', 'results', 'citeseer', 'gcn', 'model_export.json'));
    data = load_gnn_data_export( ...
        fullfile(here, '..', '..', 'results', 'citeseer', 'gcn', 'data_export.json'));

    X = double(data{1,'input'}{1});
    ei = double(data{1,'edge_index'}{1});
    n_nodes = size(X, 1);
    feat_dim = size(X, 2);
    numMP = nn.getNumMessagePassingSteps();

    [G, ~] = build_cora_graph_from_sample(n_nodes, ei);
    fprintf('CiteSeer GCN: %d nodes, %d features, %d MP steps\n', n_nodes, feat_dim, numMP);

    % Find correctly classified nodes
    Y_ref = double(data{1,'output'}{1});
    targets = double(data{1,'target_label'}{1});
    [~, pred_idx] = max(Y_ref, [], 2);
    correct_nodes = find((pred_idx - 1) == targets);

    num_test = 50;
    test_nodes = correct_nodes(1:min(num_test, numel(correct_nodes)));
    eps_list = [0.001, 0.005, 0.01];
    num_pert_edges_list = [1, 3, 5];  % for edge perturbation

    fprintf('Testing %d nodes\n\n', numel(test_nodes));

    % ── A: Target-node perturbation (polyZonotope, feat_dim generators) ──
    fprintf('=== A: Target-node feature perturbation ===\n');
    results_A = run_feature_perturbation(nn, G, X, targets, test_nodes, eps_list, ...
        feat_dim, numMP, 'target_only');

    % ── B: All-node perturbation (interval, tractable) ──
    fprintf('\n=== B: All-node feature perturbation (interval) ===\n');
    results_B = run_feature_perturbation(nn, G, X, targets, test_nodes, eps_list, ...
        feat_dim, numMP, 'all_nodes_interval');

    % ── C: Edge perturbation (CORA's approach) ──
    fprintf('\n=== C: Edge perturbation (delta=0) ===\n');
    results_C = run_edge_perturbation(nn, G, X, Y_ref, targets, test_nodes, ...
        num_pert_edges_list, numMP);

    % ── Summary ──
    fprintf('\n\n========== SUMMARY ==========\n\n');

    fprintf('A: Target-node feature perturbation\n');
    print_summary(results_A, eps_list, 'eps');

    fprintf('\nB: All-node feature perturbation (interval)\n');
    print_summary(results_B, eps_list, 'eps');

    fprintf('\nC: Edge perturbation (num_edges removed)\n');
    print_summary(results_C, num_pert_edges_list, 'edges');

    fprintf('\nDone.\n');
end


function results = run_feature_perturbation(nn, G, X, targets, test_nodes, eps_list, feat_dim, numMP, mode)
    results = struct();
    for ei = 1:numel(eps_list)
        eps_val = eps_list(ei);
        results(ei).eps = eps_val;
        results(ei).V = 0; results(ei).F = 0; results(ei).U = 0; results(ei).E = 0;
        results(ei).n = 0; results(ei).rts = [];
    end

    for ni = 1:numel(test_nodes)
        n0 = test_nodes(ni);
        target_label = targets(n0);

        try
            [khop, ~] = nearest(G, n0, numMP + 1);
            sub_nodes = [n0; khop];
            G_sub = subgraph(G, sub_nodes);
            x_vec = reshape(X(sub_nodes, :), [], 1);
            nn_red = nn.reduceGNNForNode(1, G_sub);

            options = struct;
            options.nn.graph = G_sub;
            options.nn.num_generators = 5000;
            options.nn.reuse_bounds = false;
            options.nn.idx_pert_edges = [];
            options.nn.invsqrt_order = 2;

            % Nominal
            nn_red.reset();
            clean = double(nn_red.evaluate(x_vec, options));
            clean = clean(:);
            [~, ci] = max(clean);
            if (ci - 1) ~= target_label
                continue;  % skip misclassified
            end
        catch
            continue;
        end

        num_classes = numel(clean);
        spec = build_cora_classification_spec(num_classes, target_label, 0);

        for ei = 1:numel(eps_list)
            eps_val = eps_list(ei);
            dim = numel(x_vec);

            try
                if strcmp(mode, 'target_only')
                    % Only perturb first feat_dim entries
                    gens = zeros(dim, feat_dim);
                    gens(1:feat_dim, :) = eps_val * eye(feat_dim);
                    X_set = compact(polyZonotope(x_vec, gens));
                elseif strcmp(mode, 'all_nodes_interval')
                    % Perturb all entries using interval (O(n) memory)
                    lb = x_vec - eps_val;
                    ub = x_vec + eps_val;
                    X_set = interval(lb, ub);
                end

                nn_red.reset();
                tic;
                Y = nn_red.evaluate(X_set, options);
                rt = toc;

                I_spec = interval(spec.A * Y);
                slack = I_spec.sup - spec.b;
                margin = -max(slack);

                if all(slack <= 0)
                    status = 'V';
                else
                    nom_slack = spec.A * clean - spec.b;
                    if any(nom_slack > 0)
                        status = 'F';
                    else
                        status = 'U';
                    end
                end
            catch
                status = 'E';
                rt = NaN;
                margin = NaN;
            end

            results(ei).(status) = results(ei).(status) + 1;
            results(ei).n = results(ei).n + 1;
            if ~isnan(rt)
                results(ei).rts(end+1) = rt;
            end

            fprintf('[%s] n=%d eps=%.3f status=%s margin=%+.3f rt=%.3fs\n', ...
                mode, n0-1, eps_val, status, margin, rt);
        end
    end
end


function results = run_edge_perturbation(nn, G, X, Y_ref, targets, test_nodes, num_pert_list, numMP)
    results = struct();
    for ei = 1:numel(num_pert_list)
        results(ei).eps = num_pert_list(ei);
        results(ei).V = 0; results(ei).F = 0; results(ei).U = 0; results(ei).E = 0;
        results(ei).n = 0; results(ei).rts = [];
    end

    for ni = 1:numel(test_nodes)
        n0 = test_nodes(ni);
        target_label = targets(n0) + 1;  % 1-based for CORA's argmax trick

        try
            [khop, ~] = nearest(G, n0, numMP + 1);
            sub_nodes = [n0; khop];
            G_sub = subgraph(G, sub_nodes);
            x_vec = reshape(X(sub_nodes, :), [], 1);
            nn_red = nn.reduceGNNForNode(1, G_sub);

            % Nominal check
            options = struct;
            options.nn.graph = G_sub;
            options.nn.num_generators = 5000;
            options.nn.reuse_bounds = false;
            options.nn.idx_pert_edges = [];
            options.nn.invsqrt_order = 2;

            nn_red.reset();
            clean = double(nn_red.evaluate(x_vec, options));
            clean = clean(:);
            [~, ci] = max(clean);
            if ci ~= target_label
                continue;
            end
        catch
            continue;
        end

        num_classes = numel(clean);

        for ei = 1:numel(num_pert_list)
            num_pert = num_pert_list(ei);

            try
                % Perturb edges (remove random edges, keep MST)
                [G_pert, idxPert] = aux_perturb_edges(G_sub, num_pert);
                if numel(idxPert) < num_pert
                    results(ei).E = results(ei).E + 1;
                    results(ei).n = results(ei).n + 1;
                    continue;
                end

                options.nn.graph = G_pert;
                options.nn.idx_pert_edges = idxPert;

                X_set = polyZonotope(x_vec);  % delta=0, no feature perturbation

                nn_red.reset();
                tic;
                Y = nn_red.evaluate(X_set, options);
                rt = toc;

                % Verify with argmax trick (CORA style)
                W = eye(num_classes);
                W(:, target_label) = W(:, target_label) - 1;
                Y_transformed = W * Y;
                I = interval(Y_transformed);

                if all(I.sup <= 0)
                    status = 'V';
                    margin = -max(I.sup);
                else
                    status = 'U';
                    margin = -max(I.sup);
                end
            catch
                status = 'E';
                rt = NaN;
                margin = NaN;
            end

            results(ei).(status) = results(ei).(status) + 1;
            results(ei).n = results(ei).n + 1;
            if ~isnan(rt)
                results(ei).rts(end+1) = rt;
            end

            fprintf('[edge] n=%d pert=%d status=%s margin=%+.3f rt=%.3fs\n', ...
                n0-1, num_pert, status, margin, rt);
        end
    end
end


function [G_pert, idxPertEdges] = aux_perturb_edges(G, numPertEdges)
    % Keep minimum spanning tree
    [~, n0] = max(degree(G));
    G_msp = G.minspantree('Root', n0, 'Type', 'forest');
    msp_edges = G_msp.Edges.EndNodes;
    G_removable = G.rmedge(msp_edges(:,1), msp_edges(:,2));
    % Don't perturb self-loops
    G_removable = G_removable.rmedge(1:G.numnodes, 1:G.numnodes);

    numAvail = G_removable.numedges;
    numPert = min(numPertEdges, numAvail);
    if numPert == 0
        G_pert = G;
        idxPertEdges = [];
        return;
    end

    idxSel = randsample(numAvail, numPert);
    pertEdges = G_removable.Edges.EndNodes(idxSel, :);
    idxPertEdges = G.findedge(pertEdges(:,1), pertEdges(:,2));
    G_pert = G;
end


function print_summary(results, param_list, param_name)
    for ei = 1:numel(results)
        r = results(ei);
        n = r.n;
        if n == 0; continue; end
        rt_med = NaN;
        if ~isempty(r.rts); rt_med = median(r.rts); end
        fprintf('  %s=%.3f: V=%d/%d (%.1f%%)  F=%d  U=%d  E=%d  rt_med=%.3fs\n', ...
            param_name, param_list(ei), r.V, n, r.V/n*100, r.F, r.U, r.E, rt_med);
    end
end
