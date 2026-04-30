function run_ppi_gcn_edge_pert()
%RUN_PPI_GCN_EDGE_PERT  PPI GCN edge perturbation (1/3/5 edges, delta=0,
%   per-label threshold check). Documented as infeasible: 3 MP steps on PPI's
%   dense graph produce subgraphs CORA cannot handle.

    here = fileparts(mfilename('fullpath'));
    setup_cora(fullfile(here, '..', '..', 'cora'));

    model_json = fullfile(here, '..', '..', 'results', 'ppi', 'gcn', 'model_export.json');
    data_json  = fullfile(here, '..', '..', 'results', 'ppi', 'gcn', 'data_export.json');

    nn = neuralNetwork.readGNNetwork(model_json);
    data = load_gnn_data_export(data_json);

    % PPI is single-graph per file: sample 1 has all nodes
    raw = data{1, 'input'}; if iscell(raw); raw = raw{1}; end; X = double(raw);
    raw = data{1, 'edge_index'}; if iscell(raw); raw = raw{1}; end; ei_raw = double(raw);
    raw = data{1, 'target_label'}; if iscell(raw); raw = raw{1}; end; targets = double(raw);  % [N, 121]
    raw = data{1, 'output'}; if iscell(raw); raw = raw{1}; end; Y_ref = double(raw);  % [N, 121]

    num_nodes = size(X, 1);
    num_labels = size(targets, 2);
    fprintf('PPI GCN edge pert: %d nodes, %d labels\n', num_nodes, num_labels);

    [G, ~] = build_cora_graph_from_sample(num_nodes, ei_raw);
    numMP = nn.getNumMessagePassingSteps();

    % Find fully-correct nodes
    preds_ref = double(Y_ref > 0);
    correct_nodes = find(all(preds_ref == targets, 2));
    fprintf('Correct nodes: %d/%d\n', numel(correct_nodes), num_nodes);

    % Test on a subset
    num_test = min(50, numel(correct_nodes));
    test_nodes = correct_nodes(1:num_test);

    num_pert_list = [1, 3, 5];
    results = struct();
    for ei = 1:numel(num_pert_list)
        results(ei).pert = num_pert_list(ei);
        results(ei).V = 0; results(ei).F = 0; results(ei).U = 0; results(ei).E = 0; results(ei).S = 0;
        results(ei).n = 0; results(ei).rts = [];
    end

    threshold = 0.0;

    for ni = 1:numel(test_nodes)
        n0 = test_nodes(ni);
        target_labels = targets(n0, :);

        try
            [khop, ~] = nearest(G, n0, numMP + 1);
            sub_nodes = [n0; khop];
            G_sub = subgraph(G, sub_nodes);
            x_vec = reshape(X(sub_nodes, :), [], 1);

            if G_sub.numnodes > 200  % skip very large subgraphs
                for ei = 1:numel(num_pert_list)
                    results(ei).S = results(ei).S + 1;
                    results(ei).n = results(ei).n + 1;
                end
                fprintf('  n=%d SKIPPED (subgraph too large: %d nodes)\n', n0-1, G_sub.numnodes);
                continue;
            end

            nn_red = nn.reduceGNNForNode(1, G_sub);

            options = struct;
            options.nn.graph = G_sub;
            options.nn.num_generators = 3000;
            options.nn.reuse_bounds = false;
            options.nn.idx_pert_edges = [];
            options.nn.invsqrt_order = 2;

            nn_red.reset();
            clean = double(nn_red.evaluate(x_vec, options));
            clean = clean(:);
            clean_pred = double(clean > threshold)';
            if ~isequal(clean_pred, target_labels)
                continue;
            end
        catch ME
            fprintf('  n=%d FAIL in nominal: %s\n', n0-1, ME.message);
            continue;
        end

        % Build multilabel spec
        spec = build_cora_multilabel_spec(num_labels, target_labels, threshold);

        for ei = 1:numel(num_pert_list)
            num_pert = num_pert_list(ei);
            try
                [G_pert, idxPert] = aux_perturb_edges(G_sub, num_pert);
                if numel(idxPert) < num_pert
                    results(ei).S = results(ei).S + 1;
                    results(ei).n = results(ei).n + 1;
                    continue;
                end

                options.nn.graph = G_pert;
                options.nn.idx_pert_edges = idxPert;

                X_set = polyZonotope(x_vec);  % delta=0
                nn_red.reset();
                tic;
                Y = nn_red.evaluate(X_set, options);
                rt = toc;

                I_spec = interval(spec.A * Y);
                slack = I_spec.sup - spec.b;
                if all(slack <= 0)
                    status = 'V';
                else
                    status = 'U';
                end
            catch
                status = 'E';
                rt = NaN;
            end

            results(ei).(status) = results(ei).(status) + 1;
            results(ei).n = results(ei).n + 1;
            if ~isnan(rt); results(ei).rts(end+1) = rt; end
        end

        fprintf('  n=%d done (subgraph=%d nodes)\n', n0-1, G_sub.numnodes);
    end

    fprintf('\n=== PPI GCN Edge Perturbation ===\n');
    for ei = 1:numel(results)
        r = results(ei);
        rt_med = NaN; if ~isempty(r.rts); rt_med = median(r.rts); end
        fprintf('  edges=%d: V=%d/%d (%.1f%%)  U=%d  E=%d  Skip=%d  rt_med=%.3fs\n', ...
            r.pert, r.V, r.n, r.V/max(r.n,1)*100, r.U, r.E, r.S, rt_med);
    end
    fprintf('Done.\n');
end


function [G_pert, idxPertEdges] = aux_perturb_edges(G, numPertEdges)
    [~, n0] = max(degree(G));
    G_msp = G.minspantree('Root', n0, 'Type', 'forest');
    msp_edges = G_msp.Edges.EndNodes;
    G_removable = G.rmedge(msp_edges(:,1), msp_edges(:,2));
    G_removable = G_removable.rmedge(1:G.numnodes, 1:G.numnodes);

    numAvail = G_removable.numedges;
    numPert = min(numPertEdges, numAvail);
    if numPert == 0
        G_pert = G; idxPertEdges = []; return;
    end

    idxSel = randsample(numAvail, numPert);
    pertEdges = G_removable.Edges.EndNodes(idxSel, :);
    idxPertEdges = G.findedge(pertEdges(:,1), pertEdges(:,2));
    G_pert = G;
end
