function run_enzymes_edge_perturbation()
%RUN_ENZYMES_EDGE_PERTURBATION  ENZYMES GCN edge perturbation (1/3/5 edges,
%   delta=0, argmax check) using CORA's native idx_pert_edges mechanism.

    here = fileparts(mfilename('fullpath'));
    setup_cora(fullfile(here, '..', '..', 'cora'));

    % Load GCN
    model_json = fullfile(here, '..', '..', 'results', 'gcn', 'model_export.json');
    data_json  = fullfile(here, '..', '..', 'results', 'gcn', 'data_export.json');

    nn = neuralNetwork.readGNNetwork(model_json);
    data = load_gnn_data_export(data_json);
    num_samples = min(30, height(data));  % reduced for tractability

    fprintf('ENZYMES GCN edge perturbation: %d samples\n', num_samples);

    num_pert_list = [1, 3, 5];
    results = struct();
    for ei = 1:numel(num_pert_list)
        results(ei).pert = num_pert_list(ei);
        results(ei).V = 0; results(ei).F = 0; results(ei).U = 0; results(ei).E = 0;
        results(ei).n = 0; results(ei).rts = [];
    end

    for i = 1:num_samples
        % Extract sample
        raw = data{i, 'input'}; if iscell(raw); raw = raw{1}; end; X = double(raw);
        raw = data{i, 'edge_index'}; if iscell(raw); raw = raw{1}; end; ei_raw = double(raw);
        raw = data{i, 'target_label'}; if iscell(raw); raw = raw{1}; end; target_label = double(raw);
        num_nodes = size(X, 1);

        % Build graph
        [G, ~] = build_cora_graph_from_sample(num_nodes, ei_raw);
        x_vec = reshape(X, [], 1);

        % Nominal check
        options = struct;
        options.nn.graph = G;
        options.nn.num_generators = 2000;
        options.nn.reuse_bounds = false;
        options.nn.idx_pert_edges = [];
        options.nn.invsqrt_order = 2;

        nn.reset();
        try
            clean = double(nn.evaluate(x_vec, options));
            clean = clean(:);
            [~, ci] = max(clean);
            if (ci - 1) ~= target_label
                continue;  % skip misclassified
            end
        catch
            continue;
        end

        num_classes = numel(clean);
        target_idx = target_label + 1;

        for ei = 1:numel(num_pert_list)
            num_pert = num_pert_list(ei);
            try
                [G_pert, idxPert] = aux_perturb_edges(G, num_pert);
                if numel(idxPert) < num_pert
                    results(ei).E = results(ei).E + 1;
                    results(ei).n = results(ei).n + 1;
                    continue;
                end

                options.nn.graph = G_pert;
                options.nn.idx_pert_edges = idxPert;

                X_set = polyZonotope(x_vec);  % delta=0
                nn.reset();
                tic;
                Y = nn.evaluate(X_set, options);
                rt = toc;

                % Argmax check
                W = eye(num_classes);
                W(:, target_idx) = W(:, target_idx) - 1;
                I = interval(W * Y);
                if all(I.sup <= 0)
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
            if ~isnan(rt)
                results(ei).rts(end+1) = rt;
            end
        end

        fprintf('  sample %d/%d done\n', i, num_samples);
    end

    % Print results
    fprintf('\n=== ENZYMES GCN Edge Perturbation Results ===\n');
    for ei = 1:numel(results)
        r = results(ei);
        rt_med = NaN;
        if ~isempty(r.rts); rt_med = median(r.rts); end
        fprintf('  edges=%d: V=%d/%d (%.1f%%)  U=%d  E=%d  rt_med=%.3fs\n', ...
            r.pert, r.V, r.n, r.V/max(r.n,1)*100, r.U, r.E, rt_med);
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
