function [net_sample, info] = build_cora_sign_network_for_sample(net_info, X_nodes, edge_index)
%BUILD_CORA_SIGN_NETWORK_FOR_SAMPLE  Compose graph-level SIGN projection P
%   into the first linear layer (W1' = W1*P) so the verifier sees a flat MLP
%   over flatten(X_nodes). Supports legacy K-hop and PST models.

    num_nodes = size(X_nodes, 1);
    feature_dim = size(X_nodes, 2);
    json_layers = net_info.json_layers;

    % Decide whether to use PST or legacy path
    sc = net_info.sign_config;
    use_pst = ~isempty(sc.operator_names) && (sc.p + sc.s + sc.t > 0);

    if use_pst
        P = aux_build_pst_projection(num_nodes, feature_dim, edge_index, sc);
    else
        num_hops = net_info.sign_hops;
        operator_family = net_info.sign_operator_family;
        if isempty(operator_family)
            operator_family = 'multi_scale_concat';
        end
        P = aux_build_sign_projection(num_nodes, feature_dim, edge_index, num_hops, operator_family);
    end

    layers = {};
    for k = 1:numel(json_layers)
        L = json_layers(k);
        W = double(L.W);
        b = double(L.b(:));

        if k == 1
            W = W * P;
        end

        layers{end + 1} = nnLinearLayer(W, b); %#ok<AGROW>
        act = char(L.act);
        if strcmp(act, 'linear')
            act = '';
        end
        if ~isempty(act)
            layers{end + 1} = nnActivationLayer.instantiateFromString(act); %#ok<AGROW>
        end
    end

    net_sample = neuralNetwork(layers);
    info = struct;
    info.input_dim = size(P, 2);
    info.sign_dim = size(P, 1);
    if use_pst
        info.num_hops = sc.p + sc.s + sc.t;
        info.operator_family = 'pst';
    else
        info.num_hops = net_info.sign_hops;
        info.operator_family = net_info.sign_operator_family;
    end
end


%% ── PST projection (Power / Spectral / Triangle) ─────────────────────────

function P = aux_build_pst_projection(num_nodes, feature_dim, edge_index, sc)
%AUX_BUILD_PST_PROJECTION  Build the linear projection for SIGN(p,s,t).
%
% Each operator branch produces a mean-pooled row vector per hop. The
% projection maps flattened node features to the concatenated pooled SIGN
% vector, matching the Python precompute_sign.compute_pooled_sign_vector.

    A_raw = aux_make_undirected_adj(num_nodes, edge_index);
    n = num_nodes;
    I = speye(n);
    mean_row = ones(1, n) / n;

    blocks = {};

    % x0: raw features, mean-pooled
    blocks{end + 1} = kron(speye(feature_dim), mean_row);

    % Power operators: GCN-normalized (D^{-1/2} A D^{-1/2} with self-loops)
    if sc.p > 0
        A_hat = aux_gcn_norm(A_raw + I);
        cur = A_hat;
        for hop = 1:sc.p
            blocks{end + 1} = kron(speye(feature_dim), mean_row * cur); %#ok<AGROW>
            cur = cur * A_hat;
        end
    end

    % Spectral operators: PPR diffusion
    if sc.s > 0
        ppr = aux_ppr_operator(A_raw + I, sc.ppr_alpha);
        cur = ppr;
        for hop = 1:sc.s
            blocks{end + 1} = kron(speye(feature_dim), mean_row * cur); %#ok<AGROW>
            cur = cur * ppr;
        end
    end

    % Triangle operators: triangle-induced diffusion
    if sc.t > 0
        tri = aux_triangle_operator(A_raw, sc.triangle_keep_self_loops);
        cur = tri;
        for hop = 1:sc.t
            blocks{end + 1} = kron(speye(feature_dim), mean_row * cur); %#ok<AGROW>
            cur = cur * tri;
        end
    end

    P = full(vertcat(blocks{:}));
end


function PPR = aux_ppr_operator(A_with_self_loops, alpha)
%AUX_PPR_OPERATOR  Exact PPR: alpha * (I - (1-alpha) * D^{-1} A)^{-1}
    n = size(A_with_self_loops, 1);
    T = aux_row_norm(A_with_self_loops);
    I = speye(n);
    system = I - (1 - alpha) * T;
    PPR = alpha * (system \ I);
end


function T = aux_triangle_operator(A_raw, keep_self_loops)
%AUX_TRIANGLE_OPERATOR  Triangle-induced adjacency with row normalization.
%   Edge weight (i,j) proportional to common neighbors, restricted to edges.
    B = A_raw;
    B = spones(B);
    if ~keep_self_loops
        B = B - spdiags(diag(B), 0, size(B,1), size(B,2));
    end
    common_neighbors = B * B;
    tri = B .* common_neighbors;
    T = aux_row_norm(tri);
end


function N = aux_row_norm(A)
%AUX_ROW_NORM  D^{-1} A row normalization.
    d = full(sum(A, 2));
    d_inv = zeros(size(d));
    mask = d > 0;
    d_inv(mask) = 1 ./ d(mask);
    N = spdiags(d_inv, 0, size(A, 1), size(A, 2)) * A;
end


%% ── Legacy K-hop projection ──────────────────────────────────────────────

function P = aux_build_sign_projection(num_nodes, feature_dim, edge_index, num_hops, operator_family)
    A = aux_make_undirected_adj(num_nodes, edge_index);
    A = A + speye(num_nodes);
    A_hat = aux_gcn_norm(A);

    mean_row = ones(1, num_nodes) / num_nodes;
    M = speye(num_nodes);
    blocks = cell(num_hops + 1, 1);

    for hop = 0:num_hops
        pooled_row = mean_row * M;
        blocks{hop + 1} = kron(speye(feature_dim), pooled_row);
        M = A_hat * M;
    end

    switch operator_family
        case 'multi_scale_concat'
            P = full(vertcat(blocks{:}));
        case 'final_hop_only'
            P = full(blocks{end});
        otherwise
            error('sign:CoraUnsupportedModel', ...
                'Unsupported SIGN operator family ''%s''.', operator_family);
    end
end


%% ── Shared helpers ───────────────────────────────────────────────────────

function A = aux_make_undirected_adj(num_nodes, edge_index)
    edge_index = double(edge_index);
    if isvector(edge_index)
        edge_index = reshape(edge_index, 2, []);
    end

    src = edge_index(1, :) + 1;
    dst = edge_index(2, :) + 1;

    src_full = [src, dst];
    dst_full = [dst, src];

    mask = src_full >= 1 & src_full <= num_nodes & ...
        dst_full >= 1 & dst_full <= num_nodes;
    src_full = src_full(mask);
    dst_full = dst_full(mask);

    pairs = unique([src_full(:), dst_full(:)], 'rows');
    values = ones(size(pairs, 1), 1);
    A = sparse(pairs(:, 1), pairs(:, 2), values, num_nodes, num_nodes);
end


function A_hat = aux_gcn_norm(A)
    d = full(sum(A, 2));
    d_inv_sqrt = zeros(size(d));
    mask = d > 0;
    d_inv_sqrt(mask) = 1 ./ sqrt(d(mask));
    D_inv_sqrt = spdiags(d_inv_sqrt, 0, size(A, 1), size(A, 2));
    A_hat = D_inv_sqrt * A * D_inv_sqrt;
end
