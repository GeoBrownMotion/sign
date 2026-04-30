function [net_node, info] = build_node_sign_network(net_info, X_nodes, edge_index, node_idx)
%BUILD_NODE_SIGN_NETWORK  Compose per-node SIGN projection P_i into the first
%   linear layer (W1' = W1*P_i). P_i: flatten(X) -> sign_features(node_idx),
%   so the verifier perturbs raw node features rather than precomputed SIGN.

    num_nodes = size(X_nodes, 1);
    feature_dim = size(X_nodes, 2);
    json_layers = net_info.json_layers;
    sc = net_info.sign_config;

    % Build per-node SIGN projection: maps flatten(X) -> sign_features(node_idx)
    P = aux_build_node_projection(num_nodes, feature_dim, edge_index, sc, node_idx);
    % P is [sign_dim, num_nodes * feature_dim]

    % Compose into network layers
    layers = {};
    for k = 1:numel(json_layers)
        L = json_layers(k);
        W = double(L.W);
        b = double(L.b(:));

        if k == 1
            W = W * P;  % compose projection into first layer
        end

        layers{end + 1} = nnLinearLayer(W, b); %#ok<AGROW>
        act = char(L.act);
        if strcmp(act, 'linear'); act = ''; end
        if ~isempty(act)
            layers{end + 1} = nnActivationLayer.instantiateFromString(act); %#ok<AGROW>
        end
    end

    net_node = neuralNetwork(layers);
    info = struct;
    info.input_dim = size(P, 2);  % num_nodes * feat_dim
    info.sign_dim = size(P, 1);
    info.node_idx = node_idx;
end


function P = aux_build_node_projection(num_nodes, feature_dim, edge_index, sc, node_idx)
%AUX_BUILD_NODE_PROJECTION  Build projection for a single node's SIGN features.
%
% Maps flatten(X) [num_nodes*feat_dim, 1] -> sign_features_i [sign_dim, 1]
%
% sign_features_i = [X(i,:), (A_hat*X)(i,:), (A_hat^2*X)(i,:), ...]
% Each block extracts row node_idx from the operator applied to X.

    A_raw = aux_make_undirected_adj(num_nodes, edge_index);
    n = num_nodes;
    I_n = speye(n);
    e_i = sparse(node_idx, 1, 1, n, 1);  % unit vector for node_idx

    blocks = {};

    % x0: raw features of node i -> e_i' * X -> kron(I_d, e_i')
    blocks{end + 1} = kron(speye(feature_dim), e_i');

    % Power operators
    if sc.p > 0
        A_hat = aux_gcn_norm(A_raw + I_n);
        cur = A_hat;
        for hop = 1:sc.p
            % (A^k * X)(i,:) = e_i' * A^k * X = kron(I_d, e_i' * A^k)
            blocks{end + 1} = kron(speye(feature_dim), e_i' * cur); %#ok<AGROW>
            cur = cur * A_hat;
        end
    end

    % PPR operators
    if sc.s > 0
        ppr = aux_ppr_operator(A_raw + I_n, sc.ppr_alpha);
        cur = ppr;
        for hop = 1:sc.s
            blocks{end + 1} = kron(speye(feature_dim), e_i' * cur); %#ok<AGROW>
            cur = cur * ppr;
        end
    end

    % Triangle operators
    if sc.t > 0
        tri = aux_triangle_operator(A_raw, sc.triangle_keep_self_loops);
        cur = tri;
        for hop = 1:sc.t
            blocks{end + 1} = kron(speye(feature_dim), e_i' * cur); %#ok<AGROW>
            cur = cur * tri;
        end
    end

    P = full(vertcat(blocks{:}));
end


%% ── Operator helpers (shared with build_cora_sign_network_for_sample) ──

function A = aux_make_undirected_adj(num_nodes, edge_index)
    edge_index = double(edge_index);
    if isvector(edge_index); edge_index = reshape(edge_index, 2, []); end
    src = edge_index(1,:)+1; dst = edge_index(2,:)+1;
    src_full = [src, dst]; dst_full = [dst, src];
    mask = src_full>=1 & src_full<=num_nodes & dst_full>=1 & dst_full<=num_nodes;
    pairs = unique([src_full(mask)', dst_full(mask)'], 'rows');
    A = sparse(pairs(:,1), pairs(:,2), ones(size(pairs,1),1), num_nodes, num_nodes);
end

function A_hat = aux_gcn_norm(A)
    d = full(sum(A,2)); d_inv = zeros(size(d));
    d_inv(d>0) = 1./sqrt(d(d>0));
    D = spdiags(d_inv, 0, size(A,1), size(A,2));
    A_hat = D * A * D;
end

function PPR = aux_ppr_operator(A_sl, alpha)
    n = size(A_sl,1); d = full(sum(A_sl,2));
    d_inv = zeros(size(d)); d_inv(d>0) = 1./d(d>0);
    T = spdiags(d_inv, 0, n, n) * A_sl;
    I_n = speye(n);
    PPR = alpha * ((I_n - (1-alpha)*T) \ I_n);
end

function T = aux_triangle_operator(A_raw, keep_sl)
    B = spones(A_raw);
    if ~keep_sl; B = B - spdiags(diag(B),0,size(B,1),size(B,2)); end
    cn = B * B; tri = B .* cn;
    d = full(sum(tri,2)); d_inv = zeros(size(d)); d_inv(d>0) = 1./d(d>0);
    T = spdiags(d_inv, 0, size(tri,1), size(tri,2)) * tri;
end
