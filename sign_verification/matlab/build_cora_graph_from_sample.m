function [G, info] = build_cora_graph_from_sample(num_nodes, edge_index)
%BUILD_CORA_GRAPH_FROM_SAMPLE Construct an undirected MATLAB graph for CORA.

    if isempty(edge_index)
        error('sign:CoraInvalidGraph', 'edge_index is empty.');
    end

    edge_index = double(edge_index);
    if isvector(edge_index)
        edge_index = reshape(edge_index, 2, []);
    end
    if size(edge_index, 1) ~= 2
        error('sign:CoraInvalidGraph', 'edge_index must have shape [2, m].');
    end

    src = edge_index(1, :) + 1;
    dst = edge_index(2, :) + 1;
    valid = src >= 1 & src <= num_nodes & dst >= 1 & dst <= num_nodes;
    src = src(valid);
    dst = dst(valid);

    G = graph(src, dst);
    missing_nodes = max(0, num_nodes - height(G.Nodes));
    if missing_nodes > 0
        G = addnode(G, missing_nodes);
    end

    edges_before = G.numedges;
    G = addedge(G, 1:num_nodes, 1:num_nodes);
    G = simplify(G, 'keepselfloops');
    edges_after = G.numedges;

    info = struct;
    info.num_nodes = G.numnodes;
    info.num_edges = G.numedges;
    info.self_loops_added = max(0, edges_after - edges_before);
    info.graph_type = 'undirected+self_loops';
end
