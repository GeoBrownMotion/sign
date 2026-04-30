function [net_red, G_sub, x_vec, y_ref, target_label, info] = extract_cora_node_problem(net, X_nodes, edge_index, output_logits, target_labels, node_index)
%EXTRACT_CORA_NODE_PROBLEM  Reduce a node-classification GNN to its k-hop
%   subgraph problem (mirrors CORA's reduceGNNForNode flow).

    [G_full, ~] = build_cora_graph_from_sample(size(X_nodes, 1), edge_index);
    num_mp_steps = net.getNumMessagePassingSteps();
    khop_neighbors = G_full.nearest(node_index, num_mp_steps + 1);
    sub_nodes = unique([node_index; khop_neighbors], 'stable');

    root_pos = find(sub_nodes == node_index, 1);
    if root_pos ~= 1
        sub_nodes([1, root_pos]) = sub_nodes([root_pos, 1]);
    end

    G_sub = subgraph(G_full, sub_nodes);
    x_vec = reshape(double(X_nodes(sub_nodes, :)), [], 1);
    y_ref = double(output_logits(node_index, :))';
    target_label = double(target_labels(node_index));
    net_red = net.reduceGNNForNode(1, G_sub);

    info = struct;
    info.node_index = node_index;
    info.sub_nodes = sub_nodes;
    info.num_message_passing_steps = num_mp_steps;
end
