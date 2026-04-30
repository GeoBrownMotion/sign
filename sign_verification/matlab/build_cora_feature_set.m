function [x0_vec, X_set, info] = build_cora_feature_set(X_nodes, eps_value)
%BUILD_CORA_FEATURE_SET  L_inf eps-ball over flatten(X_nodes) as a polyZonotope.

    x0_vec = reshape(double(X_nodes), [], 1);
    lb = x0_vec - eps_value;
    ub = x0_vec + eps_value;

    if eps_value == 0
        X_set = polyZonotope(x0_vec);
    else
        generators = eps_value * eye(numel(x0_vec));
        X_set = compact(polyZonotope(x0_vec, generators));
    end

    info = struct;
    info.input_set_type = 'interval-box/polyZonotope';
    info.lb = lb;
    info.ub = ub;
end
