function spec = build_cora_classification_spec(num_classes, target_label, margin)
%BUILD_CORA_CLASSIFICATION_SPEC  Argmax polytope: y_target - y_c >= margin
%   for all c != target. target_label is 0-based on input.

    if nargin < 3
        margin = 0;
    end

    target_idx = target_label + 1;
    rows = [];
    bounds = [];
    for j = 1:num_classes
        if j == target_idx
            continue;
        end
        row = zeros(1, num_classes);
        row(j) = 1;
        row(target_idx) = -1;
        rows = [rows; row]; %#ok<AGROW>
        bounds = [bounds; -margin]; %#ok<AGROW>
    end

    spec = struct;
    spec.type = 'argmax_polytope';
    spec.A = rows;
    spec.b = bounds;
    spec.polytope = polytope(rows, bounds);
    spec.target_label = target_label;
    spec.target_idx = target_idx;
    spec.margin = margin;
end
