function spec = build_cora_multilabel_spec(num_outputs, target_labels, threshold)
%BUILD_CORA_MULTILABEL_SPEC  Per-output threshold polytope for multi-label tasks:
%   y_i >= threshold for positives, y_j <= threshold for negatives.

    if nargin < 3
        threshold = 0;
    end

    target_labels = target_labels(:)';  % ensure row vector
    assert(numel(target_labels) == num_outputs, ...
        'target_labels length (%d) must equal num_outputs (%d)', ...
        numel(target_labels), num_outputs);

    rows = [];
    bounds = [];
    constraint_labels = [];
    constraint_sense = {};

    for i = 1:num_outputs
        row = zeros(1, num_outputs);
        if target_labels(i) == 1
            % Positive label: output(i) >= threshold  =>  -output(i) <= -threshold
            row(i) = -1;
            rows = [rows; row]; %#ok<AGROW>
            bounds = [bounds; -threshold]; %#ok<AGROW>
            constraint_labels = [constraint_labels; i]; %#ok<AGROW>
            constraint_sense{end+1} = 'pos'; %#ok<AGROW>
        else
            % Negative label: output(i) <= threshold  =>  output(i) <= threshold
            row(i) = 1;
            rows = [rows; row]; %#ok<AGROW>
            bounds = [bounds; threshold]; %#ok<AGROW>
            constraint_labels = [constraint_labels; i]; %#ok<AGROW>
            constraint_sense{end+1} = 'neg'; %#ok<AGROW>
        end
    end

    spec = struct;
    spec.type = 'multilabel_threshold';
    spec.A = rows;
    spec.b = bounds;
    spec.polytope = polytope(rows, bounds);
    spec.target_labels = target_labels;
    spec.threshold = threshold;
    spec.num_pos = sum(target_labels == 1);
    spec.num_neg = sum(target_labels == 0);
    spec.constraint_labels = constraint_labels;
    spec.constraint_sense = constraint_sense;
end
