function result = run_cora_verification_case(net, x0_vec, X_set, options, spec, clean_scores, clean_correct, num_falsifier_samples)
%RUN_CORA_VERIFICATION_CASE Verify one graph/sample/epsilon combination.

    if nargin < 7
        clean_correct = true;
    end
    if nargin < 8
        num_falsifier_samples = 32;
    end

    result = struct;
    result.status = 'ERROR';
    result.status_reason = 'not_run';
    result.runtime_sec = NaN;
    result.margin_lb = NaN;
    result.counterexample_available = 0;
    result.counterexample = [];
    result.cora_result = [];

    try
        tic;
        Y = net.evaluate(X_set, options);
        result.runtime_sec = toc;
        result.cora_result = Y;

        if aux_is_empty_result(Y)
            result.status = 'UNKNOWN';
            result.status_reason = 'CORA returned an empty/indeterminate reachable set.';
            return;
        end

        I = interval(spec.A * Y);
        spec_slack = I.sup - spec.b;
        result.margin_lb = -max(spec_slack);

        if all(spec_slack <= 0)
            result.status = 'VERIFIED';
            result.status_reason = 'The reachable output set satisfies the argmax specification.';
            return;
        end

        clean_margin = aux_margin_from_scores(clean_scores, spec.target_idx);
        if ~clean_correct || clean_margin <= 0
            result.status = 'FALSIFIED';
            result.status_reason = 'The nominal input already violates the target-label specification.';
            result.margin_lb = clean_margin;
            result.counterexample_available = 1;
            result.counterexample = x0_vec;
            return;
        end

        [found, witness, witness_margin] = aux_find_counterexample( ...
            net, x0_vec, options, spec, num_falsifier_samples);
        if found
            result.status = 'FALSIFIED';
            result.status_reason = 'A sampled witness violates the target-label specification.';
            result.margin_lb = witness_margin;
            result.counterexample_available = 1;
            result.counterexample = witness;
            return;
        end

        result.status = 'UNKNOWN';
        result.status_reason = [ ...
            'CORA did not prove the argmax specification, and sampled falsification ', ...
            'did not find a witness.'];

    catch ME
        result.status = 'ERROR';
        result.status_reason = sprintf('%s: %s', ME.identifier, ME.message);
    end
end


function tf = aux_is_empty_result(Y)
    if isa(Y, 'contSet')
        tf = representsa(Y, 'emptySet');
    else
        tf = isempty(Y);
    end
end


function margin = aux_margin_from_scores(scores, target_idx)
    scores = double(scores(:));
    other_idx = [1:target_idx-1, target_idx+1:numel(scores)];
    margin = scores(target_idx) - max(scores(other_idx));
end


function [found, witness, witness_margin] = aux_find_counterexample(net, x0_vec, options, spec, num_samples)
    found = false;
    witness = [];
    witness_margin = NaN;

    lb = x0_vec - options.nn.feature_eps;
    ub = x0_vec + options.nn.feature_eps;
    dim = numel(x0_vec);

    for k = 1:num_samples
        alpha = rand(dim, 1);
        candidate = lb + alpha .* (ub - lb);
        scores = double(net.evaluate(candidate, options));
        margin = aux_margin_from_scores(scores, spec.target_idx);
        if margin <= 0
            found = true;
            witness = candidate;
            witness_margin = margin;
            return;
        end
    end
end
