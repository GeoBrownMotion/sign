function [net, info] = load_cora_network_from_json(json_path)
%LOAD_CORA_NETWORK_FROM_JSON  Build a CORA neuralNetwork from cora-json-v1
%   (gcn / lin / global_mean_pool / global_add_pool). Edge-feature GNNs rejected.

    raw = fileread(json_path);
    obj = jsondecode(raw);

    if ~isfield(obj, 'layers')
        error('sign:CoraUnsupportedModel', ...
            'Model export must contain a top-level ''layers'' field.');
    end

    metadata = struct;
    if isfield(obj, 'metadata')
        metadata = obj.metadata;
    end
    if aux_get_bool(metadata, 'uses_edge_features', false) || ...
            aux_get_num(metadata, 'edge_feature_dim', 0) > 0
        error('sign:CoraUnsupportedModel', ...
            'Edge-feature GNNs are unsupported by the CORA backend in this repo.');
    end

    json_layers = obj.layers;
    aux_validate_gcn_head_layout(json_layers);

    layers = {};
    input_mode = aux_get_str(metadata, 'input_mode', '');
    pooled = strcmp(input_mode, 'pooled_vector') || strcmp(input_mode, 'sign_pooled_vector');
    has_gcn = false;
    pooling_type = '';

    for k = 1:numel(json_layers)
        L = json_layers(k);
        layer_type = char(L.type);
        act = char(L.act);
        if strcmp(act, 'linear')
            act = '';
        end

        switch layer_type
            case 'gcn'
                if pooled
                    error('sign:CoraUnsupportedModel', ...
                        'GCN layers after global pooling are unsupported.');
                end
                has_gcn = true;
                layers{end+1} = nnGCNLayer(); %#ok<AGROW>
                layers{end+1} = nnGNNLinearLayer(double(L.W), double(L.b(:))); %#ok<AGROW>

            case 'lin'
                if pooled
                    layers{end+1} = nnLinearLayer(double(L.W), double(L.b(:))); %#ok<AGROW>
                else
                    layers{end+1} = nnGNNLinearLayer(double(L.W), double(L.b(:))); %#ok<AGROW>
                end

            case 'global_mean_pool'
                pooled = true;
                pooling_type = 'mean';
                layers{end+1} = nnGNNGlobalPoolingLayer('mean'); %#ok<AGROW>

            case 'global_add_pool'
                pooled = true;
                pooling_type = 'add';
                layers{end+1} = nnGNNGlobalPoolingLayer('add'); %#ok<AGROW>

            otherwise
                error('sign:CoraUnsupportedModel', ...
                    'Unsupported layer type ''%s'' in %s.', layer_type, json_path);
        end

        if ~isempty(act)
            layers{end+1} = nnActivationLayer.instantiateFromString(act); %#ok<AGROW>
        end
    end

    net = neuralNetwork(layers);
    info = struct;
    info.json_path = json_path;
    info.model_family = aux_infer_family(metadata, has_gcn);
    info.task_level = aux_infer_task_level(metadata, pooled);
    info.pooling_type = pooling_type;
    info.input_mode = input_mode;
    info.uses_edge_features = false;
    info.num_layers = numel(json_layers);
    info.json_layers = json_layers;
    info.sign_hops = aux_get_num(metadata, 'sign_hops', 0);
    info.sign_operator_family = aux_get_str(metadata, 'sign_operator_family', '');
    info.feature_dim = aux_get_num(metadata, 'feature_dim', 0);
    info.sign_config = aux_get_sign_config(metadata);
end


function sc = aux_get_sign_config(metadata)
%AUX_GET_SIGN_CONFIG  Extract PST sign_config from metadata, or empty struct.
    sc = struct('p', 0, 's', 0, 't', 0, 'ppr_alpha', 0.05, ...
                'triangle_keep_self_loops', false, 'operator_names', {{}});
    if ~isstruct(metadata) || ~isfield(metadata, 'sign_config')
        return;
    end
    raw = metadata.sign_config;
    if isempty(raw)
        return;
    end
    if isfield(raw, 'p'), sc.p = double(raw.p); end
    if isfield(raw, 's'), sc.s = double(raw.s); end
    if isfield(raw, 't'), sc.t = double(raw.t); end
    if isfield(raw, 'ppr_alpha'), sc.ppr_alpha = double(raw.ppr_alpha); end
    if isfield(raw, 'triangle_keep_self_loops')
        sc.triangle_keep_self_loops = logical(raw.triangle_keep_self_loops);
    end
    if isfield(raw, 'operator_names')
        names = raw.operator_names;
        if ischar(names)
            sc.operator_names = {names};
        elseif iscell(names)
            sc.operator_names = cellfun(@char, names, 'UniformOutput', false);
        else
            sc.operator_names = cellstr(names);
        end
    end
end


function aux_validate_gcn_head_layout(json_layers)
    types = arrayfun(@(layer) string(layer.type), json_layers);
    gcn_idx = find(types == "gcn");
    if isempty(gcn_idx)
        return;
    end

    pool_idx = find(types == "global_mean_pool" | types == "global_add_pool");
    if isempty(pool_idx)
        return;
    end

    last_gcn = gcn_idx(end);
    if numel(pool_idx) == 1 && pool_idx == numel(json_layers)
        tail = json_layers(last_gcn + 1:pool_idx - 1);
        if isempty(tail)
            return;
        end
        tail_types = arrayfun(@(layer) string(layer.type), tail);
        if ~all(tail_types == "lin")
            error('sign:CoraUnsupportedModel', [ ...
                'Ambiguous GCN export: layers after the last GCN and before the final ', ...
                'global pooling must remain linear layers only.']);
        end
    end
end


function value = aux_get_bool(s, field_name, default_value)
    value = default_value;
    if isstruct(s) && isfield(s, field_name)
        value = logical(s.(field_name));
    end
end


function value = aux_get_num(s, field_name, default_value)
    value = default_value;
    if isstruct(s) && isfield(s, field_name)
        value = double(s.(field_name));
    end
end


function value = aux_get_str(s, field_name, default_value)
    value = default_value;
    if isstruct(s) && isfield(s, field_name)
        value = char(s.(field_name));
    end
end


function family = aux_infer_family(metadata, has_gcn)
    if isstruct(metadata) && isfield(metadata, 'model_family')
        family = char(metadata.model_family);
    elseif has_gcn
        family = 'gcn';
    else
        family = 'mlp';
    end
end


function task_level = aux_infer_task_level(metadata, pooled)
    if isstruct(metadata) && isfield(metadata, 'task_level')
        task_level = char(metadata.task_level);
    elseif pooled
        task_level = 'graph';
    else
        task_level = 'node';
    end
end
