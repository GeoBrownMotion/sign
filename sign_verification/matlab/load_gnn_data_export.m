function data = load_gnn_data_export(json_path)
%LOAD_GNN_DATA_EXPORT  Load graph samples from data_export.json into a table.

    raw = fileread(json_path);
    obj = jsondecode(raw);

    if iscell(obj)
        jsondata = obj;
    elseif isstruct(obj) && isfield(obj, 'data')
        jsondata = obj.data;
    else
        error('Unsupported data_export format in %s', json_path);
    end

    if isempty(jsondata)
        data = table();
        return;
    end

    headers = jsondata{1};
    rows = repmat(struct(), max(numel(jsondata) - 1, 0), 1);
    for i = 2:numel(jsondata)
        row = jsondata{i};
        for j = 1:numel(headers)
            rows(i - 1).(headers{j}) = row{j};
        end
    end

    data = struct2table(rows, 'AsArray', true);
end
