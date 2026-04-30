function cora_root = setup_cora(cora_root)
%SETUP_CORA Add the local CORA checkout to the MATLAB path.

    if nargin < 1 || isempty(cora_root)
        here = fileparts(mfilename('fullpath'));
        cora_root = fullfile(here, '..', '..', 'cora');
    end

    if ~isfolder(cora_root)
        error('CORA root not found: %s', cora_root);
    end

    addpath(genpath(cora_root));
    fprintf('[cora] Added to path: %s\n', cora_root);
end
