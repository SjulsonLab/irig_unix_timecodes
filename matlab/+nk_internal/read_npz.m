function s = read_npz(npz_path)
%READ_NPZ  Load a .clocktable.npz file into a struct.
%
%   s = nk_internal.read_npz(npz_path)
%
%   Returns a struct with fields:
%     source, reference, nominal_rate, source_units, metadata,
%     sync_stratum, sync_dispersion_upperbound_ms
%
%   Missing fields are set to [].

    % Unzip to a temp directory
    tmp_dir = tempname();
    mkdir(tmp_dir);
    cleanup = onCleanup(@() rmdir(tmp_dir, 's'));
    unzip(npz_path, tmp_dir);

    % Initialize output struct with defaults
    s = struct( ...
        'source', [], ...
        'reference', [], ...
        'nominal_rate', [], ...
        'source_units', '', ...
        'metadata', [], ...
        'sync_stratum', [], ...
        'sync_dispersion_upperbound_ms', [] ...
    );

    % Map of .npy filenames to struct field names
    field_map = struct( ...
        'source', 'source', ...
        'reference', 'reference', ...
        'nominal_rate', 'nominal_rate', ...
        'source_units', 'source_units', ...
        'x_metadata', '_metadata', ...
        'sync_stratum', 'sync_stratum', ...
        'sync_dispersion_upperbound_ms', 'sync_dispersion_upperbound_ms' ...
    );

    fields = fieldnames(field_map);
    for i = 1:numel(fields)
        field_name = fields{i};
        npy_name = field_map.(field_name);
        npy_path = fullfile(tmp_dir, [npy_name '.npy']);

        if ~isfile(npy_path)
            continue;
        end

        data = nk_internal.read_npy(npy_path);

        % Special handling for metadata: parse JSON string
        if strcmp(field_name, 'x_metadata')
            s.metadata = jsondecode(data);
        else
            s.(field_name) = data;
        end
    end
end
