function data = read_npy(filepath)
%READ_NPY  Parse a single .npy file and return its contents.
%
%   data = nk_internal.read_npy(filepath)
%
%   Supported dtypes:
%     '<f8' (float64) — returns double array
%     '<U*' (unicode)  — returns char/string
%
%   0-d arrays (scalars) return a MATLAB scalar or string.
%   1-d arrays return column vectors.

    fid = fopen(filepath, 'r', 'l');  % little-endian
    if fid == -1
        error('read_npy:fileOpen', 'Cannot open %s', filepath);
    end
    cleanup = onCleanup(@() fclose(fid));

    % -- Magic bytes: \x93NUMPY --
    magic = fread(fid, 6, '*uint8')';
    expected_magic = [147 78 85 77 80 89];  % \x93NUMPY
    if ~isequal(magic, expected_magic)
        error('read_npy:badMagic', 'Not a valid .npy file: %s', filepath);
    end

    % -- Version --
    major = fread(fid, 1, 'uint8');
    minor = fread(fid, 1, 'uint8'); %#ok<NASGU>

    % -- Header length --
    if major == 1
        header_len = fread(fid, 1, 'uint16');
    else
        header_len = fread(fid, 1, 'uint32');
    end

    % -- Parse header dict --
    header_str = fread(fid, header_len, '*char')';
    descr = parse_field(header_str, 'descr');
    shape = parse_shape(header_str);
    fortran_order = contains(header_str, '''fortran_order'': True');

    % -- Read data based on dtype --
    if strcmp(descr, '<f8') || strcmp(descr, '=f8')
        % float64
        n_elements = prod(max(shape, 1));
        data = fread(fid, n_elements, 'float64');
        if isempty(shape) || isequal(shape, 0)
            % 0-d scalar
            data = data(1);
        elseif isscalar(shape)
            % 1-d → column vector
            data = data(:);
        elseif fortran_order
            data = reshape(data, shape);
        else
            data = reshape(data, fliplr(shape))';
        end
    elseif startsWith(descr, '<U') || startsWith(descr, '=U')
        % Unicode string — 4 bytes per character (UTF-32LE)
        n_chars = str2double(descr(3:end));
        raw = fread(fid, n_chars * 4, '*uint8');
        % Decode UTF-32LE: take every 4th byte group as uint32
        codepoints = typecast(raw, 'uint32');
        % Convert to MATLAB char (works for BMP characters)
        data = strtrim(native2unicode(raw', 'UTF-32LE'));
    else
        error('read_npy:unsupportedDtype', ...
            'Unsupported dtype: %s in %s', descr, filepath);
    end
end


function val = parse_field(header, field_name)
%PARSE_FIELD  Extract a string field value from the numpy header dict.
    pattern = ['''', field_name, ''': '''];
    idx = strfind(header, pattern);
    if isempty(idx)
        val = '';
        return;
    end
    start = idx + length(pattern);
    rest = header(start:end);
    end_idx = strfind(rest, '''');
    val = rest(1:end_idx(1)-1);
end


function shape = parse_shape(header)
%PARSE_SHAPE  Extract the shape tuple from the numpy header dict.
%   Returns [] for 0-d (scalar), a scalar for 1-d, or a vector for N-d.
    pattern = '''shape'': (';
    idx = strfind(header, pattern);
    if isempty(idx)
        shape = [];
        return;
    end
    start = idx + length(pattern);
    rest = header(start:end);
    end_idx = strfind(rest, ')');
    shape_str = strtrim(rest(1:end_idx(1)-1));

    % Handle empty shape () → scalar
    if isempty(shape_str) || strcmp(shape_str, '')
        shape = [];
        return;
    end

    % Remove trailing comma for 1-element tuples like (5,)
    shape_str = strrep(shape_str, ',', ' ');
    shape = sscanf(shape_str, '%d')';
end
