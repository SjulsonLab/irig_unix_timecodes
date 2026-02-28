function ct = decode_irig(input_path, varargin)
%DECODE_IRIG  Decode IRIG timecodes via Python and return a ClockTable.
%
%   ct = decode_irig(path)
%   ct = decode_irig(path, 'n_channels', 3, 'irig_channel', 2)
%   ct = decode_irig(path, 'format', 'sglx', 'irig_channel', 'sync')
%
%   Name-value pairs:
%     'format'       — 'auto' (default), 'dat', 'sglx'
%     'n_channels'   — required for .dat files
%     'irig_channel' — channel index or 'sync' (default for sglx)
%     'python'       — path to Python executable (default: 'python3')

    % Parse name-value arguments
    p = inputParser;
    p.addRequired('input_path', @ischar);
    p.addParameter('format', 'auto', @ischar);
    p.addParameter('n_channels', [], @isnumeric);
    p.addParameter('irig_channel', [], @(x) isnumeric(x) || ischar(x));
    p.addParameter('python', 'python3', @ischar);
    p.parse(input_path, varargin{:});
    args = p.Results;

    % Determine format from file extension if 'auto'
    fmt = args.format;
    if strcmp(fmt, 'auto')
        [~, ~, ext] = fileparts(input_path);
        if strcmp(ext, '.bin')
            fmt = 'sglx';
        elseif strcmp(ext, '.dat')
            fmt = 'dat';
        else
            error('decode_irig:unknownFormat', ...
                'Cannot auto-detect format for extension: %s', ext);
        end
    end

    % Build the Python command
    switch fmt
        case 'sglx'
            irig_ch = args.irig_channel;
            if isempty(irig_ch)
                irig_ch = 'sync';
            end
            if isnumeric(irig_ch)
                ch_arg = sprintf('%d', irig_ch);
            else
                ch_arg = sprintf('''%s''', irig_ch);
            end
            py_code = sprintf( ...
                ['from neurokairos import decode_sglx_irig; ' ...
                 'decode_sglx_irig(''%s'', irig_channel=%s, save=True)'], ...
                strrep(input_path, '''', ''''''), ch_arg);

        case 'dat'
            if isempty(args.n_channels)
                error('decode_irig:missingNChannels', ...
                    'n_channels is required for .dat files');
            end
            irig_ch = args.irig_channel;
            if isempty(irig_ch)
                error('decode_irig:missingIrigChannel', ...
                    'irig_channel is required for .dat files');
            end
            py_code = sprintf( ...
                ['from neurokairos import decode_dat_irig; ' ...
                 'decode_dat_irig(''%s'', %d, %d, save=True)'], ...
                strrep(input_path, '''', ''''''), ...
                args.n_channels, irig_ch);

        otherwise
            error('decode_irig:unknownFormat', ...
                'Unknown format: %s', fmt);
    end

    % Run Python
    cmd = sprintf('%s -c "%s"', args.python, py_code);
    [status, output] = system(cmd);
    if status ~= 0
        error('decode_irig:pythonFailed', ...
            'Python decoding failed (exit %d):\n%s', status, output);
    end

    % Load the resulting .clocktable.npz
    % Python saves as <full_filename>.clocktable.npz (appended, not replaced)
    npz_path = [input_path '.clocktable.npz'];
    if ~isfile(npz_path)
        error('decode_irig:noOutput', ...
            'Expected output not found: %s', npz_path);
    end
    ct = ClockTable.load(npz_path);
end
