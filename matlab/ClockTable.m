classdef ClockTable
%CLOCKTABLE  Sparse mapping between source and reference clock domains.
%
%   Loads NeuroKairos .clocktable.npz files and provides bidirectional
%   interpolation matching the Python ClockTable behavior exactly.
%
%   ct = ClockTable.load('recording.clocktable.npz');
%   utc = ct.source_to_reference(sample_indices);
%   samples = ct.reference_to_source(utc_timestamps);

    properties (SetAccess = immutable)
        source              % float64 column vector
        reference           % float64 column vector
        nominal_rate        % scalar double
        source_units        % string or ''
        metadata            % struct or []
        sync_stratum        % float64 column vector or []
        sync_dispersion_upperbound_ms  % float64 column vector or []
    end

    properties (Constant, Access = private)
        EXTRAP_LIMIT_S = 1.5;  % max extrapolation distance (seconds)
    end

    methods
        function obj = ClockTable(s)
        %CLOCKTABLE  Construct from struct (as returned by read_npz).
            obj.source = s.source(:);
            obj.reference = s.reference(:);
            obj.nominal_rate = s.nominal_rate;
            obj.source_units = s.source_units;
            obj.metadata = s.metadata;

            if isfield(s, 'sync_stratum')
                obj.sync_stratum = s.sync_stratum(:);
            else
                obj.sync_stratum = [];
            end
            if isfield(s, 'sync_dispersion_upperbound_ms')
                obj.sync_dispersion_upperbound_ms = ...
                    s.sync_dispersion_upperbound_ms(:);
            else
                obj.sync_dispersion_upperbound_ms = [];
            end
        end

        function result = source_to_reference(obj, values)
        %SOURCE_TO_REFERENCE  Convert source-domain values to reference.
        %
        %   ref = ct.source_to_reference(source_values)
        %
        %   Linearly interpolates within bounds. Extrapolates up to 1.5 s
        %   beyond boundaries. Returns NaN beyond that.
            result = interp1( ...
                obj.source, obj.reference, values(:), 'linear');
            result = obj.extrapolate_below( ...
                result, values(:), obj.source, obj.reference);
            result = obj.extrapolate_above( ...
                result, values(:), obj.source, obj.reference);
            % Preserve input shape
            result = reshape(result, size(values));
        end

        function result = reference_to_source(obj, values)
        %REFERENCE_TO_SOURCE  Convert reference-domain values to source.
        %
        %   src = ct.reference_to_source(reference_values)
        %
        %   Linearly interpolates within bounds. Extrapolates up to 1.5 s
        %   beyond boundaries. Returns NaN beyond that.
            result = interp1( ...
                obj.reference, obj.source, values(:), 'linear');
            result = obj.extrapolate_below( ...
                result, values(:), obj.reference, obj.source);
            result = obj.extrapolate_above( ...
                result, values(:), obj.reference, obj.source);
            % Preserve input shape
            result = reshape(result, size(values));
        end

        function disp(obj)
        %DISP  Formatted display matching Python ClockTable.__repr__.
            n = numel(obj.source);
            if isempty(obj.source_units)
                units_str = '';
            else
                units_str = sprintf(' (%s)', obj.source_units);
            end
            fprintf('ClockTable: %d entries%s, rate=%.1f\n', ...
                n, units_str, obj.nominal_rate);

            fprintf('  source=[%.1f..%.1f], reference=[%.1f..%.1f]\n', ...
                obj.source(1), obj.source(end), ...
                obj.reference(1), obj.reference(end));

            if ~isempty(obj.sync_stratum)
                s_min = min(obj.sync_stratum);
                s_max = max(obj.sync_stratum);
                d_min = min(obj.sync_dispersion_upperbound_ms);
                d_max = max(obj.sync_dispersion_upperbound_ms);
                if s_min == s_max
                    s_str = sprintf('%d', s_min);
                else
                    s_str = sprintf('%d-%d', s_min, s_max);
                end
                if d_min == d_max
                    d_str = sprintf('< %.2g ms', d_min);
                else
                    d_str = sprintf('< %.2g-%.2g ms', d_min, d_max);
                end
                fprintf('  sync: stratum %s, dispersion %s\n', ...
                    s_str, d_str);
            end
        end
    end

    methods (Static)
        function obj = load(path)
        %LOAD  Load a ClockTable from a .clocktable.npz file.
        %
        %   ct = ClockTable.load('recording.clocktable.npz')
            s = nk_internal.read_npz(path);
            obj = ClockTable(s);
        end
    end

    methods (Access = private)
        function result = extrapolate_below(obj, result, v, from, to)
        %EXTRAPOLATE_BELOW  Handle values below the first anchor point.
            mask = isnan(result) & (v < from(1));
            if ~any(mask)
                return;
            end
            slope = (to(2) - to(1)) / (from(2) - from(1));
            extrap = to(1) + (v(mask) - from(1)) * slope;

            % Compute distance in the "to" domain (reference for s2r)
            dist = to(1) - extrap;
            within = dist <= obj.EXTRAP_LIMIT_S;
            extrap(~within) = NaN;
            result(mask) = extrap;
        end

        function result = extrapolate_above(obj, result, v, from, to)
        %EXTRAPOLATE_ABOVE  Handle values above the last anchor point.
            mask = isnan(result) & (v > from(end));
            if ~any(mask)
                return;
            end
            slope = (to(end) - to(end-1)) / (from(end) - from(end-1));
            extrap = to(end) + (v(mask) - from(end)) * slope;

            % Compute distance in the "to" domain
            dist = extrap - to(end);
            within = dist <= obj.EXTRAP_LIMIT_S;
            extrap(~within) = NaN;
            result(mask) = extrap;
        end
    end
end
