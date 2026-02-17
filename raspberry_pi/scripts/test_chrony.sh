#!/bin/bash

# Diagnostic script for checking chrony and gpsd status.
#
# Checks:
#   - chrony service status
#   - chronyc tracking output (stratum, root dispersion, sync status)
#   - chronyc sources output (which sources are reachable)
#   - gpsd status (if installed, i.e., server mode)
#   - PPS device availability
#
# Usage:
#   ./test_chrony.sh

set -uo pipefail

echo "=== NeuroKairos Time Sync Diagnostics ==="
echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# --- chrony service status ---
echo "--- chrony service status ---"
if systemctl is-active --quiet chrony 2>/dev/null; then
    echo "chrony: RUNNING"
else
    echo "chrony: NOT RUNNING"
    systemctl status chrony --no-pager 2>/dev/null || echo "(chrony not installed)"
fi
echo ""

# --- chronyc tracking ---
echo "--- chronyc tracking ---"
if command -v chronyc >/dev/null 2>&1; then
    chronyc tracking 2>/dev/null || echo "(chronyc tracking failed)"
else
    echo "(chronyc not installed)"
fi
echo ""

# --- chronyc sources ---
echo "--- chronyc sources ---"
if command -v chronyc >/dev/null 2>&1; then
    chronyc sources -v 2>/dev/null || echo "(chronyc sources failed)"
else
    echo "(chronyc not installed)"
fi
echo ""

# --- chronyc sourcestats ---
echo "--- chronyc sourcestats ---"
if command -v chronyc >/dev/null 2>&1; then
    chronyc sourcestats 2>/dev/null || echo "(chronyc sourcestats failed)"
fi
echo ""

# --- PPS device ---
echo "--- PPS device ---"
if [ -e /dev/pps0 ]; then
    echo "/dev/pps0: PRESENT"
    if command -v ppstest >/dev/null 2>&1; then
        echo "Running ppstest (3 second timeout)..."
        timeout 3 ppstest /dev/pps0 2>/dev/null || echo "(no PPS pulses detected in 3 seconds)"
    else
        echo "(ppstest not installed — install pps-tools for PPS verification)"
    fi
else
    echo "/dev/pps0: NOT FOUND (GPS PPS may not be configured)"
fi
echo ""

# --- gpsd status (server only) ---
echo "--- gpsd status ---"
if command -v gpsd >/dev/null 2>&1; then
    if systemctl is-active --quiet gpsd 2>/dev/null; then
        echo "gpsd: RUNNING"
    else
        echo "gpsd: NOT RUNNING"
    fi

    if command -v gpspipe >/dev/null 2>&1; then
        echo "GPS data (5 second sample):"
        timeout 5 gpspipe -w 2>/dev/null | head -5 || echo "(no GPS data received)"
    fi
else
    echo "gpsd: NOT INSTALLED (this is expected on client-only machines)"
fi
echo ""

echo "=== Diagnostics complete ==="
