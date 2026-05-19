#!/bin/bash
# Launch UE -game with Unreal Insights tracing enabled.
# User drives the scenario manually (load → sprint → mine), then closes UE.
# Trace file lands at /d/traces/<name>.utrace.
# After UE exits, convert to CSV with convert_trace.sh.
set -u
NAME="${1:-session}"
TRACE="D:/traces/${NAME}.utrace"
mkdir -p /d/traces
rm -f "$TRACE"

UE_EXE='D:\Epic Games\UE_5.7\Engine\Binaries\Win64\UnrealEditor.exe'
UE_PROJ='D:\Unreal Projects\Mithril2026\Mithril2026.uproject'
TRACE_WIN="D:\\traces\\${NAME}.utrace"

echo "Launching UE with trace → $TRACE_WIN"
PID=$(powershell.exe -Command "Start-Process -FilePath '$UE_EXE' -ArgumentList '\"$UE_PROJ\"', '-game', '-ResX=1280', '-ResY=720', '-windowed', '-nosteam', '-UseAllAvailableCores', '-trace=default,cpu,frame,bookmark,file,loadtime', '-statnamedevents', '-tracefile=$TRACE_WIN' -PassThru | Select-Object -ExpandProperty Id" | tr -d '\r')
echo "UE PID=$PID"
echo "Drive the scenario, then close UE. Trace will be at $TRACE"
echo ""
echo "Suggested scenario markers (optional — won't affect the trace):"
echo "  • Wait for initial world load to finish"
echo "  • Sprint forward for ~5 seconds"
echo "  • Hold LMB to mine for ~10 seconds"
echo "  • Close UE"
