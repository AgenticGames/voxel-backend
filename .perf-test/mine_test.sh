#!/bin/bash
# Launch UE -game, sprint+look+hold_mine, collect all emitted burst profiles.
set -u
TAG="${1:-mine_baseline}"
CMDPATH="/d/Unreal Projects/Mithril2026/Saved/claude_cmd.json"
STATEPATH="/d/Unreal Projects/Mithril2026/Saved/claude_state.json"
PROFDIR="/d/Unreal Projects/Mithril2026/Saved/Profiling"
UE_EXE='D:\Epic Games\UE_5.7\Engine\Binaries\Win64\UnrealEditor.exe'
UE_PROJ='D:\Unreal Projects\Mithril2026\Mithril2026.uproject'

ls "$PROFDIR" | grep streaming_profile | sort > /tmp/prof_before.txt
rm -f "$CMDPATH" "$STATEPATH"

PID=$(powershell.exe -Command "Start-Process -FilePath '$UE_EXE' -ArgumentList '\"$UE_PROJ\"', '-game', '-ResX=1280', '-ResY=720', '-windowed', '-nosteam', '-UseAllAvailableCores' -PassThru | Select-Object -ExpandProperty Id" | tr -d '\r')
echo "Launched UE PID=$PID (tag=$TAG)"

for i in $(seq 1 90); do
  ls "$PROFDIR" | grep streaming_profile | sort > /tmp/prof_now.txt
  NEW=$(comm -13 /tmp/prof_before.txt /tmp/prof_now.txt | grep initial_load | head -1)
  [ -n "$NEW" ] && break
  sleep 1
done
echo "initial_load ready"

sleep 3
echo '{"cmd":"teleport","x":0,"y":0,"z":1500,"yaw":0}' > "$CMDPATH"
sleep 3
echo '{"cmd":"sprint","seconds":2}' > "$CMDPATH"
sleep 4
echo '{"cmd":"look_at","yaw":0,"pitch":-45}' > "$CMDPATH"
sleep 5  # long wait for sprint bursts to drain fully

# Snapshot now — anything after is the mine window.
ls "$PROFDIR" | grep streaming_profile | sort > /tmp/prof_mine_start.txt

# Start an explicit tagged profile session RIGHT before mining begins
echo "{\"cmd\":\"profile\",\"action\":\"start\",\"tag\":\"$TAG\"}" > "$CMDPATH"
sleep 1
echo '{"cmd":"hold_mine","seconds":15,"range":10000}' > "$CMDPATH"
sleep 17
# Stop the profile explicitly to force report emission even if session hasn't auto-ended
echo '{"cmd":"profile","action":"stop"}' > "$CMDPATH"
sleep 3

ls "$PROFDIR" | grep streaming_profile | sort > /tmp/prof_after.txt
NEW_PROFILES=$(comm -13 /tmp/prof_mine_start.txt /tmp/prof_after.txt)
echo "--- mine-window profiles ---"
echo "$NEW_PROFILES"

# Copy them with tag prefix
OUT="/c/Users/Shazbot/voxel-backend/.perf-test/$TAG"
mkdir -p "$OUT"
for f in $NEW_PROFILES; do cp "$PROFDIR/$f" "$OUT/"; done

powershell.exe -Command "Stop-Process -Id $PID -Force" 2>&1 | head -1
echo "DONE tag=$TAG → $OUT"
