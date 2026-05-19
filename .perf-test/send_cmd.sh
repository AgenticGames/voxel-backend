#!/bin/bash
# Usage: send_cmd.sh '<json>'
JSON="$1"
CMDPATH="/d/Unreal Projects/Mithril2026/Saved/claude_cmd.json"
echo "$JSON" > "$CMDPATH"
echo "sent: $JSON to $CMDPATH"
