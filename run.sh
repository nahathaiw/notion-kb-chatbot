#!/usr/bin/env bash
# Launch the Notion KB Chatbot safely.
#
# Why this script exists: several local projects each have an `app.py`, so a
# blanket `pkill -f "streamlit run app.py"` would kill ALL of them. This frees
# only THIS project's port (8502, set in .streamlit/config.toml) and then starts
# the app — never touching other Streamlit apps you have running.

set -e
cd "$(dirname "$0")"

PORT=8502

# Free only the process listening on our port, if any.
OLD=$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)
if [ -n "$OLD" ]; then
  echo "Stopping existing app on port $PORT (pid $OLD)..."
  kill "$OLD" 2>/dev/null || true
  sleep 1
fi

echo "Starting Notion KB Chatbot on http://localhost:$PORT"
exec streamlit run app.py
