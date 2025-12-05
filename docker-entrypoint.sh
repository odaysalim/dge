#!/bin/bash
# Disable CrewAI tracing before starting the API
# This prevents the interactive prompt that blocks API responses

echo "Disabling CrewAI tracing..."
crewai traces disable 2>/dev/null || echo "crewai traces command not available, using env vars"

# Start the API
exec uvicorn api:app --host 0.0.0.0 --port 8000
