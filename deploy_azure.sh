#!/usr/bin/env bash
# One-command Azure deploy of the Outlet Profiler to Azure Container Apps.
# Builds the Dockerfile in the cloud (ACR) and exposes a public HTTPS URL.
#
# Prereqs:
#   - Azure CLI:            brew install azure-cli
#   - Logged in:            az login   (and an active Azure *subscription*)
#   - Run from repo root:   ANTHROPIC_API_KEY=sk-ant-... ./deploy_azure.sh
#
# The Anthropic key is read from the environment — it is never written to the repo.
set -euo pipefail

RG="${AZ_RESOURCE_GROUP:-fa-ai}"
LOC="${AZ_LOCATION:-centralindia}"
ENVN="${AZ_ENV:-fa-ai-env}"
APP="${AZ_APP:-outlet-profiler}"
TRINO="trino.fieldassist.io,trino-slmg.fieldassist.io,trino-haldiram.fieldassist.io,trino-gulf.fieldassist.io,trino-colpal.fieldassist.io"

echo "▸ enabling Container Apps providers…"
az extension add --name containerapp --upgrade --yes >/dev/null
az provider register --namespace Microsoft.App >/dev/null
az provider register --namespace Microsoft.OperationalInsights >/dev/null

echo "▸ building the Dockerfile in the cloud and deploying (torch build ~10-15 min)…"
az containerapp up -n "$APP" -g "$RG" -l "$LOC" --environment "$ENVN" \
  --source . --ingress external --target-port 8100

echo "▸ setting resources (torch needs RAM) + env…"
az containerapp update -n "$APP" -g "$RG" --cpu 2.0 --memory 4.0Gi \
  --set-env-vars \
    PROFILER_REQUIRE_AUTH=0 \
    PROFILER_WORKERS=4 \
    PROFILER_DATA_DIR=/app/data \
    PROFILER_DB=/app/data/agent.db \
    PROFILER_LLM_MODEL=claude-haiku-4-5-20251001 \
    PROFILER_TRINO_HOSTS="$TRINO" \
    ${ANTHROPIC_API_KEY:+ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"} >/dev/null

FQDN="$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
echo ""
echo "✅ LIVE:      https://$FQDN"
echo "   UI:        https://$FQDN/"
echo "   manifest:  https://$FQDN/.well-known/agent.json   (give this to the supervisor)"
echo "   health:    https://$FQDN/health/ready"
echo ""
echo "Note: storage is ephemeral (onboarded companies/runs reset on restart)."
echo "For persistence, mount Azure Files at /app/data — see AZURE notes in GO_LIVE.md."
