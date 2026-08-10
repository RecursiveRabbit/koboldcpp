#!/bin/bash
# Test attention_version endpoint

echo "Testing /api/extra/attention_version endpoint"
echo "=============================================="
echo ""

curl -s http://localhost:5001/api/extra/attention_version | jq .

echo ""
echo "Expected: {\"version\": 2}"
