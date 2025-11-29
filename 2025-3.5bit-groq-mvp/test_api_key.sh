#!/bin/bash
# Quick API key tester

if [ -z "$GROQ_API_KEY" ]; then
    echo "❌ GROQ_API_KEY not set"
    echo "Run: export GROQ_API_KEY='your_key_here'"
    exit 1
fi

echo "Testing API key: ${GROQ_API_KEY:0:10}...${GROQ_API_KEY: -4}"
echo ""

RESPONSE=$(curl -s https://api.groq.com/openai/v1/models \
    -H "Authorization: Bearer $GROQ_API_KEY")

if echo "$RESPONSE" | grep -q '"data"'; then
    MODEL_COUNT=$(echo "$RESPONSE" | python3 -c "import sys, json; print(len(json.load(sys.stdin)['data']))" 2>/dev/null)
    echo "✅ API Key is VALID!"
    echo "   Available models: $MODEL_COUNT"
    echo ""
    echo "You can now run the full demo:"
    echo "  cd groq && ./compile_and_run.sh"
else
    echo "❌ API Key is INVALID"
    echo ""
    if echo "$RESPONSE" | grep -q "error"; then
        ERROR=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['error']['message'])" 2>/dev/null)
        echo "Error: $ERROR"
    fi
    echo ""
    echo "Get a new key at: https://console.groq.com/keys"
    echo "Then run: export GROQ_API_KEY='your_new_key'"
fi
