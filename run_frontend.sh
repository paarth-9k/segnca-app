#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/frontend"
if [ ! -d "node_modules" ]; then
  npm install
fi
if [ ! -f ".env" ]; then
  cp .env.example .env
fi
echo "Starting dev server on http://localhost:5173 ..."
npm run dev -- --host 0.0.0.0
