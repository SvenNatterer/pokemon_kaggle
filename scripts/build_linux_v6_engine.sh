#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

if ! command -v zig >/dev/null 2>&1; then
  echo "Missing Zig cross-compiler. Install it with: brew install zig" >&2
  exit 1
fi

SOURCE_DIR="pokemon-tcg-ai-battle/ptcg_engine/ptcgProgram 22"
OUTPUT="${1:-src/cg/libcg.so}"
CACHE_ROOT="${TMPDIR:-/tmp}/pokemon-zig-cache"
BUILD_LOG="$CACHE_ROOT/build.log"

mkdir -p "$CACHE_ROOT/global" "$CACHE_ROOT/local" "$(dirname "$OUTPUT")"
if ! env \
    ZIG_GLOBAL_CACHE_DIR="$CACHE_ROOT/global" \
    ZIG_LOCAL_CACHE_DIR="$CACHE_ROOT/local" \
    zig c++ \
      -O3 -shared -std=c++20 -fPIC -target x86_64-linux-gnu.2.17 \
      -I"$SOURCE_DIR" \
      "$SOURCE_DIR/Export.cpp" \
      "$SOURCE_DIR/RelationalObservation.cpp" \
      -Wl,-s \
      -o "$OUTPUT" >"$BUILD_LOG" 2>&1; then
  cat "$BUILD_LOG" >&2
  exit 1
fi

file "$OUTPUT"
if ! nm -D "$OUTPUT" 2>/dev/null | grep " GetV6Observation$" >/dev/null; then
  echo "Built library does not export GetV6Observation: $OUTPUT" >&2
  exit 1
fi
echo "Built native V6 Linux engine: $OUTPUT"
