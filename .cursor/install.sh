#!/usr/bin/env bash
# Cursor Cloud Agent install script for combine-for-AI (`install` in .cursor/environment.json).
#
# Cursor runs this from the repository root during every Build, on its default
# Ubuntu base image (CPU only: cloud agents have no GPU), then snapshots the disk.
# It must be idempotent. Shell exports don't survive into agent runs, so the tools
# it installs are exposed through /etc/profile.d and /usr/local/bin.
# See https://cursor.com/docs/cloud-agent/setup
#
# Installs only what this repo's CI and manifests need:
#   - apt: curl, ca-certificates
#   - uv + Python from .python-version
#   - uv sync --frozen
#   - tool directories exposed to later shells (/etc/profile.d + /usr/local/bin links)
#
# It ends with a dependency fetch/prebuild, not a test run.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

# Match CI (.github/workflows/ci.yml setup-uv version: "0.6.x").
UV_VERSION="${UV_VERSION:-0.6.17}"

# Install apt packages that are not already present.
apt_install() {
  local missing=() pkg
  for pkg in "$@"; do
    if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
      missing+=("$pkg")
    fi
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    $SUDO apt-get -o Acquire::Retries=5 update -qq
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=5 install -y --no-install-recommends "${missing[@]}"
  fi
}

# --- System packages (curl + CA certs for the installers below) ---
apt_install curl ca-certificates

# --- Python (uv) ---
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  installer="$(mktemp)"
  curl -LsSf -o "$installer" https://astral.sh/uv/install.sh
  UV_VERSION="$UV_VERSION" sh "$installer"
  rm -f "$installer"
  if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv installation failed" >&2
    exit 1
  fi
fi

if [ ! -f .python-version ]; then
  echo "Error: .python-version not found in repository root" >&2
  exit 1
fi
py_version="$(tr -d '[:space:]' < .python-version)"

if ! uv python install "$py_version"; then
  echo "Error: Failed to install Python ${py_version} with uv" >&2
  exit 1
fi

if [ ! -f uv.lock ]; then
  echo "Error: uv.lock not found. Run 'uv lock' to generate it." >&2
  exit 1
fi
if ! uv sync --frozen; then
  echo "Error: uv sync --frozen failed. Check pyproject.toml and uv.lock." >&2
  exit 1
fi

# --- Expose the tools to later shells ---
# The PATH exports above last only for this script; Cursor starts the agent's shells
# separately. Login shells get these directories from /etc/profile.d, and every other
# shell finds the entry points through symlinks in /usr/local/bin (on the default PATH).
repo_root="$(pwd)"
tool_dirs=("$repo_root/.venv/bin" "$HOME/.local/bin")
{
  for dir in "${tool_dirs[@]}"; do
    # shellcheck disable=SC2016 # $PATH must expand when the profile is sourced, not now.
    printf 'export PATH=%q:$PATH\n' "$dir"
  done
} | $SUDO tee /etc/profile.d/cursor-env-combine-for-AI.sh >/dev/null

shopt -s nullglob
for dir in "${tool_dirs[@]}"; do
  [ -d "$dir" ] || continue
  for tool in "$dir"/*; do
    [ -e "$tool" ] || continue
    name="${tool##*/}"
    case "$name" in
      # uv's installer also drops env/env.fish (sourced, not run) in ~/.local/bin.
      pip* | activate* | deactivate | Activate.ps1 | env | env.fish) continue ;;
    esac
    if [ -f "$tool" ] && [ -x "$tool" ]; then
      $SUDO ln -sfn "$tool" "/usr/local/bin/$name"
    fi
  done
done
shopt -u nullglob

echo "Cursor install for combine-for-AI finished."
