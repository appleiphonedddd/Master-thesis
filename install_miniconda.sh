#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="$(mktemp -d)"
LOG_FILE="$LOG_DIR/install_miniconda.log"
exec > >(tee -a "$LOG_FILE") 2>&1

PREFIX_DEFAULT="$HOME/miniconda3"
SHELLS_DEFAULT="auto"   # auto | comma list: bash,zsh,fish
FORCE="false"
DO_INIT="true"
USE_MAMBA="false"
ENV_FILE=""
PREFIX="$PREFIX_DEFAULT"
SHA256_EXPECTED=""
REFRESH="true"          

usage() {
  cat <<EOF
Usage: $0 [options]
  --prefix <path>        Install prefix (default: $PREFIX_DEFAULT)
  --no-init              Skip "conda init"
  --shells <list>        Shells to init: auto | bash,zsh,fish (default: auto)
  --mamba                Install mamba in base
  --env-file <path>      environment.yml to create/update env
  --force                Reinstall even if prefix exists (backup old dir)
  --sha256 <hash>        Verify installer SHA256
  --no-refresh           Do NOT auto-refresh shell after install
  -h, --help             Show this help
EOF
}

# -------- Parse args --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --no-init) DO_INIT="false"; shift ;;
    --shells) SHELLS_DEFAULT="$2"; shift 2 ;;
    --mamba) USE_MAMBA="true"; shift ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --force) FORCE="true"; shift ;;
    --sha256) SHA256_EXPECTED="$2"; shift 2 ;;
    --no-refresh) REFRESH="false"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

# -------- Helpers --------
have() { command -v "$1" >/dev/null 2>&1; }

download() {
  local url="$1" out="$2"
  echo "Downloading: $url"
  if have curl; then
    curl -fL --retry 3 --connect-timeout 15 -o "$out" "$url"
  elif have wget; then
    wget -O "$out" "$url"
  else
    echo "Error: neither curl nor wget found." >&2
    exit 1
  fi
}

verify_sha256() {
  local file="$1" expected="$2"
  if [[ -z "$expected" ]]; then
    echo "No SHA256 provided; skipping integrity check."
    return 0
  fi
  echo "Verifying SHA256..."
  local actual
  if have sha256sum; then
    actual="$(sha256sum "$file" | awk '{print $1}')"
  elif have shasum; then
    actual="$(shasum -a 256 "$file" | awk '{print $1}')"
  else
    echo "No sha256 tool found; skipping integrity check."
    return 0
  fi
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA256 mismatch! expected=$expected actual=$actual" >&2
    exit 1
  fi
  echo "SHA256 OK."
}

backup_existing_prefix() {
  local pfx="$1"
  if [[ -d "$pfx" ]]; then
    if [[ "$FORCE" == "true" ]]; then
      local bak="${pfx}.bak.$(date +%s)"
      echo "Backing up existing prefix to: $bak"
      mv "$pfx" "$bak"
    else
      echo "Prefix already exists: $pfx"
      echo "Use --force to backup and reinstall, or --prefix to choose another path."
      exit 0
    fi
  fi
}

detect_arch() {
  local arch
  arch="$(uname -m)"
  case "$arch" in
    x86_64) echo "x86_64" ;;
    aarch64|arm64) echo "aarch64" ;;
    *) echo "Unsupported arch: $arch" >&2; exit 1 ;;
  esac
}

choose_installer_url() {
  local arch="$1"
  case "$arch" in
    x86_64)  echo "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" ;;
    aarch64) echo "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh" ;;
  esac
}

init_shells() {
  local pfx="$1" shells="$2"
  if [[ "$DO_INIT" != "true" ]]; then
    echo "Skipping conda init (--no-init)."
    return
  fi

  # auto-detect current shell if requested
  if [[ "$shells" == "auto" ]]; then
    if [[ -n "${BASH_VERSION:-}" ]]; then
      shells="bash"
    elif [[ -n "${ZSH_VERSION:-}" ]]; then
      shells="zsh"
    else
      shells="bash"
    fi
  fi

  IFS=',' read -r -a arr <<< "$shells"
  for s in "${arr[@]}"; do
    case "$s" in
      bash|zsh|fish)
        echo "Running: $pfx/bin/conda init $s"
        "$pfx/bin/conda" init "$s" || true
        ;;
      *)
        echo "Unknown shell '$s' (supported: bash,zsh,fish). Skipping."
        ;;
    esac
  done

  echo "RC files were modified by 'conda init'."
}

setup_conda_defaults() {
  echo "Configuring conda defaults..."
  conda config --set auto_activate_base false || true
  conda config --set channel_priority strict || true
  conda config --add channels conda-forge || true
}

create_or_update_env() {
  local file="$1"
  if [[ -n "$file" && -f "$file" ]]; then
    echo "Applying environment file: $file"
    if conda env list | grep -qE '^\s*[^#]'; then
      echo "Trying 'conda env update -f $file'"
      conda env update -f "$file" || conda env create -f "$file"
    else
      echo "Creating env from $file"
      conda env create -f "$file"
    fi
  else
    [[ -n "$file" ]] && echo "Env file not found: $file (skipping)"
  fi
}

is_sourced() {
  # bash
  if [[ -n "${BASH_VERSION:-}" ]]; then
    [[ "${BASH_SOURCE[0]}" != "$0" ]] && return 0 || return 1
  fi
  # zsh
  if [[ -n "${ZSH_VERSION:-}" ]]; then
    [[ "${ZSH_EVAL_CONTEXT:-}" == *":file"* ]] && return 0 || return 1
  fi
  return 1
}

activate_in_current_shell() {
  if [[ -n "${BASH_VERSION:-}" ]]; then
    eval "$("$PREFIX/bin/conda" shell.bash hook)"
    return
  fi
  if [[ -n "${ZSH_VERSION:-}" ]]; then
    eval "$("$PREFIX/bin/conda" shell.zsh hook)"
    return
  fi
  return 1
}

refresh_shell() {
  [[ "$REFRESH" != "true" ]] && return 0
  [[ "$DO_INIT" != "true" ]] && return 0

  if is_sourced; then
    echo "Activating conda in this shell..."
    activate_in_current_shell || true
    return 0
  fi

  local target="${SHELL:-}"
  if [[ -z "$target" ]]; then
    if have zsh; then target="$(command -v zsh)"
    elif have bash; then target="$(command -v bash)"
    else echo "No bash/zsh found to re-open shell."; return 0
    fi
  fi
  echo "Opening a new login shell ($target) with conda initialized..."
  exec "$target" -l
}

cleanup() {
  rm -rf "$LOG_DIR" 2>/dev/null || true
}
trap 'echo "Error occurred. See log: $LOG_FILE"' ERR
trap cleanup EXIT

echo "== Miniconda Bootstrap =="

if have conda; then
  EXISTING_PREFIX="$(conda info --base 2>/dev/null || true)"
  if [[ -n "$EXISTING_PREFIX" && "$EXISTING_PREFIX" != "$PREFIX" ]]; then
    echo "Warning: a different conda is already on PATH: $EXISTING_PREFIX"
    echo "Proceeding may lead to multiple conda installs on PATH."
  fi
fi

backup_existing_prefix "$PREFIX"

ARCH="$(detect_arch)"
URL="$(choose_installer_url "$ARCH")"
INST="$(mktemp -p /tmp Miniconda3-XXXXXX.sh)"

echo "Installer: $URL"
download "$URL" "$INST"
chmod +x "$INST"
verify_sha256 "$INST" "$SHA256_EXPECTED"

echo "Installing to: $PREFIX (batch mode)"
bash "$INST" -b -p "$PREFIX"
rm -f "$INST"

export PATH="$PREFIX/bin:$PATH"

echo "Conda version:"
conda --version || true

init_shells "$PREFIX" "$SHELLS_DEFAULT"
setup_conda_defaults

if [[ "$USE_MAMBA" == "true" ]]; then
  echo "Installing mamba in base..."
  conda install -n base -y mamba -c conda-forge || true
fi

create_or_update_env "$ENV_FILE"

echo "Verifying conda info..."
conda info || true

refresh_shell

echo "Done."
