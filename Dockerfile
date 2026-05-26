# pinn-bath: PyTorch + CUDA 12.4 + Python 3.13 image for the inverse SWE
# bathymetry sweeps. Built on the NVIDIA CUDA runtime image; works under
# `docker run --gpus all` (Linux) or Docker Desktop with WSL2 (Windows).
#
# Build:    docker build -t pinn-bath .
# Run:      docker run --rm -it --gpus all -v "$PWD:/workspace" pinn-bath bash
# Compose:  docker compose run --rm pinn

# CUDA 12.8 runtime image — required for RTX 50-series (Blackwell, sm_120)
# native kernels in PyTorch >=2.7. Older 4060/3090/A100 hosts also work
# (forward-compat). "runtime" = nvidia driver libs + CUDA libs, no devel
# toolchain (we don't compile CUDA kernels — PyTorch ships its own).
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Non-interactive apt + UTF-8 locale.
ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Base system deps. Ubuntu 22.04 ships Python 3.10; we let `uv` install
# 3.13 standalone from upstream (next layer). git is needed to install
# editable packages from a workdir that's a git repo.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package + interpreter manager).
# Pin the install script to a release tag for reproducibility.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install Python 3.13 via uv. ~30s, ~25 MB download.
RUN uv python install 3.13

# --- Project setup --------------------------------------------------------

WORKDIR /workspace

# Copy ONLY the dep manifests first so layer cache survives code changes.
# README.md is referenced by pyproject.toml (`readme = "README.md"`).
COPY pyproject.toml uv.lock README.md ./

# Create the venv at a fixed path and install deps (without the project
# itself — we install -e . later once the source is mounted/copied).
RUN uv venv --python 3.13 /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv"

# Install runtime + dev deps from the lockfile. --no-install-project
# skips the editable install of pinn-bath itself (done next). --active
# tells uv to use the venv at VIRTUAL_ENV (/opt/venv) instead of creating
# a project-local .venv at /workspace/.venv (which COPY . . would clobber).
RUN uv sync --active --frozen --no-install-project --extra dev

# Now copy the source. .dockerignore keeps .venv/, runs/, Notes/, etc. out.
COPY . .

# Install the package itself in editable mode so import works under the
# unified PATH. Doing this after COPY . means the source layer is the
# last expensive step and re-runs on every code change (which is fine).
RUN uv pip install --python /opt/venv/bin/python --no-deps -e .

# Smoke check at build time: imports + CUDA detection script (not actual
# CUDA — that needs the runtime GPU passthrough).
RUN python -c "import pinn_bath, torch; \
    print(f'pinn_bath OK, torch={torch.__version__}, cuda_built={torch.version.cuda}')"

# Default: drop into bash so the user can run sweeps interactively.
# Override with `docker run ... pytest -m fast` or similar.
CMD ["/bin/bash"]
