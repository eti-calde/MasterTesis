# Running the experiments via Docker

This is the **easiest path** for someone with a fresh machine + NVIDIA
GPU who wants to reproduce the sweeps without installing Python 3.13,
`uv`, or our 30+ deps locally. The container is built on
`nvidia/cuda:12.4.1-runtime-ubuntu22.04` and ships PyTorch 2.6 + the
full `pinn_bath` stack.

**Total disk for the image**: ~6 GB (CUDA + PyTorch + deps).
**Build time**: ~5-10 min on a decent connection.

---

## Prerequisites

### Linux host

1. **NVIDIA driver** >= 535 (so it supports CUDA 12.4):
   ```bash
   nvidia-smi   # should report the GPU + a driver version
   ```
2. **Docker Engine** 20.10+ (or Docker Desktop on Linux):
   ```bash
   docker --version
   docker compose version   # plugin v2 must be available
   ```
   Install:
   ```bash
   # Ubuntu / Debian
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER   # log out + back in
   ```
3. **NVIDIA Container Toolkit** (gives Docker GPU passthrough):
   ```bash
   # Ubuntu / Debian — official install
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
     sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
4. **Verify GPU passthrough**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```
   Should print the GPU info from inside a container.

### Windows host (with WSL2)

1. **Windows 11** or recent Windows 10 (build 21H2+).
2. **NVIDIA driver** >= 535 (the Windows GeForce driver, NOT a Linux driver inside WSL).
3. **WSL2** with Ubuntu 22.04+:
   ```powershell
   wsl --install -d Ubuntu
   ```
4. **Docker Desktop for Windows**:
   - Download from <https://docs.docker.com/desktop/install/windows-install/>.
   - In Settings → General → enable **"Use WSL 2 based engine"**.
   - In Settings → Resources → WSL Integration → enable for your Ubuntu distro.
5. **GPU passthrough** is automatic on recent Docker Desktop + driver
   combos — no extra container-toolkit install needed.
6. **Verify** from inside Ubuntu/WSL:
   ```bash
   nvidia-smi                                                  # native WSL2
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```

If `docker run --gpus all ... nvidia-smi` works on either platform, you
are ready.

---

## Setup

```bash
# Inside a Linux shell or WSL2 Ubuntu shell:
git clone https://github.com/eti-calde/MasterTesis.git
cd MasterTesis

# Build the image once. Downloads CUDA base + installs Python 3.13 + deps.
# Cached layers reused on subsequent builds (~10s if only code changed).
docker compose build
```

---

## Regenerate the ground-truth datasets

The per-experiment `.npz` files (~25 MB total) are gitignored and
deterministic — regenerate once per host:

```bash
docker compose run --rm pinn bash scripts/regenerate_datasets.sh
```

This creates `Experiments/0X-*/data/ground_truth_*.npz` for Exps 1-5.
Exp 6 (Angel real data) uses the small processed `.npz` files already
shipped in the repo under `Experiments/datasets/angel2024/processed/`.

---

## Run everything in one command

For an unattended run that does dataset regen + all 7 sweeps + packages
the results into `runs.tar.gz`:

```bash
docker compose run --rm pinn bash scripts/run_all.sh
```

Wall-time on an RTX 4060: ~6 hours. Resumable — re-running picks up where
the last attempt left off. On a 4 GB GPU (e.g. GTX 1650) set `HEAVY=0`
to skip the three 2D sweeps that need more VRAM:
`docker compose run --rm pinn bash -c 'HEAVY=0 bash scripts/run_all.sh'`.

The steps below are the manual breakdown if you want to run sweeps one
at a time.

## Run the sweeps (manual)

### Fast smoke before everything

```bash
docker compose run --rm pinn pytest -m fast --quiet
```
Should report ~260 tests passing in ~10s. If any fail, fix before
launching multi-hour sweeps.

### Single sweep (interactive)

```bash
docker compose run --rm pinn \
  python -m studies.arch_scaling --device cuda --study-dir runs/arch_scaling
```

The `runs/` directory is bind-mounted, so partial results survive
container exits and you can re-launch to resume.

### All 6 sweeps in sequence (overnight)

```bash
docker compose run --rm pinn bash scripts/run_local_tonight.sh
```

This runs the 4 lighter sweeps that fit comfortably on a 4 GB GPU. For
the heavier 2D ones (Exp 3, Exp 5) + the remaining N_t sweeps, the
4060 has the VRAM headroom — run them in additional commands:

```bash
docker compose run --rm pinn python -m studies.arch_scaling \
  --device cuda --study-dir runs/arch_scaling --cases exp3,exp5
docker compose run --rm pinn python -m studies.exp2_n_t_sweep \
  --device cuda --study-dir runs/exp2_n_t
docker compose run --rm pinn python -m studies.exp5_n_t_sweep \
  --device cuda --study-dir runs/exp5_n_t
```

Each sweep is resumable: re-running picks up where the last attempt
left off. Per-run failures don't abort the sweep (logged + skipped).

### Detached / overnight runs

To launch a sweep and detach (so closing the terminal doesn't kill it):

```bash
# Linux + WSL2 (recommended): use nohup + & disown
docker compose run -d --rm pinn bash scripts/run_local_tonight.sh
# (or run inside `screen` / `tmux` if installed)
```

Monitor:

```bash
docker compose logs -f                # follow stdout
docker compose ps                     # check container is alive
```

---

## Aggregating + syncing results

Aggregate tables from inside the container:

```bash
docker compose run --rm pinn python -c "
from studies.aggregate import collect, format_text_table
for study in ['arch_scaling', 'ablation_forms', 'exp1_sensitivity',
              'exp2_n_t', 'exp5_n_t', 'exp6']:
    rows = collect(f'runs/{study}')
    if not rows: continue
    print(f'\\n=== {study} ({len(rows)} runs) ===')
    print(format_text_table(rows))
"
```

To send results back to the original machine:

```bash
# From the friend's machine, after sweeps finish:
tar -czf runs.tar.gz runs/
# transfer runs.tar.gz via scp / gdrive / whatever

# On the original machine:
tar -xzf runs.tar.gz
```

---

## Troubleshooting

**`docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`**
→ NVIDIA Container Toolkit isn't installed or Docker daemon wasn't restarted.

**`CUDA not available` inside the container**
→ Either the toolkit is missing, the host driver is too old (need >= 535),
or you didn't pass `--gpus all` / aren't running through `docker compose`.

**Image build fails on `uv sync` step**
→ Check network — `uv` downloads PyTorch wheels (~2 GB) over HTTPS.

**`runs/` dir owned by `root` after a compose run**
→ Linux gotcha; the container writes as root. Fix once:
```bash
sudo chown -R "$USER:$USER" runs/
```
Or run with the host UID:
```bash
docker compose run --rm --user "$(id -u):$(id -g)" pinn bash ...
```

**Docker Desktop on Windows says "WSL integration is not enabled"**
→ Settings → Resources → WSL Integration → toggle the distro.

---

## What lives where (host vs container)

| Path | Host | Container | Purpose |
|---|---|---|---|
| `.` | your working copy | `/workspace` (bind mount) | source, edits sync both ways |
| `runs/` | `./runs/` | `/workspace/runs/` (bind mount) | sweep outputs, persistent |
| `Experiments/datasets/` | gitignored + Angel processed tracked | inside image | benchmark data |
| `.venv/` | local (host) | n/a | venv only on host (Docker uses `/opt/venv`) |

The container's venv (`/opt/venv`) is independent of any host `.venv/`,
so the friend doesn't need Python installed locally — only Docker.
