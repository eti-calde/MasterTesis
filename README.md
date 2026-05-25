# pinn-bath — Inversión de batimetría con PINN

Código de la tesis "Inversión de batimetría mediante redes neuronales
informadas por física" (Etienne Calderón). El paquete `pinn_bath` provee la
infraestructura compartida (arquitecturas A1/A2/A3, formas del residuo SWE,
entrenamiento, métricas, checkpointing) usada por los experimentos en
`Experiments/`.

## Estructura

```
MasterTesis/
├─ pyproject.toml          # deps pinned (S11)
├─ src/pinn_bath/          # biblioteca compartida
├─ Experiments/            # un directorio por caso (Exp 1 a 6)
├─ studies/                # orquestación de §5.1 y §5.4
├─ tests/                  # pytest
├─ Report/                 # LaTeX
└─ Slides/                 # presentación (slidev)
```

> Notas personales (vault Obsidian con literatura PDF, ~330 MB) y los
> datasets externos (Tian / Liu / Angel raw, ~1.4 GB) quedan fuera del
> repo público — ver `.gitignore`. Para reproducir un experimento que
> consuma datasets externos hay que bajarlos del paper original.

## Setup

### Docker (recomendado para reproducir desde cero)

Full instructions in [`docs/DOCKER.md`](docs/DOCKER.md) (Linux + Windows/WSL2):

```bash
docker compose build
docker compose run --rm pinn bash scripts/regenerate_datasets.sh
docker compose run --rm pinn pytest -m fast        # ~260 verdes en ~10s
docker compose run --rm pinn bash scripts/run_local_tonight.sh
```

Requiere Docker + NVIDIA Container Toolkit (Linux) o Docker Desktop con WSL2 (Windows).

### Native (uv)

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev]"
bash scripts/regenerate_datasets.sh
pytest -m fast
```
