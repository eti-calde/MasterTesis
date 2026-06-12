"""Legacy v1 dataset modules (operator pivot).

``generator`` and ``operator_dataset`` are the v1 bank generator and split
builder, kept so the published v1 sweeps remain exactly reproducible; the
v2 pipeline lives in :mod:`pinn_bath.datagen`. Both modules here are
torch-free (data generation can run on a host without torch).
"""
