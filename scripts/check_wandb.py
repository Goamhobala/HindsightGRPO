"""Read-only authentication/connectivity check; never print or persist the key."""
import os
import wandb

if not os.environ.get("WANDB_API_KEY", "").strip():
    raise SystemExit("Export WANDB_API_KEY in the shell used to submit jobs.")
if os.environ.get("WANDB_MODE", "online") not in ("online", "run"):
    raise SystemExit("Online W&B logging is required.")
viewer = wandb.Api(timeout=30).viewer
if not viewer:
    raise SystemExit("W&B authentication failed.")
print("W&B authentication and API connectivity verified.")
