"""Training speed per model on each device available to this interpreter (synthetic batches of the real shape).

Results are merged into results/benchmark.csv by (model, device), so running it from .venv (CPU + Intel XPU) and
from .venv-cuda (CPU + NVIDIA) builds one table; results/benchmark.md is regenerated each time.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from grf2com import preprocess as pp  # noqa: E402
from grf2com.config import ROOT, add_config_args, load_config, project_path  # noqa: E402
from grf2com.data import FeatureBuilder, make_folds, rate_mult  # noqa: E402
from grf2com.device import available_devices, describe, synchronize  # noqa: E402
from grf2com.models import MODELS, build_model, count_parameters  # noqa: E402
from grf2com.reporting import md_table  # noqa: E402


def time_steps(cfg: dict, device: torch.device, steps: int, warmup: int = 5) -> tuple[float, int]:
    fb = FeatureBuilder(cfg)
    k = rate_mult(cfg)
    length = int(round(cfg["data"]["window_s"] * pp.COM_RATE))
    batch = int(cfg["train"]["batch_size"])
    model = build_model(cfg, fb.n_inputs, fb.n_subject).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(batch, length * k, fb.n_inputs, device=device)
    s = torch.randn(batch, fb.n_subject, device=device)
    y = torch.randn(batch, length, 3, device=device)
    for i in range(warmup + steps):
        if i == warmup:
            synchronize(device)
            t0 = time.perf_counter()
        loss = ((model(x, s)[:, ::k] - y) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    synchronize(device)
    return (time.perf_counter() - t0) / steps, count_parameters(model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, model_config=False)
    parser.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--devices", nargs="*", help="default: every device this interpreter can use")
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    out = project_path(load_config(), "results_dir")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for dev_name in args.devices or available_devices():
        device = torch.device(dev_name)
        for name in args.models:
            cfg = load_config(ROOT / "configs" / "models" / f"{name}.yaml", args.set)
            steps = args.steps if dev_name != "cpu" else max(5, args.steps // 3)
            sec, n_params = time_steps(cfg, device, steps)
            epoch_s = sec * int(cfg["train"]["steps_per_epoch"])
            n_folds = len(make_folds(cfg))
            loso_h = epoch_s * int(cfg["train"]["epochs"]) * n_folds / 3600
            rows.append({"model": name, "device": dev_name, "device_name": describe(device),
                         "ms_per_step": 1000 * sec, "s_per_epoch": epoch_s, "loso_hours_max": loso_h,
                         "n_params": n_params, "torch": torch.__version__})
            print(f"{name:12s} {describe(device):40s} {1000 * sec:8.1f} ms/step  {epoch_s:7.1f} s/epoch  "
                  f"full LOSO <= {loso_h:5.2f} h", flush=True)
    path = out / "benchmark.csv"
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old.set_index(["model", "device"]).index.isin(new.set_index(["model", "device"]).index)]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(["model", "device"]).reset_index(drop=True)
    new.to_csv(path, index=False)
    cfg = load_config()
    show = new.assign(**{"ms/step": new.ms_per_step.round(1), "s/epoch": new.s_per_epoch.round(1),
                         "full LOSO (h, upper bound)": new.loso_hours_max.round(2)})[
        ["model", "device_name", "ms/step", "s/epoch", "full LOSO (h, upper bound)", "n_params", "torch"]]
    md = ["# Training speed", "",
          f"Batch {cfg['train']['batch_size']} × {cfg['data']['window_s']} s windows at "
          f"{cfg['preprocess']['grf_rate']} Hz, {cfg['train']['steps_per_epoch']} steps per epoch. 'Full LOSO' = "
          f"{cfg['train']['epochs']} epochs × {len(make_folds(cfg))} folds for one config and seed, ignoring early "
          "stopping and evaluation time (so an upper bound on training time).", "",
          md_table(show, index=False, digits=2)]
    (out / "benchmark.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
