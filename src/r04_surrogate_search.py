from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import psutil
from catboost import CatBoostRegressor
from scipy.stats import qmc
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resource_plan(
    reserve_cpus=0,
    reserve_ram_gb=1.0,
    ram_per_worker_gb=2.0,
    parallel_trials=2,
):
    logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    mem = psutil.virtual_memory()
    gib = 1024**3
    available = mem.available / gib

    cpu_capacity = max(1, logical - max(0, reserve_cpus))
    memory_capacity = max(
        1,
        int(max(0.0, available - max(0.0, reserve_ram_gb))
            // max(ram_per_worker_gb, 0.25)),
    )

    # Full-machine budget for final fits and surface generation.
    workers = max(1, min(cpu_capacity, memory_capacity))

    # During Optuna search, run several trials concurrently while partitioning
    # the available CPU budget between them. This avoids nested oversubscription.
    search_jobs = max(
        1,
        min(int(max(1, parallel_trials)), cpu_capacity, memory_capacity),
    )
    search_workers_per_model = max(
        1,
        min(workers, cpu_capacity // search_jobs, memory_capacity // search_jobs),
    )

    return {
        "logical_cpus": logical,
        "total_ram_gb": mem.total / gib,
        "available_ram_gb": available,
        "reserve_cpus": reserve_cpus,
        "reserve_ram_gb": reserve_ram_gb,
        "ram_per_worker_gb": ram_per_worker_gb,
        "workers": workers,
        "search_jobs": search_jobs,
        "search_workers_per_model": search_workers_per_model,
    }


def normalized_metrics(y_true, y_pred, scales):
    err = np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2, axis=0))
    norm = err / scales
    median = float(np.median(norm))
    q90 = float(np.quantile(norm, 0.90))
    return median, q90, 0.70 * median + 0.30 * q90


class LatentHGB:
    def __init__(self, params, scales, seed):
        self.params = dict(params)
        self.scales = np.asarray(scales, float)
        self.seed = int(seed)
        self.center = None
        self.pca = None
        self.models = []

    def fit(self, X, Y):
        Yn = np.asarray(Y, float) / self.scales
        self.center = Yn.mean(axis=0)
        Z0 = Yn - self.center
        ncomp = max(
            1,
            min(
                int(self.params["latent_components"]),
                Z0.shape[0] - 1,
                Z0.shape[1],
            ),
        )
        self.pca = PCA(n_components=ncomp, random_state=self.seed)
        Z = self.pca.fit_transform(Z0)
        self.models = []
        for j in range(Z.shape[1]):
            m = HistGradientBoostingRegressor(
                learning_rate=float(self.params["learning_rate"]),
                max_iter=int(self.params["max_iter"]),
                max_leaf_nodes=int(self.params["max_leaf_nodes"]),
                min_samples_leaf=int(self.params["min_samples_leaf"]),
                l2_regularization=float(self.params["l2_regularization"]),
                max_bins=int(self.params["max_bins"]),
                random_state=self.seed + 104729 * j,
            )
            m.fit(np.asarray(X, float), Z[:, j])
            self.models.append(m)
        return self

    def predict(self, X):
        Z = np.column_stack([m.predict(np.asarray(X, float)) for m in self.models])
        Yn = self.pca.inverse_transform(Z) + self.center
        return Yn * self.scales


class StableCatBoost:
    """
    CatBoost receives tolerance-normalized responses, then a per-response
    translation and one common global divisor. Translation and a common positive
    scale preserve the relative MultiRMSE geometry while preventing numerically
    unsafe targets (the private R04 failure reached ~1e14).
    """

    def __init__(self, params, scales, workers, seed):
        self.params = dict(params)
        self.scales = np.asarray(scales, float)
        self.workers = int(workers)
        self.seed = int(seed)
        self.center = None
        self.global_divisor = None
        self.active_mask = None
        self.model = None

    def fit(self, X, Y):
        Yn = np.asarray(Y, float) / self.scales
        self.center = Yn.mean(axis=0)
        centered = Yn - self.center

        if not np.isfinite(centered).all():
            raise RuntimeError("Non-finite normalized CatBoost target.")

        # CatBoost rejects multi-output training when one or more response
        # dimensions are constant inside a CV fold. Train only dimensions with
        # meaningful fold-level variation, then reconstruct constant dimensions
        # from their normalized training means at prediction time.
        spread = np.ptp(centered, axis=0)
        scale_ref = np.maximum(1.0, np.max(np.abs(Yn), axis=0))
        self.active_mask = spread > (1e-12 * scale_ref)
        if not np.any(self.active_mask):
            raise RuntimeError("All normalized CatBoost targets are constant.")

        active = centered[:, self.active_mask]
        max_abs = float(np.max(np.abs(active)))
        self.global_divisor = max(1.0, max_abs / 1000.0)
        stable = active / self.global_divisor

        if not np.isfinite(stable).all():
            raise RuntimeError("Non-finite stabilized CatBoost target.")

        one_dimensional = stable.shape[1] == 1
        train_target = stable[:, 0] if one_dimensional else stable
        loss = "RMSE" if one_dimensional else "MultiRMSE"

        self.model = CatBoostRegressor(
            loss_function=loss,
            eval_metric=loss,
            iterations=int(self.params["iterations"]),
            depth=int(self.params["depth"]),
            learning_rate=float(self.params["learning_rate"]),
            l2_leaf_reg=float(self.params["l2_leaf_reg"]),
            random_strength=float(self.params["random_strength"]),
            bagging_temperature=float(self.params["bagging_temperature"]),
            border_count=int(self.params["border_count"]),
            random_seed=self.seed,
            thread_count=self.workers,
            task_type="CPU",
            verbose=False,
            allow_writing_files=False,
        )
        self.model.fit(np.asarray(X, float), train_target)
        return self

    def predict(self, X):
        active_pred = np.asarray(self.model.predict(np.asarray(X, float)), float)
        if active_pred.ndim == 1:
            active_pred = active_pred[:, None]
        n = active_pred.shape[0]
        Yn = np.tile(self.center, (n, 1))
        Yn[:, self.active_mask] = active_pred * self.global_divisor + self.center[self.active_mask]
        return Yn * self.scales


def suggest(trial, model):
    if model == "extratrees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 500, 1400, step=100),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            "max_features": trial.suggest_float("max_features", 0.55, 1.0),
            "max_depth": trial.suggest_categorical("max_depth", [None, 10, 16, 24, 32]),
        }
    if model == "hgb":
        return {
            "latent_components": trial.suggest_categorical(
                "latent_components", [8, 10, 12, 14, 16, 20, 24]
            ),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "max_iter": trial.suggest_int("max_iter", 250, 650, step=50),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 31, 79, step=8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 16, 40),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 0.002, 1.5, log=True
            ),
            "max_bins": trial.suggest_categorical("max_bins", [127, 255]),
        }
    if model == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 300, 1200, step=100),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature", 0.0, 2.0
            ),
            "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        }
    raise ValueError(model)


def build_model(model, params, scales, workers, seed):
    if model == "extratrees":
        return ExtraTreesRegressor(random_state=seed, n_jobs=workers, **params)
    if model == "hgb":
        return LatentHGB(params, scales, seed)
    if model == "catboost":
        return StableCatBoost(params, scales, workers, seed)
    raise ValueError(model)


def load_contract(input_dir: Path):
    evidence = pd.read_csv(input_dir / "evidence.csv")
    scales_df = pd.read_csv(input_dir / "scales.csv")
    bounds = pd.read_csv(input_dir / "parameter_bounds.csv")
    screen = pd.read_csv(input_dir / "screen_map.csv")
    residual_path = input_dir / "residual_targets.csv"
    residual = (
        set(pd.read_csv(residual_path)["target_id"].astype(str))
        if residual_path.exists()
        else set()
    )

    xcols = sorted(c for c in evidence.columns if c.startswith("X"))
    ycols = sorted(c for c in evidence.columns if c.startswith("Y"))
    if not xcols or not ycols:
        raise RuntimeError("Neutral X/Y columns not found.")

    scale_map = dict(
        zip(scales_df["response_id"].astype(str), scales_df["scale"].astype(float))
    )
    scales = np.asarray([scale_map[c] for c in ycols], float)
    if (scales <= 0).any() or not np.isfinite(scales).all():
        raise RuntimeError("Invalid response scales.")

    pids = sorted(evidence["P00"].astype(str).unique())
    for pid in pids:
        evidence[f"PROFILE_{pid}"] = (evidence["P00"].astype(str) == pid).astype(float)
    feature_cols = xcols + [f"PROFILE_{p}" for p in pids]

    bound_map = bounds.set_index("feature_id")[["lower", "upper"]].astype(float)
    return evidence, feature_cols, xcols, ycols, scales, screen, residual, bound_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["extratrees", "hgb", "catboost"], required=True)
    ap.add_argument("--input-dir", default="inputs/r04")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--virtual-points", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=260923)
    ap.add_argument("--reserve-cpus", type=int, default=0)
    ap.add_argument("--reserve-ram-gb", type=float, default=1.0)
    ap.add_argument("--ram-per-worker-gb", type=float, default=2.0)
    ap.add_argument("--parallel-trials", type=int, default=2)
    args = ap.parse_args()

    input_dir = ROOT / args.input_dir
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    evidence, feature_cols, xcols, ycols, scales, screen, residual, bound_map = load_contract(
        input_dir
    )
    dev = evidence.loc[evidence["LOCKBOX"].astype(int).eq(0)].reset_index(drop=True)
    lock = evidence.loc[evidence["LOCKBOX"].astype(int).eq(1)].reset_index(drop=True)
    X = dev[feature_cols].astype(float)
    Y = dev[ycols].astype(float)
    Xlock = lock[feature_cols].astype(float)
    Ylock = lock[ycols].astype(float)
    groups = dev["GROUP_ID"].astype(str).to_numpy()

    resources = resource_plan(
        args.reserve_cpus,
        args.reserve_ram_gb,
        args.ram_per_worker_gb,
        args.parallel_trials,
    )
    workers = int(resources["workers"])
    search_jobs = int(resources["search_jobs"])
    search_workers = int(resources["search_workers_per_model"])
    write_csv(out / "resource_plan.csv", [{**resources, "backend": "CPU"}])

    splitter = GroupKFold(n_splits=5)

    def objective(trial):
        params = suggest(trial, args.model)
        fold_scores = []
        for fold, (tr, va) in enumerate(splitter.split(X, Y, groups)):
            model = build_model(
                args.model, params, scales, search_workers, args.seed + 1009 * fold
            )
            model.fit(X.iloc[tr], Y.iloc[tr])
            pred = model.predict(X.iloc[va])
            fold_scores.append(
                normalized_metrics(Y.iloc[va].to_numpy(float), pred, scales)
            )
            del model
            gc.collect()
        trial.set_user_attr("fold_median", [x[0] for x in fold_scores])
        trial.set_user_attr("fold_q90", [x[1] for x in fold_scores])
        return float(np.mean([x[2] for x in fold_scores]))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=args.seed, multivariate=True, group=True
        ),
    )
    study.optimize(
        objective,
        n_trials=args.trials,
        n_jobs=search_jobs,
        gc_after_trial=True,
    )
    best = dict(study.best_params)

    trial_rows = []
    for t in study.trials:
        trial_rows.append(
            {
                "trial": t.number,
                "state": str(t.state),
                "objective": t.value,
                "fold_median": json.dumps(t.user_attrs.get("fold_median", [])),
                "fold_q90": json.dumps(t.user_attrs.get("fold_q90", [])),
                **t.params,
            }
        )
    write_csv(out / "optuna_trials.csv", trial_rows)
    (out / "best_params.json").write_text(
        json.dumps(best, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    cv_pred = np.empty_like(Y.to_numpy(float))
    fold_models = []
    for fold, (tr, va) in enumerate(splitter.split(X, Y, groups)):
        model = build_model(args.model, best, scales, workers, args.seed + 1009 * fold)
        model.fit(X.iloc[tr], Y.iloc[tr])
        cv_pred[va] = model.predict(X.iloc[va])
        fold_models.append(model)

    final_model = build_model(args.model, best, scales, workers, args.seed)
    final_model.fit(X, Y)
    lock_pred = final_model.predict(Xlock)

    cv_median, cv_q90, cv_comp = normalized_metrics(Y.to_numpy(float), cv_pred, scales)
    lb_median, lb_q90, lb_comp = normalized_metrics(
        Ylock.to_numpy(float), lock_pred, scales
    )

    # Per-response RMSE for quality gating.
    cv_rmse = np.sqrt(np.mean((Y.to_numpy(float) - cv_pred) ** 2, axis=0))
    lb_rmse = np.sqrt(np.mean((Ylock.to_numpy(float) - lock_pred) ** 2, axis=0))
    idx = {c: i for i, c in enumerate(ycols)}

    target_rows = []
    eligible_targets = []
    for row in screen.to_dict("records"):
        tid = str(row["target_id"])
        refs = []
        if str(row["rule"]) == "INTERVAL":
            refs = [str(row["lo_response"]), str(row["hi_response"])]
        else:
            refs = [str(row["value_response"])]
        ok_refs = all(r in idx for r in refs)
        if ok_refs:
            worst_cv = max(cv_rmse[idx[r]] / scales[idx[r]] for r in refs)
            worst_lb = max(lb_rmse[idx[r]] / scales[idx[r]] for r in refs)
        else:
            worst_cv = worst_lb = np.inf
        ok = bool(ok_refs and worst_cv <= 1.0 and worst_lb <= 1.0)
        if ok:
            eligible_targets.append(row)
        target_rows.append(
            {
                "target_id": tid,
                "cv_worst_normalized_rmse": worst_cv,
                "lockbox_worst_normalized_rmse": worst_lb,
                "eligible": str(ok).lower(),
                "residual_flag": str(tid in residual).lower(),
            }
        )
    write_csv(out / "quality.csv", target_rows)

    # Common deterministic Sobol surface.
    engine = qmc.Sobol(d=len(xcols), scramble=True, seed=args.seed + 17)
    power = 1 << int(np.ceil(np.log2(args.virtual_points)))
    unit = engine.random_base2(int(np.log2(power)))[: args.virtual_points]
    lows = np.asarray([float(bound_map.loc[c, "lower"]) for c in xcols])
    highs = np.asarray([float(bound_map.loc[c, "upper"]) for c in xcols])
    values = qmc.scale(unit, lows, highs)

    pids = sorted(evidence["P00"].astype(str).unique())
    profiles = np.asarray([pids[i % len(pids)] for i in range(args.virtual_points)])
    VX = pd.DataFrame(values, columns=xcols)
    for pid in pids:
        VX[f"PROFILE_{pid}"] = (profiles == pid).astype(float)
    pred = np.asarray(final_model.predict(VX[feature_cols]), float)

    # Fold-model disagreement in normalized response units.
    uncertainty = np.empty(args.virtual_points, float)
    batch = 1024
    for start in range(0, args.virtual_points, batch):
        stop = min(start + batch, args.virtual_points)
        block = VX.iloc[start:stop][feature_cols]
        fp = np.stack([m.predict(block) for m in fold_models], axis=0)
        uncertainty[start:stop] = np.median(np.std(fp, axis=0) / scales, axis=1)

    impl_by_target = {}
    for row in eligible_targets:
        tid = str(row["target_id"])
        lo = float(row["lower"])
        hi = float(row["upper"])
        half = max((hi - lo) / 2.0, 1e-12)
        if str(row["rule"]) == "INTERVAL":
            rlo = str(row["lo_response"])
            rhi = str(row["hi_response"])
            plo = pred[:, idx[rlo]]
            phi = pred[:, idx[rhi]]
            low = np.minimum(plo, phi)
            high = np.maximum(plo, phi)
            gap = np.where(high < lo, lo - high, np.where(low > hi, low - hi, 0.0))
            erm = max(
                cv_rmse[idx[rlo]], cv_rmse[idx[rhi]],
                lb_rmse[idx[rlo]], lb_rmse[idx[rhi]],
            )
        else:
            rid = str(row["value_response"])
            pv = pred[:, idx[rid]]
            gap = np.maximum(lo - pv, np.maximum(pv - hi, 0.0))
            erm = max(cv_rmse[idx[rid]], lb_rmse[idx[rid]])
        impl_by_target[tid] = gap / max(np.sqrt(half * half + erm * erm), 1e-12)

    if impl_by_target:
        tids = list(impl_by_target)
        M = np.column_stack([impl_by_target[t] for t in tids])
        max_i = M.max(axis=1)
        rms_i = np.sqrt(np.mean(M * M, axis=1))
    else:
        max_i = np.full(args.virtual_points, np.inf)
        rms_i = np.full(args.virtual_points, np.inf)

    resid_ids = [t for t in impl_by_target if t in residual]
    guard_ids = [t for t in impl_by_target if t not in residual]
    if resid_ids:
        R = np.column_stack([impl_by_target[t] for t in resid_ids])
        resid_rms = np.sqrt(np.mean(R * R, axis=1))
    else:
        resid_rms = np.full(args.virtual_points, np.inf)
    if guard_ids:
        G = np.column_stack([impl_by_target[t] for t in guard_ids])
        guard_max = G.max(axis=1)
    else:
        guard_max = np.zeros(args.virtual_points)
    residual_score = resid_rms + np.maximum(guard_max - 1.0, 0.0)

    rows = []
    for i in range(args.virtual_points):
        row = {
            "virtual_index": i,
            "P00": profiles[i],
            "max_i": float(max_i[i]),
            "rms_i": float(rms_i[i]),
            "uncertainty": float(uncertainty[i]),
            "residual_score": float(residual_score[i]),
            "nroy": str(max_i[i] <= 3.0).lower(),
        }
        for j, c in enumerate(xcols):
            row[c] = float(values[i, j])
        rows.append(row)

    write_csv(out / "surface.csv", rows)

    chosen = set()
    queue = []

    def add(order, bucket, limit):
        count = 0
        for i in order:
            if i in chosen:
                continue
            chosen.add(int(i))
            row = dict(rows[int(i)])
            row["bucket"] = bucket
            row["model"] = args.model
            queue.append(row)
            count += 1
            if count >= limit:
                break

    add(np.lexsort((rms_i, max_i)), "A", 8)
    add(np.argsort(-uncertainty), "B", 8)
    add(np.argsort(np.abs(max_i - 3.0)), "C", 8)
    add(np.lexsort((max_i, residual_score + (max_i > 3.0) * 1000.0)), "D", 8)
    for i, row in enumerate(queue, 1):
        row["queue_order"] = i
    write_csv(out / "queue.csv", queue)

    write_csv(
        out / "summary.csv",
        [{
            "model": args.model,
            "development_rows": len(dev),
            "lockbox_rows": len(lock),
            "trials": args.trials,
            "objective": "0.70_median+0.30_q90",
            "best_objective": study.best_value,
            "cv_median": cv_median,
            "cv_q90": cv_q90,
            "cv_composite": cv_comp,
            "lockbox_median": lb_median,
            "lockbox_q90": lb_q90,
            "lockbox_composite": lb_comp,
            "eligible_targets": len(eligible_targets),
            "virtual_points": args.virtual_points,
            "nroy_count": int((max_i <= 3.0).sum()),
            "queue_points": len(queue),
            "catboost_stabilization": (
                "center_per_response+single_global_divisor"
                if args.model == "catboost" else ""
            ),
        }],
    )


if __name__ == "__main__":
    raise SystemExit(main())
