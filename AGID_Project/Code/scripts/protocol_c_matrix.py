"""Build Protocol C frozen-model audit matrix.

This script expands the existing AGID audit into a large, reproducible matrix of
closed-form bottleneck interventions. It does not retrain any model.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np


CONCEPT_PAIR = {"jpeg_quant", "freq_radial"}

FIELDNAMES = [
    "source",
    "seed",
    "family",
    "subset_size",
    "channels",
    "imputation",
    "direction",
    "generator",
    "n",
    "metric",
    "value",
    "baseline_acc",
    "acc",
    "delta_pp",
    "effect_pp",
    "auc",
    "real_acc",
    "fake_acc",
    "protocol",
    "claim_scope",
    "notes",
]


def _round(value):
    if value is None:
        return ""
    try:
        if np.isnan(value):
            return ""
    except TypeError:
        pass
    return round(float(value), 6)


def logit_to_prob(logits):
    logits = np.asarray(logits, dtype=np.float64)
    return np.where(
        logits >= 0,
        1.0 / (1.0 + np.exp(-logits)),
        np.exp(logits) / (1.0 + np.exp(logits)),
    )


def safe_auc(labels, probs):
    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, probs))
    except Exception:
        return None


def metrics_for(probs, labels):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs)
    preds = (probs > 0.5).astype(int)
    real_mask = labels == 0
    fake_mask = labels == 1
    return {
        "n": int(len(labels)),
        "acc": float((preds == labels).mean()),
        "auc": safe_auc(labels, probs),
        "real_acc": float((preds[real_mask] == 0).mean()) if real_mask.any() else None,
        "fake_acc": float((preds[fake_mask] == 1).mean()) if fake_mask.any() else None,
    }


def grouped_metrics(probs, labels, generators):
    groups = sorted(set(generators.tolist()))
    out = {}
    for group in groups:
        mask = generators == group
        out[group] = metrics_for(probs[mask], labels[mask])
    out["__ALL__"] = metrics_for(probs, labels)
    return out


def _row(
    *,
    source,
    seed,
    family,
    subset_size,
    channels,
    generator,
    metric,
    value,
    protocol,
    claim_scope,
    baseline_acc=None,
    acc=None,
    delta_pp=None,
    effect_pp=None,
    auc=None,
    real_acc=None,
    fake_acc=None,
    n=None,
    imputation="",
    direction="",
    notes="",
):
    return {
        "source": source,
        "seed": seed,
        "family": family,
        "subset_size": subset_size,
        "channels": channels,
        "imputation": imputation,
        "direction": direction,
        "generator": generator,
        "n": n if n is not None else "",
        "metric": metric,
        "value": _round(value),
        "baseline_acc": _round(baseline_acc),
        "acc": _round(acc),
        "delta_pp": _round(delta_pp),
        "effect_pp": _round(effect_pp),
        "auc": _round(auc),
        "real_acc": _round(real_acc),
        "fake_acc": _round(fake_acc),
        "protocol": protocol,
        "claim_scope": claim_scope,
        "notes": notes,
    }


def compute_subset_interventions(
    concepts,
    labels,
    generators,
    weights,
    bias,
    names,
    *,
    seed,
    source,
    subset_sizes=(1, 2, 3),
):
    concepts = np.asarray(concepts, dtype=np.float32)
    labels = np.asarray(labels).astype(int)
    generators = np.asarray(generators)
    weights = np.asarray(weights, dtype=np.float32)

    baseline_probs = logit_to_prob(concepts @ weights + float(bias))
    baseline = grouped_metrics(baseline_probs, labels, generators)
    rows = []
    protocol = "frozen_bottleneck_closed_form"
    claim_scope = "audit_intervention_not_retraining"

    for subset_size in subset_sizes:
        for idxs in itertools.combinations(range(len(names)), subset_size):
            channel_names = [names[i] for i in idxs]
            channels = "+".join(channel_names)

            dropped = concepts.copy()
            dropped[:, idxs] = 0.0
            drop_metrics = grouped_metrics(
                logit_to_prob(dropped @ weights + float(bias)), labels, generators
            )
            rows.extend(
                _metric_rows(
                    drop_metrics,
                    baseline,
                    source=source,
                    seed=seed,
                    family="drop",
                    subset_size=subset_size,
                    channels=channels,
                    imputation="zero",
                    protocol=protocol,
                    claim_scope=claim_scope,
                )
            )

            kept = np.zeros_like(concepts)
            kept[:, idxs] = concepts[:, idxs]
            keep_metrics = grouped_metrics(
                logit_to_prob(kept @ weights + float(bias)), labels, generators
            )
            rows.extend(
                _metric_rows(
                    keep_metrics,
                    baseline,
                    source=source,
                    seed=seed,
                    family="keep_only",
                    subset_size=subset_size,
                    channels=channels,
                    imputation="zero_elsewhere",
                    protocol=protocol,
                    claim_scope=claim_scope,
                )
            )

    return rows


def _metric_rows(metrics, baseline, **meta):
    rows = []
    for generator, vals in metrics.items():
        base_acc = baseline[generator]["acc"]
        delta_pp = (vals["acc"] - base_acc) * 100.0
        rows.append(
            _row(
                generator=generator,
                n=vals["n"],
                metric="accuracy",
                value=vals["acc"],
                baseline_acc=base_acc,
                acc=vals["acc"],
                delta_pp=delta_pp,
                effect_pp=delta_pp,
                auc=vals["auc"],
                real_acc=vals["real_acc"],
                fake_acc=vals["fake_acc"],
                **meta,
            )
        )
    return rows


def compute_single_imputation_interventions(
    concepts,
    labels,
    generators,
    weights,
    bias,
    names,
    *,
    seed,
    source,
):
    concepts = np.asarray(concepts, dtype=np.float32)
    labels = np.asarray(labels).astype(int)
    generators = np.asarray(generators)
    weights = np.asarray(weights, dtype=np.float32)
    baseline_probs = logit_to_prob(concepts @ weights + float(bias))
    baseline = grouped_metrics(baseline_probs, labels, generators)
    rows = []

    global_means = concepts.mean(axis=0)
    class_means = {
        int(label): concepts[labels == label].mean(axis=0)
        for label in sorted(set(labels.tolist()))
    }

    for idx, name in enumerate(names):
        for mode in ("zero", "global_mean", "class_mean"):
            edited = concepts.copy()
            if mode == "zero":
                edited[:, idx] = 0.0
            elif mode == "global_mean":
                edited[:, idx] = global_means[idx]
            else:
                for label, mean_vec in class_means.items():
                    edited[labels == label, idx] = mean_vec[idx]
            metrics = grouped_metrics(
                logit_to_prob(edited @ weights + float(bias)), labels, generators
            )
            rows.extend(
                _metric_rows(
                    metrics,
                    baseline,
                    source=source,
                    seed=seed,
                    family="impute_single",
                    subset_size=1,
                    channels=name,
                    imputation=mode,
                    protocol="frozen_bottleneck_closed_form",
                    claim_scope="audit_intervention_not_retraining",
                )
            )
    return rows


def compute_reliance_summary(rows):
    single = [
        r
        for r in rows
        if r.get("family") == "drop"
        and int(r.get("subset_size") or 0) == 1
        and r.get("generator", "__ALL__") == "__ALL__"
        and r.get("delta_pp", "") != ""
    ]
    if not single:
        single = [
            r
            for r in rows
            if r.get("family") == "drop"
            and int(r.get("subset_size") or 0) == 1
            and r.get("delta_pp", "") != ""
        ]

    abs_total = sum(abs(float(r["delta_pp"])) for r in single)
    compression_total = sum(
        abs(float(r["delta_pp"]))
        for r in single
        if set(str(r["channels"]).split("+")) & CONCEPT_PAIR
    )
    max_single = max((abs(float(r["delta_pp"])) for r in single), default=0.0)
    families = {}
    for row in rows:
        families[row["family"]] = families.get(row["family"], 0) + 1
    return {
        "n_rows": len(rows),
        "families": families,
        "single_drop_rows": len(single),
        "compression_share_single_drop": round(compression_total / abs_total, 6)
        if abs_total
        else None,
        "reliance_concentration_single_drop": round(max_single / abs_total, 6)
        if abs_total
        else None,
        "claim_scope": "Frozen-model causal/interventional audit matrix; not a full retraining matrix.",
    }


def load_dump_rows(results_dir):
    dump_specs = [
        ("42", results_dir / "full_inference_dump.npz"),
        ("1", results_dir / "full_inference_dump_seed1.npz"),
        ("2", results_dir / "full_inference_dump_seed2.npz"),
    ]
    rows = []
    for seed, path in dump_specs:
        if not path.exists():
            continue
        dump = np.load(path, allow_pickle=True)
        names = [str(x) for x in dump["concept_names"].tolist()]
        rows.extend(
            compute_subset_interventions(
                dump["concepts"],
                dump["labels"],
                dump["generators"],
                dump["classifier_weight"],
                float(dump["classifier_bias"]),
                names,
                seed=seed,
                source=str(path.relative_to(results_dir.parent)),
            )
        )
        rows.extend(
            compute_single_imputation_interventions(
                dump["concepts"],
                dump["labels"],
                dump["generators"],
                dump["classifier_weight"],
                float(dump["classifier_bias"]),
                names,
                seed=seed,
                source=str(path.relative_to(results_dir.parent)),
            )
        )
    return rows


def load_counterfactual_rows(results_dir):
    specs = [
        ("42", results_dir / "seed42_analysis"),
        ("1", results_dir / "seed1_analysis"),
        ("2", results_dir / "seed2_analysis"),
    ]
    rows = []
    for seed, directory in specs:
        for filename, direction in [
            ("A4_real_to_fake_swap.csv", "real_to_fake"),
            ("A4_fake_to_real_swap.csv", "fake_to_real"),
        ]:
            path = directory / filename
            if not path.exists():
                continue
            with path.open(newline="", encoding="utf-8") as f:
                for record in csv.DictReader(f):
                    flip_rate = float(record["flip_rate"])
                    rows.append(
                        _row(
                            source=str(path.relative_to(results_dir.parent)),
                            seed=seed,
                            family="counterfactual_swap",
                            subset_size=1,
                            channels=record["concept"],
                            direction=direction,
                            generator=record["generator"],
                            n=record.get("n_correct_real")
                            or record.get("n_correct_fake")
                            or "",
                            metric="flip_rate",
                            value=flip_rate,
                            effect_pp=flip_rate * 100.0,
                            protocol="existing_frozen_counterfactual",
                            claim_scope="audit_intervention_not_retraining",
                        )
                    )
    return rows


def load_perturbation_rows(results_dir):
    rows = []
    batch_dir = results_dir / "batch1_analysis"
    for path, variant_field, family in [
        (batch_dir / "B3_jpeg_quality_curve.csv", "jpeg_q", "jpeg_quality_sweep"),
        (batch_dir / "B4_resolution_curve.csv", "resolution", "resolution_sweep"),
    ]:
        if not path.exists():
            continue
        baselines = {}
        with path.open(newline="", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
        for record in records:
            marker = record[variant_field]
            if marker in {"100", "native"}:
                baselines[record["generator"]] = float(record["acc"])
        for record in records:
            gen = record["generator"]
            acc = float(record["acc"])
            base = baselines.get(gen)
            delta = (acc - base) * 100.0 if base is not None else None
            rows.append(
                _row(
                    source=str(path.relative_to(results_dir.parent)),
                    seed="42",
                    family=family,
                    subset_size="input",
                    channels="input",
                    direction=f"{variant_field}={record[variant_field]}",
                    generator=gen,
                    metric="accuracy",
                    value=acc,
                    baseline_acc=base,
                    acc=acc,
                    delta_pp=delta,
                    effect_pp=delta,
                    real_acc=float(record["real_acc"]),
                    fake_acc=float(record["fake_acc"]),
                    protocol="existing_input_perturbation",
                    claim_scope="input_perturbation_not_retraining",
                )
            )
    return rows


def load_edelta_rows(results_dir):
    specs = [
        ("42", results_dir / "edelta" / "gate3_audit" / "gate3_full_audit.json"),
        ("1", results_dir / "edelta_seed1_audit" / "gate3_full_audit.json"),
        ("2", results_dir / "edelta_seed2_audit" / "gate3_full_audit.json"),
    ]
    rows = []
    for seed, path in specs:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        ablation = payload.get("a3_ablation_debiased", {})
        for concept, info in ablation.items():
            mean_acc = info.get("mean_acc")
            rows.append(
                _row(
                    source=str(path.relative_to(results_dir.parent)),
                    seed=seed,
                    family="edelta_single_drop",
                    subset_size=1,
                    channels=concept,
                    imputation="zero",
                    generator="__Q96_MEAN__",
                    metric="accuracy",
                    value=mean_acc,
                    acc=mean_acc,
                    delta_pp=info.get("delta_pp"),
                    effect_pp=info.get("delta_pp"),
                    protocol="existing_debiased_frozen_audit",
                    claim_scope="debiased_model_audit_not_retraining_matrix",
                    notes="E-Delta trained model audited after Q96 debiasing.",
                )
            )
            for gen, vals in info.get("per_gen", {}).items():
                rows.append(
                    _row(
                        source=str(path.relative_to(results_dir.parent)),
                        seed=seed,
                        family="edelta_single_drop",
                        subset_size=1,
                        channels=concept,
                        imputation="zero",
                        generator=gen,
                        n=vals.get("n"),
                        metric="accuracy",
                        value=vals.get("acc"),
                        acc=vals.get("acc"),
                        delta_pp=info.get("delta_pp"),
                        effect_pp=info.get("delta_pp"),
                        auc=vals.get("auc"),
                        real_acc=vals.get("real_acc"),
                        fake_acc=vals.get("fake_acc"),
                        protocol="existing_debiased_frozen_audit",
                        claim_scope="debiased_model_audit_not_retraining_matrix",
                        notes="Per-generator row inherits mean delta_pp from E-Delta audit JSON.",
                    )
                )
    return rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def write_summary_md(summary, path):
    lines = [
        "# Protocol C Frozen Audit Matrix",
        "",
        f"- Rows: {summary['n_rows']}",
        f"- Single-drop rows used for reliance summary: {summary['single_drop_rows']}",
        f"- Compression share (single drop): {summary['compression_share_single_drop']}",
        f"- Reliance concentration (single drop): {summary['reliance_concentration_single_drop']}",
        f"- Claim scope: {summary['claim_scope']}",
        "",
        "## Rows by Family",
        "",
        "| Family | Rows |",
        "|---|---:|",
    ]
    for family, count in sorted(summary["families"].items()):
        lines.append(f"| {family} | {count} |")
    lines.extend(
        [
            "",
            "This matrix is generated from frozen bottleneck dumps and existing perturbation/audit outputs.",
            "It is not a full retraining matrix.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_protocol_c_matrix(results_dir, out_dir):
    rows = []
    rows.extend(load_dump_rows(results_dir))
    rows.extend(load_counterfactual_rows(results_dir))
    rows.extend(load_perturbation_rows(results_dir))
    rows.extend(load_edelta_rows(results_dir))

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "protocol_c_frozen_audit_matrix.csv"
    summary_path = out_dir / "protocol_c_summary.json"
    md_path = out_dir / "protocol_c_summary.md"
    write_csv(rows, csv_path)
    summary = compute_reliance_summary(rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_md(summary, md_path)
    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="Results")
    parser.add_argument("--out-dir", default="Results/protocol_c_matrix")
    args = parser.parse_args()

    rows, summary = build_protocol_c_matrix(Path(args.results_dir), Path(args.out_dir))
    print(f"[DONE] wrote {summary['n_rows']} rows")
    print(f"[DONE] families: {summary['families']}")
    print(f"[DONE] out_dir: {args.out_dir}")


if __name__ == "__main__":
    main()
