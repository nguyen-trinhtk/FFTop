#!/usr/bin/env python3
"""Load bench/config/default.yaml, measure the variant union, plot each comparison."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CONFIG = HERE / "config" / "default.yaml"

SCHEMA = 3
SPEC_COLUMNS = (
    "backend,traversal,radix,simd,execution,threads,size"
)
_CPU_FIELDS = (
    "backend", "traversal", "radix", "simd",
    "execution", "threads",
)
Scale = Literal["log", "linear"]


def parse_scale(value: str) -> Scale:
    key = value.lower()
    if key in ("linear", "scalar", "lin"):
        return "linear"
    if key in ("log", "log2"):
        return "log"
    raise ValueError(f"scale must be log or linear/scalar, got {value!r}")


@dataclass(frozen=True)
class Host:
    cpu_threads: int = 1
    simd_kernel: str = "scalar"
    openmp: bool = False
    cuda: bool = False


@dataclass(frozen=True)
class Variant:
    backend: str = "CPU"
    traversal: str = "iterative"
    radix: str = "2"
    simd: str = "scalar"
    execution: str = "serial"
    threads: int | str = 1
    label: str | None = None

    def series_name(self) -> str:
        if self.label:
            return self.label
        if self.backend == "naive-dft":
            return "naive DFT"
        parts = [self.traversal, f"radix-{self.radix}", self.simd]
        if self.execution == "parallel":
            parts.append(f"omp/{self.threads}")
        return " / ".join(p for p in parts if p not in ("", "-"))


@dataclass(frozen=True)
class Comparison:
    id: str
    title: str
    question: str
    variants: tuple[Variant, ...]
    min_k: int = 6
    max_k: int = 24
    gates: tuple[str, ...] = ()
    plot: str = "time_vs_n"
    thread_sweep: bool = False
    hold_fixed: bool = True
    xscale: str = "log"
    yscale: str = "linear"

    def __post_init__(self) -> None:
        object.__setattr__(self, "xscale", parse_scale(self.xscale))
        object.__setattr__(self, "yscale", parse_scale(self.yscale))


@dataclass(frozen=True)
class Job:
    variant: Variant
    size: int
    label: str

    def spec_line(self) -> str:
        v = self.variant
        return ",".join((
            v.backend, v.traversal, v.radix, v.simd, v.execution,
            str(v.threads), str(self.size),
        ))


def _variant(data: dict, base: dict) -> Variant:
    merged = {**base, **data}
    radix = merged.get("radix", "2")
    threads = merged.get("threads", 1)
    return Variant(
        backend=str(merged.get("backend", "CPU")),
        traversal=str(merged.get("traversal", "iterative")),
        radix="-" if radix is None else str(radix),
        simd=str(merged.get("simd", "scalar")),
        execution=str(merged.get("execution", "serial")),
        threads=threads,
        label=merged.get("label"),
    )


def load_config(path: Path) -> tuple[Comparison, ...]:
    raw = yaml.safe_load(path.read_text()) or {}
    defaults = raw.get("defaults") or {}
    base = defaults.get("variant") or {}
    named = {
        name: _variant(spec or {}, base)
        for name, spec in (raw.get("variants") or {}).items()
    }
    out = []
    for row in raw.get("comparisons") or []:
        try:
            variants = tuple(named[name] for name in row["variants"])
        except KeyError as exc:
            raise KeyError(f"{row.get('id')}: unknown variant {exc}") from exc
        out.append(Comparison(
            id=row["id"],
            title=row["title"],
            question=row.get("question", ""),
            variants=variants,
            min_k=int(row.get("min_k", defaults.get("min_k", 6))),
            max_k=int(row.get("max_k", defaults.get("max_k", 24))),
            gates=tuple(row.get("gates") or ()),
            plot=row.get("plot", defaults.get("plot", "time_vs_n")),
            thread_sweep=bool(row.get("thread_sweep", False)),
            hold_fixed=bool(row.get("hold_fixed", defaults.get("hold_fixed", True))),
            xscale=row.get("xscale", defaults.get("xscale", "log")),
            yscale=row.get("yscale", defaults.get("yscale", "linear")),
        ))
    comparisons = tuple(out)
    check_comparisons(comparisons)
    return comparisons


def is_power_of(n: int, base: int) -> bool:
    if n == 0 or base < 2:
        return False
    while n % base == 0:
        n //= base
    return n == 1


def thread_counts(host_threads: int) -> list[int]:
    out = [1]
    t = 2
    while t <= host_threads:
        out.append(t)
        t *= 2
    if host_threads not in out:
        out.append(host_threads)
    return out


def _vary_axes(variants: Iterable[Variant]) -> tuple[str, ...]:
    variants = tuple(variants)
    if len(variants) < 2:
        return ()
    backends = {v.backend for v in variants}
    if backends == {"GPU"}:
        return tuple(
            field for field in ("traversal",)
            if len({getattr(v, field) for v in variants}) > 1
        )
    if any(v.backend != "CPU" for v in variants):
        return ("backend",) if len(backends) > 1 else ()
    return tuple(
        field for field in _CPU_FIELDS
        if len({getattr(v, field) for v in variants}) > 1
    )


def check_comparisons(comparisons: Iterable[Comparison]) -> None:
    seen = set()
    for cmp in comparisons:
        if cmp.id in seen:
            raise ValueError(f"duplicate comparison id {cmp.id}")
        seen.add(cmp.id)
        if not cmp.variants:
            raise ValueError(f"{cmp.id}: no variants")
        if cmp.hold_fixed and not cmp.thread_sweep:
            axes = _vary_axes(cmp.variants)
            if len(axes) > 1:
                raise ValueError(f"{cmp.id}: varies {axes}, expected one axis")


def comparison_by_id(cmp_id: str, comparisons: Iterable[Comparison]) -> Comparison:
    for cmp in comparisons:
        if cmp.id == cmp_id:
            return cmp
    known = ", ".join(c.id for c in comparisons)
    raise KeyError(f"unknown comparison {cmp_id!r} (have {known})")


def expand_variants(cmp: Comparison, host: Host) -> tuple[Variant, ...]:
    if not cmp.thread_sweep:
        return cmp.variants
    out = []
    for base in cmp.variants:
        for t in thread_counts(host.cpu_threads):
            out.append(replace(
                base,
                execution="serial" if t == 1 else "parallel",
                threads=t,
                label="serial" if t == 1 else f"{t} threads",
            ))
    return tuple(out)


def resolve(variant: Variant, host: Host, n: int) -> Variant:
    simd = host.simd_kernel if variant.simd == "host" else variant.simd
    threads = host.cpu_threads if variant.threads == "host" else int(variant.threads)
    execution = variant.execution
    if execution == "parallel" and not host.openmp:
        execution = "serial"
        threads = 1
    radix = variant.radix
    if radix == "auto":
        radix = "4" if is_power_of(n, 4) else "2"
    return replace(variant, simd=simd, threads=threads, execution=execution, radix=radix)


def host_ok(cmp: Comparison, host: Host) -> str | None:
    if "needs_simd" in cmp.gates and host.simd_kernel == "scalar":
        return "no SIMD kernel on this host"
    if "needs_omp" in cmp.gates and not host.openmp:
        return "OpenMP not enabled"
    if "needs_gpu" in cmp.gates and not host.cuda:
        return "no CUDA device"
    return None


def size_ok(cmp: Comparison, variant: Variant, n: int) -> bool:
    if "power_of_4" in cmp.gates and not is_power_of(n, 4):
        return False
    if variant.radix == "4" and not is_power_of(n, 4):
        return False
    if variant.execution == "parallel" and n < 1024:
        return False
    return True


def jobs_for(host: Host, comparisons: Iterable[Comparison]) -> list[Job]:
    check_comparisons(comparisons)
    unique: dict[str, Job] = {}
    for cmp in comparisons:
        if host_ok(cmp, host):
            continue
        for variant in expand_variants(cmp, host):
            for k in range(cmp.min_k, cmp.max_k + 1):
                n = 1 << k
                if not size_ok(cmp, variant, n):
                    continue
                resolved = resolve(variant, host, n)
                job = Job(variant=resolved, size=n, label=resolved.series_name())
                unique[job.spec_line()] = job
    return list(unique.values())


def load_rows(path: str | Path | None) -> list[dict[str, str]]:
    source = open(path, newline="") if path else sys.stdin
    try:
        lines = [line for line in source if line.strip() and not line.startswith("#")]
        return list(csv.DictReader(lines))
    finally:
        if path:
            source.close()


def host_from_system_file(path: Path | None) -> Host | None:
    if path is None or not path.is_file():
        return None
    return host_from_dict(json.loads(path.read_text()))


def host_from_dict(data: dict) -> Host:
    return Host(
        cpu_threads=int(data.get("cpu_threads", 1)),
        simd_kernel=str(data.get("simd", data.get("simd_kernel", "scalar"))),
        openmp=bool(data.get("openmp", False)),
        cuda=bool(data.get("cuda", False)),
    )


def _norm(row: dict[str, str], key: str, default: str = "-") -> str:
    value = row.get(key, default)
    return default if value in ("", None) else value


def row_matches(row: dict[str, str], variant: Variant) -> bool:
    if _norm(row, "backend") != variant.backend:
        return False
    if variant.backend == "naive-dft":
        return True
    return (
        _norm(row, "traversal") == variant.traversal
        and _norm(row, "radix") == str(variant.radix)
        and _norm(row, "simd") == variant.simd
        and _norm(row, "execution") == variant.execution
        and int(row.get("threads") or 1) == int(variant.threads)
    )


def infer_host(rows: list[dict[str, str]]) -> Host:
    simd, threads, openmp, cuda = "scalar", 1, False, False
    for row in rows:
        value = _norm(row, "simd", "scalar")
        if value not in ("scalar", "-"):
            simd = value
        threads = max(threads, int(row.get("threads") or 1))
        if _norm(row, "execution") == "parallel":
            openmp = True
        if _norm(row, "backend") == "GPU":
            cuda = True
    return Host(cpu_threads=threads, simd_kernel=simd, openmp=openmp, cuda=cuda)


def series_for(rows, cmp: Comparison, host: Host, y_key: str = "ms") -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    variants = expand_variants(cmp, host)
    for row in rows:
        n = int(row["size"])
        k = n.bit_length() - 1
        if k < cmp.min_k or k > cmp.max_k:
            continue
        if y_key not in row:
            continue
        for variant in variants:
            if not size_ok(cmp, variant, n):
                continue
            resolved = resolve(variant, host, n)
            if not row_matches(row, resolved):
                continue
            series[resolved.series_name()].append((n, float(row[y_key])))
            break
    for points in series.values():
        points.sort()
    return {k: v for k, v in series.items() if v}


def series_order(cmp: Comparison, host: Host, series: dict) -> list[str]:
    order = []
    for variant in expand_variants(cmp, host):
        name = variant.series_name()
        if name in series and name not in order:
            order.append(name)
    for name in series:
        if name not in order:
            order.append(name)
    return order


def apply_scale(ax, axis: str, scale: str) -> None:
    if parse_scale(scale) != "log":
        return
    if axis == "x":
        ax.set_xscale("log", base=2)
    else:
        ax.set_yscale("log")


def plot_time_vs_n(series, order, output: Path, title: str, xscale: str, yscale: str,
                   ylabel: str = "time (ms)") -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in order:
        points = series[name]
        ax.plot([n for n, _ in points], [y for _, y in points], marker="o", label=name)
    apply_scale(ax, "x", xscale)
    apply_scale(ax, "y", yscale)
    ax.set_xlabel("N")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_speedup(series, output: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    baseline = series.get("serial")
    if not baseline:
        print("skip speedup: no serial series", file=sys.stderr)
        return
    base = dict(baseline)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, points in series.items():
        if name == "serial":
            continue
        xs, ys = [], []
        for n, ms in points:
            if n in base and ms > 0:
                xs.append(n)
                ys.append(base[n] / ms)
        if xs:
            ax.plot(xs, ys, marker="o", label=name)
    apply_scale(ax, "x", "log")
    ax.set_xlabel("N")
    ax.set_ylabel("speedup vs serial")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


GPU_METRICS = (
    ("ms", "Runtime (ms)", "time (ms)", "linear"),
    ("effective_bandwidth_gbps", "Effective bandwidth (GB/s)", "GB/s", "linear"),
    ("throughput_gsamples_s", "Throughput (Gsamples/s)", "Gsamples/s", "linear"),
    ("global_mem_bytes", "Estimated global memory traffic", "bytes", "log"),
)


def plot_gpu_metrics(rows, cmp: Comparison, out_dir: Path, host: Host) -> list[Path]:
    import matplotlib.pyplot as plt

    written = []
    panels = []
    for key, title, ylabel, yscale in GPU_METRICS:
        series = series_for(rows, cmp, host, y_key=key)
        if not series:
            print(f"skip plot {cmp.id} {key}: no matching rows", file=sys.stderr)
            continue
        order = series_order(cmp, host, series)
        path = out_dir / f"{cmp.id}_{key}.png"
        plot_time_vs_n(series, order, path, f"{cmp.title}: {title}", cmp.xscale, yscale, ylabel)
        written.append(path)
        panels.append((series, order, title, ylabel, yscale))

    if len(panels) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for ax, (series, order, title, ylabel, yscale) in zip(axes.ravel(), panels):
            for name in order:
                points = series[name]
                ax.plot([n for n, _ in points], [y for _, y in points], marker="o", label=name)
            apply_scale(ax, "x", cmp.xscale)
            apply_scale(ax, "y", yscale)
            ax.set_title(title)
            ax.set_xlabel("N")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", linestyle=":", linewidth=0.6)
        axes[0, 0].legend()
        combined = out_dir / f"{cmp.id}_metrics.png"
        fig.savefig(combined, dpi=150)
        plt.close(fig)
        written.append(combined)
    return written


def plot_comparison(rows, cmp: Comparison, out_dir: Path, host: Host | None = None) -> list[Path]:
    host = host or infer_host(rows)
    if cmp.plot == "gpu_metrics":
        return plot_gpu_metrics(rows, cmp, out_dir, host)
    series = series_for(rows, cmp, host)
    if not series:
        print(f"skip plot {cmp.id}: no matching rows", file=sys.stderr)
        return []
    order = series_order(cmp, host, series)
    written = []
    time_path = out_dir / f"{cmp.id}.png"
    plot_time_vs_n(series, order, time_path, cmp.title, cmp.xscale, cmp.yscale)
    written.append(time_path)
    if cmp.plot == "openmp":
        speed_path = out_dir / f"{cmp.id}_speedup.png"
        plot_speedup(series, speed_path, cmp.title + " (speedup)")
        if speed_path.is_file():
            written.append(speed_path)
    return written


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def read_system(bench: Path) -> dict:
    proc = subprocess.run(
        [str(bench), "--system"], check=True, capture_output=True, text=True,
    )
    return json.loads(proc.stdout)


def measure(bench: Path, jobs_path: Path, csv_path: Path) -> None:
    with jobs_path.open() as stdin, csv_path.open("w") as stdout:
        proc = subprocess.run([str(bench)], stdin=stdin, stdout=stdout, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"fftop_bench exited {proc.returncode}")


def select_comparisons(all_cmps: tuple[Comparison, ...], args) -> tuple[Comparison, ...]:
    comparisons = all_cmps
    if args.only:
        comparisons = tuple(comparison_by_id(i, all_cmps) for i in args.only)
    if args.xscale or args.yscale:
        comparisons = tuple(
            replace(c, xscale=args.xscale or c.xscale, yscale=args.yscale or c.yscale)
            for c in comparisons
        )
    return comparisons


def write_plots(csv_path: Path, out_dir: Path, comparisons: tuple[Comparison, ...]) -> None:
    rows = load_rows(csv_path)
    host = host_from_system_file(out_dir / "system.json")
    for cmp in comparisons:
        for path in plot_comparison(rows, cmp, out_dir, host):
            print(f"wrote {path}")


def self_check(comparisons: tuple[Comparison, ...]) -> None:
    check_comparisons(comparisons)
    if parse_scale("scalar") != "linear" or parse_scale("log2") != "log":
        raise AssertionError("scale aliases")
    by_id = {c.id: c for c in comparisons}
    if "scalar-vs-simd" not in by_id:
        host = Host(cpu_threads=8, simd_kernel="neon", openmp=True, cuda=True)
        jobs = jobs_for(host, comparisons)
        if not jobs:
            raise AssertionError("expected GPU jobs on a cuda host")
        none = Host(cpu_threads=1, simd_kernel="scalar", openmp=False, cuda=False)
        skipped = {c.id for c in comparisons if host_ok(c, none)}
        if skipped != {c.id for c in comparisons if "needs_gpu" in c.gates}:
            raise AssertionError(f"unexpected skips on CPU host: {skipped}")
        return
    first = comparisons[0]
    if first.xscale != "log" or first.yscale != "linear":
        raise AssertionError("default axis scales")
    host = Host(cpu_threads=8, simd_kernel="neon", openmp=True, cuda=True)
    jobs = jobs_for(host, comparisons)
    if not jobs:
        raise AssertionError("expected jobs on a neon/8/omp host")
    lines = [j.spec_line() for j in jobs]
    if len(lines) != len(set(lines)):
        raise AssertionError("duplicate spec lines")
    none = Host(cpu_threads=1, simd_kernel="scalar", openmp=False)
    skipped = {c.id for c in comparisons if host_ok(c, none)}
    if skipped != {"scalar-vs-simd", "serial-vs-openmp"}:
        raise AssertionError(f"unexpected skips on scalar host: {skipped}")
    if any(not is_power_of(j.size, 4) for j in jobs if j.variant.radix == "4"):
        raise AssertionError("radix-4 job on a non 4^p size")
    check_comparisons([Comparison(
        id="one-axis", title="x", question="x",
        variants=by_id["scalar-vs-simd"].variants,
    )])
    try:
        check_comparisons([Comparison(
            id="two-axes", title="x", question="x",
            variants=by_id["conclusion"].variants,
        )])
    except ValueError:
        pass
    else:
        raise AssertionError("expected two-axis comparison to fail")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("-o", "--output-dir", type=Path)
    parser.add_argument("--only", action="append", default=[], metavar="ID")
    parser.add_argument("--bench", type=Path, default=ROOT / "build/bench/fftop_bench")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--plot", type=Path, metavar="CSV", help="replot this CSV")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--xscale", help="log | linear | scalar")
    parser.add_argument("--yscale", help="log | linear | scalar")
    args = parser.parse_args()

    comparisons = load_config(args.config)
    if args.check:
        self_check(comparisons)
        host = Host(cpu_threads=8, simd_kernel="neon", openmp=True, cuda=True)
        jobs = jobs_for(host, comparisons)
        print(f"{len(jobs)} jobs on neon/8/omp")
        for cmp in comparisons:
            n = sum(
                1
                for variant in expand_variants(cmp, host)
                for k in range(cmp.min_k, cmp.max_k + 1)
                if size_ok(cmp, variant, 1 << k)
            )
            print(f"  {cmp.id}: {n} points")
        print("ok")
        return

    comparisons = select_comparisons(comparisons, args)

    if args.plot:
        csv_path = args.plot
        out_dir = args.output_dir or csv_path.resolve().parent
        write_plots(csv_path, out_dir, comparisons)
        return

    if args.plot_only:
        out_dir = args.output_dir or ROOT / "log" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        csv_path = out_dir / "bench.csv"
        if not csv_path.is_file():
            raise SystemExit(f"no {csv_path}")
        write_plots(csv_path, out_dir, comparisons)
        return

    if not args.dry_run and not args.bench.is_file():
        raise SystemExit(f"missing {args.bench} (build fftop_bench first)")

    out_dir = args.output_dir or ROOT / "log" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        host = Host(cpu_threads=8, simd_kernel="neon", openmp=True, cuda=True)
        system = {
            "schema": SCHEMA,
            "note": "dry-run host fixture",
            "cpu_threads": host.cpu_threads,
            "simd": host.simd_kernel,
            "openmp": host.openmp,
        }
    else:
        system = read_system(args.bench)
        host = host_from_dict(system)

    system["schema"] = SCHEMA
    system["git"] = git_sha()
    system["bench"] = str(args.bench)
    system["config"] = str(args.config)

    jobs = jobs_for(host, comparisons)
    if not jobs and any("needs_gpu" in c.gates for c in comparisons):
        raise SystemExit(
            f"0 GPU jobs (cuda={getattr(host, 'cuda', False)}). "
            "Build with -DFFTOP_ENABLE_CUDA=ON on a machine with a CUDA device."
        )
    (out_dir / "system.json").write_text(json.dumps(system, indent=2) + "\n")
    shutil.copy2(args.config, out_dir / "config.yaml")
    (out_dir / "jobs.txt").write_text(
        "# " + SPEC_COLUMNS + "\n" + "\n".join(j.spec_line() for j in jobs) + "\n"
    )

    for cmp in comparisons:
        reason = host_ok(cmp, host)
        if reason:
            print(f"skip comparison {cmp.id}: {reason}", file=sys.stderr)
    print(f"{len(jobs)} jobs → {out_dir}", file=sys.stderr)

    if args.dry_run:
        for job in jobs:
            print(job.spec_line())
        return

    csv_path = out_dir / "bench.csv"
    measure(args.bench, out_dir / "jobs.txt", csv_path)
    if not args.no_plot:
        write_plots(csv_path, out_dir, comparisons)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
