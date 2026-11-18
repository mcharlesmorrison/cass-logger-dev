"""
Example: Tap test. How well the IMUs' sample times line up, and the file's size.

Record with the logger and its IMUs on a table, and hit the table a few times
(at the start, middle and end of the recording). Every IMU feels each hit at
the same instant, give or take ~0.1 ms for the shock to cross the table, so the
hits show how well the logger's timestamps for the different IMUs agree, and
whether that changes over the recording.

Sections
--------
1. ``timing_quality`` — each IMU's sample rate against nominal, sample-spacing
   jitter, gaps, FIFO events (overruns, sync losses) and when they happened.
2. ``find_taps`` / ``tap_offsets`` — finds each hit in every IMU's accel and
   measures each IMU's time offset from the internal IMU (S) two ways:
   - onset: when the hit first reaches 20% of its peak (interpolated between
     samples, so it doesn't depend on how hard each IMU felt it);
   - xcorr: the shift that best lines up the hit's shape (cross-correlation).
   Each IMU's digital low-pass filter delays its data by a fixed amount, so a
   constant offset of a millisecond or two between different parts (ICM-20948
   vs ICM-45686) is expected. What matters is that the offset stays the same
   from the first hit to the last (no drift), and that two identical IMUs
   (two ICM-45686s at the same rate) line up with each other.
3. ``size_breakdown`` — where the file's bytes go (RECORD and FIFO blocks per
   IMU), compared with the pre-0.13 format (one fixed-size record per 2 ms).

Usage
-----
::

    python examples/tap_test_ex.py                       # the bundled example
    python examples/tap_test_ex.py tmp_<unix time>       # a download folder
    python examples/tap_test_ex.py path/to/LOG_x.bin --save-dir plots

With a directory, the newest ``.bin`` in it is used. Without ``--save-dir``
the plots are shown.

The bundled example (``examples/data/0.14-auto_icm45686-x2_tap-test``) is
the first 9 s of a 7.7 min tap test: ICM-45686s on ports C and E, three hits
at 7–8 s. Drift needs hits spread over minutes: record your own, with hits at
the start, middle and end.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cass_logger_dev import stream_format as sf
from cass_logger_dev.firmware_structs import FIRMWARE_DTYPES
from download_and_inspect_ex import bad_stamps  # next to this script

TAP_MG = 500  # a hit: accel jumps by at least this much (mg) from the samples before
QUIET_S = 0.15  # a new hit needs this long below TAP_MG / 2 first
MATCH_S = 0.03  # one hit's onsets in different IMUs are within this of each other
BASELINE_S = 0.03  # the "before" level: the mean of the previous 30 ms of samples
ONSET_FRACTION = 0.2  # onset = first crossing of 20% of the hit's peak
XCORR_WINDOW_S = (-0.005, 0.025)  # around the onset
XCORR_STEP_S = 10e-6  # grid the hits are resampled onto for the cross-correlation
REF = "S"  # offsets are measured from the internal IMU
COLORS = {"S": "black", "C": "tab:orange", "E": "tab:purple"}


def section(title: str):
    print(f"\n── {title} " + "─" * max(0, 74 - len(title)))


# ── 1. Sample timing ──────────────────────────────────────────────────────────


def timing_quality(stream: sf.StreamFile, data: sf.StreamData):
    section("Recording")
    rec = data.records
    duration = rec["t"].iloc[-1] - rec["t"].iloc[0]
    print(f"  {stream.fw_version}, device {stream.device_id or '(not set)'}")
    print(f"  {duration:.1f} s ({duration / 60:.1f} min), {len(rec):,d} records")
    for port, info in stream.ports.items():
        print(f"  port {port}: {info.state_name} {info.device}")

    section("Sample timing")
    nominal = {info.port: info for info in stream.sources.values()}
    print(
        f"  {'imu':4s} {'sensor':6s} {'samples':>9s} {'rate Hz':>9s} {'vs nom':>7s} "
        f"{'dt med':>7s} {'p0.1':>6s} {'p99.9':>6s} {'max':>7s}  (ms)  timed"
    )
    for port, imu in data.imu.items():
        info = nominal.get(port)
        for name, nom in [("accel", info.accel_hz), ("gyro", info.gyro_hz)]:
            frame = getattr(imu, name)
            if len(frame) < 2:
                continue
            t_us = frame["tmicros"].to_numpy(np.int64)
            t_us = np.delete(t_us, bad_stamps(t_us))  # listed below
            dt = np.diff(t_us) / 1000
            rate = (len(t_us) - 1) / (t_us[-1] - t_us[0]) * 1e6
            timed = (frame["t_edge"] | frame["t_fit"]).mean()
            print(
                f"  {port:4s} {name:6s} {len(frame):9,d} {rate:9.1f} {rate / nom - 1:+7.1%} "
                f"{np.median(dt):7.3f} {np.percentile(dt, 0.1):6.3f} {np.percentile(dt, 99.9):6.3f} "
                f"{dt.max():7.2f}        {timed:.1%}"
            )
        a = imu.accel
        bad = bad_stamps(a["tmicros"].to_numpy(np.int64))
        for i in bad:
            print(
                f"       {port} accel sample {i:,d} stamped {a['t'].iloc[i]:.3f} s, between samples at "
                f"{a['t'].iloc[i - 1]:.3f} and {a['t'].iloc[i + 1]:.3f} s: a bad time, not a gap"
            )
        a = a.drop(a.index[bad])
        dt = np.diff(a["tmicros"].to_numpy(np.int64))
        period = np.median(dt)
        for g in np.flatnonzero(dt > 3 * period):
            lost = round(dt[g] / period) - 1
            print(
                f"       {port} accel gap at {a['t'].iloc[g]:.3f} s: {dt[g] / 1000:.1f} ms "
                f"with no samples (~{lost} samples lost)"
            )
        for _, ev in imu.events.iterrows():
            flags = [f for f in sf.FIFO_BLOCK_FLAGS.values() if f in ev and ev[f]]
            print(f"       {port} FIFO event at {ev['t']:.3f} s: {', '.join(flags)}")


# ── 2. Taps ───────────────────────────────────────────────────────────────────


def jump_mg(imu: sf.ImuData) -> tuple[np.ndarray, np.ndarray]:
    """|accel - mean of the previous BASELINE_S|, in mg, and its times (s).

    The baseline only uses samples from before each one, so a hit can't leak
    into the samples before it (as a centred filter would).
    """
    a = imu.accel
    t = a["t"].to_numpy()
    xyz = a[["x", "y", "z"]].to_numpy(float)
    rate = (len(t) - 1) / (t[-1] - t[0])
    w = max(2, int(BASELINE_S * rate))
    base = pd.DataFrame(xyz).rolling(w, min_periods=1).mean().shift(1).bfill().to_numpy()
    return t, np.linalg.norm(xyz - base, axis=1)


def hit_onsets(t: np.ndarray, m: np.ndarray) -> list[int]:
    """Index of the start of each hit: the first sample over TAP_MG / 2 after a
    quiet spell, if the jump then reaches TAP_MG within 10 ms."""
    onsets, last_loud = [], -np.inf
    for i in np.flatnonzero(m > TAP_MG / 2):
        if t[i] - last_loud > QUIET_S:
            end = np.searchsorted(t, t[i] + 0.01)
            if m[i:end].max() > TAP_MG:
                onsets.append(i)
        last_loud = t[i]
    return onsets


def refine_onset(t: np.ndarray, m: np.ndarray, i: int) -> tuple[float, float]:
    """(onset time, peak mg): the 20%-of-peak crossing, interpolated."""
    end = np.searchsorted(t, t[i] + 0.01)
    peak = m[i:end].max()
    level = ONSET_FRACTION * peak
    j = i
    while j > 0 and m[j - 1] >= level:
        j -= 1
    while m[j] < level:
        j += 1
    if j == 0:
        return t[0], peak
    frac = (level - m[j - 1]) / (m[j] - m[j - 1])
    return t[j - 1] + frac * (t[j] - t[j - 1]), peak


def find_taps(data: sf.StreamData) -> pd.DataFrame:
    """One row per hit that every IMU felt: each IMU's onset time and peak."""
    signals = {port: jump_mg(imu) for port, imu in data.imu.items()}
    found = []
    for port, (t, m) in signals.items():
        for i in hit_onsets(t, m):
            onset, peak = refine_onset(t, m, i)
            found.append((onset, port, peak))
    found.sort()

    taps, group = [], []
    for onset, port, peak in found + [(np.inf, "", 0)]:
        if group and onset - group[0][0] > MATCH_S:
            ports = {p for _, p, _ in group}
            if ports == set(signals):  # felt by every IMU, once each
                row = {}
                for o, p, pk in group:
                    if p not in row:
                        row[f"t_{p}"], row[f"peak_{p}"] = o, pk
                taps.append(row)
            group = []
        group.append((onset, port, peak))
    return pd.DataFrame(taps), signals


def xcorr_lag(ref: tuple, other: tuple, t0: float) -> float:
    """How much later (s) ``other``'s hit is than ``ref``'s, from their shapes."""
    grid = np.arange(t0 + XCORR_WINDOW_S[0], t0 + XCORR_WINDOW_S[1], XCORR_STEP_S)
    a = np.interp(grid, *ref)
    b = np.interp(grid, *other)
    a, b = (a - a.mean()) / a.std(), (b - b.mean()) / b.std()
    lags = np.arange(-len(grid) + 1, len(grid))
    c = np.correlate(b, a, mode="full")
    keep = np.abs(lags * XCORR_STEP_S) <= 0.005  # within ±5 ms
    k = lags[keep][np.argmax(c[keep])]
    return k * XCORR_STEP_S


def tap_offsets(taps: pd.DataFrame, signals: dict) -> pd.DataFrame:
    section("Taps")
    if taps.empty:
        print(f"  no hits over {TAP_MG} mg felt by every IMU")
        return taps
    ports = [p for p in signals if p != REF]
    pairs = [(p, REF) for p in ports] + ([(ports[0], ports[1])] if len(ports) == 2 else [])
    for a, b in pairs:
        taps[f"onset_{a}-{b}"] = (taps[f"t_{a}"] - taps[f"t_{b}"]) * 1000
        taps[f"xcorr_{a}-{b}"] = [
            xcorr_lag(signals[b], signals[a], t0) * 1000 for t0 in taps[f"t_{REF}"]
        ]

    head = "  ".join(f"{f'{a}-{b} onset':>10s} {'xcorr':>6s}" for a, b in pairs)
    print(f"  {len(taps)} hits felt by every IMU. Offsets in ms (positive: the first one later)")
    print(f"  {'t (s)':>8s}  {'peak g ' + '/'.join(signals):>16s}  {head}")
    for _, r in taps.iterrows():
        peaks = "/".join(f"{r[f'peak_{p}'] / 1000:.1f}" for p in signals)
        cols = "  ".join(f"{r[f'onset_{a}-{b}']:10.3f} {r[f'xcorr_{a}-{b}']:6.3f}" for a, b in pairs)
        print(f"  {r[f't_{REF}']:8.3f}  {peaks:>16s}  {cols}")

    section("Offsets: constant, and drifting?")
    span_min = (taps[f"t_{REF}"].iloc[-1] - taps[f"t_{REF}"].iloc[0]) / 60
    for a, b in pairs:
        for how in ("onset", "xcorr"):
            v = taps[f"{how}_{a}-{b}"].to_numpy()
            line = (
                f"  {a}-{b} {how:5s}  mean {v.mean():+7.3f} ms, spread (std) {v.std():6.3f} ms, "
                f"min {v.min():+7.3f}, max {v.max():+7.3f}"
            )
            if span_min >= 1:
                slope = np.polyfit(taps[f"t_{REF}"], v, 1)[0] * 60 * 1000
                line += f";  drift {slope:+6.1f} µs/min over {span_min:.1f} min"
            print(line)
    if span_min < 1:
        print(f"  (hits span {span_min * 60:.1f} s: too short to measure drift)")
    return taps


def plot_taps(name: str, taps: pd.DataFrame, signals: dict, data: sf.StreamData):
    plt.style.use("ggplot")
    figs = []

    fig, axs = plt.subplots(len(signals), 1, sharex=True, figsize=(13, 2.2 * len(signals) + 1))
    for ax, (port, (t, m)) in zip(axs, signals.items()):
        ax.plot(t, m / 1000, lw=0.6, color=COLORS.get(port))
        for t0 in taps.get(f"t_{port}", []):
            ax.axvline(t0, color="tab:red", lw=0.8, ls=":")
        for _, ev in data.imu[port].events.iterrows():
            ax.axvline(ev["t"], color="tab:blue", lw=1)
        ax.set_ylabel(f"{port} jump [g]")
    axs[0].set_title(f"{name}: accel jumps (red dotted: hits found; blue: FIFO events)")
    axs[-1].set_xlabel("time [s]")
    fig.tight_layout()
    figs.append(("overview", fig))

    if not taps.empty:
        n = len(taps)
        cols = min(n, 4)
        rows = int(np.ceil(n / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows), squeeze=False)
        for ax, (_, r) in zip(axs.flat, taps.iterrows()):
            t0 = r[f"t_{REF}"]
            for port, (t, m) in signals.items():
                sel = (t > t0 - 0.004) & (t < t0 + 0.012)
                ax.plot((t[sel] - t0) * 1000, m[sel] / r[f"peak_{port}"], "-o", ms=2.5, lw=1,
                        color=COLORS.get(port), label=port)
                ax.axvline((r[f"t_{port}"] - t0) * 1000, color=COLORS.get(port), lw=0.6, ls="--")
            ax.set_title(f"{t0:.2f} s", fontsize=9)
            ax.set_xlabel(f"ms from {REF} onset", fontsize=8)
            ax.tick_params(labelsize=7)
        for ax in axs.flat[n:]:
            ax.axis("off")
        axs[0, 0].set_ylabel("jump / its peak")
        axs[0, 0].legend(fontsize=7)
        fig.suptitle(f"{name}: each hit, every sample (dashed: onset at {ONSET_FRACTION:.0%} of peak)")
        fig.tight_layout()
        figs.append(("taps", fig))

        fig, ax = plt.subplots(figsize=(10, 4.5))
        for c in [c for c in taps.columns if c.startswith(("onset_", "xcorr_"))]:
            marker = "o" if c.startswith("onset") else "s"
            ax.plot(taps[f"t_{REF}"], taps[c], marker, ms=5, alpha=0.8, label=c.replace("_", " "))
        ax.set_xlabel("time of hit [s]")
        ax.set_ylabel("offset [ms]")
        ax.set_title(f"{name}: offsets between IMUs at each hit (flat = no drift)")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        figs.append(("offsets", fig))
    return figs


# ── 3. File size ──────────────────────────────────────────────────────────────


def size_breakdown(path: Path, stream: sf.StreamFile, data: sf.StreamData):
    section("File size")
    size = path.stat().st_size
    rec = data.records
    duration = rec["t"].iloc[-1] - rec["t"].iloc[0]
    ticks = len(rec)
    H, FH, E = sf.HEADER_LEN, sf.FIFO_HEADER.size, sf.FIFO_ENTRY_DTYPE.itemsize

    rows = [("RECORD (analog, 500 Hz)", ticks * (H + stream.record_len), ticks, None)]
    for src in sorted(stream.fifo):
        info = stream.sources[src]
        entries = stream.fifo[src]
        n_blocks = len(stream.blocks[src])
        kinds = pd.Series(entries["tag"] >> 3).map(
            {sf.TAG_ACCEL: "accel", sf.TAG_GYRO: "gyro", sf.TAG_MAG: "mag",
             sf.TAG_ACCEL_HG: "accel_hg", sf.TAG_TIMESTAMP: "timestamp"}).fillna("other")
        per_s = (kinds.value_counts() / duration).round().astype(int).to_dict()
        rows.append((f"FIFO {info.port} {info.device}", n_blocks * (H + FH) + len(entries) * E, n_blocks, per_s))
    other = size - sum(r[1] for r in rows)

    print(f"  {size:,d} bytes ({size / 1e6:.1f} MB) for {duration:.1f} s: {size / duration / 1e3:.1f} kB/s, "
          f"{size / duration * 3600 / 1e6:.0f} MB per hour")
    print(f"  {'':34s} {'bytes':>12s} {'share':>6s} {'per 2 ms':>9s}  readings per second")
    for label, nbytes, _, per_s in rows:
        readings = ", ".join(f"{k} {v:,d}" for k, v in sorted(per_s.items())) if per_s else ""
        print(f"  {label:34s} {nbytes:12,d} {nbytes / size:6.1%} {nbytes / ticks:8.1f}B  {readings}")
    print(f"  {'headers (FILE_HEADER, SOURCE_INFO)':34s} {other:12,d}")
    print(f"  {'total':34s} {size:12,d} {'':6s} {size / ticks:8.1f}B")
    print(
        f"  Every FIFO entry is {E} bytes: one 3-axis reading (6) + its own timestamp (4) + tag and"
        f" flags (2).\n  Every block adds a {H}-byte header, and a FIFO block {FH} more."
    )

    section("Compared with the pre-0.13 format")
    # Before 0.13 every 2 ms was one datrec: the analog channels plus every
    # IMU's accel, gyro and mag as float32, polled once per record.
    n_port_imus = sum(1 for info in stream.sources.values() if info.port in "CE")
    old_key = {0: "std", 1: "i2c_1", 2: "i2c_2"}[n_port_imus]
    old = FIRMWARE_DTYPES[old_key]().itemsize
    new = size / ticks
    print(f"  Same sensors in the old format ('{old_key}': {n_port_imus} port IMU(s) + the internal one):")
    print(f"  {'':22s} {'old':>10s} {'new':>10s} {'new/old':>8s}")
    for label, o, n in [
        ("per 2 ms", f"{old} B", f"{new:.0f} B"),
        ("per second", f"{old * 500 / 1e3:.1f} kB", f"{new * 500 / 1e3:.1f} kB"),
        ("per hour", f"{old * 500 * 3600 / 1e6:.0f} MB", f"{new * 500 * 3600 / 1e6:.0f} MB"),
        ("this recording", f"{old * ticks / 1e6:.1f} MB", f"{size / 1e6:.1f} MB"),
    ]:
        print(f"  {label:22s} {o:>10s} {n:>10s} {new / old:8.2f}x" if label == "per 2 ms" else
              f"  {label:22s} {o:>10s} {n:>10s}")
    imu_readings_old = 3 * (1 + n_port_imus) * 500  # accel, gyro, mag per IMU per record
    imu_readings_new = sum(len(stream.fifo[s]) for s in stream.fifo) / duration
    print(f"  IMU readings stored per second: old {imu_readings_old:,d}, new {imu_readings_new:,.0f} "
          f"({imu_readings_new / imu_readings_old:.2f}x)")
    old_per_reading = 12  # 3 x float32
    new_per_reading = sum(r[1] for r in rows[1:]) / sum(len(stream.fifo[s]) for s in stream.fifo)
    print(f"  Bytes per IMU reading: old {old_per_reading} (3 x float32, no timestamp), "
          f"new {new_per_reading:.1f} (entry + share of block headers)")


# ── Driver ────────────────────────────────────────────────────────────────────


def main(path: Path, save_dir: Path | None):
    if path.is_dir():
        path = max(path.glob("*.bin"), key=lambda p: p.stat().st_mtime)
    print(path)
    stream = sf.read_stream(path)
    data = sf.process_stream(stream)

    timing_quality(stream, data)
    taps, signals = find_taps(data)
    taps = tap_offsets(taps, signals)
    size_breakdown(path, stream, data)

    figs = plot_taps(path.name, taps, signals, data)
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        for short, fig in figs:
            out = save_dir / f"{path.stem}_{short}.png"
            fig.savefig(out, dpi=120)
            print(f"  saved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path(__file__).parent / "data" / "0.14-auto_icm45686-x2_tap-test",
        help=".bin file, or a folder (its newest .bin); default: the bundled example",
    )
    parser.add_argument("--save-dir", type=Path, help="save the plots here instead of showing them")
    args = parser.parse_args()
    main(args.path, args.save_dir)
