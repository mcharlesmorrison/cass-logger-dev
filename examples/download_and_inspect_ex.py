"""
Example: Download every file from a Cass Logger and check it over.

A bench tool for new firmware and new sensor setups. For each ``.bin`` file it
prints what the file says about itself, walks its first blocks byte by byte,
runs sanity checks on every channel, and plots everything.

Workflow
--------
1. ``download`` — reads the device's firmware version, ID and file list,
   downloads every file with ``CassCommands.download_all()``, and checks each
   downloaded file is exactly the size the device reported.
2. ``print_file_header`` / ``print_sources`` — the FILE_HEADER (firmware,
   device ID, record layout, analog channel mask, what is on ports C and E)
   and one SOURCE_INFO per IMU (part, port, sensors, rates, scales).
3. ``dump_blocks`` — the first few blocks of the file decoded by hand with
   ``struct``, to show how the format is laid out.
4. ``check_ports`` / ``check_records`` / ``check_analog`` / ``check_imu`` —
   ports C and E against what was recorded (an IMU with a broken I2C wire
   recorded as analog: one line pinned high), record rate and gaps, analog
   channel ranges, each IMU's sample rates against nominal, timing quality,
   FIFO events, gravity and gyro bias.
5. ``plot_file`` — one figure per IMU (accel, gyro, mag), one for the analog
   channels, and one for sample timing.

Files from before firmware 0.13 (raw records, no header) are checked too,
using the ``fw_ver`` from the download's ``metadata.txt``; steps 2 and 3 don't
apply to them.

Ports
-----
- ``S`` : the logger's internal ICM-20948 (SPI). Every logger has it.
- ``C``, ``E`` : the two external ports that can take an I2C IMU (ICM-45686 or
  LSM6DSV320X), or 3 analog channels. From 0.14 the logger detects which.
- ``A``, ``B``, ``D``, ``F`` : analog-only ports, 3 channels each
  (``a0``–``a2``, ...), raw 12-bit ADC counts.

The file format is specified in ``docs/stream-format.md`` in cass-logger-fw;
the parser is ``cass_logger_dev/stream_format.py``.

Usage
-----
Edit the ``if __name__ == "__main__":`` block at the bottom, then run::

    python examples/download_and_inspect_ex.py

- ``file_dir = download(cass)`` downloads from the connected logger; swap it
  for ``file_dir = Path("./tmp_example")`` to inspect a folder of ``.bin``
  files (or one file) already on disk, with no logger needed.
- ``inspect_path(...)`` options: ``fw_ver`` for pre-0.13 files without a
  metadata.txt, ``n_blocks`` to decode, ``plots``, ``save_dir`` for PNGs.
- Uncomment ``delete_all(cass)`` to wipe the logger's SD card afterwards.

The other ``CassCommands`` calls (``list_files``, ``list_file_sizes``,
``get_fw_ver``, ``get_device_ID``, ...) work as before and can be added there.
"""

import struct
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cass_logger_dev import stream_format as sf
from cass_logger_dev.cass_commands import CassCommands
from cass_logger_dev.firmware_structs import FIRMWARE_DTYPES
from cass_logger_dev.stream_format import ImuData, StreamData, StreamFile

ADC_MAX = 4095  # 12-bit analog channels
RATE_TOLERANCE = 0.05  # IMU clocks run up to ~3% off nominal
PORT_NAMES = {"S": "internal IMU (S)", "C": "port C", "E": "port E"}
SOURCE_NAMES = {0: "logger", 1: "port C", 2: "port E", 3: "internal (S)"}
BLOCK_NAMES = {
    sf.BLOCK_FILE_HEADER: "FILE_HEADER",
    sf.BLOCK_SOURCE_INFO: "SOURCE_INFO",
    sf.BLOCK_RECORD: "RECORD",
    sf.BLOCK_FIFO: "FIFO",
}
SENSOR_CODES = {
    sf.TAG_GYRO: "gyro",
    sf.TAG_ACCEL: "accel",
    sf.TAG_TIMESTAMP: "timestamp",
    sf.TAG_MAG: "mag",
    sf.TAG_MAG_NACK: "mag_nack",
    sf.TAG_ACCEL_HG: "accel_hg",
}
# datrec field n (analog_mask bit n), from firmware 0.14: the 18-channel layout.
ANALOG_18_FIELDS = [n for n in FIRMWARE_DTYPES["analog_18"]().names if n != "tmicros"]


class Report:
    """Collects one file's check results so they can be summarised at the end."""

    def __init__(self, name: str):
        self.name = name
        self.warnings: list[str] = []

    def ok(self, msg: str):
        print(f"  [ OK ] {msg}")

    def info(self, msg: str):
        print(f"  [info] {msg}")

    def warn(self, msg: str):
        print(f"  [WARN] {msg}")
        self.warnings.append(msg)


def section(title: str):
    print(f"\n── {title} " + "─" * max(0, 74 - len(title)))


# ── 1. Download ───────────────────────────────────────────────────────────────


def download(cass: CassCommands) -> Path:
    """Download every file on the logger and check each one's size."""
    section("Device")
    print(f"  firmware   {cass.get_fw_ver()}")
    print(f"  device ID  {cass.get_device_ID()}")
    names = cass.list_files()
    sizes = cass.list_file_sizes()
    if not names:
        raise SystemExit("No files on the device.")
    for name, size in zip(names, sizes):
        print(f"  {name:24s} {size:>12,d} bytes")

    section("Download")
    out_dir = Path(cass.download_all())
    report = Report("download")
    for name, size in zip(names, sizes):
        local = out_dir / name
        got = local.stat().st_size if local.exists() else -1
        if got == size:
            report.ok(f"{name}: {got:,d} bytes, matches the device")
        else:
            report.warn(f"{name}: {got:,d} bytes on disk, device said {size:,d}")
    print(f"  saved to {out_dir.resolve()}")
    return out_dir


# ── 2. FILE_HEADER and SOURCE_INFO ────────────────────────────────────────────


def _channels(mask: int) -> str:
    return (
        ", ".join(f for bit, f in enumerate(ANALOG_18_FIELDS) if mask >> bit & 1)
        or "none"
    )


def print_file_header(path: Path, stream: StreamFile):
    """The FILE_HEADER block: the first block of every 0.13+ file."""
    section("FILE_HEADER")
    with open(path, "rb") as f:
        head = f.read(sf.HEADER_LEN + sf.FILE_HEADER_MAX)
    n_sources = sf.FILE_HEADER.unpack_from(head, sf.HEADER_LEN)[-1]
    print(f"  format_version   {stream.format_version}  (3 = firmware 0.13, 4 = 0.14)")
    print(f"  fw_version       {stream.fw_version}")
    print(f"  device_id        {stream.device_id or '(not set)'}")
    print(f"  record_rate_hz   {stream.record_rate_hz}")
    print(f"  record_len       {stream.record_len} bytes per RECORD")
    print(f"  n_sources        {n_sources} SOURCE_INFO blocks follow")
    if stream.analog_mask is not None:
        print(
            f"  analog_mask      0x{stream.analog_mask:05X}  real channels: {_channels(stream.analog_mask)}"
        )
        print(
            f"  pullup_mask      0x{stream.pullup_mask:05X}  pull-ups on: {_channels(stream.pullup_mask)}"
        )
        for port, info in stream.ports.items():
            if info.device_type:
                what = f"{info.device} at 0x{info.i2c_addr:02X}"
            elif info.i2c_addr:
                what = f"unknown I2C device at 0x{info.i2c_addr:02X}"
            elif info.state_name == "fault":
                what = (
                    "I2C pull-ups but no device answered (an IMU with a broken wire?)"
                )
            else:
                what = "no I2C device"
            print(
                f"  port_{port.lower()}           state {info.state} ({info.state_name}), {what}"
            )
    else:
        print(
            "  (format 3: no analog_mask or port info; the variant in fw_version says what's on the ports)"
        )


def print_sources(stream: StreamFile):
    """One SOURCE_INFO block per IMU: what it is and how to scale its data."""
    section("SOURCE_INFO")
    for src, info in sorted(stream.sources.items()):
        sensors = [
            name for bit, name in sf.IMU_SENSOR_NAMES.items() if info.sensors & bit
        ]
        print(
            f"  source {src} = {SOURCE_NAMES.get(src, '?')}: {info.device}, port '{info.port}', "
            f"i2c 0x{info.i2c_addr:02X}, sensors 0x{info.sensors:02X} ({', '.join(sensors)})"
        )
        for name, hz, scale, unit in [
            ("accel", info.accel_hz, info.accel_mg_per_lsb, "mg/LSB"),
            ("gyro", info.gyro_hz, info.gyro_mdps_per_lsb, "mdps/LSB"),
            ("mag", info.mag_hz, info.mag_mG_per_lsb, "mG/LSB"),
            ("accel_hg", info.hg_hz, info.hg_mg_per_lsb, "mg/LSB"),
        ]:
            if hz:
                print(f"      {name:9s} {hz:7.1f} Hz nominal, {scale:8.4f} {unit}")


# ── 3. Raw block walk ─────────────────────────────────────────────────────────


def dump_blocks(path: Path, n_blocks: int, records_to_show: int = 2):
    """Decode the file's first blocks by hand, to show the layout.

    Every block is a 5-byte header (sync 0xA5, type, source, uint16 len)
    followed by ``len`` payload bytes. Only the first ``records_to_show``
    RECORD blocks are printed, so FIFO blocks from several ticks show up.
    """
    section(f"First blocks (of {path.name})")
    buf = path.read_bytes()
    i = shown = records = 0
    while shown < n_blocks and i + sf.HEADER_LEN <= len(buf):
        sync, btype, src, length = struct.unpack_from("<BBBH", buf, i)
        if sync != sf.SYNC:
            print(f"  @{i:<7d} no sync byte here (0x{sync:02X}): corrupt, stopping")
            return
        p = i + sf.HEADER_LEN
        payload = buf[p : p + length]
        name = BLOCK_NAMES.get(btype, f"type 0x{btype:02X}")
        records += btype == sf.BLOCK_RECORD
        if btype != sf.BLOCK_RECORD or records <= records_to_show:
            hdr = buf[i:p].hex(" ")
            print(
                f"  @{i:<7d} [{hdr}]  {name:11s} source {src} ({SOURCE_NAMES.get(src, '?')}), {length} bytes"
            )
            for line in _describe_payload(btype, payload):
                print(f"  {'':8s}   {line}")
            shown += 1
        i = p + length


def _describe_payload(btype: int, payload: bytes) -> list[str]:
    if btype == sf.BLOCK_FILE_HEADER:
        magic, fmt, fw, dev, rate, rec_len, n_src = sf.FILE_HEADER.unpack_from(payload)
        lines = [
            f"magic {magic!r}, format {fmt}, fw {fw.rstrip(b'\0').decode()!r}, "
            f"device {dev.rstrip(b'\0').decode()!r}, {rate} Hz, record_len {rec_len}, {n_src} sources"
        ]
        if len(payload) > sf.FILE_HEADER.size:
            mask, pullups, *ports = sf.FILE_HEADER_V4_EXTRA.unpack_from(
                payload, sf.FILE_HEADER.size
            )
            lines.append(
                f"analog_mask 0x{mask:05X}, pullup_mask 0x{pullups:05X}, "
                f"port C (state, device, addr) {tuple(ports[:3])}, port E {tuple(ports[3:])}"
            )
        return lines
    if btype == sf.BLOCK_SOURCE_INFO:
        kind, dev, port = sf.SOURCE_HEADER.unpack_from(payload)
        addr, sensors = sf.IMU_INFO.unpack_from(payload, sf.SOURCE_HEADER.size)
        pairs = [
            sf.IMU_SENSOR_PAIR.unpack_from(
                payload, sf.SOURCE_HEADER.size + sf.IMU_INFO.size + 8 * k
            )
            for k in range(bin(sensors).count("1"))
        ]
        return [
            f"kind {kind}, device_type {dev} ({sf.device_name(dev)}), port {port.decode()!r}, "
            f"i2c 0x{addr:02X}, sensors 0x{sensors:02X}",
            "(rate_hz, scale) pairs: "
            + ", ".join(f"({r:.1f}, {s:.4f})" for r, s in pairs),
        ]
    if btype == sf.BLOCK_RECORD:
        tmicros = struct.unpack_from("<I", payload)[0]
        channels = struct.unpack_from(f"<{(len(payload) - 4) // 2}h", payload, 4)
        return [f"tmicros {tmicros}, {len(channels)} int16 fields: {list(channels)}"]
    if btype == sf.BLOCK_FIFO:
        flags, count, read_us = sf.FIFO_HEADER.unpack_from(payload)
        events = [n for bit, n in sf.FIFO_BLOCK_FLAGS.items() if flags & bit]
        lines = [
            f"flags 0x{flags:02X} {events or ''}, {count} entries, read at {read_us} µs"
        ]
        entries = np.frombuffer(
            payload, dtype=sf.FIFO_ENTRY_DTYPE, count=count, offset=sf.FIFO_HEADER.size
        )
        for e in entries[:4]:
            sensor = SENSOR_CODES.get(e["tag"] >> 3, f"code 0x{e['tag'] >> 3:02X}")
            timing = (
                "edge"
                if e["flags"] & sf.FIFO_ENTRY_T_EDGE
                else "fit" if e["flags"] & sf.FIFO_ENTRY_T_FIT else "est"
            )
            if e["tag"] >> 3 == sf.TAG_TIMESTAMP:
                raw = f"ticks {int(e['data'][:4].copy().view('<u4')[0])}"  # uint32 in bytes 0-3
            else:
                raw = f"raw {e['data'].copy().view('<i2').tolist()}"  # int16 x, y, z
            lines.append(
                f"  t {e['t_us']} µs ({timing}), tag 0x{e['tag']:02X} -> {sensor:9s} {raw}"
            )
        if count > 4:
            lines.append(f"  ... {count - 4} more")
        return lines
    return [payload[:16].hex(" ")]


def print_block_counts(stream: StreamFile, report: Report):
    section("Blocks")
    print(f"  RECORD   {len(stream.records):9,d} blocks")
    for src in sorted(stream.blocks):
        print(
            f"  FIFO     {len(stream.blocks[src]):9,d} blocks, {len(stream.fifo.get(src, [])):9,d} entries"
            f"  source {src} ({SOURCE_NAMES.get(src, '?')})"
        )
    if stream.truncated:
        report.info(
            "the last block is cut short (normal if the logger stopped mid-write)"
        )
    if stream.skipped_bytes:
        report.warn(f"{stream.skipped_bytes} corrupt bytes skipped while resyncing")
    else:
        report.ok("no corrupt bytes")


# ── 4. Checks ─────────────────────────────────────────────────────────────────


def check_records(records: pd.DataFrame, rate_hz: int, report: Report):
    section("Records (500 Hz analog)")
    t_us = records["tmicros"].to_numpy(np.int64)
    if len(t_us) < 2:
        report.warn(f"only {len(t_us)} records")
        return
    dt = np.diff(t_us)
    period = 1e6 / rate_hz
    rate = (len(t_us) - 1) / (t_us[-1] - t_us[0]) * 1e6
    report.info(f"{len(t_us):,d} records over {records['t'].iloc[-1]:.2f} s")
    msg = f"record rate {rate:.2f} Hz (expected {rate_hz})"
    report.ok(msg) if abs(rate - rate_hz) < 0.01 * rate_hz else report.warn(msg)
    gaps = dt > 1.5 * period
    if gaps.any():
        report.warn(
            f"{gaps.sum()} gaps between records, longest {dt.max() / 1000:.1f} ms"
        )
    else:
        report.ok(f"no gaps (record spacing {dt.min()}–{dt.max()} µs)")
    if (dt <= 0).any():
        report.warn(f"{(dt <= 0).sum()} records out of order")


def check_analog(records: pd.DataFrame, pullup_mask: Optional[int], report: Report):
    section("Analog channels (raw ADC counts, 0–4095)")
    cols = [
        c
        for c in records.columns
        if c not in ("tmicros", "t") and c[0] in "abcdef" and c[1:].isdigit()
    ]
    print(f"  {'ch':4s} {'min':>6s} {'max':>6s} {'mean':>8s} {'std':>8s}")
    for c in sorted(cols):
        v = records[c].to_numpy()
        note = ""
        if v.std() == 0:
            note = "  flat (nothing connected?)"
        elif (v >= ADC_MAX - 5).mean() > 0.95 or (v <= 5).mean() > 0.95:
            note = "  at a rail (open switch / short?)"
        print(
            f"  {c:4s} {v.min():6d} {v.max():6d} {v.mean():8.1f} {v.std():8.1f}{note}"
        )
    missing = sorted(
        {c[0].upper() for c in ANALOG_18_FIELDS} - {c[0].upper() for c in cols}
    )
    if missing:
        report.info(
            f"no analog channels for port(s) {', '.join(missing)} (an IMU is there)"
        )
    if pullup_mask:
        report.info(f"pull-ups on: {_channels(pullup_mask)}")


# Which of x1 / x2 is SDA and SCL on the two I2C ports.
I2C_LINES = {"C": {"SCL": 1, "SDA": 2}, "E": {"SDA": 1, "SCL": 2}}
PORT_BITS = {"C": 9, "E": 3}  # analog_mask bit of the port's x0
RAIL = 3900  # counts: a line an I2C board's pull-up holds high reads above this


def _pinned_high(records: pd.DataFrame, col: str) -> bool:
    return col in records and (records[col].to_numpy() >= RAIL).mean() > 0.99


def check_ports(stream: StreamFile, data: StreamData, report: Report):
    """Ports C and E: the header's port state against the data actually recorded.

    An IMU board pulls SDA and SCL up. Until firmware 0.14 caught it, an IMU
    with one broken wire answered nothing and the port was recorded as analog,
    with no error. Its analog channels then show the board's pull-up: one I2C
    line pinned at the rail, the other floating.
    """
    section("Ports C and E")
    records = data.records
    for port, info in stream.ports.items():
        bits = 0x7 << PORT_BITS[port]
        real = bool(stream.analog_mask & bits)
        name = f"port {port}"
        if real != info.is_analog:
            report.warn(
                f"{name}: state {info.state_name} but analog_mask says its channels are "
                f"{'real' if real else 'not real'}"
            )

        if info.is_analog:
            if stream.pullup_mask & bits:
                report.info(
                    f"{name}: analog, with pull-ups on (I2C line check doesn't apply)"
                )
                continue
            p = port.lower()
            high = {
                line: _pinned_high(records, f"{p}{x}")
                for line, x in I2C_LINES[port].items()
            }
            if all(high.values()):
                report.warn(
                    f"{name}: recorded as analog, but SDA ({p}{I2C_LINES[port]['SDA']}) and "
                    f"SCL ({p}{I2C_LINES[port]['SCL']}) are both pinned high: an I2C board "
                    f"that didn't answer?"
                )
            elif any(high.values()):
                up = "SDA" if high["SDA"] else "SCL"
                down = "SCL" if up == "SDA" else "SDA"
                report.warn(
                    f"{name}: recorded as analog, but {up} ({p}{I2C_LINES[port][up]}) is pinned "
                    f"high and {down} ({p}{I2C_LINES[port][down]}) isn't: an IMU with a broken "
                    f"{down} wire? (Or an analog sensor at full scale on {p}{I2C_LINES[port][up]}.)"
                )
            else:
                report.ok(f"{name}: analog, no sign of an I2C board")
            if port in data.imu:
                report.warn(f"{name}: recorded as analog, but there is IMU data for it")
            continue

        if info.state_name in ("imu", "imu_no_mag"):
            has_source = any(s.port == port for s in stream.sources.values())
            imu = data.imu.get(port)
            if not has_source:
                report.warn(f"{name}: {info.device}, but no SOURCE_INFO block for it")
            if imu is None or imu.accel.empty:
                report.warn(f"{name}: {info.device}, but no IMU data recorded")
            else:
                report.ok(
                    f"{name}: {info.device} at 0x{info.i2c_addr:02X}, data recorded"
                )
            if info.state_name == "imu_no_mag":
                report.warn(
                    f"{name}: the IMU's magnetometer didn't answer (recorded 6-DoF)"
                )
        else:
            # The logger refuses to record with a port in a fault state.
            report.warn(f"{name}: state {info.state_name} in a recording")


def _rate(frame: pd.DataFrame) -> float:
    span = frame["t"].iloc[-1] - frame["t"].iloc[0] if len(frame) > 1 else 0
    return (len(frame) - 1) / span if span > 0 else 0.0


def bad_stamps(t_us: np.ndarray) -> np.ndarray:
    """Indices of samples whose time is out of line with both neighbours.

    A lone sample stamped far ahead (or behind) of the samples around it, which
    are in order with each other. (Firmware 0.14 before the fix stamped the
    first sample after a FIFO overrun up to seconds late.) Dropping these
    keeps one bad time from looking like a long gap.
    """
    t = np.asarray(t_us, dtype=np.int64)
    if len(t) < 3:
        return np.array([], dtype=int)
    prev, cur, nxt = t[:-2], t[1:-1], t[2:]
    out = (nxt >= prev) & ((cur > nxt) | (cur < prev))
    return np.flatnonzero(out) + 1


def check_imu(
    port: str,
    imu: ImuData,
    info: Optional[sf.SourceInfo],
    records: pd.DataFrame,
    report: Report,
):
    section(f"{PORT_NAMES.get(port, port)}: {imu.device}")
    nominal = {}
    if info is not None:
        nominal = {
            "accel": info.accel_hz,
            "gyro": info.gyro_hz,
            "accel_hg": info.hg_hz,
            "mag": info.mag_hz,
        }

    for name in ["accel", "accel_hg", "gyro", "mag"]:
        frame = getattr(imu, name)
        if frame.empty:
            continue
        rate = _rate(frame)
        line = f"{name:9s} {len(frame):8,d} samples, {rate:7.1f} Hz"
        t_us = frame["tmicros"].to_numpy(np.int64)
        bad = bad_stamps(t_us) if name != "mag" else []
        if len(bad):
            when = ", ".join(f"{frame['t'].iloc[i]:.3f}" for i in bad[:5])
            report.warn(
                f"{port} {name}: {len(bad)} sample(s) with a time out of line with the samples around them (stamped {when} s)"
            )
        dt = np.diff(np.delete(t_us, bad))
        median_dt = np.median(dt) if len(dt) else 0
        gaps = (dt > 3 * median_dt).sum() if median_dt > 0 else 0
        if gaps and name != "mag":
            line += f", {gaps} gaps (longest {dt.max() / 1000:.1f} ms)"
        if info is None:
            report.info(line + " (pre-0.13: one sample per record)")
        elif name == "mag":
            # Magnetometers write only new samples, below their poll rate.
            report.info(line + f" (polled at {nominal['mag']:.0f} Hz)")
        else:
            off = rate / nominal[name] - 1 if nominal.get(name) else 0
            line += f" ({off:+.1%} vs {nominal[name]:.0f} Hz nominal)"
            (
                report.ok(line)
                if abs(off) < RATE_TOLERANCE and not gaps
                else report.warn(line)
            )

    # Coverage: an IMU that stops before the records did dropped out.
    if not imu.accel.empty and len(records):
        end_gap = records["t"].iloc[-1] - imu.accel["t"].iloc[-1]
        if end_gap > 1.0:
            report.warn(f"data stops {end_gap:.1f} s before the end of the recording")

    if info is not None:
        timed = (
            (imu.accel["t_edge"] | imu.accel["t_fit"]).mean() if len(imu.accel) else 0
        )
        how = (
            "INT1 edge"
            if imu.accel["t_edge"].any()
            else "clock fit, which takes ~1 s to lock"
        )
        msg = f"{timed:.0%} of accel samples precisely timed ({how})"
        report.ok(msg) if timed > 0.5 else report.warn(msg)
        for flag in sf.FIFO_BLOCK_FLAGS.values():
            n = int(imu.events[flag].sum()) if flag in imu.events else 0
            if n and flag in ("overrun", "read_failed", "late"):
                report.warn(f"{n} FIFO '{flag}' events: samples may be lost")
            elif n:
                report.info(f"{n} '{flag}' events")
        if imu.events.empty:
            report.ok("no FIFO events (overruns, failed reads, sync losses)")
        if imu.unknown_words:
            report.warn(f"{imu.unknown_words} FIFO words with an unknown tag")
        if imu.mag_nacks:
            report.info(f"{imu.mag_nacks} magnetometer NACKs")

    # Physical sanity: gravity and gyro bias (meaningful if the logger sat still).
    if len(imu.accel):
        g = np.linalg.norm(imu.accel[["x", "y", "z"]].to_numpy(), axis=1) / 1000
        report.info(
            f"|accel| median {np.median(g):.3f} g (≈1 g if the logger was still)"
        )
    if len(imu.gyro):
        bias = imu.gyro[["x", "y", "z"]].median().round(2).tolist()
        report.info(f"gyro median {bias} dps (≈0 if still)")
    if len(imu.mag):
        b = np.linalg.norm(imu.mag[["x", "y", "z"]].to_numpy(), axis=1)
        report.info(f"|mag| median {np.median(b):.1f} µT (Earth's field is 25–65 µT)")

    # The LSM6DSV320X's two accelerometers sample the same instants. They should
    # track each other while the low-g one (±16 g) isn't clipping; the high-g
    # one has its own zero-g offset (a few hundred mg, uncalibrated).
    if len(imu.accel_hg) and len(imu.accel):
        both = pd.merge_asof(
            imu.accel[["t", "x", "y", "z"]],
            imu.accel_hg[["t", "x", "y", "z"]],
            on="t",
            direction="nearest",
            tolerance=5e-4,
            suffixes=("", "_hg"),
        ).dropna()
        low = both[["x", "y", "z"]].to_numpy()
        high = both[["x_hg", "y_hg", "z_hg"]].to_numpy()
        clipped = (np.abs(low) > 15_500).any(axis=1)
        if clipped.any():
            report.info(
                f"low-g accel clipped at ±16 g in {clipped.sum()} samples (use accel_hg there)"
            )
        diff = high[~clipped] - low[~clipped]
        if len(diff):
            offset = np.median(diff, axis=0)
            residual = np.median(np.abs(diff - offset))
            report.info(
                f"high-g minus low-g offset {offset.round(0).tolist()} mg (high-g zero-g offset)"
            )
            msg = f"after the offset, low-g and high-g agree to {residual:.0f} mg (median; high-g LSB is 10.4 mg)"
            report.ok(msg) if residual < 50 else report.warn(msg)


# ── 5. Plots ──────────────────────────────────────────────────────────────────

AXIS_COLORS = {"x": "tab:red", "y": "tab:blue", "z": "tab:green"}


def _legend(ax, ncol: int = 3):
    leg = ax.legend(loc="upper right", ncol=ncol, fontsize=8)
    for line in leg.get_lines():
        line.set_linewidth(2.5)


def _thin(frame: pd.DataFrame, max_points: int = 200_000) -> pd.DataFrame:
    step = max(1, len(frame) // max_points)
    return frame.iloc[::step]


def plot_file(name: str, data: StreamData) -> list[tuple[str, plt.Figure]]:
    """Every figure for one file, as (short name, figure)."""
    plt.style.use("ggplot")
    figs = []

    for port, imu in data.imu.items():
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 9))
        for axis, color in AXIS_COLORS.items():
            for ax, frame in zip(axs, [imu.accel, imu.gyro, imu.mag]):
                f = _thin(frame)
                ax.plot(f["t"], f[axis], lw=0.8, color=color, label=axis)
            if len(imu.accel_hg):
                # Same colour per axis, dashed: the ±320 g channel behind the ±16 g one.
                hg = _thin(imu.accel_hg)
                axs[0].plot(
                    hg["t"],
                    hg[axis],
                    lw=0.8,
                    ls="--",
                    alpha=0.6,
                    color=color,
                    label=f"{axis} high-g",
                )
        for ax, ylabel in zip(axs, ["accel [mg]", "gyro [dps]", "mag [µT]"]):
            ax.set_ylabel(ylabel)
            _legend(ax, ncol=6)
            ax.tick_params(axis="x", labelbottom=True)
        axs[0].set_title(f"{name}: {PORT_NAMES.get(port, port)}, {imu.device}")
        axs[-1].set_xlabel("time [s]")
        fig.tight_layout()
        figs.append((f"imu_{port}", fig))

    rec = _thin(data.records)
    ports = sorted({c[0] for c in rec.columns if c[0] in "abcdef" and c[1:].isdigit()})
    if ports:
        fig, axs = plt.subplots(
            len(ports),
            1,
            sharex=True,
            figsize=(12, 2 + 1.6 * len(ports)),
            squeeze=False,
        )
        for ax, p in zip(axs[:, 0], ports):
            for ch in range(3):
                col = f"{p}{ch}"
                if col in rec:
                    ax.plot(rec["t"], rec[col], lw=0.8, label=col)
            ax.set_ylabel(f"port {p.upper()}\n[counts]")
            _legend(ax)
        axs[0, 0].set_title(f"{name}: analog channels")
        axs[-1, 0].set_xlabel("time [s]")
        fig.tight_layout()
        figs.append(("analog", fig))

    # Timing: spacing between consecutive samples; spikes are gaps or lost data.
    fig, ax = plt.subplots(figsize=(12, 5))
    t = data.records["t"].to_numpy()
    ax.plot(
        t[1:],
        np.diff(data.records["tmicros"].to_numpy()) / 1000,
        lw=0.8,
        color="black",
        label="records (500 Hz)",
    )
    for (port, imu), color in zip(
        data.imu.items(), ["tab:orange", "tab:purple", "tab:cyan"]
    ):
        a = imu.accel
        if len(a) > 1:
            ax.plot(
                a["t"].to_numpy()[1:],
                np.diff(a["tmicros"].to_numpy()) / 1000,
                lw=0.8,
                color=color,
                label=f"{port} accel ({imu.device})",
            )
        for _, ev in imu.events.iterrows():
            ax.axvline(ev["t"], color=color, lw=1, ls=":")
    ax.set_yscale("log")
    ax.set_ylabel("time since previous sample [ms]")
    ax.set_xlabel("time [s]")
    ax.set_title(f"{name}: sample spacing (dotted lines: FIFO events; spikes are gaps)")
    _legend(ax, ncol=1)
    fig.tight_layout()
    figs.append(("timing", fig))
    return figs


# ── Driver ────────────────────────────────────────────────────────────────────


def inspect_file(
    path: Path,
    fw_ver: Optional[str],
    n_blocks: int,
    plots: bool,
    save_dir: Optional[Path],
) -> Report:
    print("\n" + "═" * 80 + f"\n{path}\n" + "═" * 80)
    report = Report(path.name)

    if sf.is_stream_file(path):
        stream = sf.read_stream(path)
        print_file_header(path, stream)
        print_sources(stream)
        dump_blocks(path, n_blocks)
        print_block_counts(stream, report)
        data = sf.process_stream(stream)
        sources = {info.port: info for info in stream.sources.values()}
        rate_hz = stream.record_rate_hz
    else:
        if fw_ver is None:
            report.warn(
                "pre-0.13 file and no metadata.txt: pass fw_ver to inspect_path"
            )
            return report
        section("Pre-0.13 file")
        print(
            f"  no header: raw {fw_ver} records (fw_ver from metadata.txt or inspect_path)"
        )
        data = CassCommands.process_stream_file(path, fw_ver)
        stream = None
        sources = {}
        rate_hz = 500

    if stream is not None and stream.analog_mask is not None:
        check_ports(stream, data, report)
    check_records(data.records, rate_hz, report)
    check_analog(data.records, data.pullup_mask, report)
    for port, imu in data.imu.items():
        check_imu(port, imu, sources.get(port), data.records, report)

    if plots and len(data.records):
        figs = plot_file(path.name, data)
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            for short, fig in figs:
                out = save_dir / f"{path.stem}_{short}.png"
                fig.savefig(out, dpi=120)
                plt.close(fig)
                print(f"  saved {out}")
        else:
            plt.show()
    return report


def inspect_path(
    path,
    fw_ver: Optional[str] = None,
    n_blocks: int = 12,
    plots: bool = True,
    save_dir=None,
) -> list[Report]:
    """Inspect one .bin file, or every .bin file under a directory.

    ``fw_ver`` is only needed for pre-0.13 files without a metadata.txt next
    to them. ``save_dir`` writes the plots there as PNGs instead of showing them.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path.resolve()} doesn't exist")
    files = [path] if path.is_file() else sorted(path.rglob("*.bin"))
    if not files:
        raise SystemExit(f"No .bin files in {path.resolve()}")
    save_dir = Path(save_dir) if save_dir else None

    reports = []
    for f in files:
        file_fw_ver = fw_ver
        if file_fw_ver is None:
            meta = CassCommands.find_and_parse_metadata(str(f.parent), recursive=False)
            file_fw_ver = meta.get("firmware_version") if meta else None
        reports.append(inspect_file(f, file_fw_ver, n_blocks, plots, save_dir))

    section("Summary")
    for r in reports:
        status = "OK" if not r.warnings else f"{len(r.warnings)} warning(s)"
        print(f"  {r.name:40s} {status}")
        for w in r.warnings:
            print(f"      - {w}")
    return reports


def delete_all(cass: CassCommands):
    """Delete every file on the logger's SD card (asks y/n first)."""
    section("Delete")
    print(f"  on the device: {cass.list_files()}")
    result = cass.delete_all_files(prompt_user=True)
    if result is True:
        print("  deleted; the device is empty")
    elif result is False:
        print(f"  some files were not deleted: {cass.list_files()}")


if __name__ == "__main__":
    cass = CassCommands()
    print(cass.list_files())

    # Pick the files to inspect: download them from the connected logger, or
    # point at a folder of .bin files (or one .bin file) already on disk.
    # Relative paths are from the directory you run the script in.
    # # file_dir = Path("./tmp_example")
    # # file_dir = Path(__file__).parent / "data" / "0.14-auto_icm45686-lsm6dsv320x"
    file_dir = download(cass)  # saves to ./tmp_<unix time>

    inspect_path(
        file_dir,
        fw_ver=None,  # pre-0.13 files without metadata.txt: e.g. "0.10-i2c_45686_9dof"
        n_blocks=12,  # blocks to decode by hand
        plots=True,
        save_dir=None,  # e.g. "./plots" to save PNGs instead of showing them
    )

    # Wipe the logger's SD card once you've checked the download (asks y/n).
    # delete_all(cass)
