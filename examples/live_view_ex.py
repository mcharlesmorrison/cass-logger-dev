"""
Firmware bench tool: poll live view snapshots and print every channel.

This is for firmware development and bench checks (see Phase 1.4 in
``docs/live-view-plan.md``), not a product feature. Live view for users runs
in the Cass Logger app.

It sends ``'k'`` at ``--rate`` Hz using ``CassCommands.snapshot()`` and
redraws a table of all channels, the frame flags, and running error counts
(timeouts, CRC errors, false syncs, skipped bytes). It keeps polling while
the device is recording, so it can be used for the recording-interaction
test. Press Ctrl-C to stop.

Usage
-----
    python examples/live_view_ex.py
    python examples/live_view_ex.py --rate 20
    python examples/live_view_ex.py --fw 0.12-i2c_2   # skip the version query
"""

import argparse
import shutil
import sys
import time

from cass_logger_dev.cass_commands import CassCommands
from cass_logger_dev.live import FrameParser, live_columns, supports_live_view

CLEAR = "\033[H\033[J"


def format_value(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:10.3f}"
    return f"{value:10d}"


def render(fw_ver, columns, snap, counts, parser, rate, achieved_hz, last_dt_us) -> str:
    lines = [
        f"Cass live view bench  |  fw {fw_ver}  |  target {rate:g} Hz, "
        f"actual {achieved_hz:5.1f} Hz  |  Ctrl-C to stop",
        "",
    ]

    if snap is None:
        status = "waiting for first frame"
    elif snap.logging:
        status = "LOGGING (device is recording, no sensor data)"
    else:
        status = "ok"
    if snap is not None and snap.sensor_init_failed:
        status += "  |  SENSOR_INIT_FAILED"
    lines.append(f"Status: {status}")

    s = parser.stats
    lines.append(
        f"Frames: {counts['ok']}  logging: {counts['logging']}  timeouts: {counts['timeouts']}  "
        f"errors: {counts['errors']}  |  CRC errors: {s.crc_errors}  false syncs: {s.false_syncs}  "
        f"skipped bytes: {s.skipped_bytes}"
    )
    if last_dt_us is not None:
        lines.append(f"Device dt between frames: {last_dt_us / 1000:.1f} ms")
    if counts["last_error"]:
        lines.append(f"Last error: {counts['last_error']}")
    lines.append("")

    values = snap.values if snap is not None and snap.values is not None else {}
    cells = [f"{name:>10} {format_value(values.get(name))}" for name in columns]
    cell_w = 22
    ncols = max(1, shutil.get_terminal_size((100, 40)).columns // (cell_w + 2))
    for i in range(0, len(cells), ncols):
        lines.append("  ".join(c.ljust(cell_w) for c in cells[i : i + ncols]))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", type=float, default=10.0, help="poll rate in Hz (default 10)")
    ap.add_argument("--timeout", type=float, default=0.25, help="reply timeout in seconds (default 0.25)")
    ap.add_argument("--fw", default=None, help="firmware version string; read from the device if omitted")
    args = ap.parse_args()

    cass = CassCommands()
    fw_ver = args.fw or cass.get_fw_ver()
    if not supports_live_view(fw_ver):
        print(f"Firmware '{fw_ver}' does not support live view (needs 0.12 or newer).")
        return 1
    columns = live_columns(fw_ver)

    parser = FrameParser()
    counts = {"ok": 0, "logging": 0, "timeouts": 0, "errors": 0, "last_error": ""}
    snap = None
    last_tmicros = None
    last_dt_us = None
    period = 1.0 / args.rate
    recent = []  # host times of recent good replies, for the achieved rate

    cass._flush_all()
    next_tick = time.monotonic()
    try:
        while True:
            try:
                new = cass.snapshot(fw_ver, timeout=args.timeout, parser=parser)
                snap = new
                recent.append(new.host_time)
                if new.logging:
                    counts["logging"] += 1
                    last_tmicros = None
                else:
                    counts["ok"] += 1
                    tmicros = new.values["tmicros"]
                    if last_tmicros is not None:
                        last_dt_us = (tmicros - last_tmicros) & 0xFFFFFFFF
                    last_tmicros = tmicros
            except TimeoutError as e:
                counts["timeouts"] += 1
                counts["last_error"] = str(e)
            except ValueError as e:  # variant mismatch or protocol error
                counts["errors"] += 1
                counts["last_error"] = str(e)

            now = time.monotonic()
            recent = [t for t in recent if now - t < 2.0]
            achieved = len(recent) / 2.0
            sys.stdout.write(CLEAR + render(fw_ver, columns, snap, counts, parser, args.rate, achieved, last_dt_us) + "\n")
            sys.stdout.flush()

            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()  # fell behind; don't try to catch up
    except KeyboardInterrupt:
        pass
    finally:
        cass._close_serial()

    s = parser.stats
    print(
        f"\nStopped. frames={counts['ok']} logging={counts['logging']} timeouts={counts['timeouts']} "
        f"errors={counts['errors']} crc_errors={s.crc_errors} false_syncs={s.false_syncs}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
