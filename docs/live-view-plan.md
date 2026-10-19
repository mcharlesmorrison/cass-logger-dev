# Live View — Implementation Plan

Live sensor readout in the Device panel of the Cass Logger app, at about 10 Hz, via a new snapshot command.

This same file lives in all three repos (`cass-logger-fw`, `cass-logger-dev`, `cass-logger-app`) at `docs/live-view-plan.md`. If you change the protocol, update all three copies.

| Repo | Role | Section |
|---|---|---|
| `cass-logger-fw` | New `'k'` snapshot command, framing, cleanup. Firmware 0.11 → **0.12** | [Phase 1](#phase-1--firmware-cass-logger-fw) |
| `cass-logger-dev` | Frame parser + decoder, `snapshot()` / `get_snapshot()`, bench example. 0.1.0 → **0.2.0** | [Phase 2](#phase-2--host-library-cass-logger-dev) |
| `cass-logger-app` | Device lock, live polling thread, API endpoints, Live view card in the Device panel | [Phase 3](#phase-3--app-cass-logger-app) |

`cass-utils` needs no changes for v1. It only matters later, for calibrated units (see [Later](#later-not-in-v1)).

---

## Decisions (v1)

1. **Polling, not continuous streaming.** The host sends `'k'` and the device replies with one framed record. The device never sends anything it wasn't asked for, which is what lets live view share the ports with other commands (decision 5).
2. **Default rate is 10 Hz**, with 5 / 10 / 20 Hz selectable. Every frame carries the device `microtime`, so host timing jitter doesn't affect the data.
3. **Live view only runs in the app, in the Device panel.** It stops automatically when the user switches to another panel (Data, Field Test), on disconnect, and before a firmware flash. There's no polling thread in `cass-logger-dev`. That repo only provides the protocol (parser, decoder, single-snapshot call) and a small bench script for firmware development.
4. **No live view while recording.** If the device is recording, it answers `'k'` with a `LOGGING` flag and no sensor data. It never reads sensors from `loop()` while the 500 Hz logger interrupt is running. The app refuses to start live view while recording, and stops it as soon as a frame comes back with `LOGGING` set (e.g. someone pressed the hardware record button).
5. **Other device commands run alongside live view.** A single device I/O lock in the app makes every command and every live poll take turns on the serial ports. While a command such as setting the RTC holds the lock, live view skips its polls and shows "paused". The rate doesn't make this possible: polling plus the lock does, and it would work the same at 20 Hz. See [Why a lock](#why-a-lock-and-not-just-a-lower-rate).
6. **The payload is the raw `datrec` struct.** The host decodes it with the existing `FIRMWARE_DTYPES`, so all five variants work without new per-variant code.
7. **Values are raw.** ADC channels are counts (0–4095). IMU values are mg, dps and mag units, as the firmware writes them today.

### Why a lock and not just a lower rate

Every reply, from both commands and snapshots, comes back on the same data port (`Serial`). Existing replies have no framing: the host reads "everything in the buffer until an `'x'`" (e.g. `get_fw_ver`), or a fixed-size raw block (`'t'` transfers). Several existing commands also close and reopen the ports (`_flush_all`, `_close_serial`).

If a snapshot frame arrived in the middle of another reply, that reply would be corrupted, and a frame can easily contain the byte `'x'`. Lowering the rate only makes a collision less likely, not impossible.

The lock removes the problem entirely: a poll and a command can never be in flight at the same time. Because the device only speaks when spoken to, holding the lock for the length of one request/reply is enough.

---

## Protocol spec (shared contract)

### Command

| Direction | Port | Bytes |
|---|---|---|
| host → device | command port (`SerialUSB1`) | `k` (0x6B), single byte, no terminator |
| device → host | data port (`Serial`) | one snapshot frame (below) |

Old firmware (≤ 0.11) silently ignores `k`. The host **must** use a timeout and check the firmware version before using it (see `supports_live_view`).

### Snapshot frame

All multi-byte fields are little-endian.

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 2 | `sync` | `0xA5 0x5A` |
| 2 | 1 | `frame_type` | `0x01` = SNAPSHOT. Other values are reserved. |
| 3 | 1 | `flags` | see below |
| 4 | 2 | `payload_len` | `uint16`. Equals `sizeof(datrec)` (76 / 108 / 136), or `0` when recording |
| 6 | N | `payload` | raw `datrec` bytes, layout per `include/deviceConfig.h` |
| 6+N | 2 | `crc16` | CRC-16/CCITT-FALSE over bytes `[2, 6+N)`, i.e. type + flags + len + payload |

**Flags**

| Bit | Name | Meaning |
|---|---|---|
| 0 | `LOGGING` | Device is recording. `payload_len` is 0 and no sensors were read. The host must stop live view. |
| 1 | `SENSOR_INIT_FAILED` | `sensorInitOk == false` at boot. Some IMU fields may be stale or zero. |
| 2–7 | reserved | Must be 0 |

**CRC-16/CCITT-FALSE:** poly `0x1021`, init `0xFFFF`, not reflected, no final XOR. In Python this is `binascii.crc_hqx(data, 0xFFFF)`. Check value: `crc("123456789") == 0x29B1`.

**Parser rules (host):**
- Scan for `A5 5A`, and discard any bytes before it. Debug text can still show up on the data port.
- If `payload_len > 512`, it's a false sync: drop one byte and rescan.
- If the CRC doesn't match: drop one byte and rescan.
- If `payload_len == 0` without `LOGGING`, that's a protocol error.
- If `payload_len != dtype.itemsize` for the connected firmware variant, raise a clear variant-mismatch error. Don't decode it.

---

## Phase 1 — Firmware (`cass-logger-fw`)

Branch: `feature/live-view`. Version: `FW_VERSION "0.12"` in `include/variantConfig.h`, plus a changelog line in the `src/main.cpp` header.

### 1.1 Snapshot command
New `include/live.h` + `src/live.cpp`, following the `commands.cpp` / `fileIO.cpp` pattern:

- `uint16_t crc16_ccitt(const uint8_t *data, size_t len, uint16_t crc = 0xFFFF);`
- `void sendSnapshot();`
  - **If `isLogging`:** send a frame with `flags = LOGGING` and `payload_len = 0`. **Don't touch any sensor or bus.** Only write it if `Serial.availableForWrite() >= 8`, so a host that stopped reading can never block `loop()` and starve `CheckLogger()` (the ring buffer only holds 100 ms).
  - **Else:** `datrec rec; myCollector(&rec);`. Calling the collector directly is safe here because the logger's timer interrupt only runs while recording. `isLogging` only changes in `loop()` context (via `toggleLogging()` / `'r'` / `'q'`), so it can't flip in the middle of a snapshot.
  - If `!sensorInitOk`, set `flags |= SENSOR_INIT_FAILED`.
  - Build the frame in a local buffer (6 + 136 + 2 = 144 bytes max), compute the CRC, then write it once with `Serial.write(buf, n)`.
- `src/main.cpp`: add `if (ch == 'k') sendSnapshot();` to the dispatch list.

### 1.2 Remove serial prints from the collector
`src/collector.cpp` calls `Serial.println("didn't get ... data!!!")`. It does this inside the 500 Hz interrupt while recording, and now also on the snapshot path. The host parser skips stray text, but it's noise on the data port and bad practice in an ISR.
- Replace each one with a `volatile uint32_t` failure counter (`n_fail_spi`, `n_fail_i2c_c`, `n_fail_i2c_e`), declared in `collector.h`.
- Print the counters in `myBinaryDisplay` under `#ifdef TIMING_DEBUG`, next to the existing timing output.

### 1.3 Overflow visibility
`GetStatus()` in `src/commands.cpp`: add `Serial.printf("Buffer overflows: %lu\n", tsp->bufferoverflows);`. The recording test needs it.

### 1.4 Firmware acceptance
- [ ] `pio run -e std`, `-e i2c_1`, `-e i2c_2`, `-e i2c_45686`, `-e i2c_45686_2` all build.
- [ ] Bench test with `cass-logger-dev/examples/live_view_ex.py` on at least `std` and one I2C variant. Values respond to touching the inputs, and the IMU responds to motion.
- [ ] While not recording, the bench script runs 10 minutes at 20 Hz without CRC errors.
- [ ] **Recording interaction:** run the bench script, then press the hardware record button. The next frame has `LOGGING` set with `payload_len == 0`. Leave the script sending `'k'` at 20 Hz while recording for 10 minutes, then stop and send `s`. Overflows must be 0. Parse the `.bin` and confirm `diff(tmicros)` is about 2000 µs (500 Hz) everywhere, with no gaps.
- [ ] Existing commands (download, list, RTC, device ID) are unaffected.

---

## Phase 2 — Host library (`cass-logger-dev`)

Branch: `feature/live-view`. Version: `0.2.0` in `pyproject.toml`. Tag `v0.2.0` after merge, since the app pins to it.

This repo owns the **protocol**: framing, decoding, and a single snapshot request. It doesn't own the polling loop or the locking. Those are app-only (Phase 3).

### 2.1 `cass_logger_dev/live.py` (new)
Pure parsing code with no serial I/O, so it can be unit-tested without hardware.

- Constants: `SYNC = b"\xA5\x5A"`, `FRAME_SNAPSHOT = 0x01`, `FLAG_LOGGING = 0x01`, `FLAG_SENSOR_INIT_FAILED = 0x02`, `MAX_PAYLOAD = 512`, `MIN_LIVE_FW = (0, 12)`.
- `crc16(data: bytes) -> int` returns `binascii.crc_hqx(data, 0xFFFF)`.
- `class FrameParser` with `feed(chunk: bytes) -> list[RawFrame]`. It keeps an internal `bytearray` and applies the parser rules above. `RawFrame` holds `frame_type`, `flags`, `payload`.
- `@dataclass Snapshot` with fields `host_time: float` (`time.monotonic()`), `logging: bool`, `sensor_init_failed: bool`, `values: dict[str, int | float] | None` (`None` when `logging`).
- `decode_payload(payload: bytes, fw_ver: str) -> dict`: look up `FIRMWARE_DTYPES[CassCommands.get_dtype_key(fw_ver)]()`, check that `itemsize == len(payload)` (otherwise raise `ValueError` describing the variant mismatch), `np.frombuffer` it, and return `{name: python_scalar}` in dtype field order.
- `supports_live_view(fw_ver: str) -> bool`: parse the leading `X.Y` from `"0.12-std"` and compare it against `MIN_LIVE_FW`.

### 2.2 `CassCommands` additions (`cass_commands.py`)
- `snapshot(self, fw_ver: str, timeout: float = 0.25) -> Snapshot` is the **low-level call for repeated polling**:
  - send `b"k"` on `ser_command`;
  - feed `ser_data` bytes into a `FrameParser` until one frame arrives or `timeout` passes;
  - decode it;
  - on timeout, call `ser_data.reset_input_buffer()` so a late frame can't leak into the next command's reply, then raise `TimeoutError`.

  It must **not** call `_flush_all()` or `_close_serial()`, so the ports stay open between polls. It goes through the `ser_data` / `ser_command` properties, which reopen the ports if another command closed them.
- `get_snapshot(self, fw_ver: str | None = None, timeout: float = 0.5) -> Snapshot` takes a single snapshot in the existing per-command style: `_flush_all()`, then `snapshot()`, then `_close_serial()`. If `fw_ver` is `None`, call `get_fw_ver()`.

### 2.3 Read timeouts (recommended; can be a separate PR)
`get_fw_ver`, `get_device_ID`, `get_RTC_time` and `get_rtc_install_timestamp` currently wait forever (`while b"x" not in ...`). The app calls these on connect and from the Device panel. While live view shares the ports, a hung command means a frozen live view too. Add a `_read_until(terminator: bytes, timeout: float = 2.0) -> bytes` helper that raises `TimeoutError`, and use it in those four methods.

### 2.4 Tests, example, docs
- `tests/test_live.py` (pytest; add `[project.optional-dependencies] dev = ["pytest>=8"]`). Build synthetic frames from the spec and cover:
  - a clean frame;
  - garbage or ASCII text before the sync bytes;
  - a frame split across many `feed()` calls;
  - two frames in one chunk;
  - a bad CRC followed by a good frame;
  - a false `A5 5A` inside the payload;
  - a `LOGGING` frame with `payload_len == 0`;
  - `payload_len > 512`;
  - a variant mismatch in `decode_payload`;
  - `supports_live_view("0.11-std")` is false and `("0.12-i2c_2")` is true;
  - the CRC check value `0x29B1`.
- `examples/live_view_ex.py` is the **firmware bench tool** for Phase 1.4, not a product feature. It's a plain loop around `snapshot()` at a `--rate` argument that prints a refreshing table of all channels, flags and CRC/timeout counts. Ctrl-C to stop.
- `README.md`: add one row for `get_snapshot()`, and note that live view is driven from the app.

### 2.5 Host library acceptance
- [ ] `pytest` passes.
- [ ] `live_view_ex.py` decodes correctly on every variant available on the bench.
- [ ] Against 0.11 firmware, `get_snapshot()` raises `TimeoutError` within its timeout, with no hang.

---

## Phase 3 — App (`cass-logger-app`)

Branch: `feature/live-view`. Follow the existing conventions: tabs in `gui/`, type hints, and `ok()`/`err()` results.

### 3.1 Dependency
`pyproject.toml`: pin `cass-logger-dev @ git+https://github.com/mcharlesmorrison/cass-logger-dev@v0.2.0`. It's currently unpinned on `main`.

### 3.2 Device I/O lock (`gui/services/cass_service.py`)
pywebview runs each JS→Python call on its own thread, so device calls can already overlap today (e.g. the 5 s status poll during `set_rtc_time`). This lock fixes that existing race, and it's also what lets live view run alongside other commands.

- Add `_io_lock = threading.RLock()`. It's reentrant because some `CassCommands` methods call others.
- The `cass` property returns a thin proxy around `CassCommands` whose `__getattr__` wraps every callable so that it runs `with self._io_lock:`. Every existing `MainApi` call (`self._svc.cass.foo()`) is then serialized **without changing `main_api.py`**.
- `check_alive()`: try `self._io_lock.acquire(blocking=False)`. If the lock is busy, a command is in progress, so the device is alive: return `True` without touching the port. Otherwise run the existing probe and release the lock.

### 3.3 Live polling (`gui/services/cass_service.py`)
- State: `_live_thread`, `_live_stop: threading.Event`, `_live_frames: deque[Snapshot]` (`maxlen` around 600), `_live_paused: bool`, `_live_missed: int`, `_live_stopped_reason: Optional[str]`, `_live_last_ok: float`.
- `start_live(rate_hz: float) -> tuple[bool, str]`:
  1. Require a connection and `supports_live_view(self._fw_ver)`. Otherwise return "Live view needs firmware 0.12 or newer — update it below."
  2. Take one synchronous `snapshot()` under the lock. If `logging` is set, return "The logger is recording — stop recording to use live view."
  3. Start the daemon thread.
- Each tick of the thread:
  - `self._io_lock.acquire(timeout=0.02)`. If that fails, set `_live_paused = True`, skip the tick and continue.
  - Otherwise call `self._cass.snapshot(self._fw_ver)` (the raw object, not the proxy, since the lock is already held), then release.
  - `TimeoutError`: increment `_live_missed`. After 10 in a row, stop with reason "Device not responding".
  - `snapshot.logging`: stop with reason "Recording started — live view stopped".
  - `SerialException` / `OSError`: stop with reason "Device disconnected", then `disconnect()`.
  - Otherwise append the frame, clear `_live_paused`, update `_live_last_ok`, and sleep the rest of the period.
- `stop_live()` is idempotent: set the event, join the thread, then call `_close_serial()` under the lock to go back to the normal "ports closed between commands" state.
- `disconnect()` calls `stop_live()` first.
- `check_alive()` while live returns `time.monotonic() - _live_last_ok < 3` and doesn't touch the port.

### 3.4 `gui/api/main_api.py`
- `start_live(rate_hz: float = 10) -> dict` → `ok({"columns": [...], "rate_hz": ...})`.
- `stop_live() -> dict`.
- `get_live_frames() -> dict` drains the deque and returns a **columnar** batch to keep the JSON small:
  ```json
  {"live": true, "paused": false, "stopped_reason": null, "missed": 0,
   "sensor_init_failed": false,
   "t": [12.31, 12.41, ...],              // host seconds since start_live
   "device_us": [123456, ...],             // datrec microtime
   "values": {"d0": [...], "gx": [...], ...}}
  ```
  Put the conversion in a pure helper, `_frames_to_batch(frames, t0)`, so it can be tested.
- `start_firmware_flash`: call `self._svc.stop_live()` first. A flash reboots the device, so live view can't continue.
- Window close (`gui/app.py`): stop live view on shutdown so the ports are released.

### 3.5 Frontend
- `npm install uplot`. It draws on canvas and copes with 10–20 Hz × many series. Recharts (SVG) will stutter at that rate, so keep Recharts for the existing static plots.
- `src/types.ts`: add a `LiveBatch` interface and `start_live` / `stop_live` / `get_live_frames` to `PyApi`.
- `src/App.tsx`: pass `active={panel === 'device'}` to `DevicePanel`. Panels stay mounted and are only hidden with `display: none`, so the prop is how live view knows to stop. This is the same pattern `DataPanel` already uses.
- `src/components/LiveView.tsx` (new), rendered as a card inside `DevicePanel.tsx`:
  - A Start/Stop button and a rate selector (5 / **10** / 20 Hz).
  - **Auto-stop** when `active` becomes false, on unmount, when `connected` becomes false, and when the user starts a firmware flash.
  - Poll `get_live_frames()` every 100 ms. Keep a rolling 10 s window in **refs** rather than React state, and push it into uPlot with `setData()`. That avoids re-rendering on every poll.
  - **Channel tiles**, grouped by logger port (A, B, C, D, E, F, then IMU SPI / IMU C / IMU E), built from the `columns` that `start_live` returns. That makes them variant-aware for free. Each tile shows the latest value, and clicking it adds that channel to the chart.
  - Default charts: SPI IMU accel (gx/gy/gz), plus one chart of the selected analog channels.
  - Status chips: **Paused — device busy** while `paused` is set, **Sensor init failed** when that flag is set, and a missed-poll count if it's above 0.
  - When live view stops because of `stopped_reason`, show the reason inline (recording started, not responding, disconnected).
  - Firmware too old: show the message, with the firmware updater on the same panel.
  - Footnote: "Snapshots at N Hz — short pulses (reed / wheel-speed switches) can be missed."
- Other Device panel controls (RTC get/set, install timestamp, manual connect) stay enabled during live view. They queue behind the lock, and live view briefly shows "paused".

### 3.6 Tests and docs
- `tests/test_live_service.py`, using a fake `CassCommands` whose `snapshot()` returns synthetic frames:
  - while one thread holds `_io_lock`, the live thread doesn't call `snapshot()` and sets `paused`;
  - a `logging` frame stops live view with the recording reason;
  - 10 timeouts in a row stop it with "not responding";
  - `check_alive` doesn't touch the port while live or while the lock is held;
  - `disconnect` stops live view;
  - the shape of `_frames_to_batch`.
- `CLAUDE.md`: amend "Serial ports are opened lazily per command and closed after each operation" to note that live view holds them open between `start_live` and `stop_live`. Document `_io_lock` (all device I/O goes through it), and add `LiveView.tsx` to Key Files.

### 3.7 App acceptance
- [ ] Live view in the Device panel runs at 10 Hz for 10 minutes on macOS and Windows, and the UI stays smooth with every channel tile visible.
- [ ] Switching to Data or Field Test stops live view. Coming back shows it stopped, and the user restarts it.
- [ ] Get/set RTC and install timestamp work during live view. Live view shows "paused" briefly and resumes, with no garbled values on either side.
- [ ] Starting live view while recording is refused. Pressing the record button during live view stops it within one poll with the recording message, and the downloaded file has no gaps.
- [ ] Unplugging during live view: the app shows disconnected within about 5 s with no hang. Plugging back in auto-reconnects.
- [ ] A device on 0.11 shows the "update firmware" message.
- [ ] `npm run build` and `pytest tests` pass.

---

## Sequencing

```
Phase 1 firmware ──────────────┐
                               ├─► bench integration (live_view_ex.py) ─► Phase 3 app ─► release
Phase 2.1/2.4 parser + tests ──┘         (needs both)                       (UI can start against a fake snapshot())
```

1. Firmware (Phase 1) and the host parser and its tests (2.1, 2.4) can run **in parallel**: the parser is tested against synthetic frames built from the spec.
2. Build `snapshot()` / `get_snapshot()` and the bench example (2.2, 2.4) against real hardware running the Phase 1 firmware, then pass the Phase 1.4 checks.
3. App lock, polling and UI (Phase 3). The lock (3.2) can land first as its own PR, because it fixes an existing race whether or not live view ships. The UI can be built before step 2 is done by using a fake `snapshot()` that generates sine waves.
4. **Merge and release order:**
   1. merge firmware;
   2. merge `cass-logger-dev` and tag `v0.2.0`;
   3. merge the app (pinned to `v0.2.0`);
   4. publish firmware 0.12 for all variants with `cass-logger-app/scripts/release_firmware.py`;
   5. release the app.

   Firmware has to be on R2 before the app ships, so users can update from the Device panel when live view asks them to.

## Rough effort

| Piece | Estimate |
|---|---|
| Firmware (1.1–1.3) + variant builds + bench/recording checks | 0.5 day |
| `cass-logger-dev` parser, `snapshot()`, tests, bench example | 0.5–1 day (+0.5 for 2.3 timeouts) |
| App lock + polling thread + API + tests | 1 day |
| App Live view card (uPlot, tiles, states) | 1–1.5 days |
| Cross-platform testing (Windows port holding) | 0.5 day |

## Later (not in v1)

- **Catch short pulses on digital channels.** Add a live-mode `IntervalTimer` that runs the collector at `SAMPLERATE` into a scratch record and tracks edges and min/max per digital channel, reset on each snapshot. The timer must stop before `StartLogging()`. This would add a new frame type or append fields; `payload_len` and `frame_type` leave room for both.
- **Calibrated units** using sensor calibrations from `cass-utils` (`sensor_db`).
- **Continuous streaming** (device-paced frames) if more than ~50 Hz is ever needed. The cleanest way is to send stream frames on `SerialUSB1` TX, which is unused today, so streaming never mixes with command replies on `Serial`.
- Gate the other non-ISR debug prints (`SetDBPrint(true)`, `enableDebugging()`) behind a build flag.
