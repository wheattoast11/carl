---
last_updated: 2026-04-20
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.9.0
scope: software interfaces for Terminals modular hardware
status: forward spec (software lands first, hardware follows)
---

# Modular hardware × software interface spec

Hardware hasn't shipped yet. This spec lets us build the software **now**
so the hardware team (or OEM partners) can target a stable contract
when modules land. Every path below has a software-only fallback; the
dashboard is fully functional without any device plugged in.

The product fantasy, grounded:

- **USB drive** — bootable Terminals OS + hardware wallet + TOTP-style
  2FA + decorative lighting. Detachable from the watch strap.
- **Wristband** — mic + push-to-talk + battery for the USB + biometric
  sensor (PPG optional).
- **Baseball cap** — bone-conduction sensor array on an Arduino Nano
  BLE; 4h battery; streams frisson / vibration to the agent; can emit
  as actuator.
- **Watch (Inland 37)** — body-worn metric monitor; read-only from
  CARL's perspective.

Everything is modular. Every module is optional. Every module signs
its messages with a hardware key. The user sees one button.

## 1 · Device taxonomy

| Device     | Transport                    | Role                                 | ID surface                     |
| ---------- | ---------------------------- | ------------------------------------ | ------------------------------ |
| USB        | `/dev/terminals-usb` (USB-C) | Identity + boot + wallet + 2FA       | Serial + device certificate    |
| Wristband  | BLE (custom profile)         | Mic stream + PTT button + biometric  | BLE MAC + device certificate   |
| Cap        | BLE + WebMIDI fallback       | Bone-conduction sensors + actuator   | BLE MAC + device certificate   |
| Watch      | BLE (standard GATT HRS)      | Read-only HR/HRV/temp                | BLE MAC (no private cert)      |

The USB is the root of trust. Other modules cross-sign through it at
pairing time — similar to Apple's secure enclave chain. Without the
USB, other modules fall back to a lower trust tier.

## 2 · Software interfaces

### 2.1 · `/dev/terminals-usb` device node

Linux: udev rule adds a symlink at `/dev/terminals-usb` when the device
is enumerated. The rule matches `idVendor=<tbd>` and
`idProduct=<tbd>` and sets permissions for the `terminals` group.

```udev
# /etc/udev/rules.d/99-terminals.rules
SUBSYSTEM=="usb", ATTRS{idVendor}=="<tbd>", ATTRS{idProduct}=="<tbd>", \
    SYMLINK+="terminals-usb", MODE="0660", GROUP="terminals", TAG+="uaccess"
```

The device exposes four logical endpoints:

| Endpoint | Path suffix    | Protocol             | Use                                   |
| -------- | -------------- | -------------------- | ------------------------------------- |
| `ident`  | `-ident`       | HID report           | Read serial + device cert             |
| `sign`   | `-sign`        | HID report           | Sign 32-byte payloads (Ed25519)       |
| `wallet` | `-wallet`      | HID report           | Wallet operations (locked by PIN)     |
| `led`    | `-led`         | HID report           | Set LED patterns                      |

macOS: we ship a DriverKit extension (code-signed by Apple team id);
Windows: WinUSB driver via INF. Library code uses `hid` Python bindings
under the hood but only imports lazily to preserve the carl-studio
import-time budget.

### 2.2 · `carl admin attest-device` command

Adds to `carl_studio/cli/` alongside the existing `carl admin unlock`.

```bash
carl admin attest-device
```

Flow:

1. Ensure `/dev/terminals-usb` (or OS equivalent) is present.
2. Read `ident` report: serial, device certificate, firmware hash.
3. Send a random 32-byte challenge to `sign` endpoint.
4. Verify the returned signature against the device certificate's
   public key.
5. Verify the device certificate chains to the Terminals root CA
   (pinned in `carl-studio` at `src/carl_studio/admin.py`).
6. Write `~/.carl/devices/<serial>.json` with:
   ```json
   {
     "serial": "...",
     "device_cert": "<PEM>",
     "attested_at": "2026-04-20T12:34:56Z",
     "fw_hash": "...",
     "tier": "admin",
     "binding_hash": "<HMAC of device pubkey + hw fingerprint>"
   }
   ```
7. Update `carl_studio.admin.is_admin()` to also accept
   device-cert-based unlock (in addition to env var + hw fingerprint).

No private keys leave the device; CARL only ever holds the cert and a
binding hash.

### 2.3 · WebSerial API shim

For browser-based pairing (Summercamp), we expose a thin shim at
`carl.camp/device/pair`:

```ts
const port = await navigator.serial.requestPort({
  filters: [{ usbVendorId: 0xTBD, usbProductId: 0xTBD }],
});
await port.open({ baudRate: 0 }); // HID, baudRate ignored
const cert = await deviceReadIdent(port);
const sig  = await deviceSign(port, challenge);
await fetch("/api/device/attest", { method: "POST", body: JSON.stringify({ cert, sig, challenge }) });
```

The page degrades gracefully: if `navigator.serial` is unavailable
(Firefox, Safari), it prompts the user to attest via CLI and paste the
resulting attestation code — a short base32 string good for 5 minutes.

### 2.4 · Bluetooth LE profile — wristband PTT

Custom GATT service. UUID root `C4E1-...-0000` (placeholder until
registered).

| Characteristic       | UUID suffix | Operation  | Semantics                                           |
| -------------------- | ----------- | ---------- | --------------------------------------------------- |
| `PTT_STATE`          | `0001`      | notify     | `0` released, `1` pressed; rising edge triggers mic |
| `MIC_STREAM`         | `0002`      | indication | 20 ms Opus frames, 16 kHz mono                      |
| `BATTERY_LEVEL`      | `0003`      | read+notify| 0–100, matches std BAS                              |
| `BIOMETRIC_PPG`      | `0004`      | notify     | 50 Hz PPG samples                                   |
| `HAPTIC_CMD`         | `0005`      | write      | Haptic pattern id (0–63)                            |
| `DEVICE_CERT`        | `0010`      | read       | Cert chain (chunked)                                |

Pairing uses LE Secure Connections with OOB data printed on the USB's
LED screen (four-digit code).

The agent's consumer side lives in
`src/carl_studio/hardware/wristband.py` (new module, lazy-imported).
Public API:

```python
from carl_studio.hardware import Wristband

async with Wristband.connect() as w:
    async for frame in w.mic_stream():
        agent.on_audio(frame)
```

### 2.5 · WebMIDI / BLE — cap sensors

The bone-conduction cap streams a 1 kHz MIDI-encoded vector: 8
channels, 14-bit values per channel. Two transports:

1. **BLE** (default): custom GATT service with a single
   `SENSOR_STREAM` indication.
2. **WebMIDI** (fallback for desktop testing): the cap pretends to be a
   MIDI controller; control changes carry the sensor values.

Consumer module: `src/carl_studio/hardware/cap.py`.

```python
from carl_studio.hardware import Cap

async with Cap.connect() as c:
    async for frame in c.sensor_stream():
        # frame.channels: list[float], len 8
        # frame.timestamp: float (monotonic)
        ...
```

The cap can also be driven as an actuator: `await c.vibrate(intensity,
duration_ms)`. A Resonant can bind its output to this actuator.

## 3 · A2A device-to-agent protocol

Each module talks to the CARL agent over the existing a2a bus
(`src/carl_studio/a2a/bus.py`). We reuse the Signal protocol rather
than inventing a new one.

### 3.1 · Message format

```json
{
  "type": "device.sensor_frame",
  "device_id": "cap-<serial>",
  "timestamp": 1714058400.123,
  "payload": { "channels": [0.12, 0.34, ...] },
  "signature": "<base64-ed25519>"
}
```

### 3.2 · Auth flow

1. Device signs message with its device key.
2. Agent verifies against the device certificate it already holds from
   attestation.
3. Messages without a valid signature are dropped (logged to ledger as
   `carl.device.auth_failed`).

Replay protection: every device maintains a 64-bit monotonic counter
included in the signed payload. Agent rejects non-monotonic counters.

## 4 · Resonant ↔ device bindings

The core payoff of the hardware is that Resonants can subscribe to
device streams and emit to actuators with low latency. Binding lives
in `carl.yaml`:

```yaml
resonants:
  breathing:
    hash: 3f1c...
    inputs:
      - source: cap.channel[2]    # bone-conduction channel 2
        normalize: zscore
      - source: watch.hrv
    outputs:
      - target: cap.vibration     # drive actuator
        clamp: [0, 1]
        smoothing_ms: 50
```

The binding layer lives in
`src/carl_studio/hardware/bindings.py`. On start, carl-studio reads
the config, connects to the named devices, and spins a tight asyncio
loop that reads inputs, calls `resonant.forward(obs)`, and writes the
output to the target actuator.

### 4.1 · Latency target

Round-trip latency (sensor sample → Resonant forward → actuator
response) target: **< 10 ms** measured end-to-end.

Budget:

| Stage                                   | Budget |
| --------------------------------------- | ------ |
| BLE indication delivery                 | 3 ms   |
| Deserialization + normalization         | 1 ms   |
| `Resonant.perceive → cognize → act`     | 2 ms   |
| BLE write (actuator command)            | 3 ms   |
| Slack                                   | 1 ms   |

At tree depth ≤ 3, `Resonant.forward` is well under 1 ms on any modern
CPU (numpy, closed-form eval). Depth 4 is ~2 ms; that eats the slack
and is the practical reason for `MAX_DEPTH=4`.

## 5 · Boot-from-USB spec

### 5.1 · Lite mode (default for v0.9.0)

The USB carries a small cross-platform launcher. On insert, the user
double-clicks `Terminals.app` (macOS), `Terminals.exe` (Windows), or
`terminals` (Linux). The launcher:

1. Presents a full-screen "CARL OS" skin (dark base, pulse pane,
   palette) that is actually an Electron/Tauri wrapper over the host
   OS.
2. Shells out to host `python` + carl-studio installed at
   `~/.carl/python/` (or bundled on the USB for air-gapped cases).
3. Locks the UI when the USB is unplugged; Resonants persist to host
   `~/.carl/` via T5's `OptimizerStateStore`.

The lite app is **signed** with our Apple Developer ID on macOS,
Authenticode on Windows, and AppImage signed + GPG-detached on Linux.

### 5.2 · Full mode (v0.10.0+)

A signed Linux image (≤ 2 GB) with Alpine + terminals-tech stack
pre-installed. Boot order: BIOS → USB → `grub.cfg` loads the image →
runs a systemd target that starts carl-studio + Summercamp as a local
web service on `http://localhost:7777`.

Image signing uses Ed25519 signatures over the image blob, verified at
boot by a shim loaded by the BIOS. The hardware USB itself holds the
verification key.

### 5.3 · Signing rule

All signed boot artifacts:

- Ed25519 signature over `sha256(image_bytes)`.
- Signed by the Terminals root key (held offline).
- Revocation list synced nightly when online.

## 6 · Fallback paths (no devices)

Every path listed must work with zero hardware. The dashboard's
first-run flow explicitly exercises the software-only path.

| With hardware                 | Without hardware                                       |
| ----------------------------- | ------------------------------------------------------ |
| Identity via USB cert         | Identity via carl.camp email + JWT                     |
| PTT via wristband             | Browser mic with `navigator.mediaDevices.getUserMedia` |
| Sensor stream from cap        | Simulated sensor stream from `simulated_channels.py`   |
| Hardware-signed uploads       | Software-signed uploads (lower-trust flag in metadata) |
| Attested marketplace download | Ed25519-signed download (no attestation chain)         |
| Boot from USB                 | Host install via `pipx install carl-studio[all]`       |

Software-only mode flags every durable artifact with
`trust_tier: "software"`. Marketplace listings from software-only users
are allowed at the paid tier but not the admin tier. This preserves the
hardware's real value (higher trust) without locking out early users.

## 7 · One-button experience mapping

The "Apple-ification" is a mapping from physical events to CARL
actions. It lives in `src/carl_studio/hardware/mapping.py`:

| Physical event               | Effect                                                        |
| ---------------------------- | ------------------------------------------------------------- |
| Power-on (laptop + USB in)   | Dashboard opens → heartbeat pane → no login screen            |
| Press wristband PTT          | Mic opens → 5 s window → audio → agent                        |
| Release PTT                  | Agent replies; reply routed to audio out + dashboard ledger   |
| Frisson spike on cap         | Logged as `bioreaction` event; Resonants with that binding can train against it |
| Unplug USB                   | Session locks; dashboard dims; Resonants persist to disk      |
| Re-plug USB within 5 min     | Session resumes without re-auth                               |
| Re-plug after > 5 min        | Re-attest via `carl admin attest-device` (silent if cert still valid) |
| Double-tap wristband         | Opens composition sheet (quick action)                         |
| Long-hold PTT 3 s            | "What did CARL notice?" — ledger observation spoken aloud     |

All mappings are overridable in `carl.yaml > hardware.mapping`.
Defaults stay opinionated; power users tune.

## 8 · Implementation order (software-first)

Build in this order; hardware can slot in at any stage.

1. `carl_studio/hardware/` package scaffolding + imports in lazy
   style (so carl-studio import time stays flat).
2. Simulated device shims (`SimulatedWristband`, `SimulatedCap`) that
   match the real API. Dashboard tests run against these in CI.
3. `carl admin attest-device` command against simulated USB.
4. Binding layer and Resonant ↔ simulated-device round-trip benchmark
   (verifies the < 10 ms budget).
5. WebSerial shim + Summercamp pairing page (against simulated).
6. Full BLE + WebMIDI + HID wiring against prototype hardware.
7. Boot-from-USB lite (Tauri), signed build artifacts.
8. Full-mode bootable image.

At each step, software-only fallback must remain fully functional.
Every step ships with a feature flag and defaults off until validated.

## 9 · Out of scope

- Hardware firmware development (external team).
- FCC / CE compliance (handled by hardware vendor).
- Mobile-native pairing (iOS / Android apps come after v0.9.0).
- Bluetooth Classic support; BLE only.
- Any binding that exceeds `MAX_DEPTH=4` after composition — the
  bindings layer refuses on construction.

## 10 · Acceptance checklist

The hardware interface ships when:

- [ ] `carl admin attest-device` works against a simulated USB in CI
      and produces a cert file the rest of the stack accepts.
- [ ] Simulated wristband + cap complete a full Resonant round trip in
      < 10 ms on the reference laptop.
- [ ] Dashboard renders heartbeat from simulated streams with no
      hardware present.
- [ ] WebSerial pairing degrades cleanly on Firefox/Safari.
- [ ] Lite-mode launcher ships signed on all three OSes.
- [ ] All hardware modules are lazy-imported; `import carl_studio`
      stays under the existing import-time budget.
- [ ] Consent flags (`hardware_enabled`, `bioreaction_logging`) remain
      off by default per `carl_studio.consent` policy.
