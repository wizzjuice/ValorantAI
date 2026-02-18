# Valorant AI - TensorRT YOLOv8

A real-time object detection and aim-assistance tool built in C++ using **NVIDIA TensorRT**, **CUDA**, **DirectX 11 Desktop Duplication**, and an **Arduino HID** interface for mouse control.

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**. Using aim-assistance software in online multiplayer games violates the Terms of Service of virtually all game publishers and may result in permanent bans. The author takes no responsibility for misuse.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Dependencies](#dependencies)
- [Hardware Setup](#hardware-setup)
- [Build Instructions](#build-instructions)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Key Parameters](#key-parameters)
- [Known Limitations](#known-limitations)

---

## Overview

This program captures the screen in real time, runs a YOLOv8-based TensorRT inference engine on each frame, detects targets within a configurable circular field-of-view overlay, and moves the mouse toward the best target using a smoothed, adaptive aim algorithm — all communicated through a serial-connected Arduino acting as a hardware HID device.

---

## Features

- **Real-time screen capture** via DXGI Desktop Duplication API (DirectX 11)
- **TensorRT inference** for fast GPU-accelerated object detection
- **YOLOv8 output parsing** with confidence thresholding and Non-Maximum Suppression (NMS)
- **Persistent target tracking** with lock-on confidence, velocity estimation, and predictive aiming
- **Virtual hitbox system** that offsets the aim point toward the head of a detected target
- **Adaptive aim controller** using a second-order spring-damper motion model
- **Detection filtering** to reduce false positives by requiring targets to persist across multiple frames
- **Circular FOV overlay** rendered as a transparent always-on-top window
- **Arduino HID mouse control** over serial (COM port) to bypass software-level input hooks
- **Humanization layer** that adds subtle jitter to movement to appear more natural

---

## System Requirements

| Component | Minimum |
|-----------|---------|
| OS | Windows 10/11 (64-bit) |
| GPU | NVIDIA GPU with CUDA support (Turing or newer recommended) |
| CUDA | 11.x or later |
| TensorRT | 10.x (`nvinfer_10.lib`, `nvonnxparser_10.lib`) |
| DirectX | DirectX 11 |
| Arduino | Any Arduino with HID/Serial capability (e.g., Arduino Leonardo, Pro Micro) |
| COM Port | Arduino must be connected and visible as a COM port (default: `COM10`) |

---

## Dependencies

### Libraries (link via `#pragma comment`)
- `nvinfer_10.lib` — NVIDIA TensorRT inference engine
- `nvonnxparser_10.lib` — ONNX model parser for TensorRT
- `cudart.lib` — CUDA runtime
- `d3d11.lib` — Direct3D 11
- `dxgi.lib` — DXGI for desktop duplication
- `user32.lib` — Windows UI / input

### Headers Required
- `NvInfer.h`
- `NvOnnxParser.h`
- `cuda_runtime_api.h`
- `d3d11.h`, `dxgi1_2.h`

### TensorRT Engine File
Place your compiled `.engine` file at the path specified in `main()`:
```
C:\Users\<username>\OneDrive\Documents\ONNX\best.engine
```
This path is hardcoded and must be updated to match your environment.

---

## Hardware Setup

1. Flash your Arduino with firmware that accepts 3-byte serial commands:
   - `{'M', dx, dy}` — relative mouse move
   - `{'C', button, 0}` — mouse click
2. Connect the Arduino via USB.
3. Identify the COM port in Device Manager (e.g., `COM10`).
4. Update the `arduino.connect("\\\\.\\COM10")` call in `main()` if needed.

---

## Build Instructions

1. **Install prerequisites:**
   - Visual Studio 2019/2022 with C++17 support
   - CUDA Toolkit (matching your TensorRT version)
   - TensorRT SDK (headers + `.lib` files)

2. **Configure your project:**
   - Add TensorRT and CUDA include directories to the project include paths
   - Add TensorRT, CUDA, DirectX lib directories to linker paths
   - Set the Runtime Library to **Multi-threaded (/MT)** or **Multi-threaded DLL (/MD)** as appropriate

3. **Compile:**
   ```
   Build → Build Solution (Release x64)
   ```

4. **Place assets:**
   - Copy your `best.engine` file to the expected path
   - Ensure `cudart64_<version>.dll` and TensorRT DLLs are in the executable directory or on `PATH`

---

## Configuration

All primary tuning parameters are set in `main()` inside the `processingThread` lambda and in the `AdaptiveAimConfig` struct.

### Detection
| Parameter | Default | Description |
|-----------|---------|-------------|
| `confThreshold` | `0.40` | Minimum confidence score to consider a detection valid |
| `nmsThreshold` | `0.50` | IoU threshold for Non-Maximum Suppression |
| `targetClassId` | `1` | YOLO class index to target |
| `CIRCLE_RADIUS` | `80` | FOV circle radius in screen pixels |

### Aim Controller (`AdaptiveAimConfig`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `pixelsPerCount` | `2.2` | Mouse sensitivity scaling (pixels per HID count) |
| `minGain` | `0.35` | Gain applied when target is close |
| `maxGain` | `0.70` | Gain applied when target is far |
| `gainTransitionDistance` | `40.0` | Error distance (px) at which gain starts scaling up |
| `deadzone` | `0.8` | Minimum error in counts before any movement is sent |
| `maxSpeed` | `127.0` | Maximum HID movement per frame |
| `omega` | `0.55` | Spring-damper natural frequency |
| `zeta` | `1.0` | Spring-damper damping ratio (1.0 = critically damped) |
| `maxPredictionFrames` | `1.5` | Max frames of motion prediction applied |
| `minSpeedForPrediction` | `3.0` | Target speed (px/frame) required to enable prediction |
| `jitterAmount` | `0.2` | Humanization jitter magnitude |

### Target Tracker (`PersistentTargetTracker`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_LOCK_CONFIDENCE` | `0.55` | Minimum confidence to engage aim |
| `MAX_FRAMES_MISSING` | `15` | Frames without detection before target is dropped |
| `POSITION_MATCH_THRESHOLD` | `100.0` | Pixel distance to consider a detection the same target |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        Main Thread                       │
│   Keyboard polling (BACKSPACE to exit) + lifecycle mgmt  │
└───────────────┬────────────────────────────┬─────────────┘
                │                            │
     ┌──────────▼──────────┐     ┌───────────▼──────────┐
     │   Capture Thread    │     │   Overlay Thread     │
     │  DXGI Duplication   │     │  Transparent circle  │
     │  → FrameData queue  │     │  win32 window        │
     └──────────┬──────────┘     └──────────────────────┘
                │ (lock-free queue, max depth 1)
     ┌──────────▼───────────────────────────────────────┐
     │              Processing Thread                   │
     │  1. Preprocess (bilinear resize → CHW float)     │
     │  2. CUDA memcpy → TensorRT inference             │
     │  3. Parse detections + NMS                       │
     │  4. DetectionFilter (multi-frame confirmation)   │
     │  5. FOV circle intersection test                 │
     │  6. PersistentTargetTracker (lock + velocity)    │
     │  7. VirtualHitbox (head offset calculation)      │
     │  8. AdaptiveAimController → Arduino serial       │
     └──────────────────────────────────────────────────┘
```

---

## How It Works

### 1. Screen Capture
The DXGI Desktop Duplication API acquires frames directly from the GPU, copies them to a CPU-readable staging texture, and enqueues raw BGRA pixel data.

### 2. Preprocessing
Each frame is bilinearly downsampled to 640×640 and converted from BGRA uint8 to CHW float32 (normalized to [0, 1]) for TensorRT input.

### 3. Inference
The TensorRT engine runs asynchronously on a CUDA stream. Output is in the standard YOLOv8 format: `[batch, attributes, predictions]` where attributes are `[cx, cy, w, h, class0_conf, ..., classN_conf]`.

### 4. Detection Filtering
Raw detections pass through NMS and then a `DetectionFilter` that requires a bounding box to be confirmed across at least 2 consecutive frames, reducing noise and flickering.

### 5. FOV Gating
Only targets whose virtual hitbox intersects the circular overlay (centered on screen) are eligible for targeting.

### 6. Target Selection & Tracking
The best target is selected by a weighted score of proximity to center and bounding box size. `PersistentTargetTracker` maintains a locked target across frames, building confidence and estimating velocity for predictive aiming.

### 7. Aim Control
A second-order spring-damper system smoothly drives the error toward zero. Movement is scaled by an adaptive gain that increases with distance. Velocity prediction compensates for moving targets. Final HID delta values are sent to the Arduino over serial.

---

## Key Parameters to Tune First

1. **`pixelsPerCount`** — Must match your in-game mouse sensitivity. Increase if the aim overshoots; decrease if it undershoots.
2. **`CIRCLE_RADIUS`** — Controls how wide your targeting FOV is.
3. **`confThreshold`** — Lower values detect more targets but increase false positives.
4. **`targetClassId`** — Set to match the YOLO class index of your target object.
5. **COM Port** — Update `"\\\\.\\COM10"` to your Arduino's actual port.
6. **Engine path** — Update the hardcoded path to `best.engine`.

---

## Known Limitations

- The engine file path and COM port are **hardcoded** — they must be manually updated before building.
- Preprocessing is **CPU-bound**; moving it to CUDA would improve throughput.
- No support for **multi-monitor** setups — always captures the primary display (`EnumOutputs(0, ...)`).
- The overlay circle is **cosmetic only** and does not dynamically reflect the inference-space FOV radius when resolution differs from 640×640.
- The `DetectionFilter` uses a simple distance heuristic and may fail to correctly associate targets during fast motion.
