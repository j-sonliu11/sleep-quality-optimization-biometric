# sleep-quality-optimization-biometric  
**ML-Driven Biometric Wearable for Sleep Quality Optimization**  
*UC Riverside — Senior Design Project*

---

## Overview
This project builds an **ML-driven biometric wearable** that uses multimodal physiological signals to estimate sleep state in real time and support **gentle, stage-aware sleep optimization**. The system is designed to transform raw sensor streams into:

- **Live sleep staging and prediction**
- **Gentle sleep quality optimization interventions** (sound, light, temperature, etc.)
- **Actionable sleep recommendations** (bedtime guidance, room temperature targets, routines, etc.)

---

## What It Uses (Biometrics & Sensors)
Biometric signals and inputs may include:

- **EEG** (brain activity proxies for sleep staging)
- **EOG** (eye movement proxies for REM detection)
- **PPG** (heart rate / HRV signals)
- **Accelerometer** (motion, posture, restlessness)
- **Thermometer / temperature sensor** (skin or ambient temperature)
- *(Optional / future)* additional sensors for improved robustness

> Exact sensor selection depends on hardware revision and availability.

---

## Core Capabilities

### 1) Live Sleep Staging & Prediction
- Continuous signal streaming
- Preprocessing (filtering, artifact mitigation)
- Epoching (typically 30s windows; configurable)
- Feature extraction (time/frequency/HRV/motion/temperature)
- ML model inference to classify stages (e.g., **W, N1, N2, N3, REM**)

### 2) Gentle Optimization Interventions
Stage-aware interventions intended to be **non-disruptive**:
- Sound (e.g., subtle noise shaping / ambient sound)
- Light (e.g., dimming, warm light cues)
- Temperature (e.g., fan / thermostat recommendations or control hooks)

### 3) Actionable Recommendations
User-facing insights for sleep quality optimization:
- Suggested sleep schedule timing
- Environmental parameter recommendations (especially **temperature**)
- Habit and consistency reminders based on observed patterns

---

## Repository Notes
This README describes the intent and scope of the project. As the repo evolves, we recommend organizing into:

- `hardware/` (mechanical + electrical + BOM)
- `firmware/` (embedded drivers, BLE, sampling)
- `software/` (data collection, training, inference, dashboard)
- `docs/` (architecture, requirements, test plans, results)

---

## Data & Privacy
Sleep and biometric signals are sensitive.

- **Do not commit raw personal recordings** to a public repository.
- Prefer: local storage, de-identified samples, or private datasets.
- Add a `.gitignore` to exclude raw data and run artifacts.

Suggested `.gitignore` snippet:
```gitignore
data/raw/
data/processed/
runs/
.venv/
__pycache__/
.env
.vscode/
