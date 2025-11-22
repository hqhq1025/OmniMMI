# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

如果我的要求或者指令不够清晰, 不够让你明确任务的详细要求, 请你先不要执行, 而是先向我确认, 明确需求和要求.

## Overview

OmniMMI is a comprehensive multi-modal interaction benchmark for evaluating OmniLLMs (Large Omni Language Models) in streaming video contexts. The benchmark includes 1,121+ interactive videos and 2,290+ questions across six subtasks: State Grounding (SG), Action Prediction (AP), Multi-turn Dependencies (MD), Proactive Turn-taking (PT), Proactive Alerting (PA), and Speaker Identification (SI).

## Key Architectural Components

### Model Integration Architecture

The codebase uses a plugin-style architecture where each baseline model has its own modeling file (e.g., `videochatgpt_modeling.py`, `longva_modeling.py`) that wraps the model-specific implementation. All model wrappers:

- Inherit from `ViLLMBaseModel` base class in `baselines/base.py`
- Implement a `generate()` method that accepts instruction and video path
- Are loaded via the `load_model()` function in `model_testing_zoo.py` or `evaluations/inference.py`

### Online vs Offline Models

The benchmark distinguishes between two model categories:

**Offline models** (most baselines): Process entire videos at once via `generate(instruction, video_path)`.

**Online models** (`VideoOnline`, `VideoLLaMBOnline`, `M4Online`, `M4-AudioOnline`): Process streaming video frame-by-frame with methods:
- `load_video(video_path)` - Load video for streaming
- `input_video_stream(timestamp)` - Feed frames incrementally
- `input_query_stream(question)` - Input queries at specific timestamps
- `()` - Get responses when available

### Configuration System

All paths are centralized in `baselines/config.json`:
- `CKPT_DIR`: Checkpoint directory (default: `checkpoints/`)
- `DATA_DIR`: Data directory (default: `../omnibench`)

The actual benchmark data is in `omnimmi/` with JSON files for each subtask.

## Common Development Commands

### Quick Model Testing

Debug a single model's inference pipeline:

```bash
cd baselines
python ../model_testing_zoo.py --model_name Gemini-1.5-pro
```

Supported models: `VideoChatGPT`, `VideoChat2`, `VideoLLaVA`, `LLaMA-VID`, `PLLaVA`, `PLLaVA-13B`, `PLLaVA-34B`, `LLaVA-NeXT-Video`, `LLaVA-NeXT-Video-34B`, `LongVA`, `LongVILA`, `LongLLaVA`, `VideoLLaMB`, `VideoOnline`, `VideoLLaMBOnline`, `Gemini-1.5-pro`, `GPT4O`, `InternLMXCO`, `VideoLLaMA2`, `M4`, `M4-Audio`, `MiniCPM-o`

### Running Full Benchmark Evaluation

**One-step evaluation (all subtasks):**

```bash
cd baselines
bash run_all.sh    # Run inference for all subtasks
bash eval_all.sh   # Evaluate all results
```

**Subtask-specific evaluation:**

```bash
cd baselines

# Action Prediction
bash run_ap.sh
bash eval_ap.sh

# State Grounding
bash run_sg.sh
bash eval_sg.sh

# Multi-turn Dependencies
bash run_md.sh
bash eval_md.sh

# Speaker Identification
bash run_si.sh
bash eval_si.sh

# Proactive Alerting
bash run_pa.sh
bash eval_pa.sh

# Proactive Turn-taking
bash run_pt.sh
bash eval_pt.sh
```

### Evaluation Pipeline Details

Each `run_*.sh` script:
1. Activates appropriate conda environment for each model
2. Runs `evaluations/inference.py` with multi-GPU support (splits work across chunks)
3. Concatenates chunk results into `results/{benchmark_name}_{model_name}.jsonl`

Each `eval_*.sh` script:
1. Runs `evaluations/evaluate.py` on the inference outputs
2. Uses GPT-based evaluation or rule-based scoring depending on task

### Custom Evaluation Script Setup

When modifying `run_*.sh` scripts, configure:

```bash
# Define models and their conda environments
model_names=("LongVA" "PLLaVA-13B")
environments=("llongva" "pllava")

# Set your conda path
source ~/scratch/anaconda3/bin/activate

# The script will iterate through models, activating each environment
```

## Dataset Structure

```
omnimmi/
├── clips/                          # Video clips for offline models
├── videos/                         # Full videos
├── action_prediction.json          # AP task annotations
├── dynamic_state_grounding.json    # SG task annotations
├── multiturn_dependency_reasoning.json  # MD task annotations
├── proactive_alerting.json         # PA task annotations
├── proactive_turntaking.json       # PT task annotations
└── speaker_identification.json     # SI task annotations
```

Each JSON contains entries with:
- `video`: Video filename
- `qa`: List of question-answer pairs with timestamps (for multi-turn tasks)

## Important Environment Setup Notes

1. **Decord configuration**: Set `export DECORD_EOF_RETRY_MAX=20480` before running evaluations to prevent decord-related errors.

2. **Audio input models**: For models supporting speech input (e.g., VideoLLaMA2, M4-Audio, VITA), install ChatTTS for TTS conversion:
   ```bash
   pip install ChatTTS num2words
   ```

3. **API-based models**:
   - Gemini: Requires Google API credentials
   - GPT-4o: Requires `.env` file in `baselines/gpt4o/` with `API_BASE` and `API_KEY`

## Adding a New Baseline Model

1. Create `baselines/{model_name}_modeling.py` implementing the `ViLLMBaseModel` interface
2. Add model loading logic to `load_model()` in both `model_testing_zoo.py` and `evaluations/inference.py`
3. Add checkpoint path mapping in the model loader (follows pattern: `f"{CKPT_DIR}/{model_checkpoint_dir}"`)
4. Add model name to the choices in argparse and to the appropriate model list in run scripts
5. Create or specify the conda environment for the model
6. For online models: implement streaming methods (`load_video`, `input_video_stream`, `input_query_stream`)

## Working with Baselines Directory

Each model in `baselines/` has its own subdirectory containing:
- Model-specific code and configurations
- Custom processor/encoder implementations
- Model wrapper that interfaces with the evaluation framework

Common baseline directories:
- `video_chatgpt/`, `videochat2/`, `videollava/` - Standard video-language models
- `longva/`, `longvila/`, `longllava/` - Long-context video models
- `intersuit/`, `intersuit_av/` - M4 model variants (video-only and audio-visual)
- `gemini/`, `gpt4o/` - API-based commercial models

## Task-Specific Characteristics

**Streaming-aware tasks** (require temporal state understanding):
- **AP** (Action Prediction): Predict next action before it occurs
- **SG** (State Grounding): Ground current state in streaming context
- **MD** (Multi-turn Dependencies): Resolve references across conversation turns (uses `##ANSWER##` token for dependencies)

**Proactive reasoning tasks** (require turn-taking and initiative):
- **PT** (Proactive Turn-taking): Distinguish between noise and legitimate queries
- **PA** (Proactive Alerting): Initiate responses at appropriate times
- **SI** (Speaker Identification): Identify who is speaking in the video

Multi-turn tasks (MD, SG) chain answers using `##ANSWER##` placeholder that gets replaced with previous answer.
