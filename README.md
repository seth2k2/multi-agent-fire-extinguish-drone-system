## Multi-Agent Fire Extinguish Drone System

This repository contains a multi-agent simulation built with Pygame and CrewAI. The simulation models
a forest playground where user-created fires spread, autonomous drones extinguish fires, and an
LLM-based interpreter (CrewAI) converts operator chat into structured drone commands.

Entry point: `main.py`.

New / notable features (not in earlier READMEs)
- Sidebar control panel (right side of the window) with buttons: Spawn Birds, Add Drone, Remove Drone, Restart.
- In-GUI chat input in the sidebar with a "Send" button (disabled when the input is empty). The chat goes to the CrewAI interpreter — terminal input is no longer required.
- Starts with 4 drones by default; the "Restart" button resets the simulation and returns the drone count to 4.
- Per-drone override semantics: LLM commands may target specific drones (e.g. "dock drone3") or all drones. Targeted commands only affect listed drones; global commands affect everyone.
- Proportional allocation across multiple fires: drones are split among fires in proportion to a fire's severity (radius and spread rate), with fractional rounding and nearest-drone selection.
- Docking protection: drones en route to the dock are not pulled away mid-dock by new allocations (avoids interrupting docking operations).
- Drones target fire centers (with a small jitter) so they actually enter and extinguish fires reliably.
- Debug overlay: press D to toggle a debug view that draws per-drone target lines and shows per-drone state in the sidebar (useful for diagnosing assignment/resume issues).
- Clean runtime: the program no longer prints routine logs to the terminal (keeps the terminal clean).

Basic idea / how the system works
- Fires: left-click on the playground area to start a new fire. Fires grow over time and create ash (burned area). If the whole forest becomes ash, the simulation displays "FOREST DESTROYED!" and freezes.
- Drones: drones have position, a target, and an optional assigned fire center. In automatic mode (`auto`) drones are assigned to fires using the proportional allocation algorithm described above. Drones move to targets and reduce a fire's radius when inside the fire area.
- Birds: spawn birds using the "Spawn Birds" button in the sidebar. Drones will steer away from nearby birds.
- LLM / CrewAI interpreter: the in-GUI chat sends natural-language instructions to the CrewAI interpreter. The interpreter maps inputs to one of the supported actions (dock, move, check fire, resume, none) and the simulation applies the structured command.

Supported LLM actions (examples)
- dock (e.g. `dock drone3` or `everyone dock`) — send drone(s) back to the docking station.
- move (e.g. `move drone1 to 650,800` or shape commands like `make a circle`) — move specified drones to coordinates or computed formation points.
- check fire — instruct drones to patrol corners and inspect the playground.
- resume — resume normal operation (targeted resume clears override only for listed drones; `resume` with no targets resumes all).

Important behavior details
- Targeted vs global commands: When the LLM command includes specific drone names, only those drones are overridden (their `override` flag is set). Commands that apply to "all" switch the global mode to `override` and set overrides on all drones.
- Resume handling: resuming a drone clears its `override`, `target`, and `assigned_fire` so it becomes eligible for immediate reassignment by the main allocator. The main thread performs assignments (no cross-thread assignment calls) to avoid races.
- Chat UI: the Send button in the sidebar is disabled while the input is empty to prevent accidental empty messages. The sidebar shows the last user message.

Controls
- On startup the simulation creates 4 drones by default. Use the sidebar buttons to add/remove drones at runtime.
- In the Pygame window:
  - Left-click the playground (left area) to start a fire (clicks on ash are ignored).
  - Use the sidebar buttons:
    - Spawn Birds — create a flock that traverses the playground.
    - Add Drone — create a new drone at the dock.
    - Remove Drone — remove the most recently-created drone.
    - Restart — reset fires, ash, birds and user instructions, and reset the drone count and positions to the default (4 drones docked).
  - Debug overlay: press `D` to toggle debug drawing (lines to targets, per-drone statuses in sidebar).

CrewAI chat
- Use the text field in the right-side sidebar and click "Send" (or press Enter while the chat box is focused) to send a natural-language instruction to the interpreter.
- Example inputs:
  - everyone dock
  - dock drone1
  - move drone1 to 650,800
  - make a circle
  - resume

Requirements
- Python 3.10+
- pygame
- python-dotenv (optional)
- crewai (and any provider-specific libraries required by your chosen LLM)

See `requirements.txt` for the full list. Installing the full list enables the CrewAI features; installing only `pygame` and `python-dotenv` is enough to run the simulation without LLM control.

Run (Windows PowerShell)
1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies (recommended):

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Set your LLM API key in a `.env` file or environment variable (if you want CrewAI features):

```text
OPENAI_API_KEY=sk-...
```

or in PowerShell:

```powershell
$env:OPENAI_API_KEY = "your_api_key_here"
```

4. Run the simulation:

```powershell
python main.py
```

Notes & troubleshooting
- The runtime intentionally suppresses routine console logs to keep the terminal clean. If you need debugging output, enable the debug overlay (`D`) or temporarily re-enable prints in `main.py`.
- If CrewAI/LLM features fail, verify your API key and network access. The interpreter uses `gpt-4o` by default; change the model in `main.py` if needed.
- If images are missing, ensure the `images/` folder contains `forest.png` and `drone.png`; `bird.png` is optional (the code draws circles for birds when the image is missing).
- If Pygame cannot open a display (headless systems), run locally with a GUI-enabled session.
