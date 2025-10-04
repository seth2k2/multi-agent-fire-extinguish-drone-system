## Multi-Agent Fire Extinguish Drone System

This repository contains a simple multi-agent simulation built with Pygame and CrewAI.
The simulation models a forest playground where user-created fires spread, autonomous drones attempt
to extinguish fires, and a small LLM-based interpreter (CrewAI) maps operator chat commands into
structured drone commands.

The main entry point is `main.py`.

## Basic idea / how the system works
- Fires: left-click on the playground window to start a new fire. Fires grow over time and leave
  ash (burned area) behind. If entire forest is turned to ash, the simulation reports
  "FOREST DESTROYED!".
- Drones: a number of drone agents are created at startup (you'll be prompted for a count). Each
  drone has a position, a target and an optional assigned fire center. In `auto` mode the system
  assigns drones to active fires (closest-first). Drones move toward their target and will reduce
  a fire's radius when close enough.
- Birds: press `B` in the Pygame window to spawn a flock of birds. Drones try to avoid nearby birds.
- CrewAI interpreter: a simple LLM-based agent listens on a terminal chat thread for operator
  instructions and attempts to map them into actions like docking, moving, or patrolling. When the
  interpreter accepts a command it places a structured instruction into the shared state and the
  simulation switches into `override` mode to follow the instruction.

Core behaviors implemented in `main.py` (high level):
- fire spreading and ash conversion
- drone assignment and docking
- LLM-based instruction parsing and override mode
- bird flock spawning and drone avoidance
- matrix/grid serialization for LLM context

## Controls
- On startup you'll be prompted in the terminal for the number of drones (default: 4).
- In the Pygame window:
  - Left-click to start a fire at the mouse position (click on grass only — fires are blocked on ash).
  - Press `B` to spawn a flock of birds.
  - Close the window or type `quit` / `exit` in the terminal to stop the simulation.
- Terminal chat: type natural language operator instructions in the terminal where `main.py` is running.
  The CrewAI interpreter will attempt to translate those instructions into one of the supported
  actions (dock, move, check fire, resume). Example inputs:
  - "everyone dock"
  - "dock drone1"
  - "move drone1 to 650,800"
  - move drone1 to 650,800, drone 2 to 500,500"    `note that 500,500 are x,y coordinates in the playgorund which is a 1000 into 1000 square`
  - "make a circle"

If the interpreter can map your instruction it will print the parsed JSON command in the terminal
and the simulation will follow it.

## Requirements
This project uses many packages in `requirements.txt`. The minimal packages needed to run the
simulation core are (at least):
- Python 3.10+ (recommended)
- pygame
- python-dotenv (optional, used to load `OPENAI_API_KEY` from a `.env` file)
- crewai (the LLM orchestration library used in the project) and any credentials it needs

The repository already contains a full `requirements.txt`. Installing the full list will ensure
the CrewAI integrations and LLM clients work correctly.

## Run (Windows PowerShell)
1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies (recommended to install the full `requirements.txt`):

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# OR if you want a minimal install:
# python -m pip install pygame python-dotenv crewai
```

3. Set your OpenAI (or another LLM provider) API key. You can create a `.env` file in the repo root
   with the following content:

```text
OPENAI_API_KEY=sk-...
```

   Or set it in the PowerShell session for a one-off run:

```powershell
$env:OPENAI_API_KEY = "your_api_key_here"
```

4. Run the simulation:

```powershell
python main.py
```

You will be prompted for the number of drones. After the Pygame window opens you can interact
with it as described in Controls.

## Notes & troubleshooting
- If Pygame fails to open a display on headless or remote environments, run locally with an
  available display session.
- If the CrewAI/LLM parts don't work, check that `OPENAI_API_KEY` (or other provider credentials
  required by your LLM library) are set and that you have network access. The interpreter uses
  `gpt-4o` in the code; change the model in `main.py` if your account doesn't have access.
- If images are missing, ensure the `images/` folder is present and contains `forest.png`,
  `drone.png`, and optionally `bird.png`. The code will fall back to a circle for birds if
  `bird.png` is not available.
- The provided `requirements.txt` is large; for quick experimentation installing only `pygame`
  and `python-dotenv` lets you run the simulation without LLM control (you can still interact
  manually via the GUI). To use CrewAI features, install the rest and ensure your LLM access is
  configured.

# multi-agent-fire-extinguish-drone-system
