import os, pygame, threading, math, sys, random, copy, json, re
from crewai import Agent, Task, Crew, Process
import os
import dotenv

dotenv.load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

GRID_SIZE = 20
PLAYGROUND_SIZE = 1000
# make sidebar wider to fit chat UI
SIDEBAR_WIDTH = 360
SCREEN_WIDTH = PLAYGROUND_SIZE + SIDEBAR_WIDTH

# Shared state
shared_state = {
    "fires": [],      # active fires [(x,y,radius,spread_rate,age_seconds)]
    "burned": [],     # ash patches [(x,y,radius)]
    "drones": {},     # populated at startup based on user input
    "birds": [],      # [(x,y,vx,vy)]
    "user_instruction": None,
    "mode": "auto",
    "debug_overlay": False,
    "matrix": [],
    "game_over": False
}

# Initialize drones based on user input
def initialize_drones(count: int = 0):
    # Do not prompt on startup. Start with `count` drones (default 0).
    count = max(0, min(50, int(count)))
    drones = {}
    for i in range(count):
        name = f"drone{i+1}"
        drones[name] = {"pos": (40, 40), "target": None, "assigned_fire": None, "override": False}
    shared_state["drones"] = drones

# start with zero drones; use Add Drone button to create drones during runtime
# start with 4 drones by default
initialize_drones(4)

# JSON-safe serialization for CrewAI
def serialize_state(state):
    safe = copy.deepcopy(state)
    safe["fires"] = [list(f) for f in safe["fires"]]
    safe["burned"] = [list(b) for b in safe.get("burned",[])]
    safe["birds"] = [list(b) for b in safe["birds"]]
    for d in safe["drones"].values():
        d["pos"] = list(d["pos"])
        if d["target"]:
            d["target"] = list(d["target"])
    return json.dumps(safe)

# Create matrix representation
def update_matrix():
    N = PLAYGROUND_SIZE // GRID_SIZE
    grid = [["grass" for _ in range(N)] for _ in range(N)]
    
    for (x,y,r,_,_) in shared_state["fires"]:
        for i in range(N):
            for j in range(N):
                cell_x, cell_y = i*GRID_SIZE, j*GRID_SIZE
                if (cell_x-x)**2 + (cell_y-y)**2 < r**2:
                    grid[j][i] = "fire"
    
    for (x,y,r) in shared_state["burned"]:
        for i in range(N):
            for j in range(N):
                cell_x, cell_y = i*GRID_SIZE, j*GRID_SIZE
                if (cell_x-x)**2 + (cell_y-y)**2 < r**2:
                    grid[j][i] = "ash"
    
    for name, d in shared_state["drones"].items():
        x,y = int(d["pos"][0]//GRID_SIZE), int(d["pos"][1]//GRID_SIZE)
        if 0 <= y < N and 0 <= x < N:
            grid[y][x] = name
    
    for (bx,by,_,_) in shared_state["birds"]:
        x,y = int(bx//GRID_SIZE), int(by//GRID_SIZE)
        if 0 <= y < N and 0 <= x < N:
            grid[y][x] = "bird"
    
    shared_state["matrix"] = grid

# Movement helper
def move_towards(current, target, speed=2):
    cx, cy = current
    tx, ty = target
    dx, dy = tx - cx, ty - cy
    dist = math.hypot(dx, dy)
    if dist < speed: 
        return target
    return (cx + dx/dist*speed, cy + dy/dist*speed)

# Fire spread (indefinite until extinguished)
def spread_fires():
    # Continuous spread; also convert inner area to ash after 4s exposure
    dt = 1/60
    new_fires = []
    for (x,y,r,rate,age) in shared_state["fires"]:
        r += rate
        age += dt
        # ash radius: inner disk that's been burning > 3s becomes ash
        v = rate * 60.0  # px per second
        ash_r = max(0.0, r - v * 3.0)
        if ash_r > 0:
            # update or append burned record for this center
            found = False
            for i,(bx,by,br) in enumerate(shared_state["burned"]):
                if bx == x and by == y:
                    shared_state["burned"][i] = (bx, by, max(br, ash_r))
                    found = True
                    break
            if not found:
                shared_state["burned"].append((x, y, ash_r))
        new_fires.append((x, y, r, rate, age))
    shared_state["fires"] = new_fires

# Forest destroyed check
def forest_destroyed():
    if not shared_state["burned"]:
        return False
    N = PLAYGROUND_SIZE // GRID_SIZE
    for i in range(N):
        for j in range(N):
            cell_x, cell_y = i*GRID_SIZE, j*GRID_SIZE
            covered = False
            for (bx,by,br) in shared_state["burned"]:
                if (cell_x - bx)**2 + (cell_y - by)**2 < br**2:
                    covered = True
                    break
            if not covered:
                return False
    return True

# Fire extinguishing
def extinguish_fires():
    to_remove = []
    for idx,(x,y,r,rate,age) in enumerate(shared_state["fires"]):
        for d in shared_state["drones"].values():
            dx,dy = d["pos"]
            if math.hypot(dx-x,dy-y) < r+15:
                r -= 0.5
        if r <= 5:
            # ensure a burned record exists and is at least radius 20
            ensured = False
            for i,(bx,by,br) in enumerate(shared_state["burned"]):
                if bx == x and by == y:
                    shared_state["burned"][i] = (bx, by, max(br, 20))
                    ensured = True
                    break
            if not ensured:
                shared_state["burned"].append((x,y,20))  # ash remains when extinguished
            to_remove.append(idx)
            # release drones assigned to this fire (by center)
            for drone in shared_state["drones"].values():
                if drone["assigned_fire"] == (x, y):
                    drone["assigned_fire"] = None
                    drone["target"] = None
        else:
            shared_state["fires"][idx] = (x,y,max(r,1),rate,age)
    for i in reversed(to_remove):
        shared_state["fires"].pop(i)

# Stable fire assignment
def assign_fires():
    # Sticky assignment: drones keep their assigned_fire until that fire is extinguished.
    fires = shared_state.get("fires", [])
    docks = {name: (40, 40) for name in shared_state["drones"].keys()}
    drone_names = list(shared_state["drones"].keys())
    total_drones = max(1, len(drone_names))
    if not fires:
        # No fires: dock drones automatically
        for name in drone_names:
            shared_state["drones"][name]["assigned_fire"] = None
            shared_state["drones"][name]["target"] = docks[name]
        return

    # Compute a weight for each fire: prefer larger radius and higher spread rate
    fire_weights = []
    for (x,y,r,rate,age) in fires:
        w = max(1.0, (r*r) * (1.0 + rate*5.0))
        fire_weights.append(w)

    total_weight = sum(fire_weights) or 1.0

    # Desired allocation per fire (floating) and integer rounding
    raw_alloc = [max(0, (w/total_weight) * total_drones) for w in fire_weights]
    alloc = [int(math.floor(x)) for x in raw_alloc]
    remaining = total_drones - sum(alloc)
    fractions = [(raw_alloc[i] - alloc[i], i) for i in range(len(raw_alloc))]
    fractions.sort(reverse=True)
    for k in range(remaining):
        idx = fractions[k % len(fractions)][1]
        alloc[idx] += 1

    # Count currently assigned drones per fire center and collect unassigned drones
    assigned_lists = { (f[0], f[1]): [] for f in fires }
    unassigned = set()
    # Tolerances for considering a drone targeting the dock
    DOCK_TARGET_TOLERANCE = 6.0
    DOCK_ARRIVAL_RADIUS = 8.0
    for name in drone_names:
        drone = shared_state["drones"][name]
        # Respect per-drone override: overridden drones should not be reallocated
        if drone.get("override"):
            af = drone.get("assigned_fire")
            if af and af in assigned_lists:
                assigned_lists[af].append(name)
            # overridden but no assigned_fire -> leave it alone (not in unassigned)
            continue
        af = drone.get("assigned_fire")
        if af and af in assigned_lists:
            assigned_lists[af].append(name)
            continue
        # If the drone's current target is the dock and it hasn't arrived yet,
        # treat it as 'docking' and do not include it in unassigned so it won't be pulled away.
        dock_pos = docks.get(name, (40, 40))
        tgt = drone.get("target")
        if tgt is not None:
            # if target is essentially the dock
            if math.hypot(tgt[0] - dock_pos[0], tgt[1] - dock_pos[1]) < DOCK_TARGET_TOLERANCE:
                # if drone hasn't arrived yet, skip adding to unassigned
                if math.hypot(drone["pos"][0] - dock_pos[0], drone["pos"][1] - dock_pos[1]) > DOCK_ARRIVAL_RADIUS:
                    continue
                # otherwise it has arrived; fall through and be considered unassigned
        # Normal unassigned drone
        unassigned.add(name)

    # Precompute drone positions for distance ranking
    drone_positions = {name: shared_state["drones"][name]["pos"] for name in drone_names}

    # For each fire, assign only the number of additional drones needed (do not pull currently assigned drones)
    for i, f in enumerate(fires):
        fx, fy, fr, rate, age = f
        center = (fx, fy)
        desired = alloc[i]
        currently = len(assigned_lists.get(center, []))
        need = max(0, desired - currently)
        if need <= 0:
            continue
        # pick nearest unassigned drones
        candidates = list(unassigned)
        candidates.sort(key=lambda n: math.hypot(drone_positions[n][0] - fx, drone_positions[n][1] - fy))
        chosen = candidates[:need]
        # Target the fire center (with a small jitter) so drones enter the burning area
        for j, name in enumerate(chosen):
            jitter_x = random.uniform(-4.0, 4.0)
            jitter_y = random.uniform(-4.0, 4.0)
            tx = fx + jitter_x
            ty = fy + jitter_y
            shared_state["drones"][name]["assigned_fire"] = center
            shared_state["drones"][name]["target"] = (tx, ty)
            unassigned.discard(name)

    # Any remaining unassigned drones assist the largest fire (or dock if none)
    if unassigned:
        largest_idx = max(range(len(fires)), key=lambda i: fires[i][2] if fires else 0)
        fx, fy, fr, rate, age = fires[largest_idx]
        center = (fx, fy)
        # Send remaining unassigned drones to the fire center with jitter
        for j, name in enumerate(list(unassigned)):
            jitter_x = random.uniform(-4.0, 4.0)
            jitter_y = random.uniform(-4.0, 4.0)
            tx = fx + jitter_x
            ty = fy + jitter_y
            shared_state["drones"][name]["assigned_fire"] = center
            shared_state["drones"][name]["target"] = (tx, ty)
            unassigned.discard(name)

# Bird flock
def spawn_birds():
    # Spawn a flock that traverses the playground once and exits
    flock = []
    for i in range(8):
        side = random.choice(["left", "right", "top", "bottom"])
        base_speed = random.uniform(0.8, 1.6)
        drift = random.uniform(-0.4, 0.4)
        if side == "left":
            x, y = -20, random.randint(0, PLAYGROUND_SIZE)
            vx, vy = base_speed, drift
        elif side == "right":
            x, y = PLAYGROUND_SIZE + 20, random.randint(0, PLAYGROUND_SIZE)
            vx, vy = -base_speed, drift
        elif side == "top":
            x, y = random.randint(0, PLAYGROUND_SIZE), -20
            vx, vy = drift, base_speed
        else:  # bottom
            x, y = random.randint(0, PLAYGROUND_SIZE), PLAYGROUND_SIZE + 20
            vx, vy = drift, -base_speed
        flock.append([x, y, vx, vy])
    shared_state["birds"] = flock

def update_birds():
    # Birds traverse in straight lines and exit; they don't react to drones
    new_birds = []
    margin = 50
    for (x,y,vx,vy) in shared_state["birds"]:
        x += vx
        y += vy
        if -margin <= x <= PLAYGROUND_SIZE + margin and -margin <= y <= PLAYGROUND_SIZE + margin:
            new_birds.append([x,y,vx,vy])
    shared_state["birds"] = new_birds

# Drone avoidance
def avoid_birds(drone_pos):
    # Increase avoidance radius and steer directly away from nearest bird
    danger_radius = 70
    nearest = None
    nearest_dist = None
    for (bx,by,vx,vy) in shared_state["birds"]:
        d = math.hypot(drone_pos[0]-bx, drone_pos[1]-by)
        if d < danger_radius and (nearest_dist is None or d < nearest_dist):
            nearest = (bx, by)
            nearest_dist = d
    if nearest is not None and nearest_dist > 1e-3:
        dx = drone_pos[0] - nearest[0]
        dy = drone_pos[1] - nearest[1]
        ux, uy = dx/nearest_dist, dy/nearest_dist
        avoid_target = (drone_pos[0] + ux*25, drone_pos[1] + uy*25)
        return avoid_target
    return None

# Formation helpers
def circle_positions(n, center=None, radius=None, margin=70):
    if n <= 0:
        return []
    if center is None:
        center = (PLAYGROUND_SIZE/2, PLAYGROUND_SIZE/2)
    if radius is None:
        radius = (PLAYGROUND_SIZE/2) - margin
        radius = max(50, radius)
    cx, cy = center
    pts = []
    for i in range(n):
        theta = 2*math.pi * i / n
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        # clamp to bounds
        x = max(0+margin, min(PLAYGROUND_SIZE-margin, x))
        y = max(0+margin, min(PLAYGROUND_SIZE-margin, y))
        pts.append((x, y))
    return pts

# CrewAI LLM interpreter
interpreter_agent = Agent(
    llm="gpt-4o",
    role="Drone Command Interpreter",
    goal="Interpret user instructions and convert them into structured drone commands",
    backstory="You are the control tower AI. You receive operator instructions and map them "
              "to predefined drone actions, considering the shared matrix state of the playground.",
    memory=True,
    verbose=True
)

interpreter_task = Task(
    description=(
        "You will receive an operator instruction and the playground state. "
        "Map the operator instruction to exactly one of the allowed actions.\n\n"
        "Allowed actions:\n"
        "- everyone dock\n"
        "- dock drone1, dock drone2 (one or more specific drones)\n"
        "- check fire\n"
        "- move droneX to location (x,y)\n"
        "- resume\n\n"
    "Operator instruction: {input}\n"
    "Context: There are {agent_count} drones. Playground size is {playground_size}x{playground_size} pixels.\n\n"
    "Output format: Respond with a single JSON object using only these keys:\n"
    "- action: one of ['dock','check fire','move','resume','none']\n"
    "- targets: optional array of drone names like ['drone1','drone2'] or ['all']\n"
        "- location: when action is 'move', provide an array of strings with coordinates for each target or in agent order, e.g. ['x1,y1','x2,y2']\n\n"
        "If asked to form a shape like 'make a circle', respond with action 'move', targets ['all'], and compute 'location' with {agent_count} evenly spaced points on a circle centered near ({playground_size/2},{playground_size/2}) with a radius that keeps all points inside bounds.\n\n"
        "If the instruction cannot be mapped to the allowed actions, respond with {\"action\": \"none\"}.\n\n"
        "Example outputs (one per response):\n"
        "{\"action\": \"check fire\"}\n"
        "{\"action\": \"resume\"}\n"
        "{\"action\": \"dock\", \"targets\":[\"drone1\"]}\n"
        "{\"action\": \"dock\", \"targets\":[\"drone1\",\"drone2\"]}\n"
        "{\"action\": \"move\", \"targets\":[\"drone1\",\"drone2\"], \"location\":[\"650,800\",\"100,200\"]}\n"
        "{\"action\": \"move\", \"targets\":[\"all\"], \"location\":[\"500,150\",\"650,250\",\"650,450\",\"500,550\",\"350,450\",\"350,250\"]}"
    ),
    expected_output="A single JSON object: {\"action\":'<action>', \"targets\":[agent1,agent2...], \"location\":[\"location of agent1 in x1,y1 format\", \"location of agent2 in x2,y2 format\"] }.",
    agent=interpreter_agent
)

crew = Crew(agents=[interpreter_agent], tasks=[interpreter_task], process=Process.sequential)

# Terminal input removed: GUI chat (sidebar) is used instead. No blocking stdin thread is started.

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, PLAYGROUND_SIZE))
clock = pygame.time.Clock()

background = pygame.image.load("images/forest.png")
background = pygame.transform.scale(background, (PLAYGROUND_SIZE, PLAYGROUND_SIZE))
drone_img = pygame.image.load("images/drone.png")
drone_img = pygame.transform.scale(drone_img, (45,45))
bird_img = None
try:
    tmp = pygame.image.load("images/bird.png").convert_alpha()
    bird_img = pygame.transform.scale(tmp, (32, 32))
except Exception:
    bird_img = None

# (ui_font and ui_buttons will be initialized after the _build_ui_buttons function is defined)

# UI buttons (in-game)
def add_drone():
    # Create a new drone with next available numeric suffix up to 50
    existing = list(shared_state["drones"].keys())
    nums = []
    for n in existing:
        m = re.search(r"(\d+)$", n)
        if m:
            try:
                nums.append(int(m.group(1)))
            except Exception:
                pass
    next_idx = 1
    if nums:
        next_idx = max(nums) + 1
    next_idx = min(next_idx, 50)
    name = f"drone{next_idx}"
    # avoid collisions: find first free name if stepping over existing
    i = 1
    while name in shared_state["drones"] and i <= 100:
        i += 1
        name = f"drone{next_idx + i}"
    shared_state["drones"][name] = {"pos": (40, 40), "target": None, "assigned_fire": None, "override": False}


def remove_drone():
    # Remove the drone with the highest numeric suffix (last created)
    names = list(shared_state["drones"].keys())
    if not names:
        return
    best = None
    best_num = -1
    for n in names:
        m = re.search(r"(\d+)$", n)
        if m:
            try:
                val = int(m.group(1))
            except Exception:
                val = 0
        else:
            val = 0
        if val > best_num:
            best_num = val
            best = n
    if best is None:
        # fallback: pop last
        best = names[-1]
    shared_state["drones"].pop(best, None)

def restart_simulation():
    """Reset the simulation state so a new session can start.
    Keeps the current drone set (so Add/Remove persists), but resets their positions
    and clears fires, burned patches, birds and instructions.
    """
    # clear dynamic elements
    shared_state["fires"] = []
    shared_state["burned"] = []
    shared_state["birds"] = []
    shared_state["user_instruction"] = None
    shared_state["mode"] = "auto"
    shared_state["matrix"] = []
    shared_state["game_over"] = False
    # reset drones to default 4
    initialize_drones(4)

# Chat input (no history)
chat_input = ""
chat_active = False

def send_chat_message(msg: str):
    """Send a chat message to CrewAI in the background and store the last user message
    in shared_state['last_user_input'] for sidebar display. The AI reply is printed to
    console (no persistent chat history).
    """
    if not msg:
        return
    # record last user input for sidebar display
    shared_state["last_user_input"] = msg
    def _worker():
        try:
            result = crew.kickoff(inputs={
                "input": msg,
                "agent_count": len(shared_state["drones"]),
                "playground_size": PLAYGROUND_SIZE,
            })
            # Parse the AI reply for a JSON command object (fall back to action 'none')
            as_text = str(result)
            parsed = {"action": "none"}
            m = re.search(r"\{[\s\S]*?\}", as_text)
            if m:
                candidate = m.group(0)
                try:
                    parsed = json.loads(candidate)
                except Exception:
                    parsed = {"action": "none"}

            # Normalize the parsed action into our internal action set
            if isinstance(parsed, dict):
                act = str(parsed.get("action", "none")).strip().lower()
                if act in ("everyone dock", "dock all", "all dock", "dock everyone"):
                    parsed = {"action": "dock", "targets": ["all"]}
                elif act in ("check fire", "check_fire"):
                    parsed = {"action": "check_fire"}
                elif act in ("resume", "continue", "auto"):
                    parsed = {"action": "resume"}
                elif "circle" in act:
                    # Fallback: generate a circle formation if not provided
                    n = len(shared_state["drones"])
                    pts = circle_positions(n)
                    locs = [f"{int(x)},{int(y)}" for (x,y) in pts]
                    parsed = {"action": "move", "targets": ["all"], "location": locs}
                elif act == "dock":
                    # keep given targets if present
                    pass
                elif act == "move":
                    # keep targets and location as provided
                    pass
                else:
                    parsed = {"action": "none"}

            # If actionable, apply parsed instruction immediately for targeted drones.
            if parsed.get("action") != "none":
                shared_state["user_instruction"] = parsed
                action = parsed.get("action")
                targets = parsed.get("targets")
                if isinstance(targets, str):
                    targets = [targets]
                # determine if this instruction applies to all drones
                explicit_all = (targets is None) or (isinstance(targets, list) and any(t == "all" for t in targets))
                # normalize targets list to actual drone names when possible
                if not explicit_all:
                    targets = [t for t in (targets or []) if t in shared_state["drones"]]
                else:
                    targets = list(shared_state["drones"].keys())

                docks = {name: (40, 40) for name in shared_state["drones"].keys()}

                # Apply action immediately to targets and set per-drone override flags
                if action == "dock":
                    for t in targets:
                        shared_state["drones"][t]["target"] = docks[t]
                        shared_state["drones"][t]["override"] = True
                    # if instruction was for all, set global override mode so auto allocator pauses
                    shared_state["mode"] = "override" if explicit_all else "auto"
                elif action == "move":
                    loc = parsed.get("location")
                    # build per_targets similarly to main loop
                    per_targets = []
                    if isinstance(loc, list):
                        for item in loc:
                            if isinstance(item, str):
                                parts = [p for p in re.split(r"[,\s]+", item) if p]
                                if len(parts) >= 2:
                                    try:
                                        per_targets.append((float(parts[0]), float(parts[1])))
                                    except Exception:
                                        per_targets.append(None)
                            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                                try:
                                    per_targets.append((float(item[0]), float(item[1])))
                                except Exception:
                                    per_targets.append(None)
                    elif isinstance(loc, str):
                        parts = [p for p in re.split(r"[,\s]+", loc) if p]
                        if len(parts) >= 2:
                            try:
                                xy = (float(parts[0]), float(parts[1]))
                                per_targets = [xy] * len(targets)
                            except Exception:
                                per_targets = []
                    elif isinstance(loc, (list, tuple)) and len(loc) >= 2:
                        try:
                            xy = (float(loc[0]), float(loc[1]))
                            per_targets = [xy] * len(targets)
                        except Exception:
                            per_targets = []

                    # Assign targets accordingly
                    if per_targets:
                        for i, t in enumerate(targets):
                            idx = min(i, len(per_targets) - 1)
                            xy = per_targets[idx]
                            if xy:
                                shared_state["drones"][t]["target"] = xy
                                shared_state["drones"][t]["override"] = True
                    # set global mode only if instruction targeted all
                    shared_state["mode"] = "override" if explicit_all else "auto"
                elif action == "move_to_fire":
                    fire = parsed.get("fire")
                    if isinstance(fire, (list, tuple)) and len(fire) >= 2:
                        x, y = float(fire[0]), float(fire[1])
                        for t in targets:
                            shared_state["drones"][t]["target"] = (x, y)
                            shared_state["drones"][t]["override"] = True
                    shared_state["mode"] = "override" if explicit_all else "auto"
                elif action == "check_fire":
                    m = 50
                    maxc = PLAYGROUND_SIZE - m
                    patrol_points = [(m, m), (maxc, m), (maxc, maxc), (m, maxc)]
                    pts_len = len(patrol_points)
                    for i, (name,d) in enumerate(shared_state["drones"].items()):
                        d["target"] = patrol_points[i % pts_len]
                        d["override"] = True
                    shared_state["mode"] = "override"
                elif action == "resume":
                    # If specific targets provided, clear override only for those drones
                    if parsed.get("targets") and not (isinstance(parsed.get("targets"), list) and any(t == "all" for t in parsed.get("targets") or [])):
                        for t in targets:
                            if t in shared_state["drones"]:
                                # clear override, target and assigned_fire so the main allocator can reassign
                                shared_state["drones"][t]["override"] = False
                                shared_state["drones"][t]["target"] = None
                                shared_state["drones"][t]["assigned_fire"] = None
                    else:
                        # resume for all drones
                        for dn in shared_state["drones"].keys():
                            # clear override, target and assigned_fire for all drones
                            shared_state["drones"][dn]["override"] = False
                            shared_state["drones"][dn]["target"] = None
                            shared_state["drones"][dn]["assigned_fire"] = None
                    shared_state["mode"] = "auto"
                    # Signal main loop to reallocate on next tick so resumed drones get assignments
                    shared_state["recompute_assignments"] = True

                pass
                # Clear stored instruction (we apply commands directly in the worker)
                shared_state["user_instruction"] = None
                # If we changed overrides/targets, signal the main loop to recompute assignments
                shared_state["recompute_assignments"] = True
            else:
                pass
        except Exception:
            pass
    threading.Thread(target=_worker, daemon=True).start()

# Button layout will be computed after fonts are initialized; we store button metadata here
ui_buttons = []

# UI font and button setup
def _build_ui_buttons():
    # create button metadata each frame (positions are static relative to screen)
    labels = [
        ("Spawn Birds", spawn_birds),
        ("Add Drone", add_drone),
        ("Remove Drone", remove_drone),
        ("Restart", restart_simulation),
    ]
    padding_x = 12
    padding_y = 6
    spacing = 8
    # Sidebar lives on the right of the playground
    y_base = 60
    x_right = PLAYGROUND_SIZE + SIDEBAR_WIDTH - 10
    btns = []
    for i, (label, cb) in enumerate(labels):
        surf = ui_font.render(label, True, (255,255,255))
        w = surf.get_width() + padding_x*2
        h = surf.get_height() + padding_y*2
        rect = pygame.Rect(x_right - w, y_base + i*(h + spacing), w, h)
        btns.append({"rect": rect, "label": label, "surf": surf, "callback": cb, "w": w, "h": h, "pad_x": padding_x, "pad_y": padding_y})
    return btns

# initialize UI font and prebuild buttons now that pygame is initialized
ui_font = pygame.font.SysFont(None, 22)
# build buttons now so they are available during event handling
ui_buttons = _build_ui_buttons()


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            # If click is inside the sidebar area, handle UI; else create fires on the playground
            in_sidebar = mx >= PLAYGROUND_SIZE
            if in_sidebar:
                # translate click coords to screen coords used by UI (no change needed)
                pass
            # Check UI buttons first (buttons occupy the sidebar area)
            clicked_button = None
            for b in ui_buttons:
                if b["rect"].collidepoint(mx, my):
                    clicked_button = b
                    break
            if clicked_button is not None:
                # call the associated callback
                try:
                    clicked_button["callback"]()
                except Exception:
                    pass
            else:
                # click was on the playground (left area): Prevent new fires from starting on ash
                if mx < PLAYGROUND_SIZE and not any(math.hypot(mx - fx, my - fy) < fr for (fx, fy, fr) in shared_state["burned"]):
                    shared_state["fires"].append((mx, my, 10, 0.2, 0.0))
            # chat input box focus (centered in sidebar)
            chat_height = 28
            chat_x = PLAYGROUND_SIZE + 12
            chat_y = PLAYGROUND_SIZE//2 - 20
            input_rect = pygame.Rect(chat_x, chat_y, SIDEBAR_WIDTH - 24, chat_height)
            send_rect = pygame.Rect(chat_x + (SIDEBAR_WIDTH - 24)//2 - 40, chat_y + chat_height + 8, 80, 28)
            if in_sidebar and input_rect.collidepoint(mx, my):
                chat_active = True
            elif in_sidebar and send_rect.collidepoint(mx, my):
                # Send button clicked — only send if non-empty
                if chat_input.strip():
                    send_chat_message(chat_input.strip())
                    chat_input = ""
                chat_active = False
            elif in_sidebar:
                chat_active = False
        if event.type == pygame.KEYDOWN:
            if chat_active:
                if event.key == pygame.K_RETURN:
                    if chat_input.strip():
                        send_chat_message(chat_input.strip())
                    chat_input = ""
                    chat_active = False
                elif event.key == pygame.K_BACKSPACE:
                    chat_input = chat_input[:-1]
                else:
                    ch = event.unicode
                    if ch:
                        chat_input += ch
                # swallow other key handling when typing
                continue
            # normal key handling
            # Toggle debug overlay with D key
            if event.key == pygame.K_d:
                shared_state["debug_overlay"] = not shared_state.get("debug_overlay", False)
                continue
            
    
    if not shared_state["game_over"]:
        update_matrix()
        spread_fires()
        extinguish_fires()
        update_birds()
        if forest_destroyed():
            shared_state["game_over"] = True
            shared_state["mode"] = "override"  # freeze behavior
    
    if not shared_state["game_over"]:
        if shared_state["mode"] == "auto":
            assign_fires()
        else:
            instr = shared_state["user_instruction"]
            if instr:
                action = instr.get("action")
                targets = instr.get("targets")
                # normalize targets: if missing or ["all"], use all drones
                if isinstance(targets, str):
                    targets = [targets]
                if not targets or (isinstance(targets, list) and any(t == "all" for t in targets)):
                    targets = list(shared_state["drones"].keys())
                # ensure targets are iterable of valid drone names
                targets = [t for t in targets if t in shared_state["drones"]]

                docks = {name: (40, 40) for name in shared_state["drones"].keys()}

                # accept alias with space/underscore for check fire
                if action in ("check fire", "check_fire"):
                    action = "check_fire"
                if action == "dock":
                    # If targets provided explicitly, override only those drones.
                    # If targets is all (or missing), override all drones.
                    explicit_all = (instr.get("targets") is None) or (isinstance(instr.get("targets"), list) and any(t == "all" for t in instr.get("targets") or []))
                    for t in targets:
                        shared_state["drones"][t]["target"] = docks[t]
                        # set per-drone override
                        shared_state["drones"][t]["override"] = True
                    if explicit_all:
                        # ensure all drones are marked override
                        for dn in shared_state["drones"].keys():
                            shared_state["drones"][dn]["override"] = True
                elif action == "move":
                    loc = instr.get("location")
                    # Support a list of 'x,y' strings for per-agent targets
                    per_targets = []
                    if isinstance(loc, list):
                        for item in loc:
                            if isinstance(item, str):
                                parts = [p for p in re.split(r"[,\s]+", item) if p]
                                if len(parts) >= 2:
                                    try:
                                        per_targets.append((float(parts[0]), float(parts[1])))
                                    except Exception:
                                        per_targets.append(None)
                            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                                try:
                                    per_targets.append((float(item[0]), float(item[1])))
                                except Exception:
                                    per_targets.append(None)
                    elif isinstance(loc, str):
                        # Single string applies to all targets
                        parts = [p for p in re.split(r"[,\s]+", loc) if p]
                        if len(parts) >= 2:
                            try:
                                xy = (float(parts[0]), float(parts[1]))
                                per_targets = [xy] * len(targets)
                            except Exception:
                                per_targets = []
                    elif isinstance(loc, (list, tuple)) and len(loc) >= 2:
                        # Single pair applies to all
                        try:
                            xy = (float(loc[0]), float(loc[1]))
                            per_targets = [xy] * len(targets)
                        except Exception:
                            per_targets = []

                    # If no explicit targets provided, map by agent order
                    agent_names = list(shared_state["drones"].keys())
                    if not instr.get("targets") and per_targets:
                        # clamp length to available agents
                        count = min(len(per_targets), len(agent_names))
                        for i in range(count):
                            xy = per_targets[i]
                            if xy:
                                shared_state["drones"][agent_names[i]]["target"] = xy
                    else:
                        # Assign per provided target names; if fewer locations than targets, reuse the last
                        if per_targets:
                            for i, t in enumerate(targets):
                                idx = min(i, len(per_targets) - 1)
                                xy = per_targets[idx]
                                if xy:
                                    shared_state["drones"][t]["target"] = xy
                    # mark targeted drones as override; if instruction didn't specify explicit targets, treat as all
                    explicit_all = (instr.get("targets") is None) or (isinstance(instr.get("targets"), list) and any(t == "all" for t in instr.get("targets") or []))
                    for t in targets:
                        shared_state["drones"][t]["override"] = True
                    if explicit_all:
                        for dn in shared_state["drones"].keys():
                            shared_state["drones"][dn]["override"] = True
                elif action == "move_to_fire":
                    fire = instr.get("fire")
                    if isinstance(fire, (list, tuple)) and len(fire) >= 2:
                        x, y = float(fire[0]), float(fire[1])
                        for t in targets:
                            shared_state["drones"][t]["target"] = (x, y)
                        # mark targeted drones override
                        for t in targets:
                            shared_state["drones"][t]["override"] = True
                elif action == "check_fire":
                    # Patrol four corners cyclically for any number of drones
                    m = 50
                    maxc = PLAYGROUND_SIZE - m
                    patrol_points = [(m, m), (maxc, m), (maxc, maxc), (m, maxc)]
                    pts_len = len(patrol_points)
                    for i, (name,d) in enumerate(shared_state["drones"].items()):
                        d["target"] = patrol_points[i % pts_len]
                    # mark all drones override for this command
                    for dn in shared_state["drones"].keys():
                        shared_state["drones"][dn]["override"] = True
                elif action == "resume":
                    # Determine if this resume targets all drones
                    explicit_all = (instr.get("targets") is None) or (isinstance(instr.get("targets"), list) and any(t == "all" for t in instr.get("targets") or []))
                    # If specific targets provided, clear override only for those drones
                    if instr.get("targets") and not explicit_all:
                        for t in targets:
                            if t in shared_state["drones"]:
                                # clear override, target and assigned_fire so the allocator can reassign them
                                shared_state["drones"][t]["override"] = False
                                shared_state["drones"][t]["target"] = None
                                shared_state["drones"][t]["assigned_fire"] = None
                        # log resumed drones and clear stored user instruction so it is not reapplied repeatedly
                        print(f"Resumed drones: {targets}")
                        # clear stored user instruction so it is not reapplied repeatedly
                        shared_state["user_instruction"] = None
                    else:
                        # resume for all drones
                        for dn in shared_state["drones"].keys():
                            shared_state["drones"][dn]["override"] = False
                            shared_state["drones"][dn]["target"] = None
                            shared_state["drones"][dn]["assigned_fire"] = None
                        print("Resumed all drones")
                        # only switch global mode back to auto when resuming all
                        shared_state["mode"] = "auto"
                    # ensure resumed drones are considered by the allocator immediately
                    shared_state["recompute_assignments"] = True
    # If a background worker requested recompute, do it on the main thread to avoid races
    if shared_state.get("recompute_assignments"):
        try:
            assign_fires()
        except Exception:
            pass
        shared_state["recompute_assignments"] = False
    
    # Move drones with bird avoidance
    if not shared_state["game_over"]:
        for d in shared_state["drones"].values():
            # Bird avoidance has priority
            avoid = avoid_birds(d["pos"])
            if avoid:
                d["pos"] = move_towards(d["pos"], avoid, speed=3)
                continue
            # If not avoiding birds, move to assigned target (or dock)
            if d["target"]:
                d["pos"] = move_towards(d["pos"], d["target"]) 
    
    # Render
    screen.blit(background, (0,0))
    # Fire overlay rendering first (semi-transparent red)
    fire_layer = pygame.Surface((PLAYGROUND_SIZE, PLAYGROUND_SIZE), pygame.SRCALPHA)
    for (x,y,r,_,_) in shared_state["fires"]:
        pygame.draw.circle(fire_layer, (255, 0, 0, 130), (int(x), int(y)), int(r))
    screen.blit(fire_layer, (0,0))
    # Draw docking station (80x80 black square) at (0,0)
    pygame.draw.rect(screen, (0,0,0), pygame.Rect(0, 0, 80, 80))
    # Ash overlay rendering on top so inner area shows ash
    ash_layer = pygame.Surface((PLAYGROUND_SIZE, PLAYGROUND_SIZE), pygame.SRCALPHA)
    for (x,y,r) in shared_state["burned"]:
        pygame.draw.circle(ash_layer, (30,30,30,200), (int(x),int(y)), int(r))
    screen.blit(ash_layer, (0,0))
    # Drones
    for d in shared_state["drones"].values():
        screen.blit(drone_img, (int(d["pos"][0]) - 22, int(d["pos"][1]) - 22))
    for (bx,by,_,_) in shared_state["birds"]:
        if bird_img:
            screen.blit(bird_img, (int(bx) - 16, int(by) - 16))
        else:
            pygame.draw.circle(screen, (255,255,0), (int(bx),int(by)), 6)

    # Debug overlay: draw lines from drones to their targets and list statuses
    if shared_state.get("debug_overlay"):
        dbg_font = pygame.font.SysFont(None, 18)
        sx = PLAYGROUND_SIZE + 12
        sy = 300
        i = 0
        for name, d in shared_state["drones"].items():
            # draw line to target if exists
            if d.get("target"):
                try:
                    tx, ty = d["target"]
                    pygame.draw.line(screen, (100,220,100), (int(d["pos"][0]), int(d["pos"][1])), (int(tx), int(ty)), 1)
                    pygame.draw.circle(screen, (100,220,100), (int(tx), int(ty)), 3)
                except Exception:
                    pass
            # draw assigned_fire center marker
            if d.get("assigned_fire"):
                try:
                    fx, fy = d["assigned_fire"]
                    pygame.draw.circle(screen, (220,180,60), (int(fx), int(fy)), 4)
                except Exception:
                    pass
            # safely build status string
            try:
                px = int(d['pos'][0]); py = int(d['pos'][1])
            except Exception:
                px = int(d['pos'][0]) if d.get('pos') else 0
                py = int(d['pos'][1]) if d.get('pos') else 0
            tgt_str = 'None'
            if d.get('target'):
                try:
                    tgt_str = f"({int(d['target'][0])},{int(d['target'][1])})"
                except Exception:
                    tgt_str = str(d.get('target'))
            af_str = str(d.get('assigned_fire'))
            ov_str = str(d.get('override'))
            status = f"{name}: pos=({px},{py}) tgt={tgt_str} af={af_str} ov={ov_str}"
            surf = dbg_font.render(status, True, (200,200,200))
            screen.blit(surf, (sx, sy + i*18))
            i += 1

    # Draw UI buttons (rebuild to account for dynamic font/rendering)
    ui_buttons = _build_ui_buttons()
    # Sidebar background
    sidebar_rect = pygame.Rect(PLAYGROUND_SIZE, 0, SIDEBAR_WIDTH, PLAYGROUND_SIZE)
    pygame.draw.rect(screen, (20, 24, 30), sidebar_rect)
    # Sidebar title
    title_font = pygame.font.SysFont(None, 28)
    title_surf = title_font.render("Control Panel", True, (220,220,220))
    screen.blit(title_surf, (PLAYGROUND_SIZE + 16, 16))
    # Drone count
    count_font = pygame.font.SysFont(None, 24)
    drone_count = len(shared_state.get("drones", {}))
    count_surf = count_font.render(f"Drones: {drone_count}", True, (200,200,200))
    screen.blit(count_surf, (PLAYGROUND_SIZE + 16, 46))
    for b in ui_buttons:
        # button background
        pygame.draw.rect(screen, (30,30,30), b["rect"], border_radius=6)
        # border
        pygame.draw.rect(screen, (200,200,200), b["rect"], width=1, border_radius=6)
        # label
        label_x = b["rect"].x + b["pad_x"]
        label_y = b["rect"].y + b["pad_y"]
        screen.blit(b["surf"], (label_x, label_y))

    # Chat input area (centered)
    chat_height = 28
    chat_x = PLAYGROUND_SIZE + 12
    chat_y = PLAYGROUND_SIZE//2 - 20
    input_rect = pygame.Rect(chat_x, chat_y, SIDEBAR_WIDTH - 24, chat_height)
    title_font = pygame.font.SysFont(None, 20)
    title = title_font.render("Enter user command", True, (200,200,200))
    screen.blit(title, (chat_x, chat_y - 26))
    color = (60, 60, 60) if not chat_active else (80, 80, 120)
    pygame.draw.rect(screen, color, input_rect, border_radius=6)
    pygame.draw.rect(screen, (200,200,200), input_rect, width=1, border_radius=6)
    # render chat_input text trimmed to fit
    max_chars = 40
    txt_surf = ui_font.render(chat_input[-max_chars:], True, (240,240,240))
    screen.blit(txt_surf, (input_rect.x + 6, input_rect.y + 4))
    # Send button below input
    send_rect = pygame.Rect(chat_x + (SIDEBAR_WIDTH - 24)//2 - 40, chat_y + chat_height + 8, 80, 28)
    send_enabled = bool(chat_input.strip())
    send_bg = (50, 120, 50) if send_enabled else (80,80,80)
    pygame.draw.rect(screen, send_bg, send_rect, border_radius=6)
    pygame.draw.rect(screen, (200,200,200), send_rect, width=1, border_radius=6)
    send_label = ui_font.render("Send", True, (255,255,255) if send_enabled else (160,160,160))
    s_x = send_rect.x + (send_rect.w - send_label.get_width())//2
    s_y = send_rect.y + (send_rect.h - send_label.get_height())//2
    screen.blit(send_label, (s_x, s_y))

    # Show last user input (only the last message) below the send button
    last_msg = shared_state.get("last_user_input")
    if last_msg:
        lm_font = pygame.font.SysFont(None, 18)
        lm_surf = lm_font.render(f"Last: {last_msg[:40]}", True, (180,180,180))
        screen.blit(lm_surf, (chat_x, send_rect.y + send_rect.h + 8))

    # HUD: show mode and last accepted command
    hud_font = pygame.font.SysFont(None, 40)
    line1 = hud_font.render(f"Mode: {shared_state.get('mode')}", True, (255,255,255))
    w = line1.get_width() + 12
    h = line1.get_height() + 8
    hud_bg = pygame.Surface((w, h), pygame.SRCALPHA)
    hud_bg.fill((0, 0, 0, 140))
    # Position HUD at top-right with a 10px margin
    base_x = PLAYGROUND_SIZE - w - 10
    base_y = 10
    screen.blit(hud_bg, (base_x, base_y))
    screen.blit(line1, (base_x + 6, base_y + 4))

    if shared_state["game_over"]:
        overlay = pygame.Surface((PLAYGROUND_SIZE, PLAYGROUND_SIZE), pygame.SRCALPHA)
        overlay.fill((0,0,0,160))
        screen.blit(overlay, (0,0))
        font = pygame.font.SysFont(None, 48)
        text = font.render("FOREST DESTROYED!", True, (255, 80, 80))
        rect = text.get_rect(center=(PLAYGROUND_SIZE//2, PLAYGROUND_SIZE//2))
        screen.blit(text, rect)
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
