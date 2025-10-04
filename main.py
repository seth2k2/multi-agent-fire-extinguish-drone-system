import os, pygame, threading, math, sys, random, copy, json, re
from crewai import Agent, Task, Crew, Process
import os
import dotenv

dotenv.load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

GRID_SIZE = 20
PLAYGROUND_SIZE = 1000

# --- Shared state ---
shared_state = {
    "fires": [],      # active fires [(x,y,radius,spread_rate,age_seconds)]
    "burned": [],     # ash patches [(x,y,radius)]
    "drones": {},     # populated at startup based on user input
    "birds": [],      # [(x,y,vx,vy)]
    "user_instruction": None,
    "mode": "auto",
    "matrix": [],
    "game_over": False
}

# --- Initialize drones based on user input ---
def initialize_drones():
    try:
        raw = input("Enter number of drones (default 4): ").strip()
        count = int(raw) if raw else 4
    except Exception:
        count = 4
    count = max(1, min(50, count))  # safety bounds
    drones = {}
    for i in range(count):
        name = f"drone{i+1}"
        drones[name] = {"pos": (40, 40), "target": None, "assigned_fire": None}
    shared_state["drones"] = drones

initialize_drones()

# --- JSON-safe serialization for CrewAI ---
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

# --- Create matrix representation ---
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

# --- Movement helper ---
def move_towards(current, target, speed=2):
    cx, cy = current
    tx, ty = target
    dx, dy = tx - cx, ty - cy
    dist = math.hypot(dx, dy)
    if dist < speed: 
        return target
    return (cx + dx/dist*speed, cy + dy/dist*speed)

# --- Fire spread (indefinite until extinguished) ---
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

# --- Forest destroyed check ---
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

# --- Fire extinguishing ---
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

# --- Stable fire assignment ---
def assign_fires():
    fires = shared_state["fires"]
    # Docking station center at (40,40) for an 80x80 square starting at (0,0)
    docks = {name: (40, 40) for name in shared_state["drones"].keys()}
    if not fires:
        # No fires: dock drones automatically
        for name, drone in shared_state["drones"].items():
            drone["assigned_fire"] = None
            drone["target"] = docks[name]
        return
    
    # Drones keep current assignment if the center still exists
    fire_centers = {(f[0], f[1]) for f in fires}
    for name, drone in shared_state["drones"].items():
        if drone["assigned_fire"] and drone["assigned_fire"] in fire_centers:
            continue
        drone["assigned_fire"] = None
    # Unassigned drones should stay docked at top-left corner
        drone["target"] = docks[name]
    
    # Assign unassigned drones to closest fires
    available_fires = fires[:]
    for name, drone in shared_state["drones"].items():
        if drone["assigned_fire"] is None and available_fires:
            # find closest fire
            closest = min(available_fires, key=lambda f: math.hypot(f[0]-drone["pos"][0], f[1]-drone["pos"][1]))
            drone["assigned_fire"] = (closest[0], closest[1])
            drone["target"] = (closest[0], closest[1])
            available_fires.remove(closest)

# --- Bird flock ---
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

# --- Drone avoidance ---
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

# --- Formation helpers ---
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

# --- CrewAI LLM interpreter ---
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

# --- Terminal chat ---
def terminal_chat():
    while True:
        print("\n\nOperator chat ready. Type quit to exit.\nType commands:")
        msg = sys.stdin.readline().strip()
        if msg:
            low = msg.lower().strip()
            if low in ("quit", "exit", "q"):
                print("Quitting simulation...")
                try:
                    pygame.event.post(pygame.event.Event(pygame.QUIT))
                except Exception:
                    pass
                break
            result = crew.kickoff(inputs={
                "input": msg,
                "agent_count": len(shared_state["drones"]),
                "playground_size": PLAYGROUND_SIZE,
            })
            # Simple extraction: find first JSON object in the output; else action: none
            as_text = str(result)
            import re
            parsed = {"action": "none"}
            m = re.search(r"\{[\s\S]*?\}", as_text)
            if m:
                candidate = m.group(0)
                try:
                    parsed = json.loads(candidate)
                except Exception:
                    parsed = {"action": "none"}

            # Normalize to the small action set (add shape aliases)
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

            # Apply only if actionable
            if parsed.get("action") != "none":
                shared_state["user_instruction"] = parsed
                shared_state["mode"] = "auto" if parsed.get("action") == "resume" else "override"
                print(f"LLM parsed: {parsed}")
            else:
                print("Could not parse into a command; ignoring.")

chat_thread = threading.Thread(target=terminal_chat, daemon=True)
chat_thread.start()

# --- Pygame setup ---
pygame.init()
screen = pygame.display.set_mode((PLAYGROUND_SIZE, PLAYGROUND_SIZE))
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

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            # Prevent new fires from starting on ash
            if not any(math.hypot(mx - fx, my - fy) < fr for (fx, fy, fr) in shared_state["burned"]):
                shared_state["fires"].append((mx, my, 10, 0.2, 0.0))
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_b:  # Press B to release birds
                spawn_birds()
    
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
                    for t in targets:
                        shared_state["drones"][t]["target"] = docks[t]
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
                elif action == "move_to_fire":
                    fire = instr.get("fire")
                    if isinstance(fire, (list, tuple)) and len(fire) >= 2:
                        x, y = float(fire[0]), float(fire[1])
                        for t in targets:
                            shared_state["drones"][t]["target"] = (x, y)
                elif action == "check_fire":
                    # Patrol four corners cyclically for any number of drones
                    m = 50
                    maxc = PLAYGROUND_SIZE - m
                    patrol_points = [(m, m), (maxc, m), (maxc, maxc), (m, maxc)]
                    pts_len = len(patrol_points)
                    for i, (name,d) in enumerate(shared_state["drones"].items()):
                        d["target"] = patrol_points[i % pts_len]
                elif action == "resume":
                    shared_state["mode"] = "auto"
    
    # Move drones with bird avoidance
    if not shared_state["game_over"]:
        for d in shared_state["drones"].values():
            avoid = avoid_birds(d["pos"])
            if avoid:
                d["pos"] = move_towards(d["pos"], avoid, speed=3)
            elif d["target"]:
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
