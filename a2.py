import tkinter as tk
import heapq
import math
import time
import random

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
ROWS        = 20
COLS        = 28
CELL        = 28
PANEL_W     = 220
ANIM_DELAY  = 8           # ms per visited-node frame (lower = faster)
AGENT_DELAY = 150         # ms between agent steps
OBS_PROB    = 0.003       # dynamic obstacle spawn probability per cell per step

# Colors
C_EMPTY    = "#FFFFFF"
C_WALL     = "#2D2D2D"
C_START    = "#27AE60"
C_GOAL     = "#E74C3C"
C_VISITED  = "#AED6F1"
C_PATH     = "#82E0AA"
C_AGENT    = "#F39C12"
C_GRID     = "#BDC3C7"
C_BG       = "#F0F3F4"
C_PANEL    = "#1A252F"
C_BTN_RUN  = "#2E86C1"
C_BTN_RST  = "#1E8449"
C_BTN_MAZE = "#7D3C98"
C_BTN_CLR  = "#922B21"
C_BTN_SG   = "#145A32"
C_TEXT     = "#FFFFFF"
C_LABEL    = "#85C1E9"
C_VAL      = "#F9E79F"
C_SEC      = "#5DADE2"
C_STATUS   = "#82E0AA"
C_TIP      = "#717D7E"


# ─────────────────────────────────────────────
# HEURISTICS
# ─────────────────────────────────────────────
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


# ─────────────────────────────────────────────
# GRID HELPERS
# ─────────────────────────────────────────────
def make_grid(density=0.0):
    g = [[0]*COLS for _ in range(ROWS)]
    if density > 0:
        for r in range(ROWS):
            for c in range(COLS):
                if random.random() < density:
                    g[r][c] = 1
    return g

def get_neighbors(pos, grid):
    r, c = pos
    result = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and grid[nr][nc] == 0:
            result.append((nr, nc))
    return result

def reconstruct(came_from, goal):
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path


# ─────────────────────────────────────────────
# SEARCH ALGORITHMS
# ─────────────────────────────────────────────
def gbfs(grid, start, goal, h_fn):
    """Greedy Best-First Search."""
    counter = 0
    heap = [(h_fn(start, goal), counter, start)]
    came_from = {start: None}
    visited = {start}
    order = []

    while heap:
        _, _, cur = heapq.heappop(heap)
        order.append(cur)
        if cur == goal:
            return reconstruct(came_from, goal), order, len(order)
        for nb in get_neighbors(cur, grid):
            if nb not in visited:
                visited.add(nb)
                came_from[nb] = cur
                counter += 1
                heapq.heappush(heap, (h_fn(nb, goal), counter, nb))
    return None, order, len(order)


def astar(grid, start, goal, h_fn):
    """A* Search."""
    counter = 0
    g_cost = {start: 0}
    heap = [(h_fn(start, goal), counter, start)]
    came_from = {start: None}
    expanded = set()
    order = []

    while heap:
        f, _, cur = heapq.heappop(heap)
        if cur in expanded:
            continue
        expanded.add(cur)
        order.append(cur)
        if cur == goal:
            return reconstruct(came_from, goal), order, len(order)
        for nb in get_neighbors(cur, grid):
            ng = g_cost[cur] + 1
            if nb not in g_cost or ng < g_cost[nb]:
                g_cost[nb] = ng
                came_from[nb] = cur
                counter += 1
                heapq.heappush(heap, (ng + h_fn(nb, goal), counter, nb))
    return None, order, len(order)


# ─────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("AI2002 – Dynamic Pathfinding Agent")
        self.root.configure(bg=C_PANEL)
        self.root.resizable(False, False)

        # ── State ──
        self.grid      = make_grid()
        self.start     = (1, 1)
        self.goal      = (ROWS-2, COLS-2)
        self.grid[self.start[0]][self.start[1]] = 0
        self.grid[self.goal[0]][self.goal[1]]   = 0

        self.algorithm  = tk.StringVar(value="A*")
        self.heuristic  = tk.StringVar(value="Manhattan")
        self.dynamic_on = tk.BooleanVar(value=False)

        self.path          = []
        self.visited_set   = set()
        self.path_set      = set()
        self.agent_pos     = None
        self.agent_idx     = 0
        self._visited_list = []
        self._anim_idx     = 0
        self._drawing      = None   # True=place wall, False=erase
        self._placing      = None   # 'start' or 'goal'
        self._anim_job     = None
        self._agent_job    = None
        self._replan_count = 0

        # Metric variables
        self.var_nodes  = tk.StringVar(value="0")
        self.var_cost   = tk.StringVar(value="0")
        self.var_time   = tk.StringVar(value="0")
        self.var_replan = tk.StringVar(value="0")
        self.var_status = tk.StringVar(value="Configure settings & press Run Search")

        self._build_ui()
        self._draw_full_grid()

    # ═══════════════════════════════════════════
    # UI CONSTRUCTION
    # ═══════════════════════════════════════════
    def _build_ui(self):
        cw = COLS * CELL
        ch = ROWS * CELL

        # Canvas
        self.canvas = tk.Canvas(self.root, width=cw, height=ch,
                                bg=C_BG, highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, padx=(8,4), pady=8, sticky="n")
        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, '_drawing', None))
        self.canvas.bind("<ButtonPress-3>",   self._on_erase)
        self.canvas.bind("<B3-Motion>",       self._on_erase)

        # Side panel frame
        pf = tk.Frame(self.root, bg=C_PANEL, width=PANEL_W)
        pf.grid(row=0, column=1, sticky="ns", padx=(4,8), pady=8)
        pf.grid_propagate(False)

        r = 0
        # Header
        tk.Label(pf, text="AI2002 Pathfinding", bg=C_PANEL, fg=C_SEC,
                 font=("Arial",13,"bold")).grid(row=r, column=0, columnspan=2, pady=(14,2)); r+=1
        tk.Label(pf, text="Assignment 2 – Question 6", bg=C_PANEL, fg=C_TIP,
                 font=("Arial",9,"italic")).grid(row=r, column=0, columnspan=2, pady=(0,8)); r+=1

        # Algorithm
        self._sec(pf, "Algorithm", r); r+=1
        for alg in ["A*", "GBFS"]:
            tk.Radiobutton(pf, text=alg, variable=self.algorithm, value=alg,
                           bg=C_PANEL, fg=C_TEXT, selectcolor=C_BTN_RUN,
                           activebackground=C_PANEL, font=("Arial",10)
                           ).grid(row=r, column=0, sticky="w", padx=18); r+=1

        # Heuristic
        self._sec(pf, "Heuristic", r); r+=1
        for h in ["Manhattan", "Euclidean"]:
            tk.Radiobutton(pf, text=h, variable=self.heuristic, value=h,
                           bg=C_PANEL, fg=C_TEXT, selectcolor=C_BTN_RUN,
                           activebackground=C_PANEL, font=("Arial",10)
                           ).grid(row=r, column=0, sticky="w", padx=18); r+=1

        # Dynamic mode
        self._sec(pf, "Dynamic Mode", r); r+=1
        tk.Checkbutton(pf, text="Enable dynamic obstacles",
                       variable=self.dynamic_on, bg=C_PANEL, fg=C_TEXT,
                       selectcolor=C_BTN_RUN, activebackground=C_PANEL,
                       font=("Arial",10)).grid(row=r, column=0, sticky="w",
                                               padx=18, pady=(0,4)); r+=1

        # Control buttons
        self._sec(pf, "Controls", r); r+=1
        for txt, cmd, col in [
            ("▶  Run Search",  self._run,        C_BTN_RUN),
            ("↺  Reset Grid",  self._reset,       C_BTN_RST),
            ("⚡  New Maze",    self._new_maze,    C_BTN_MAZE),
            ("✕  Clear Walls", self._clear_walls, C_BTN_CLR),
        ]:
            tk.Button(pf, text=txt, command=cmd, bg=col, fg=C_TEXT,
                      font=("Arial",10,"bold"), relief="flat", pady=5,
                      activebackground=C_PANEL, activeforeground=C_TEXT,
                      cursor="hand2").grid(row=r, column=0, columnspan=2,
                                           sticky="ew", padx=12, pady=3); r+=1

        # Place start/goal
        self._sec(pf, "Place Start / Goal", r); r+=1
        for txt, key in [("📍 Set Start", "start"), ("🎯 Set Goal", "goal")]:
            tk.Button(pf, text=txt, bg=C_BTN_SG, fg=C_TEXT,
                      font=("Arial",9), relief="flat", pady=4, cursor="hand2",
                      command=lambda k=key: self._set_placing(k)
                      ).grid(row=r, column=0, columnspan=2, sticky="ew",
                             padx=12, pady=2); r+=1

        # Metrics
        self._sec(pf, "Metrics", r); r+=1
        for lbl, var in [("Nodes Expanded", self.var_nodes),
                         ("Path Cost",      self.var_cost),
                         ("Time (ms)",      self.var_time),
                         ("Replans",        self.var_replan)]:
            tk.Label(pf, text=lbl+":", bg=C_PANEL, fg=C_LABEL,
                     font=("Arial",9)).grid(row=r, column=0, sticky="w", padx=12)
            tk.Label(pf, textvariable=var, bg=C_PANEL, fg=C_VAL,
                     font=("Arial",10,"bold")).grid(row=r, column=1, sticky="e", padx=12)
            r+=1

        # Status
        self._sec(pf, "Status", r); r+=1
        tk.Label(pf, textvariable=self.var_status, bg=C_PANEL, fg=C_STATUS,
                 font=("Arial",9,"italic"), wraplength=PANEL_W-24,
                 justify="left").grid(row=r, column=0, columnspan=2,
                                      sticky="w", padx=12, pady=4); r+=1

        # Legend
        self._sec(pf, "Legend", r); r+=1
        for col, name in [(C_START,"Start"), (C_GOAL,"Goal"), (C_AGENT,"Agent"),
                          (C_PATH,"Final Path"), (C_VISITED,"Visited"), (C_WALL,"Wall")]:
            f = tk.Frame(pf, bg=C_PANEL)
            f.grid(row=r, column=0, columnspan=2, sticky="w", padx=12, pady=1); r+=1
            tk.Label(f, bg=col, width=2, relief="solid").pack(side="left", padx=(0,6))
            tk.Label(f, text=name, bg=C_PANEL, fg=C_TEXT,
                     font=("Arial",9)).pack(side="left")

        # Tips
        tk.Label(pf, text="Left-drag: draw walls\nRight-drag: erase walls",
                 bg=C_PANEL, fg=C_TIP, font=("Arial",8,"italic"),
                 justify="left").grid(row=r, column=0, columnspan=2,
                                      sticky="w", padx=12, pady=(10,6))

    def _sec(self, parent, text, row):
        tk.Label(parent, text=f"── {text} ──", bg=C_PANEL, fg=C_SEC,
                 font=("Arial",9,"bold")).grid(row=row, column=0, columnspan=2,
                                               sticky="w", padx=12, pady=(8,2))

    # ═══════════════════════════════════════════
    # GRID DRAWING
    # ═══════════════════════════════════════════
    def _cell_color(self, r, c):
        pos = (r, c)
        if pos == self.start:      return C_START
        if pos == self.goal:       return C_GOAL
        if pos == self.agent_pos:  return C_AGENT
        if self.grid[r][c] == 1:   return C_WALL
        if pos in self.path_set:   return C_PATH
        if pos in self.visited_set:return C_VISITED
        return C_EMPTY

    def _draw_cell(self, r, c):
        tag = f"c{r}_{c}"
        self.canvas.delete(tag)
        x1, y1 = c*CELL+1, r*CELL+1
        x2, y2 = x1+CELL-2, y1+CELL-2
        fill = self._cell_color(r, c)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline="", tags=tag)
        pos = (r, c)
        if pos == self.start:
            self.canvas.create_text(x1+CELL//2-1, y1+CELL//2-1, text="S",
                fill="white", font=("Arial",8,"bold"), tags=tag)
        elif pos == self.goal:
            self.canvas.create_text(x1+CELL//2-1, y1+CELL//2-1, text="G",
                fill="white", font=("Arial",8,"bold"), tags=tag)
        elif pos == self.agent_pos:
            self.canvas.create_oval(x1+3, y1+3, x2-3, y2-3,
                fill="#C0392B", outline="white", width=1, tags=tag)

    def _draw_full_grid(self):
        self.canvas.delete("all")
        for r in range(ROWS+1):
            self.canvas.create_line(0, r*CELL, COLS*CELL, r*CELL, fill=C_GRID)
        for c in range(COLS+1):
            self.canvas.create_line(c*CELL, 0, c*CELL, ROWS*CELL, fill=C_GRID)
        for r in range(ROWS):
            for c in range(COLS):
                self._draw_cell(r, c)

    def _redraw(self, cells):
        for r, c in cells:
            self._draw_cell(r, c)

    # ═══════════════════════════════════════════
    # MOUSE EVENTS
    # ═══════════════════════════════════════════
    def _rc(self, event):
        c, r = event.x // CELL, event.y // CELL
        if 0 <= r < ROWS and 0 <= c < COLS:
            return r, c
        return None

    def _on_press(self, event):
        rc = self._rc(event)
        if not rc: return
        r, c = rc
        # Placing start/goal
        if self._placing in ('start', 'goal'):
            old = self.start if self._placing == 'start' else self.goal
            self.grid[old[0]][old[1]] = 0
            if self._placing == 'start': self.start = (r, c)
            else:                        self.goal  = (r, c)
            self.grid[r][c] = 0
            self._placing = None
            self.var_status.set("Placed. Press Run Search.")
            self._redraw([old, (r, c)])
            return
        if (r, c) in [self.start, self.goal]: return
        self._drawing = (self.grid[r][c] == 0)
        self.grid[r][c] = 1 if self._drawing else 0
        self._redraw([(r, c)])

    def _on_drag(self, event):
        if self._drawing is None: return
        rc = self._rc(event)
        if not rc: return
        r, c = rc
        if (r, c) in [self.start, self.goal]: return
        self.grid[r][c] = 1 if self._drawing else 0
        self._redraw([(r, c)])

    def _on_erase(self, event):
        rc = self._rc(event)
        if not rc: return
        r, c = rc
        if (r, c) in [self.start, self.goal]: return
        self.grid[r][c] = 0
        self._redraw([(r, c)])

    def _set_placing(self, key):
        self._placing = key
        self.var_status.set(f"Click grid to place {'Start' if key=='start' else 'Goal'}...")

    # ═══════════════════════════════════════════
    # CONTROLS
    # ═══════════════════════════════════════════
    def _cancel_jobs(self):
        if self._anim_job:
            self.root.after_cancel(self._anim_job); self._anim_job = None
        if self._agent_job:
            self.root.after_cancel(self._agent_job); self._agent_job = None

    def _clear_state(self):
        self.path = []; self.path_set = set()
        self.visited_set = set()
        self.agent_pos = None; self.agent_idx = 0
        self._visited_list = []; self._anim_idx = 0
        self._replan_count = 0
        self.var_nodes.set("0"); self.var_cost.set("0")
        self.var_time.set("0");  self.var_replan.set("0")

    def _reset(self):
        self._cancel_jobs()
        self.grid = make_grid()
        self.grid[self.start[0]][self.start[1]] = 0
        self.grid[self.goal[0]][self.goal[1]]   = 0
        self._clear_state()
        self._draw_full_grid()
        self.var_status.set("Grid reset.")

    def _new_maze(self):
        self._cancel_jobs()
        self.grid = make_grid(density=0.28)
        self.grid[self.start[0]][self.start[1]] = 0
        self.grid[self.goal[0]][self.goal[1]]   = 0
        self._clear_state()
        self._draw_full_grid()
        self.var_status.set("New maze generated. Press Run.")

    def _clear_walls(self):
        self._cancel_jobs()
        for r in range(ROWS):
            for c in range(COLS):
                self.grid[r][c] = 0
        self._clear_state()
        self._draw_full_grid()
        self.var_status.set("Walls cleared.")

    # ═══════════════════════════════════════════
    # SEARCH & ANIMATION
    # ═══════════════════════════════════════════
    def _h(self):
        return manhattan if self.heuristic.get() == "Manhattan" else euclidean

    def _run(self):
        self._cancel_jobs()
        self._clear_state()
        self.grid[self.start[0]][self.start[1]] = 0
        self.grid[self.goal[0]][self.goal[1]]   = 0

        h_fn = self._h()
        t0   = time.time()
        if self.algorithm.get() == "A*":
            path, vis, ne = astar(self.grid, self.start, self.goal, h_fn)
        else:
            path, vis, ne = gbfs(self.grid, self.start, self.goal, h_fn)
        elapsed = round((time.time()-t0)*1000, 2)

        self.var_nodes.set(str(ne))
        self.var_time.set(str(elapsed))

        if not path:
            self.var_status.set("No path found!")
            self.var_cost.set("N/A")
            return

        self.path = path
        self.path_set = set(path)
        self._visited_list = vis
        self.var_cost.set(str(len(path)-1))
        self.var_status.set(f"Path found! Cost={len(path)-1} | Animating...")
        self._anim_idx = 0
        self._draw_full_grid()
        self._anim_job = self.root.after(ANIM_DELAY, self._step_visited)

    def _step_visited(self):
        """Animate visited cells incrementally."""
        batch = 4
        for _ in range(batch):
            if self._anim_idx >= len(self._visited_list):
                # Show final path then start agent
                self._show_path()
                self.agent_pos = self.start
                self.agent_idx = 0
                self._redraw([self.start])
                self._agent_job = self.root.after(500, self._step_agent)
                return
            node = self._visited_list[self._anim_idx]
            self.visited_set.add(node)
            self._draw_cell(node[0], node[1])
            self._anim_idx += 1
        self._anim_job = self.root.after(ANIM_DELAY, self._step_visited)

    def _show_path(self):
        for node in self.path:
            if node not in [self.start, self.goal]:
                self._draw_cell(node[0], node[1])

    def _step_agent(self):
        if self.agent_idx >= len(self.path)-1:
            self.var_status.set("Goal reached! ✓")
            return

        prev = self.agent_pos
        self.agent_idx += 1
        self.agent_pos  = self.path[self.agent_idx]
        self._redraw([prev, self.agent_pos])

        # Dynamic obstacle spawning
        if self.dynamic_on.get():
            spawned = self._spawn_obstacles()
            if spawned and self.agent_idx < len(self.path)-1:
                nxt = self.path[self.agent_idx+1]
                if self.grid[nxt[0]][nxt[1]] == 1:
                    self._replan()
                    return

        self._agent_job = self.root.after(AGENT_DELAY, self._step_agent)

    def _spawn_obstacles(self):
        changed = []
        for r in range(ROWS):
            for c in range(COLS):
                if (r, c) not in [self.start, self.goal, self.agent_pos]:
                    if self.grid[r][c] == 0 and random.random() < OBS_PROB:
                        self.grid[r][c] = 1
                        changed.append((r, c))
        self._redraw(changed)
        return len(changed) > 0

    def _replan(self):
        h_fn = self._h()
        t0   = time.time()
        if self.algorithm.get() == "A*":
            path, vis, ne = astar(self.grid, self.agent_pos, self.goal, h_fn)
        else:
            path, vis, ne = gbfs(self.grid, self.agent_pos, self.goal, h_fn)
        elapsed = round((time.time()-t0)*1000, 2)

        self._replan_count += 1
        self.var_replan.set(str(self._replan_count))
        self.var_time.set(str(elapsed))
        self.var_nodes.set(str(int(self.var_nodes.get()) + ne))

        if not path:
            self.var_status.set("Blocked! No valid path.")
            return

        # Clear old path highlight
        for node in self.path:
            self.path_set.discard(node)
            if node not in [self.start, self.goal, self.agent_pos]:
                self._draw_cell(node[0], node[1])

        self.path = path
        self.path_set = set(path)
        self.agent_idx = 0
        self.var_cost.set(str(len(path)-1))
        self.var_status.set(f"Replanned! Cost={len(path)-1}  (#{self._replan_count})")
        self._show_path()
        self._redraw([self.agent_pos])
        self._agent_job = self.root.after(AGENT_DELAY, self._step_agent)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()