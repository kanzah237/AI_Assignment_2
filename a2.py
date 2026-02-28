

import tkinter as tk
from tkinter import font as tkfont
import heapq, math, time, random

#  Grid
ROWS        = 22
COLS        = 30
CELL        = 26         
PANEL_W     = 240     
ANIM_DELAY  = 6        
AGENT_DELAY = 130       
OBS_PROB    = 0.0025     

#  Colours
# Grid
CL_BG        = "#F8F9FA"  
CL_GRID      = "#DEE2E6"   
CL_EMPTY     = "#FFFFFF"
CL_WALL      = "#343A40" 
CL_WALL_STK  = "#495057"  
CL_START     = "#2ECC71" 
CL_GOAL      = "#E74C3C"  
CL_VISITED   = "#BDD7EE" 
CL_PATH      = "#A9DFBF"   
CL_PATH_LINE = "#27AE60"   
CL_AGENT     = "#F39C12"  
CL_AGENT_RIM = "#D68910"

# Panel
CL_PANEL     = "#1C2833"  
CL_PANEL2    = "#212F3D"   

CL_WHITE     = "#FFFFFF"
CL_OFFWHITE  = "#ECF0F1"
CL_MUTED     = "#95A5A6"
CL_ACCENT    = "#3498DB"   
CL_ACCENT2   = "#5DADE2"
CL_YELLOW    = "#F1C40F"
CL_GREEN_LBL = "#2ECC71"
CL_ORANGE    = "#E67E22"

# Button palette
BTN = {
    "run":   ("#27AE60", "#1E8449"),   
    "reset": ("#2980B9", "#1F618D"),   
    "maze":  ("#8E44AD", "#6C3483"),   
    "clear": ("#E74C3C", "#CB4335"),   
    "start": ("#1ABC9C", "#148F77"),   
    "goal":  ("#E67E22", "#CA6F1E"),   
}

#  Heuristics
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])
#  Grid functions
def make_grid(density=0.0):
    g = [[0]*COLS for _ in range(ROWS)]
    if density > 0:
        for r in range(ROWS):
            for c in range(COLS):
                if random.random() < density:
                    g[r][c] = 1
    return g

def neighbors(pos, grid):
    r, c = pos
    out = []
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
        nr, nc = r+dr, c+dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and grid[nr][nc] == 0:
            out.append((nr, nc))
    return out

def rebuild_path(cf, goal):
    path, node = [], goal
    while node is not None:
        path.append(node); node = cf[node]
    path.reverse()
    return path

#  Search algo
def run_astar(grid, start, goal, h):
    counter = 0
    g = {start: 0}
    heap = [(h(start, goal), 0, start)]
    cf = {start: None}
    closed = set()
    visited_order = []
    while heap:
        _, _, cur = heapq.heappop(heap)
        if cur in closed: continue
        closed.add(cur)
        visited_order.append(cur)
        if cur == goal:
            return rebuild_path(cf, goal), visited_order, len(visited_order)
        for nb in neighbors(cur, grid):
            ng = g[cur] + 1
            if nb not in g or ng < g[nb]:
                g[nb] = ng; cf[nb] = cur; counter += 1
                heapq.heappush(heap, (ng + h(nb, goal), counter, nb))
    return None, visited_order, len(visited_order)

def run_gbfs(grid, start, goal, h):
    counter = 0
    heap = [(h(start, goal), 0, start)]
    cf = {start: None}
    seen = {start}
    visited_order = []
    while heap:
        _, _, cur = heapq.heappop(heap)
        visited_order.append(cur)
        if cur == goal:
            return rebuild_path(cf, goal), visited_order, len(visited_order)
        for nb in neighbors(cur, grid):
            if nb not in seen:
                seen.add(nb); cf[nb] = cur; counter += 1
                heapq.heappush(heap, (h(nb, goal), counter, nb))
    return None, visited_order, len(visited_order)

#  Application
class PathfinderApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Dynamic Pathfinding Agent  ·  AI2002 Assignment 2")
        self.root.configure(bg=CL_PANEL)
        self.root.resizable(False, False)

        # heuristic state
        self.alg_var  = tk.StringVar(value="A*")
        self.h_var    = tk.StringVar(value="Manhattan")
        self.dyn_var  = tk.BooleanVar(value=False)
        self.speed_var = tk.IntVar(value=5)   # 1-10

        # Grid state 
        self.grid      = make_grid()
        self.start     = (1, 1)
        self.goal      = (ROWS-2, COLS-2)
        self._clear_sg()

        # Search results
        self.path         = []
        self.path_set     = set()
        self.visited_set  = set()
        self.agent_pos    = None
        self.agent_idx    = 0
        self._vlist       = []   # visited order list
        self._vidx        = 0   # animation cursor
        self._drawing     = None # True=wall, False=erase
        self._placing     = None # 'start' | 'goal'
        self._anim_job    = None
        self._agent_job   = None
        self._replans     = 0
        self._searching   = False

        #  Metric string vars
        self.m_nodes  = tk.StringVar(value="—")
        self.m_cost   = tk.StringVar(value="—")
        self.m_time   = tk.StringVar(value="—")
        self.m_replan = tk.StringVar(value="0")
        self.m_status = tk.StringVar(value="Ready  ·  Draw walls then press   Run")
        self.m_alg_info = tk.StringVar(value="")

        self._build_ui()
        self._full_redraw()

    #  Build ui
    def _build_ui(self):
        CW = COLS * CELL
        CH = ROWS * CELL

        #Canvas
        self.canvas = tk.Canvas(
            self.root, width=CW, height=CH,
            bg=CL_BG, highlightthickness=2,
            highlightbackground=CL_ACCENT,
            cursor="crosshair"
        )
        self.canvas.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="n")
        self.canvas.bind("<ButtonPress-1>",   self._press)
        self.canvas.bind("<B1-Motion>",       self._drag)
        self.canvas.bind("<ButtonRelease-1>", lambda _: setattr(self, "_drawing", None))
        self.canvas.bind("<ButtonPress-3>",   self._erase)
        self.canvas.bind("<B3-Motion>",       self._erase)

        #Side panel
        panel = tk.Frame(self.root, bg=CL_PANEL, width=PANEL_W)
        panel.grid(row=0, column=1, sticky="ns", padx=(5, 10), pady=10)
        panel.grid_propagate(False)

        # Title block
        tk.Label(panel, bg=CL_PANEL, font=("Segoe UI Emoji", 28)
                 ).pack(pady=(18, 0))
        tk.Label(panel, text="Pathfinding Agent", bg=CL_PANEL, fg=CL_WHITE,
                 font=("Arial", 14, "bold")).pack()
        tk.Label(panel, text="AI 2002  ·  Assignment 2  ·  Q6", bg=CL_PANEL,
                 fg=CL_MUTED, font=("Arial", 8, "italic")).pack(pady=(0, 6))

        self._divider(panel)

        #  Algorithm 
        self._section_label(panel, "⚙  Algorithm")
        alg_frame = tk.Frame(panel, bg=CL_PANEL)
        alg_frame.pack(fill="x", padx=14, pady=(0, 6))
        for alg, desc in [("A*", "Optimal · Slower"), ("GBFS", "Fast · Not optimal")]:
            row = tk.Frame(alg_frame, bg=CL_PANEL)
            row.pack(fill="x", pady=2)
            rb = tk.Radiobutton(row, text=f"  {alg}", variable=self.alg_var, value=alg,
                                bg=CL_PANEL, fg=CL_WHITE, selectcolor=CL_ACCENT,
                                activebackground=CL_PANEL, activeforeground=CL_WHITE,
                                font=("Arial", 10, "bold"), command=self._update_alg_info)
            rb.pack(side="left")
            tk.Label(row, text=desc, bg=CL_PANEL, fg=CL_MUTED,
                     font=("Arial", 8, "italic")).pack(side="left", padx=(4, 0))

        self.alg_info_lbl = tk.Label(panel, textvariable=self.m_alg_info,
                                     bg=CL_PANEL2, fg=CL_ACCENT2,
                                     font=("Arial", 8, "italic"),
                                     wraplength=PANEL_W-28, justify="left",
                                     padx=8, pady=4)
        self.alg_info_lbl.pack(fill="x", padx=14, pady=(0, 4))
        self._update_alg_info()

        self._divider(panel)

        #  Heuristic
        self._section_label(panel, "  Heuristic")
        h_frame = tk.Frame(panel, bg=CL_PANEL)
        h_frame.pack(fill="x", padx=14, pady=(0, 6))
        for h, desc in [("Manhattan", "|dx|+|dy|"), ("Euclidean", "√(dx²+dy²)")]:
            row = tk.Frame(h_frame, bg=CL_PANEL)
            row.pack(fill="x", pady=2)
            rb = tk.Radiobutton(row, text=f"  {h}", variable=self.h_var, value=h,
                                bg=CL_PANEL, fg=CL_WHITE, selectcolor=CL_ACCENT,
                                activebackground=CL_PANEL, activeforeground=CL_WHITE,
                                font=("Arial", 10))
            rb.pack(side="left")
            tk.Label(row, text=desc, bg=CL_PANEL, fg=CL_MUTED,
                     font=("Courier", 8)).pack(side="left", padx=(4, 0))

        self._divider(panel)

        # Speed slider 
        self._section_label(panel, "  Animation Speed")
        spd_frame = tk.Frame(panel, bg=CL_PANEL)
        spd_frame.pack(fill="x", padx=14, pady=(0, 6))
        tk.Label(spd_frame, text="Slow", bg=CL_PANEL, fg=CL_MUTED,
                 font=("Arial", 8)).pack(side="left")
        tk.Scale(spd_frame, variable=self.speed_var, from_=1, to=10,
                 orient="horizontal", bg=CL_PANEL, fg=CL_WHITE,
                 troughcolor=CL_PANEL2, highlightthickness=0,
                 showvalue=False, length=120,
                 activebackground=CL_ACCENT).pack(side="left", padx=4)
        tk.Label(spd_frame, text="Fast", bg=CL_PANEL, fg=CL_MUTED,
                 font=("Arial", 8)).pack(side="left")

        self._divider(panel)

        # Dynamic mode
        self._section_label(panel, "  Dynamic Obstacles")
        dyn_row = tk.Frame(panel, bg=CL_PANEL)
        dyn_row.pack(fill="x", padx=14, pady=(0, 2))
        self.dyn_cb = tk.Checkbutton(
            dyn_row, text="  Enable (walls spawn mid-run)",
            variable=self.dyn_var,
            bg=CL_PANEL, fg=CL_WHITE, selectcolor=CL_ACCENT,
            activebackground=CL_PANEL, activeforeground=CL_WHITE,
            font=("Arial", 9)
        )
        self.dyn_cb.pack(anchor="w")
        tk.Label(panel, text="Agent replans automatically if blocked",
                 bg=CL_PANEL, fg=CL_MUTED, font=("Arial", 8, "italic")
                 ).pack(anchor="w", padx=22, pady=(0, 6))

        self._divider(panel)

        # ── Control buttons ───────────────────────────
        self._section_label(panel, "  Controls")
        btns = [
            ("▶   Run Search",    self._run,        "run"),
            ("↺   Reset Grid",    self._reset,       "reset"),
            ("🎲  New Random Maze", self._new_maze,   "maze"),
            ("🗑   Clear All Walls", self._clear_walls, "clear"),
        ]
        for txt, cmd, key in btns:
            bg, abg = BTN[key]
            b = tk.Button(panel, text=txt, command=cmd,
                          bg=bg, fg=CL_WHITE, activebackground=abg,
                          activeforeground=CL_WHITE, relief="flat",
                          font=("Arial", 9, "bold"), pady=6, cursor="hand2",
                          bd=0)
            b.pack(fill="x", padx=14, pady=3)
            b.bind("<Enter>", lambda e, b=b, c=abg: b.config(bg=c))
            b.bind("<Leave>", lambda e, b=b, c=bg:  b.config(bg=c))

        self._divider(panel)

        # Place start / goal
        self._section_label(panel, "  Move Start / Goal")
        for txt, key in [(" Click grid to move Start", "start"),
                          ("  Click grid to move Goal",  "goal")]:
            bg, abg = BTN[key]
            b = tk.Button(panel, text=txt, bg=bg, fg=CL_WHITE,
                          activebackground=abg, activeforeground=CL_WHITE,
                          relief="flat", font=("Arial", 8, "bold"),
                          pady=5, cursor="hand2",
                          command=lambda k=key: self._start_placing(k))
            b.pack(fill="x", padx=14, pady=3)
            b.bind("<Enter>", lambda e, b=b, c=abg: b.config(bg=c))
            b.bind("<Leave>", lambda e, b=b, c=bg:  b.config(bg=c))

        self._divider(panel)

        #  Metrics card 
        self._section_label(panel, " Live Metrics")
        mcard = tk.Frame(panel, bg=CL_PANEL2, bd=0)
        mcard.pack(fill="x", padx=14, pady=(0, 6))
        metrics = [
            ("Nodes Expanded", self.m_nodes,  CL_ACCENT2),
            ("Path Cost",      self.m_cost,   CL_GREEN_LBL),
            ("Time (ms)",      self.m_time,   CL_YELLOW),
            ("Replans",        self.m_replan, CL_ORANGE),
        ]
        for i, (label, var, col) in enumerate(metrics):
            bg = CL_PANEL2 if i % 2 == 0 else CL_PANEL
            row = tk.Frame(mcard, bg=bg)
            row.pack(fill="x")
            tk.Label(row, text=label, bg=bg, fg=CL_MUTED,
                     font=("Arial", 8), width=14, anchor="w",
                     padx=8, pady=4).pack(side="left")
            tk.Label(row, textvariable=var, bg=bg, fg=col,
                     font=("Arial", 10, "bold"), anchor="e",
                     padx=8).pack(side="right")

        self._divider(panel)

        # Status bar 
        self.status_bar = tk.Label(
            panel, textvariable=self.m_status,
            bg=CL_PANEL2, fg=CL_GREEN_LBL,
            font=("Arial", 8, "italic"),
            wraplength=PANEL_W-20, justify="left",
            padx=10, pady=8, anchor="w"
        )
        self.status_bar.pack(fill="x", padx=14, pady=(0, 4))

        self._divider(panel)

        #  Legend 
        self._section_label(panel, "  Legend")
        legend_items = [
            (CL_START,   "Start node"),
            (CL_GOAL,    "Goal node"),
            (CL_AGENT,   "Agent (moving)"),
            (CL_PATH,    "Optimal path"),
            (CL_VISITED, "Explored nodes"),
            (CL_WALL,    "Wall / obstacle"),
        ]
        lg = tk.Frame(panel, bg=CL_PANEL)
        lg.pack(fill="x", padx=14, pady=(0, 4))
        for col, name in legend_items:
            row = tk.Frame(lg, bg=CL_PANEL)
            row.pack(fill="x", pady=2)
            swatch = tk.Frame(row, bg=col, width=16, height=16,
                              relief="solid", bd=1)
            swatch.pack(side="left", padx=(0, 8))
            swatch.pack_propagate(False)
            tk.Label(row, text=name, bg=CL_PANEL, fg=CL_OFFWHITE,
                     font=("Arial", 8)).pack(side="left")

        # Tips
        tips = tk.Frame(panel, bg=CL_PANEL2)
        tips.pack(fill="x", padx=14, pady=(6, 10))
        for tip in ["Left-drag  →  draw walls",
                    "Right-drag →  erase walls"]:
            tk.Label(tips, text=tip, bg=CL_PANEL2, fg=CL_MUTED,
                     font=("Courier", 8), anchor="w", padx=8, pady=2
                     ).pack(fill="x")

    #Helper UI builders
    def _divider(self, parent):
        tk.Frame(parent, bg=CL_ACCENT, height=1
                 ).pack(fill="x", padx=12, pady=6)

    def _section_label(self, parent, text):
        tk.Label(parent, text=text, bg=CL_PANEL, fg=CL_ACCENT2,
                 font=("Arial", 9, "bold"), anchor="w"
                 ).pack(fill="x", padx=14, pady=(2, 4))

    def _update_alg_info(self):
        info = {
            "A*":  "Uses f = g + h. Guarantees the shortest path when heuristic is admissible.",
            "GBFS":"Uses f = h only. Very fast but may return a suboptimal path."
        }
        self.m_alg_info.set(info[self.alg_var.get()])

    #  Grid
    def _cell_fill(self, r, c):
        p = (r, c)
        if p == self.start:      return CL_START
        if p == self.goal:       return CL_GOAL
        if p == self.agent_pos:  return CL_AGENT
        if self.grid[r][c] == 1: return CL_WALL
        if p in self.path_set:   return CL_PATH
        if p in self.visited_set:return CL_VISITED
        return CL_EMPTY

    def _draw_cell(self, r, c):
        tag = f"cell_{r}_{c}"
        self.canvas.delete(tag)
        x1, y1 = c*CELL + 2, r*CELL + 2
        x2, y2 = x1 + CELL - 3, y1 + CELL - 3
        fill = self._cell_fill(r, c)
        p = (r, c)

        # Cell background
        self.canvas.create_rectangle(x1, y1, x2, y2,
                                     fill=fill, outline="", tags=tag)

        # Special nodes
        cx, cy = (x1+x2)//2, (y1+y2)//2
        if p == self.start:
            self.canvas.create_text(cx, cy, text="S", fill=CL_WHITE,
                font=("Arial", 9, "bold"), tags=tag)
        elif p == self.goal:
            self.canvas.create_text(cx, cy, text="G", fill=CL_WHITE,
                font=("Arial", 9, "bold"), tags=tag)
        elif p == self.agent_pos:
            # Glowing circle
            pad = 5
            self.canvas.create_oval(x1+pad-2, y1+pad-2, x2-pad+2, y2-pad+2,
                fill=CL_AGENT_RIM, outline="", tags=tag)
            self.canvas.create_oval(x1+pad, y1+pad, x2-pad, y2-pad,
                fill=CL_AGENT, outline=CL_AGENT_RIM, width=1, tags=tag)
        elif p in self.path_set and p not in [self.start, self.goal]:
            # Small dot on path
            dot = 4
            self.canvas.create_oval(cx-dot, cy-dot, cx+dot, cy+dot,
                fill=CL_PATH_LINE, outline="", tags=tag)
        elif self.grid[r][c] == 1:
            # Subtle texture on wall
            self.canvas.create_rectangle(x1+2, y1+2, x2-2, y2-2,
                fill=CL_WALL_STK, outline="", tags=tag)

    def _full_redraw(self):
        self.canvas.delete("all")
        # Grid lines
        for r in range(ROWS + 1):
            self.canvas.create_line(0, r*CELL, COLS*CELL, r*CELL,
                                    fill=CL_GRID, width=1)
        for c in range(COLS + 1):
            self.canvas.create_line(c*CELL, 0, c*CELL, ROWS*CELL,
                                    fill=CL_GRID, width=1)
        for r in range(ROWS):
            for c in range(COLS):
                self._draw_cell(r, c)

    def _redraw_cells(self, cells):
        for r, c in cells:
            self._draw_cell(r, c)
    #  Mouse use
    def _rc(self, event):
        c, r = event.x // CELL, event.y // CELL
        if 0 <= r < ROWS and 0 <= c < COLS:
            return r, c
        return None

    def _press(self, event):
        rc = self._rc(event)
        if not rc: return
        r, c = rc
        # Place start/goal mode
        if self._placing:
            old = self.start if self._placing == "start" else self.goal
            self.grid[old[0]][old[1]] = 0
            if self._placing == "start": self.start = (r, c)
            else:                        self.goal  = (r, c)
            self.grid[r][c] = 0
            self._placing = None
            self.m_status.set("✓ Placed. Press ▶ Run to search.")
            self.root.configure(cursor="")
            self._redraw_cells([old, (r, c)])
            return
        if (r, c) in (self.start, self.goal): return
        self._drawing = (self.grid[r][c] == 0)
        self.grid[r][c] = 1 if self._drawing else 0
        self._redraw_cells([(r, c)])

    def _drag(self, event):
        if self._drawing is None: return
        rc = self._rc(event)
        if not rc: return
        r, c = rc
        if (r, c) in (self.start, self.goal): return
        self.grid[r][c] = 1 if self._drawing else 0
        self._redraw_cells([(r, c)])

    def _erase(self, event):
        rc = self._rc(event)
        if not rc: return
        r, c = rc
        if (r, c) in (self.start, self.goal): return
        self.grid[r][c] = 0
        self._redraw_cells([(r, c)])

    def _start_placing(self, key):
        self._placing = key
        label = "Start (green)" if key == "start" else "Goal (red)"
        self.m_status.set(f"🖱  Click any cell to place the {label}...")
        self.root.configure(cursor="crosshair")

    #  Control
    def _cancel_jobs(self):
        for attr in ("_anim_job", "_agent_job"):
            job = getattr(self, attr)
            if job: self.root.after_cancel(job)
            setattr(self, attr, None)

    def _clear_sg(self):
        self.grid[self.start[0]][self.start[1]] = 0
        self.grid[self.goal[0]][self.goal[1]]   = 0

    def _clear_search(self):
        self.path = []; self.path_set = set()
        self.visited_set = set()
        self.agent_pos = None; self.agent_idx = 0
        self._vlist = []; self._vidx = 0
        self._replans = 0
        self.m_nodes.set("—"); self.m_cost.set("—")
        self.m_time.set("—");  self.m_replan.set("0")

    def _reset(self):
        self._cancel_jobs()
        self.grid = make_grid(); self._clear_sg()
        self._clear_search()
        self._full_redraw()
        self.m_status.set("Grid cleared. Draw walls then press  Run.")

    def _new_maze(self):
        self._cancel_jobs()
        self.grid = make_grid(density=0.27); self._clear_sg()
        self._clear_search()
        self._full_redraw()
        self.m_status.set(" New maze ready. Press  Run!")

    def _clear_walls(self):
        self._cancel_jobs()
        self.grid = make_grid(); self._clear_sg()
        self._clear_search()
        self._full_redraw()
        self.m_status.set("All walls removed.")

    #  Search
    def _hfn(self):
        return manhattan if self.h_var.get() == "Manhattan" else euclidean

    def _run(self):
        self._cancel_jobs()
        self._clear_search()
        self._clear_sg()

        alg = self.alg_var.get()
        h   = self._hfn()

        self.m_status.set(f"🔍 Running {alg} with {self.h_var.get()} heuristic...")
        self.root.update()

        t0 = time.perf_counter()
        if alg == "A*":
            path, vis, ne = run_astar(self.grid, self.start, self.goal, h)
        else:
            path, vis, ne = run_gbfs(self.grid, self.start, self.goal, h)
        elapsed = round((time.perf_counter()-t0)*1000, 2)

        self.m_nodes.set(str(ne))
        self.m_time.set(str(elapsed))

        if not path:
            self.m_cost.set("N/A")
            self.m_status.set(" No path found! Try removing some walls.")
            return

        self.path     = path
        self.path_set = set(path)
        self._vlist   = vis
        self.m_cost.set(str(len(path)-1))
        self.m_status.set(f"Path found! Cost = {len(path)-1}  |  Sweeping explored nodes...")
        self._full_redraw()
        self._vidx = 0
        self._anim_job = self.root.after(2, self._tick_visited)

    # Animation 
    def _tick_visited(self):
        speed = self.speed_var.get()
        batch = max(1, speed * 3)   # faster speed = more cells per tick
        delay = max(1, 14 - speed)  # faster speed = shorter delay

        for _ in range(batch):
            if self._vidx >= len(self._vlist):
                # Sweep done — reveal path then launch agent
                self.m_status.set(" Exploration done. Tracing path.")
                self._show_path()
                self.agent_pos = self.start
                self.agent_idx = 0
                self._redraw_cells([self.start])
                self._agent_job = self.root.after(400, self._tick_agent)
                return
            node = self._vlist[self._vidx]
            self.visited_set.add(node)
            self._draw_cell(node[0], node[1])
            self._vidx += 1

        self._anim_job = self.root.after(delay, self._tick_visited)

    def _show_path(self):
        for node in self.path:
            if node not in (self.start, self.goal):
                self._draw_cell(node[0], node[1])

    def _tick_agent(self):
        if self.agent_idx >= len(self.path) - 1:
            self.m_status.set(" Goal reached! ")
            # Pulse the goal cell
            self._pulse_goal(3)
            return

        prev = self.agent_pos
        self.agent_idx += 1
        self.agent_pos  = self.path[self.agent_idx]
        self._redraw_cells([prev, self.agent_pos])

        # Dynamic obstacles
        if self.dyn_var.get():
            changed = self._spawn_obs()
            if changed and self.agent_idx < len(self.path) - 1:
                nxt = self.path[self.agent_idx + 1]
                if self.grid[nxt[0]][nxt[1]] == 1:
                    self.m_status.set(" Path blocked! Replanning.")
                    self.root.after(60, self._replan)
                    return

        speed = self.speed_var.get()
        delay = max(30, AGENT_DELAY - speed * 10)
        self._agent_job = self.root.after(delay, self._tick_agent)

    def _spawn_obs(self):
        changed = []
        for r in range(ROWS):
            for c in range(COLS):
                if (r, c) not in (self.start, self.goal, self.agent_pos):
                    if self.grid[r][c] == 0 and random.random() < OBS_PROB:
                        self.grid[r][c] = 1
                        changed.append((r, c))
        self._redraw_cells(changed)
        return changed

    def _replan(self):
        h   = self._hfn()
        alg = self.alg_var.get()
        t0  = time.perf_counter()
        if alg == "A*":
            path, vis, ne = run_astar(self.grid, self.agent_pos, self.goal, h)
        else:
            path, vis, ne = run_gbfs(self.grid, self.agent_pos, self.goal, h)
        elapsed = round((time.perf_counter()-t0)*1000, 2)

        self._replans += 1
        self.m_replan.set(str(self._replans))
        self.m_time.set(str(elapsed))
        self.m_nodes.set(str(int(self.m_nodes.get()) + ne))

        if not path:
            self.m_status.set("No path exists  trapped!")
            return

        # Erase old path highlight
        for node in self.path:
            self.path_set.discard(node)
            if node not in (self.start, self.goal, self.agent_pos):
                self._draw_cell(node[0], node[1])

        self.path = path
        self.path_set = set(path)
        self.agent_idx = 0
        self.m_cost.set(str(len(path)-1))
        self.m_status.set(f"🔄 Replanned  #{self._replans} · new cost = {len(path)-1}")
        self._show_path()
        self._redraw_cells([self.agent_pos])

        speed = self.speed_var.get()
        delay = max(30, AGENT_DELAY - speed * 10)
        self._agent_job = self.root.after(delay, self._tick_agent)

    def _pulse_goal(self, times):
        """Flash the goal cell a few times to celebrate."""
        if times <= 0:
            self._draw_cell(self.goal[0], self.goal[1])
            return
        r, c = self.goal
        x1, y1 = c*CELL+2, r*CELL+2
        x2, y2 = x1+CELL-3, y1+CELL-3
        tag = "pulse"
        self.canvas.delete(tag)
        col = CL_YELLOW if times % 2 == 0 else CL_GOAL
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=col, outline="", tags=tag)
        self.canvas.create_text((x1+x2)//2, (y1+y2)//2, text="G",
                                fill=CL_WHITE, font=("Arial",9,"bold"), tags=tag)
        self.root.after(200, lambda: self._pulse_goal(times-1))
#main
if __name__ == "__main__":
    root = tk.Tk()
    app  = PathfinderApp(root)
    root.mainloop()



