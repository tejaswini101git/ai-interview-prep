"""
=============================================================================
AI-BASED ADAPTIVE INTERVIEW PREPARATION SYSTEM (PBL AI PROJECT)
=============================================================================
Pure Python Standard Library (Tkinter, json, urllib.request, math, random).
Compatible with Python IDLE and any standard Python 3.x environment.

Core AI Syllabus Components Implemented:
- UNIT I:
    * Classical Search: BFS, DFS, Uniform Cost Search (UCS),
      Iterative Deepening Search (IDS), Bidirectional Search.
    * Informed / Heuristic Search: Greedy Best-First Search, A* Search.
    * Local Search: Hill Climbing, Simulated Annealing.
- UNIT II:
    * Constraint Satisfaction Problems (CSP): Domain filtering,
      Constraint relaxation, Backtracking search.
    * Propositional Logic: Forward Chaining, Backward Chaining, Resolution.
- UNIT III:
    * First-Order Logic (FOL): Variable Unification, FOL Forward Chaining.
- DYNAMIC TOPIC & GEMINI INTEGRATION:
    * Generates customized technical interview question banks for ANY user topic
      using Google Gemini REST API (gemini-flash-lite-latest).
    * Built-in offline fallback ensuring the system runs seamlessly offline.
=============================================================================
"""

import json
import math
import os
import random
import threading
import time
from collections import deque
import urllib.request
import urllib.error

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

# ---------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ---------------------------------------------------------------------------
DEFAULT_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-flash-latest"
]

FALLBACK_FILE = "questions.json"

# UI Theme Color Palette (Modern Dark / Slate Indigo)
THEME = {
    "bg_main": "#0f172a",       # Slate 900
    "bg_card": "#1e293b",       # Slate 800
    "bg_card_hover": "#273549", # Slate 750
    "bg_input": "#0b132b",      # Deep navy input box
    "border": "#334155",        # Slate 700
    "border_light": "#475569",  # Slate 600
    "text_main": "#f8fafc",     # Slate 50
    "text_muted": "#94a3b8",    # Slate 400
    "text_dim": "#64748b",      # Slate 500
    "primary": "#6366f1",       # Indigo 500
    "primary_hover": "#4f46e5", # Indigo 600
    "secondary": "#06b6d4",     # Cyan 500
    "accent": "#8b5cf6",        # Violet 500
    "success": "#10b981",       # Emerald 500
    "warning": "#f59e0b",       # Amber 500
    "danger": "#ef4444",        # Rose 500
}


# ===========================================================================
# 1. GEMINI API CLIENT (Standard Library Only)
# ===========================================================================

class GeminiClient:
    """Client for Google Gemini REST API using urllib.request."""

    @staticmethod
    def test_key(api_key):
        """Test if the given API key can connect to the Gemini API."""
        key = api_key.strip()
        if not key:
            return False, "API key is empty."
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("models", [])
                return True, f"Connected successfully! ({len(models)} models available)"
        except urllib.error.HTTPError as e:
            return False, f"HTTP Error {e.code}: {e.reason}"
        except Exception as e:
            return False, f"Connection error: {str(e)}"

    @staticmethod
    def generate_questions(topic, num_questions=10, api_key=DEFAULT_API_KEY):
        """
        Dynamically generate interview questions for any topic using Gemini.
        Returns a list of dicts with id, topic, skill, difficulty, question, keywords.
        """
        key = api_key.strip() or DEFAULT_API_KEY
        prompt = f"""Generate exactly {num_questions} high-quality technical interview questions for the topic: '{topic}'.
Ensure questions vary across difficulty levels (1=Easy, 2=Medium, 3=Hard) and cover diverse subtopics/skills.
Return strictly a valid JSON array of objects with this schema:
[
  {{
    "id": 1,
    "topic": "Specific Subtopic Name",
    "skill": "Specific Concept or Skill Name",
    "difficulty": 1,
    "question": "Clear, precise technical interview question.",
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4"]
  }}
]
Do not include markdown code fences or explanatory text. Return only the JSON array."""

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.35,
                "responseMimeType": "application/json"
            }
        }
        data_bytes = json.dumps(payload).encode("utf-8")

        last_error = None
        for model in GEMINI_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
            req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=25) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    raw = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                    if raw.startswith("```json"):
                        raw = raw[7:]
                    if raw.startswith("```"):
                        raw = raw[3:]
                    if raw.endswith("```"):
                        raw = raw[:-3]
                    questions = json.loads(raw.strip())
                    if isinstance(questions, list) and len(questions) > 0:
                        # Normalize question IDs
                        for idx, q in enumerate(questions, start=1):
                            q["id"] = idx
                            if "difficulty" not in q or q["difficulty"] not in (1, 2, 3):
                                q["difficulty"] = 2
                            if "keywords" not in q or not isinstance(q["keywords"], list):
                                q["keywords"] = [q.get("skill", "concept").lower()]
                        return questions, f"Generated dynamically via Gemini ({model})"
            except Exception as e:
                last_error = str(e)
                continue

        # If all models fail, raise error
        raise RuntimeError(f"Gemini API request failed: {last_error}")


# ===========================================================================
# 2. OFFLINE FALLBACK GENERATOR
# ===========================================================================

def get_offline_questions(topic, num_questions=10):
    """
    Provides rich fallback questions when offline or if no API key is provided.
    Includes built-in banks for popular domains or adapts existing questions.json.
    """
    # 1. Try reading from questions.json if available
    if os.path.exists(FALLBACK_FILE):
        try:
            with open(FALLBACK_FILE, "r", encoding="utf-8") as f:
                qs = json.load(f)
                if qs:
                    # Filter by topic if possible or return subset
                    matching = [q for q in qs if topic.lower() in q.get("topic", "").lower()]
                    pool = matching if len(matching) >= 5 else qs
                    selected = random.sample(pool, min(num_questions, len(pool)))
                    for i, q in enumerate(selected, 1):
                        q["id"] = i
                    return selected
        except Exception:
            pass

    # 2. Hardcoded fallback bank for general CS & AI topics
    templates = [
        {"skill": "Core Fundamentals", "diff": 1, "kw": ["definition", "purpose", "concept", "advantage"]},
        {"skill": "Architecture & Flow", "diff": 2, "kw": ["process", "components", "execution", "lifecycle"]},
        {"skill": "Performance & Efficiency", "diff": 2, "kw": ["complexity", "optimization", "time", "memory"]},
        {"skill": "Edge Cases & Conflict Handling", "diff": 3, "kw": ["deadlock", "race condition", "handling", "prevention"]},
        {"skill": "System Scaling & Best Practices", "diff": 3, "kw": ["scalability", "robustness", "patterns", "latency"]},
    ]
    generated = []
    for i in range(1, num_questions + 1):
        tpl = templates[(i - 1) % len(templates)]
        generated.append({
            "id": i,
            "topic": topic,
            "skill": f"{topic} {tpl['skill']}",
            "difficulty": tpl["diff"],
            "question": f"Explain the principles of {tpl['skill']} in {topic}. Discuss its practical significance and trade-offs.",
            "keywords": tpl["kw"] + [topic.lower().split()[0]]
        })
    return generated


# ===========================================================================
# 3. UNIT I: CLASSICAL, INFORMED & LOCAL SEARCH ALGORITHMS
# ===========================================================================

def graph_from_questions(questions):
    """
    Builds a directed prerequisite dependency graph between questions.
    Questions of lower difficulty within matching or related subtopics point
    to questions of higher difficulty.
    """
    nodes = [q["id"] for q in questions]
    graph = {n: [] for n in nodes}
    by_topic = {}
    for q in questions:
        by_topic.setdefault(q["topic"], []).append(q)

    # Within topic prerequisite edges
    for q in questions:
        same = by_topic.get(q["topic"], [])
        lower = [x for x in same if x["difficulty"] < q["difficulty"]]
        if lower:
            nearest = max(lower, key=lambda x: x["difficulty"])
            graph[nearest["id"]].append(q["id"])

    # If graph is too sparse, connect sequential levels across topics
    for i in range(len(questions) - 1):
        curr_id = questions[i]["id"]
        next_id = questions[i + 1]["id"]
        if next_id not in graph[curr_id]:
            graph[curr_id].append(next_id)

    return graph


def bfs(graph, start, goal):
    """Unit I: Breadth-First Search (queue-based, explores shortest path in steps)."""
    queue = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        for nxt in graph.get(node, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append(path + [nxt])
    return [start]


def dfs(graph, start, goal):
    """Unit I: Depth-First Search (stack-based, deep concept exploration)."""
    stack = [[start]]
    visited = set()
    while stack:
        path = stack.pop()
        node = path[-1]
        if node == goal:
            return path
        if node in visited:
            continue
        visited.add(node)
        for nxt in reversed(graph.get(node, [])):
            stack.append(path + [nxt])
    return [start]


def ucs(graph, start, goal, questions_by_id):
    """Unit I: Uniform Cost Search (priority queue based on step difficulty cost)."""
    frontier = [(0, [start])]
    visited_cost = {}
    while frontier:
        frontier.sort(key=lambda x: x[0])
        cost, path = frontier.pop(0)
        node = path[-1]
        if node == goal:
            return path
        if node in visited_cost and visited_cost[node] <= cost:
            continue
        visited_cost[node] = cost
        for nxt in graph.get(node, []):
            q_info = questions_by_id.get(nxt, {"difficulty": 1})
            step = max(1, q_info.get("difficulty", 1))
            frontier.append((cost + step, path + [nxt]))
    return [start]


def ids(graph, start, goal):
    """Unit I: Iterative Deepening Search (DFS with progressive depth limits)."""
    def depth_limited(node, target, limit, path):
        if node == target:
            return path
        if limit == 0:
            return None
        for nxt in graph.get(node, []):
            if nxt not in path:
                res = depth_limited(nxt, target, limit - 1, path + [nxt])
                if res:
                    return res
        return None

    for depth in range(len(graph) + 1):
        res = depth_limited(start, goal, depth, [start])
        if res:
            return res
    return [start]


def bidirectional_search(graph, start, goal):
    """Unit I: Bidirectional Search (simultaneous forward and backward searches)."""
    if start == goal:
        return [start]
    reverse = {n: [] for n in graph}
    for u, children in graph.items():
        for v in children:
            reverse.setdefault(v, []).append(u)

    front_a = {start: [start]}
    front_b = {goal: [goal]}
    visited_a, visited_b = set(front_a), set(front_b)

    while front_a and front_b:
        next_a = {}
        for node, path in front_a.items():
            for nxt in graph.get(node, []):
                if nxt in visited_a:
                    continue
                new_path = path + [nxt]
                if nxt in visited_b:
                    other = front_b[nxt]
                    return new_path + list(reversed(other[:-1]))
                visited_a.add(nxt)
                next_a[nxt] = new_path
        front_a = next_a

        next_b = {}
        for node, path in front_b.items():
            for nxt in reverse.get(node, []):
                if nxt in visited_b:
                    continue
                new_path = path + [nxt]
                if nxt in visited_a:
                    other = front_a[nxt]
                    return other + list(reversed(new_path[:-1]))
                visited_b.add(nxt)
                next_b[nxt] = new_path
        front_b = next_b

    return [start]


def greedy_best_first(candidates, weakness_scores):
    """Unit I: Greedy Best-First Search (picks candidate maximizing weakness priority)."""
    if not candidates:
        return None
    return max(candidates, key=lambda q: weakness_scores.get(q["skill"], 50) / max(1, q["difficulty"]))


def heuristic(q, weakness_scores, target_difficulty):
    """A* Heuristic function h(n) measuring skill mismatch and target difficulty gap."""
    weakness = weakness_scores.get(q["skill"], 50)
    difficulty_gap = abs(q["difficulty"] - target_difficulty)
    # Lower heuristic is closer to optimal interview targeting
    return (100 - weakness) + (difficulty_gap * 15)


def a_star_select(candidates, weakness_scores, target_difficulty):
    """Unit I: A* Search (evaluates candidates with f(n) = g(n) + h(n))."""
    if not candidates:
        return None
    return min(candidates, key=lambda q: q["difficulty"] * 10 + heuristic(q, weakness_scores, target_difficulty))


def hill_climbing(candidates, weakness_scores, target_difficulty):
    """Unit I: Hill Climbing Local Search (greedily steps to neighbor with best score)."""
    if not candidates:
        return None

    def score(q):
        return 100 - heuristic(q, weakness_scores, target_difficulty)

    current = random.choice(candidates)
    improved = True
    iterations = 0
    while improved and iterations < 10:
        iterations += 1
        improved = False
        neighbors = random.sample(candidates, min(3, len(candidates)))
        for n in neighbors:
            if score(n) > score(current):
                current = n
                improved = True
                break
    return current


def simulated_annealing(candidates, weakness_scores, target_difficulty):
    """Unit I: Simulated Annealing (probabilistic acceptance using cooling schedule)."""
    if not candidates:
        return None

    def score(q):
        return 100 - heuristic(q, weakness_scores, target_difficulty)

    current = random.choice(candidates)
    temperature = 12.0
    for _ in range(20):
        candidate = random.choice(candidates)
        delta = score(candidate) - score(current)
        if delta > 0 or random.random() < math.exp(delta / max(temperature, 0.05)):
            current = candidate
        temperature *= 0.85
    return current


# ===========================================================================
# 4. UNIT II: CONSTRAINT SATISFACTION PROBLEMS (CSP) & PROPOSITIONAL LOGIC
# ===========================================================================

def csp_filter(questions, max_difficulty, asked_ids, weaknesses):
    """
    Unit II: CSP Filtering.
    Applies unary and relational constraints:
    - Variable: Next question
    - Domain constraints: Unasked (id not in asked_ids), difficulty <= max_difficulty.
    - Constraint relaxation applied if domain becomes empty.
    """
    result = []
    for q in questions:
        if q["id"] in asked_ids:
            continue
        if q["difficulty"] > max_difficulty:
            continue
        result.append(q)

    # Constraint Relaxation: If difficulty constraint prunes everything, relax it
    if not result:
        for q in questions:
            if q["id"] not in asked_ids:
                result.append(q)
    return result


def backtracking_pick(candidates, asked_ids):
    """Unit II: CSP Backtracking search to assign consistent next question."""
    assignment = []

    def backtrack(index):
        if len(assignment) == 1:
            return assignment[0]
        if index >= len(candidates):
            return None
        q = candidates[index]
        if q["id"] in asked_ids:
            return backtrack(index + 1)
        assignment.append(q)
        res = backtrack(index + 1)
        if res:
            return res
        assignment.pop()
        return backtrack(index + 1)

    return backtrack(0)


def propositional_forward_chaining(facts, rules):
    """
    Unit II: Propositional Logic Forward Chaining.
    Iteratively infers new facts from satisfied rule premises until a fixed point.
    """
    inferred = set(facts)
    changed = True
    trace = []
    while changed:
        changed = False
        for premises, conclusion in rules:
            if set(premises).issubset(inferred) and conclusion not in inferred:
                inferred.add(conclusion)
                trace.append(f"Fired rule {premises} -> '{conclusion}'")
                changed = True
    return inferred, trace


def propositional_backward_chaining(goal, facts, rules, visited=None):
    """Unit II: Propositional Logic Backward Chaining (goal-directed deduction)."""
    visited = visited or set()
    if goal in facts:
        return True
    if goal in visited:
        return False
    visited.add(goal)
    for premises, conclusion in rules:
        if conclusion == goal:
            if all(propositional_backward_chaining(p, facts, rules, visited.copy()) for p in premises):
                return True
    return False


def resolution_demo():
    """Unit II: Resolution refutation theorem proving demonstration: (P v Q) and (~Q) => P."""
    clauses = [
        {"P", "Q"},
        {"~Q"},
    ]
    resolvents = []
    for c1 in clauses:
        for c2 in clauses:
            for lit in list(c1):
                opposite = "~" + lit if not lit.startswith("~") else lit[1:]
                if opposite in c2:
                    resolvent = (c1 - {lit}) | (c2 - {opposite})
                    if resolvent and resolvent not in resolvents:
                        resolvents.append(resolvent)
    return resolvents


# ===========================================================================
# 5. UNIT III: FIRST-ORDER LOGIC (FOL) UNIFICATION & REASONING
# ===========================================================================

def unify(pattern, fact):
    """
    Unit III: First-Order Logic Unification.
    Matches variables (starting with '?') against ground constants/terms.
    e.g. unify('Mastered(?student, ?skill)', 'Mastered(Student, Deadlock)')
    """
    p = pattern.strip()
    f = fact.strip()
    if p == f:
        return {}
    if p.startswith("?"):
        return {p: f}
    if f.startswith("?"):
        return {f: p}

    if "(" in p and ")" in p and "(" in f and ")" in f:
        p_pred = p.split("(")[0].strip()
        f_pred = f.split("(")[0].strip()
        if p_pred != f_pred:
            return None
        p_args = [x.strip() for x in p[p.find("(") + 1:-1].split(",")]
        f_args = [x.strip() for x in f[f.find("(") + 1:-1].split(",")]
        if len(p_args) != len(f_args):
            return None
        subst = {}
        for pa, fa in zip(p_args, f_args):
            if pa.startswith("?"):
                subst[pa] = fa
            elif fa.startswith("?"):
                subst[fa] = pa
            elif pa != fa:
                return None
        return subst
    return None


def fol_forward_chaining(facts, rules):
    """
    Unit III: FOL Forward Chaining.
    Applies substitution lifting to derive ground conclusions from generalized rules.
    """
    inferred = set(facts)
    changed = True
    while changed:
        changed = False
        for premise, conclusion in rules:
            for fact in list(inferred):
                subst = unify(premise, fact)
                if subst is not None:
                    derived = conclusion
                    for var, val in subst.items():
                        derived = derived.replace(var, val)
                    if derived not in inferred:
                        inferred.add(derived)
                        changed = True
    return inferred


# ===========================================================================
# 6. ANSWER EVALUATION (Keywords & Detail)
# ===========================================================================

def evaluate_answer(answer, keywords):
    """
    Evaluates candidate answer against expected concept keywords.
    Returns: score (0-100), matched_keywords (list), missed_keywords (list).
    """
    text = answer.lower().strip()
    if not text:
        return 0, [], keywords

    words = text.split()
    matched = []
    missed = []
    for k in keywords:
        k_clean = k.lower().strip()
        if k_clean in text:
            matched.append(k)
        else:
            missed.append(k)

    hits = len(matched)
    total_kw = max(1, len(keywords))
    score = int((hits / total_kw) * 90)

    # Elaboration bonus for substantive technical response
    if len(words) >= 30:
        score = min(100, score + 10)
    elif len(words) >= 15:
        score = min(100, score + 5)

    return score, matched, missed


# ===========================================================================
# 7. MODERN TKINTER GUI APPLICATION
# ===========================================================================

class ModernInterviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Interview Preparation System (PBL Project)")
        self.root.geometry("1100x780")
        self.root.minsize(980, 700)
        self.root.configure(bg=THEME["bg_main"])

        # Session State
        self.student_name = ""
        self.custom_topic = "Operating Systems"
        self.difficulty = 2
        self.num_questions = 8
        self.api_key = DEFAULT_API_KEY
        self.questions = []
        self.asked_ids = set()
        self.history = []
        self.weakness = {}
        self.current_question = None
        self.selection_method = "A* Search"
        self.selection_reason = ""
        self.search_traces = []

        self.apply_styles()
        self.show_home_screen()

    def apply_styles(self):
        """Configure ttk styles to blend seamlessly with dark slate palette."""
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=THEME["bg_main"], foreground=THEME["text_main"], font=("Segoe UI", 10))
        style.configure("TFrame", background=THEME["bg_main"])
        style.configure("Card.TFrame", background=THEME["bg_card"], relief="flat")
        style.configure("TLabel", background=THEME["bg_main"], foreground=THEME["text_main"])
        style.configure("Card.TLabel", background=THEME["bg_card"], foreground=THEME["text_main"])
        style.configure("Muted.TLabel", background=THEME["bg_card"], foreground=THEME["text_muted"])

        # Custom Buttons
        style.configure("Primary.TButton", background=THEME["primary"], foreground="#ffffff",
                        font=("Segoe UI", 11, "bold"), borderwidth=0, focuscolor="none")
        style.map("Primary.TButton",
                  background=[("active", THEME["primary_hover"]), ("pressed", THEME["primary_hover"])])

        style.configure("Secondary.TButton", background=THEME["bg_card"], foreground=THEME["secondary"],
                        font=("Segoe UI", 10, "bold"), borderwidth=1, bordercolor=THEME["border"])
        style.map("Secondary.TButton",
                  background=[("active", THEME["border"]), ("pressed", THEME["border"])])

        # Radiobuttons & Combobox
        style.configure("TRadiobutton", background=THEME["bg_card"], foreground=THEME["text_main"],
                        font=("Segoe UI", 10))
        style.map("TRadiobutton", background=[("active", THEME["bg_card"])])

    def clear(self):
        """Removes all widgets from root window."""
        for widget in self.root.winfo_children():
            widget.destroy()

    def build_header(self, title, subtitle, badge_text=None, badge_color=None):
        """Builds a polished top banner with title, subtitle, and status pill."""
        top = tk.Frame(self.root, bg=THEME["bg_main"], padx=35, pady=20)
        top.pack(fill="x")

        row = tk.Frame(top, bg=THEME["bg_main"])
        row.pack(fill="x")

        tk.Label(row, text=title, font=("Segoe UI", 22, "bold"),
                 bg=THEME["bg_main"], fg=THEME["text_main"]).pack(side="left")

        if badge_text:
            color = badge_color or THEME["primary"]
            badge = tk.Label(row, text=f" {badge_text} ", font=("Segoe UI", 9, "bold"),
                             bg=color, fg="#ffffff", padx=8, pady=3)
            badge.pack(side="right", padx=10)

        if subtitle:
            tk.Label(top, text=subtitle, font=("Segoe UI", 11),
                     bg=THEME["bg_main"], fg=THEME["text_muted"]).pack(anchor="w", pady=(4, 0))

        # Subtle divider
        tk.Frame(self.root, bg=THEME["border"], height=1).pack(fill="x", padx=35)

    # -----------------------------------------------------------------------
    # SCREEN 1: CONFIGURATION & TOPIC SETUP
    # -----------------------------------------------------------------------
    def show_home_screen(self):
        self.clear()
        self.build_header(
            "AI Interview Preparation System",
            "Adaptive Technical Practice Powered by Core AI Search, CSP & Logical Reasoning Engines",
            "PBL SYLLABUS PROJECT",
            THEME["secondary"]
        )

        # Scrollable container for setup
        canvas = tk.Canvas(self.root, bg=THEME["bg_main"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=THEME["bg_main"], padx=35, pady=20)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        def on_canvas_configure(e):
            canvas.itemconfig(canvas_window, width=e.width)

        canvas.bind("<Configure>", on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # CARD 1: INTERVIEW TOPIC & PROFILE
        setup_card = tk.Frame(scroll_frame, bg=THEME["bg_card"], bd=1, relief="solid", padx=25, pady=20)
        setup_card.configure(highlightbackground=THEME["border"], highlightthickness=1)
        setup_card.pack(fill="x", pady=(0, 20))

        tk.Label(setup_card, text="1. Candidate Profile & Interview Topic", font=("Segoe UI", 13, "bold"),
                 bg=THEME["bg_card"], fg=THEME["text_main"]).pack(anchor="w")

        # Student Name
        tk.Label(setup_card, text="Student / Candidate Name", font=("Segoe UI", 10, "bold"),
                 bg=THEME["bg_card"], fg=THEME["text_muted"]).pack(anchor="w", pady=(15, 5))
        self.name_entry = tk.Entry(setup_card, font=("Segoe UI", 11), bg=THEME["bg_input"],
                                   fg=THEME["text_main"], insertbackground="#ffffff", bd=1, relief="solid")
        self.name_entry.insert(0, self.student_name or "Akhil")
        self.name_entry.pack(fill="x", ipady=6)

        # Custom Topic Input
        tk.Label(setup_card, text="Interview Topic (Type ANY subject or choose a preset)",
                 font=("Segoe UI", 10, "bold"), bg=THEME["bg_card"], fg=THEME["text_muted"]).pack(anchor="w", pady=(15, 5))

        self.topic_var = tk.StringVar(value=self.custom_topic)
        self.topic_entry = tk.Entry(setup_card, textvariable=self.topic_var, font=("Segoe UI", 11),
                                    bg=THEME["bg_input"], fg=THEME["text_main"], insertbackground="#ffffff",
                                    bd=1, relief="solid")
        self.topic_entry.pack(fill="x", ipady=6)

        # Quick-choice pills/chips
        pills_frame = tk.Frame(setup_card, bg=THEME["bg_card"])
        pills_frame.pack(anchor="w", pady=(10, 5))
        tk.Label(pills_frame, text="Suggestions: ", font=("Segoe UI", 9),
                 bg=THEME["bg_card"], fg=THEME["text_dim"]).pack(side="left")

        presets = [
            "Operating Systems",
            "Python Programming",
            "Data Structures & Algorithms",
            "Computer Networks",
            "Database Management (DBMS)",
            "AI: Search, CSP & Logic (Units 1-3)"
        ]

        def set_preset(text):
            self.topic_var.set(text)

        for p in presets:
            btn = tk.Button(pills_frame, text=p, font=("Segoe UI", 8),
                            bg=THEME["border"], fg=THEME["text_main"], activebackground=THEME["primary"],
                            activeforeground="#ffffff", bd=0, padx=8, pady=2, cursor="hand2",
                            command=lambda val=p: set_preset(val))
            btn.pack(side="left", padx=4, pady=2)

        # CARD 2: PARAMETERS & AI SELECTION
        param_card = tk.Frame(scroll_frame, bg=THEME["bg_card"], bd=1, relief="solid", padx=25, pady=20)
        param_card.configure(highlightbackground=THEME["border"], highlightthickness=1)
        param_card.pack(fill="x", pady=(0, 20))

        tk.Label(param_card, text="2. Adaptive Interview Settings", font=("Segoe UI", 13, "bold"),
                 bg=THEME["bg_card"], fg=THEME["text_main"]).pack(anchor="w")

        cols = tk.Frame(param_card, bg=THEME["bg_card"])
        cols.pack(fill="x", pady=15)

        # Difficulty column
        diff_col = tk.Frame(cols, bg=THEME["bg_card"])
        diff_col.pack(side="left", fill="x", expand=True)
        tk.Label(diff_col, text="Base Difficulty", font=("Segoe UI", 10, "bold"),
                 bg=THEME["bg_card"], fg=THEME["text_muted"]).pack(anchor="w", pady=(0, 6))

        self.diff_var = tk.IntVar(value=self.difficulty)
        for val, label in [(1, "Easy (Foundations)"), (2, "Medium (Standard)"), (3, "Hard (Advanced)")]:
            tk.Radiobutton(diff_col, text=label, variable=self.diff_var, value=val,
                           bg=THEME["bg_card"], fg=THEME["text_main"], selectcolor=THEME["bg_input"],
                           activebackground=THEME["bg_card"], activeforeground=THEME["text_main"]).pack(anchor="w", pady=2)

        # Question Count column
        count_col = tk.Frame(cols, bg=THEME["bg_card"])
        count_col.pack(side="left", fill="x", expand=True)
        tk.Label(count_col, text="Number of Questions", font=("Segoe UI", 10, "bold"),
                 bg=THEME["bg_card"], fg=THEME["text_muted"]).pack(anchor="w", pady=(0, 6))

        self.count_var = tk.IntVar(value=self.num_questions)
        for val, label in [(5, "5 Questions (Quick Test)"), (8, "8 Questions (Balanced)"), (12, "12 Questions (In-Depth)")]:
            tk.Radiobutton(count_col, text=label, variable=self.count_var, value=val,
                           bg=THEME["bg_card"], fg=THEME["text_main"], selectcolor=THEME["bg_input"],
                           activebackground=THEME["bg_card"], activeforeground=THEME["text_main"]).pack(anchor="w", pady=2)

        # START BUTTON & STATUS
        action_frame = tk.Frame(scroll_frame, bg=THEME["bg_main"], pady=10)
        action_frame.pack(fill="x")

        self.start_btn = tk.Button(action_frame, text="GENERATE QUESTIONS & START INTERVIEW →",
                                   font=("Segoe UI", 12, "bold"), bg=THEME["primary"], fg="#ffffff",
                                   activebackground=THEME["primary_hover"], activeforeground="#ffffff",
                                   bd=0, padx=25, pady=12, cursor="hand2", command=self.on_start_clicked)
        self.start_btn.pack(side="left")

        self.loading_label = tk.Label(action_frame, text="", font=("Segoe UI", 10),
                                      bg=THEME["bg_main"], fg=THEME["secondary"])
        self.loading_label.pack(side="left", padx=20)

    def on_start_clicked(self):
        """Handles question generation and interview initialization asynchronously."""
        self.student_name = self.name_entry.get().strip() or "Student"
        self.custom_topic = self.topic_var.get().strip() or "General Computer Science"
        self.difficulty = self.diff_var.get()
        self.num_questions = self.count_var.get()
        self.api_key = DEFAULT_API_KEY

        self.start_btn.config(state="disabled", text="Preparing Questions...")
        self.loading_label.config(text="⏳ Generating custom question bank for topic...", fg=THEME["warning"])

        # Run in a background thread to prevent Tkinter freezing
        threading.Thread(target=self._fetch_and_start_worker, daemon=True).start()

    def _fetch_and_start_worker(self):
        questions = None
        source_note = ""
        try:
            questions, source_note = GeminiClient.generate_questions(
                self.custom_topic,
                num_questions=max(10, self.num_questions + 3), # Generate a few extra for search pool
                api_key=self.api_key
            )
        except Exception as e:
            # Fallback gracefully
            source_note = f"Offline fallback mode ({str(e)})"
            questions = get_offline_questions(self.custom_topic, num_questions=max(10, self.num_questions + 3))

        self.questions = questions
        self.asked_ids = set()
        self.history = []
        self.weakness = {}
        self.search_traces = []

        # Return to main UI thread
        self.root.after(0, self.show_question_screen)

    # -----------------------------------------------------------------------
    # QUESTION SELECTION ENGINE (UNIT I & II)
    # -----------------------------------------------------------------------
    def select_next_question(self):
        """
        Selects the next question by combining:
        1. CSP filtering (domain reduction)
        2. Graph-based classical search (BFS, DFS, UCS, IDS, Bidirectional)
        3. Heuristic & Local search (A*, Greedy, Hill Climbing, Simulated Annealing)
        4. CSP Backtracking fallback
        """
        candidates = csp_filter(
            self.questions, self.difficulty, self.asked_ids, self.weakness
        )
        if not candidates:
            return None, "No remaining questions in pool"

        step = len(self.history)
        recent = self.history[-3:]
        avg_score = sum(x["score"] for x in recent) / len(recent) if recent else 60
        target_diff = self.difficulty
        if avg_score >= 80:
            target_diff = min(3, self.difficulty + 1)
        elif avg_score < 45:
            target_diff = max(1, self.difficulty - 1)

        # Graph formulation for search algorithms
        q_graph = graph_from_questions(candidates)
        start_node = candidates[0]["id"]
        goal_node = candidates[-1]["id"]
        by_id = {q["id"]: q for q in candidates}

        selected = None
        algo_name = ""
        algo_reason = ""

        # Early interview questions demonstrate Classical Uninformed Searches
        if step == 0:
            algo_name = "A* Search [Unit I]"
            selected = a_star_select(candidates, self.weakness, target_diff)
            algo_reason = f"Evaluated f(n)=g(n)+h(n) targeting difficulty {target_diff}."
        elif step == 1:
            algo_name = "Breadth-First Search (BFS) [Unit I]"
            path = bfs(q_graph, start_node, goal_node)
            target_id = path[1] if len(path) > 1 else path[0]
            selected = by_id.get(target_id, candidates[0])
            algo_reason = f"Level-order queue traversal along path: {path}."
        elif step == 2:
            algo_name = "Uniform Cost Search (UCS) [Unit I]"
            path = ucs(q_graph, start_node, goal_node, by_id)
            target_id = path[1] if len(path) > 1 else path[0]
            selected = by_id.get(target_id, candidates[0])
            algo_reason = f"Priority queue picked lowest difficulty step cost along path: {path}."
        elif step == 3:
            algo_name = "Bidirectional Search [Unit I]"
            path = bidirectional_search(q_graph, start_node, goal_node)
            target_id = path[len(path) // 2] if path else candidates[0]["id"]
            selected = by_id.get(target_id, candidates[0])
            algo_reason = f"Forward & backward frontiers met at node {target_id}."
        elif step == 4:
            algo_name = "Iterative Deepening Search (IDS) [Unit I]"
            path = ids(q_graph, start_node, goal_node)
            target_id = path[-1] if path else candidates[0]["id"]
            selected = by_id.get(target_id, candidates[0])
            algo_reason = f"Bounded DFS explored up to depth {len(path)}."
        else:
            # Subsequent adaptive questions rotate Informed & Local Search
            mode = step % 4
            if mode == 0:
                algo_name = "Greedy Best-First Search [Unit I]"
                selected = greedy_best_first(candidates, self.weakness)
                algo_reason = "Maximizes weakness prioritization score."
            elif mode == 1:
                algo_name = "A* Search [Unit I]"
                selected = a_star_select(candidates, self.weakness, target_diff)
                algo_reason = f"Optimal balance of difficulty cost g(n) and target gap h(n)."
            elif mode == 2:
                algo_name = "Hill Climbing Local Search [Unit I]"
                selected = hill_climbing(candidates, self.weakness, target_diff)
                algo_reason = "Greedy local search explored neighborhood."
            else:
                algo_name = "Simulated Annealing [Unit I]"
                selected = simulated_annealing(candidates, self.weakness, target_diff)
                algo_reason = "Probabilistic temperature exploration escaped local optima."

        # Fallback to CSP Backtracking if needed
        if not selected or selected["id"] in self.asked_ids:
            algo_name = "CSP Backtracking Assignment [Unit II]"
            selected = backtracking_pick(candidates, self.asked_ids)
            algo_reason = "Backtracking resolved variable assignment with unasked constraint."

        self.selection_method = algo_name
        self.selection_reason = algo_reason
        return selected, algo_reason

    # -----------------------------------------------------------------------
    # SCREEN 2: ACTIVE TECHNICAL INTERVIEW QUESTION
    # -----------------------------------------------------------------------
    def show_question_screen(self):
        self.clear()

        # Check if interview limit reached
        if len(self.history) >= self.num_questions:
            self.show_results_screen()
            return

        q, reason = self.select_next_question()
        if not q:
            self.show_results_screen()
            return

        self.current_question = q
        q_idx = len(self.history) + 1

        self.build_header(
            f"Technical Interview  •  Question {q_idx} of {self.num_questions}",
            f"Topic: {self.custom_topic}   |   Candidate: {self.student_name}",
            f"Difficulty: {q['difficulty']}/3",
            THEME["success"] if q["difficulty"] == 1 else (THEME["warning"] if q["difficulty"] == 2 else THEME["danger"])
        )

        content = tk.Frame(self.root, bg=THEME["bg_main"], padx=35, pady=20)
        content.pack(fill="both", expand=True)

        # AI ALGORITHM BADGE CARD
        algo_bar = tk.Frame(content, bg=THEME["bg_card"], bd=1, relief="solid", padx=20, pady=10)
        algo_bar.configure(highlightbackground=THEME["border"], highlightthickness=1)
        algo_bar.pack(fill="x", pady=(0, 15))

        tk.Label(algo_bar, text="🤖 AI Selection Strategy:", font=("Segoe UI", 9, "bold"),
                 bg=THEME["bg_card"], fg=THEME["secondary"]).pack(side="left")
        tk.Label(algo_bar, text=f" {self.selection_method} ", font=("Segoe UI", 9, "bold"),
                 bg=THEME["primary"], fg="#ffffff", padx=6, pady=2).pack(side="left", padx=10)
        tk.Label(algo_bar, text=f"— {self.selection_reason}", font=("Segoe UI", 9),
                 bg=THEME["bg_card"], fg=THEME["text_muted"]).pack(side="left")

        # QUESTION CARD
        q_card = tk.Frame(content, bg=THEME["bg_card"], bd=1, relief="solid", padx=25, pady=20)
        q_card.configure(highlightbackground=THEME["border"], highlightthickness=1)
        q_card.pack(fill="x", pady=(0, 15))

        meta_row = tk.Frame(q_card, bg=THEME["bg_card"])
        meta_row.pack(fill="x", pady=(0, 8))

        subtopic = q.get("topic", self.custom_topic)
        skill = q.get("skill", "Concept")
        tk.Label(meta_row, text=f"SUBTOPIC: {subtopic.upper()}  •  SKILL: {skill.upper()}",
                 font=("Segoe UI", 9, "bold"), bg=THEME["bg_card"], fg=THEME["text_dim"]).pack(side="left")

        tk.Label(q_card, text=q["question"], font=("Segoe UI", 16, "bold"),
                 bg=THEME["bg_card"], fg=THEME["text_main"], wraplength=1000, justify="left").pack(anchor="w")

        # ANSWER INPUT AREA
        ans_frame = tk.Frame(content, bg=THEME["bg_main"])
        ans_frame.pack(fill="both", expand=True, pady=(0, 15))

        tk.Label(ans_frame, text="Your Technical Answer (Explain clearly using key terminology):",
                 font=("Segoe UI", 10, "bold"), bg=THEME["bg_main"], fg=THEME["text_muted"]).pack(anchor="w", pady=(0, 5))

        self.answer_box = scrolledtext.ScrolledText(
            ans_frame, font=("Segoe UI", 11), bg=THEME["bg_input"], fg=THEME["text_main"],
            insertbackground="#ffffff", bd=1, relief="solid", height=9, wrap="word"
        )
        self.answer_box.pack(fill="both", expand=True)
        self.answer_box.focus_set()

        # Word counter live update
        count_label = tk.Label(ans_frame, text="Words: 0", font=("Segoe UI", 9),
                               bg=THEME["bg_main"], fg=THEME["text_dim"])
        count_label.pack(anchor="e", pady=(4, 0))

        def update_word_count(e=None):
            text = self.answer_box.get("1.0", "end").strip()
            words = len(text.split()) if text else 0
            count_label.config(text=f"Words: {words}")

        self.answer_box.bind("<KeyRelease>", update_word_count)

        # BOTTOM BUTTONS
        btn_bar = tk.Frame(content, bg=THEME["bg_main"])
        btn_bar.pack(fill="x")

        tk.Button(btn_bar, text="SUBMIT ANSWER →", font=("Segoe UI", 11, "bold"),
                  bg=THEME["primary"], fg="#ffffff", activebackground=THEME["primary_hover"],
                  activeforeground="#ffffff", bd=0, padx=22, pady=10, cursor="hand2",
                  command=self.on_submit_answer).pack(side="right")

        tk.Button(btn_bar, text="Skip Question", font=("Segoe UI", 10),
                  bg=THEME["bg_card"], fg=THEME["text_muted"], activebackground=THEME["border"],
                  activeforeground=THEME["text_main"], bd=1, relief="solid", padx=15, pady=8, cursor="hand2",
                  command=self.on_skip_question).pack(side="left")

    def on_submit_answer(self):
        ans = self.answer_box.get("1.0", "end").strip()
        score, matched, missed = evaluate_answer(ans, self.current_question["keywords"])

        # Track history
        self.history.append({
            "id": self.current_question["id"],
            "topic": self.current_question.get("topic", self.custom_topic),
            "skill": self.current_question.get("skill", "Concept"),
            "question": self.current_question["question"],
            "score": score,
            "matched": matched,
            "missed": missed,
            "algo": self.selection_method
        })
        self.asked_ids.add(self.current_question["id"])
        self.search_traces.append((self.current_question["id"], self.selection_method, score))

        # Update weakness profile (smooth moving average)
        skill = self.current_question["skill"]
        current_weakness = self.weakness.get(skill, 50)
        self.weakness[skill] = int((current_weakness + score) / 2)

        self.show_feedback_screen(score, matched, missed)

    def on_skip_question(self):
        self.asked_ids.add(self.current_question["id"])
        self.show_question_screen()

    # -----------------------------------------------------------------------
    # SCREEN 3: REAL-TIME FEEDBACK & PROPOSITIONAL LOGIC INFERENCE
    # -----------------------------------------------------------------------
    def show_feedback_screen(self, score, matched, missed):
        self.clear()
        self.build_header(
            "Evaluation & AI Knowledge Update",
            "Keyword Analysis and Propositional Logic Inference Fired",
            f"Score: {score}%",
            THEME["success"] if score >= 75 else (THEME["warning"] if score >= 45 else THEME["danger"])
        )

        content = tk.Frame(self.root, bg=THEME["bg_main"], padx=35, pady=20)
        content.pack(fill="both", expand=True)

        # SCORE CARD
        score_card = tk.Frame(content, bg=THEME["bg_card"], bd=1, relief="solid", padx=25, pady=15)
        score_card.configure(highlightbackground=THEME["border"], highlightthickness=1)
        score_card.pack(fill="x", pady=(0, 15))

        if score >= 75:
            assessment = "Strong Technical Understanding"
            color = THEME["success"]
            icon = "✓"
        elif score >= 45:
            assessment = "Partial Understanding — Key Concepts Missed"
            color = THEME["warning"]
            icon = "⚠"
        else:
            assessment = "Needs Targeted Practice"
            color = THEME["danger"]
            icon = "✗"

        tk.Label(score_card, text=f"{icon}  {score}% — {assessment}", font=("Segoe UI", 16, "bold"),
                 bg=THEME["bg_card"], fg=color).pack(anchor="w")

        # KEYWORDS BREAKDOWN CARD
        kw_card = tk.Frame(content, bg=THEME["bg_card"], bd=1, relief="solid", padx=25, pady=15)
        kw_card.configure(highlightbackground=THEME["border"], highlightthickness=1)
        kw_card.pack(fill="x", pady=(0, 15))

        tk.Label(kw_card, text="Concept Keyword Analysis:", font=("Segoe UI", 11, "bold"),
                 bg=THEME["bg_card"], fg=THEME["text_main"]).pack(anchor="w", pady=(0, 8))

        kw_row = tk.Frame(kw_card, bg=THEME["bg_card"])
        kw_row.pack(fill="x")

        # Matched
        m_frame = tk.Frame(kw_row, bg=THEME["bg_card"])
        m_frame.pack(side="left", fill="x", expand=True)
        tk.Label(m_frame, text=f"Concepts Mentioned ({len(matched)}):", font=("Segoe UI", 9, "bold"),
                 bg=THEME["bg_card"], fg=THEME["success"]).pack(anchor="w")
        m_text = ", ".join(matched) if matched else "None detected"
        tk.Label(m_frame, text=m_text, font=("Segoe UI", 10),
                 bg=THEME["bg_card"], fg=THEME["text_main"], wraplength=450).pack(anchor="w")

        # Missed
        x_frame = tk.Frame(kw_row, bg=THEME["bg_card"])
        x_frame.pack(side="left", fill="x", expand=True)
        tk.Label(x_frame, text=f"Concepts Expected / Omitted ({len(missed)}):", font=("Segoe UI", 9, "bold"),
                 bg=THEME["bg_card"], fg=THEME["warning"]).pack(anchor="w")
        x_text = ", ".join(missed) if missed else "All primary keywords covered!"
        tk.Label(x_frame, text=x_text, font=("Segoe UI", 10),
                 bg=THEME["bg_card"], fg=THEME["text_muted"], wraplength=450).pack(anchor="w")

        # LOGICAL REASONING INFERENCE CARD (Unit II)
        logic_card = tk.Frame(content, bg=THEME["bg_card"], bd=1, relief="solid", padx=25, pady=15)
        logic_card.configure(highlightbackground=THEME["border"], highlightthickness=1)
        logic_card.pack(fill="both", expand=True, pady=(0, 15))

        tk.Label(logic_card, text="🧠 Unit II Propositional Logic Inference Trace:",
                 font=("Segoe UI", 11, "bold"), bg=THEME["bg_card"], fg=THEME["secondary"]).pack(anchor="w")

        skill_clean = self.current_question["skill"].replace(" ", "_")
        initial_facts = {"Attempted_Question", f"Tested_{skill_clean}"}
        if score >= 70:
            initial_facts.add(f"Mastered_{skill_clean}")
        else:
            initial_facts.add(f"Weak_{skill_clean}")

        rules = [
            (["Attempted_Question"], "Knowledge_Base_Updated"),
            ([f"Mastered_{skill_clean}", "Knowledge_Base_Updated"], f"Eligible_For_Advanced_{skill_clean}"),
            ([f"Weak_{skill_clean}", "Knowledge_Base_Updated"], f"Schedule_Remediation_{skill_clean}")
        ]

        inferred, trace = propositional_forward_chaining(initial_facts, rules)

        trace_box = tk.Text(logic_card, height=6, font=("Consolas", 10), bg=THEME["bg_input"],
                            fg=THEME["text_main"], bd=1, relief="solid", padx=10, pady=10)
        trace_box.pack(fill="both", expand=True, pady=8)

        trace_content = f"Initial Known Facts: {sorted(list(initial_facts))}\n"
        for t in trace:
            trace_content += f"→ {t}\n"
        trace_content += f"\nFinal Inferred Knowledge Set:\n{sorted(list(inferred))}"
        trace_box.insert("1.0", trace_content)
        trace_box.config(state="disabled")

        # NEXT BUTTON
        tk.Button(content, text="PROCEED TO NEXT QUESTION →", font=("Segoe UI", 11, "bold"),
                  bg=THEME["primary"], fg="#ffffff", activebackground=THEME["primary_hover"],
                  activeforeground="#ffffff", bd=0, padx=25, pady=10, cursor="hand2",
                  command=self.show_question_screen).pack(side="right")

    # -----------------------------------------------------------------------
    # SCREEN 4: COMPREHENSIVE REPORT & AI REASONING DASHBOARD
    # -----------------------------------------------------------------------
    def show_results_screen(self):
        self.clear()
        total_q = len(self.history)
        avg_score = int(sum(x["score"] for x in self.history) / max(1, total_q))

        self.build_header(
            "Final Technical Assessment Report",
            f"Candidate: {self.student_name}   |   Domain: {self.custom_topic}   |   Total Questions: {total_q}",
            f"Overall: {avg_score}%",
            THEME["success"] if avg_score >= 70 else (THEME["warning"] if avg_score >= 50 else THEME["danger"])
        )

        canvas = tk.Canvas(self.root, bg=THEME["bg_main"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=THEME["bg_main"], padx=35, pady=20)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        def on_canvas_configure(e):
            canvas.itemconfig(canvas_window, width=e.width)

        canvas.bind("<Configure>", on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # SCORE OVERVIEW CARD
        overview = tk.Frame(scroll_frame, bg=THEME["bg_card"], bd=1, relief="solid", padx=25, pady=20)
        overview.configure(highlightbackground=THEME["border"], highlightthickness=1)
        overview.pack(fill="x", pady=(0, 20))

        tk.Label(overview, text=f"Overall Mastery: {avg_score}%", font=("Segoe UI", 20, "bold"),
                 bg=THEME["bg_card"], fg=THEME["text_main"]).pack(anchor="w")

        # Subtopic performance breakdown
        topic_scores = {}
        for item in self.history:
            topic_scores.setdefault(item["topic"], []).append(item["score"])

        breakdown_frame = tk.Frame(overview, bg=THEME["bg_card"])
        breakdown_frame.pack(fill="x", pady=(15, 0))

        tk.Label(breakdown_frame, text="Subtopic Performance Breakdown:", font=("Segoe UI", 11, "bold"),
                 bg=THEME["bg_card"], fg=THEME["text_muted"]).pack(anchor="w", pady=(0, 6))

        for t, scores in topic_scores.items():
            t_avg = int(sum(scores) / len(scores))
            row = tk.Frame(breakdown_frame, bg=THEME["bg_card"])
            row.pack(fill="x", pady=2)
            tk.Label(row, text=f"• {t}", font=("Segoe UI", 10), bg=THEME["bg_card"],
                     fg=THEME["text_main"]).pack(side="left")
            bar_color = THEME["success"] if t_avg >= 70 else (THEME["warning"] if t_avg >= 45 else THEME["danger"])
            tk.Label(row, text=f"{t_avg}%", font=("Segoe UI", 10, "bold"), bg=THEME["bg_card"],
                     fg=bar_color).pack(side="right")

        # RECOMMENDED PRACTICE
        rec_card = tk.Frame(scroll_frame, bg=THEME["bg_card"], bd=1, relief="solid", padx=25, pady=20)
        rec_card.configure(highlightbackground=THEME["border"], highlightthickness=1)
        rec_card.pack(fill="x", pady=(0, 20))

        tk.Label(rec_card, text="Targeted Revision & Weak Areas (CSP/Search Priority)",
                 font=("Segoe UI", 13, "bold"), bg=THEME["bg_card"], fg=THEME["text_main"]).pack(anchor="w")

        weak_skills = sorted(self.weakness.items(), key=lambda x: x[1])[:4]
        if weak_skills:
            for skill, score in weak_skills:
                tk.Label(rec_card, text=f"→ {skill} (Assessed Mastery: {score}%) — Recommended for focused review.",
                         font=("Segoe UI", 10), bg=THEME["bg_card"], fg=THEME["warning"]).pack(anchor="w", pady=2)
        else:
            tk.Label(rec_card, text="All assessed skills demonstrated satisfactory proficiency.",
                     font=("Segoe UI", 10), bg=THEME["bg_card"], fg=THEME["success"]).pack(anchor="w")

        # AI ALGORITHM REASONING SHOWCASE (Syllabus Units I, II, III Demonstration)
        ai_card = tk.Frame(scroll_frame, bg=THEME["bg_card"], bd=1, relief="solid", padx=25, pady=20)
        ai_card.configure(highlightbackground=THEME["border"], highlightthickness=1)
        ai_card.pack(fill="x", pady=(0, 20))

        tk.Label(ai_card, text="AI Algorithms Execution Showcase (Viva & Syllabus Verification)",
                 font=("Segoe UI", 13, "bold"), bg=THEME["bg_card"], fg=THEME["secondary"]).pack(anchor="w")

        # 1. Search History Trace
        trace_text = "1. SEARCH ALGORITHMS TRACE (Unit I):\n"
        for qid, algo, sc in self.search_traces:
            trace_text += f"   - Question ID #{qid}: Selected by '{algo}' | Student Score: {sc}%\n"

        # 2. FOL Unification Example
        first_skill = self.history[0]["skill"].replace(" ", "_") if self.history else "Core_Concept"
        subst = unify("Mastered(?candidate, ?topic)", f"Mastered({self.student_name}, {first_skill})")
        trace_text += f"\n2. FIRST-ORDER LOGIC UNIFICATION (Unit III):\n"
        trace_text += f"   - Pattern: Mastered(?candidate, ?topic)\n"
        trace_text += f"   - Fact:    Mastered({self.student_name}, {first_skill})\n"
        trace_text += f"   - MGU (Most General Unifier): {subst}\n"

        # 3. Propositional Resolution
        resolvents = resolution_demo()
        trace_text += f"\n3. PROPOSITIONAL RESOLUTION PRINCIPLE (Unit II):\n"
        trace_text += f"   - Clause 1: {{P, Q}}   Clause 2: {{~Q}}\n"
        trace_text += f"   - Resolvents derived: {resolvents}\n"

        ai_box = tk.Text(ai_card, height=10, font=("Consolas", 10), bg=THEME["bg_input"],
                         fg=THEME["text_main"], bd=1, relief="solid", padx=10, pady=10)
        ai_box.pack(fill="both", expand=True, pady=10)
        ai_box.insert("1.0", trace_text)
        ai_box.config(state="disabled")

        # ACTIONS
        action_bar = tk.Frame(scroll_frame, bg=THEME["bg_main"], pady=10)
        action_bar.pack(fill="x")

        tk.Button(action_bar, text="START NEW INTERVIEW", font=("Segoe UI", 11, "bold"),
                  bg=THEME["primary"], fg="#ffffff", activebackground=THEME["primary_hover"],
                  activeforeground="#ffffff", bd=0, padx=22, pady=10, cursor="hand2",
                  command=self.show_home_screen).pack(side="left")

        tk.Button(action_bar, text="EXIT SYSTEM", font=("Segoe UI", 11),
                  bg=THEME["bg_card"], fg=THEME["text_muted"], activebackground=THEME["border"],
                  activeforeground=THEME["text_main"], bd=1, relief="solid", padx=20, pady=10, cursor="hand2",
                  command=self.root.destroy).pack(side="right")


# ===========================================================================
# 8. APPLICATION ENTRY POINT
# ===========================================================================

def main():
    root = tk.Tk()
    ModernInterviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
