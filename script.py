import os
import sys
import re
import time
import shutil
import random
from typing import List, Dict, Tuple

# ------- Spawning Config ----------
MAX_CONCURRENT = 4                 # how many moving dinos at once
SPAWN_INTERVAL_RANGE = (0.6, 1.4)  # seconds between spawns (randomized)
SPEED_RANGE = (1, 2)               # chars per frame, used by the spawner

# ---------- Paths (project-relative) ----------
ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_ANIMS = os.path.join(ROOT, "Dinosaurs animations")
BYE_PATH = os.path.join(BASE_ANIMS, "bye", "message.txt")


BACKGROUND_DIR = os.path.join(BASE_ANIMS, "background")
PALM_PATH = os.path.join(BACKGROUND_DIR, "palm.txt")
SUN_PATH  = os.path.join(BACKGROUND_DIR, "sun.txt")


# Debug toggles
DEBUG_BACKGROUND = False  # set True to see dotted background
DRAW_PROPS = True  #swtich on for background props (sun,palm)

# ---------- Optional Windows ANSI support ----------
if os.name == 'nt':
    try:
        import colorama  # type: ignore
        colorama.init()
    except Exception:
        pass

# ---------- Terminal color codes ----------
COLOR_CODES: Dict[str, str] = {
    'red': "\033[31m",
    'green': "\033[32m",
    'yellow': "\033[33m",
    'blue': "\033[34m",
    'magenta': "\033[35m",
    'cyan': "\033[36m",
    'white': "\033[37m",
    'reset': "\033[0m"
}

# ---------- Small terminal helpers ----------

def print_buffered(buffer: List[str]) -> None:
    # No extra newlines; and ALWAYS flush so frames advance every tick
    sys.stdout.write("\033[H")
    sys.stdout.write("".join(buffer))
    sys.stdout.flush()

def clear_console() -> None:
    os.system('cls' if os.name == 'nt' else 'clear')

def hide_cursor() -> None:
    print("\033[?25l", end='')

def show_cursor() -> None:
    print("\033[?25h", end='')


def _sun_size() -> Tuple[int, int]:
    
    if not DRAW_PROPS:  # if props aren't drawn, don't reserve space
        return (0, 0)
    lines = load_ascii_block(SUN_PATH)
    if not lines:
        return (0, 0)
    w = max((len(line) for line in lines), default=0)
    h = len(lines)
    return (w, h)

# ---------- Loading/IO ----------
def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def load_animation_frames_from_dir(dir_path: str) -> List[List[str]]:
    if not os.path.isdir(dir_path):
        return []
    frame_files = sorted(
        (f for f in os.listdir(dir_path) if f.endswith('.txt')),
        key=_natural_key
    )
    frames: List[List[str]] = []
    for filename in frame_files:
        with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as f:
            # keep trailing spaces; only strip newline
            frames.append([line.rstrip('\n') for line in f])
    return frames

def display_ascii_file(file_path: str) -> None:
    if not os.path.exists(file_path):
        print("Goodbye message file not found.")
        return
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f.read())


def load_ascii_block(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


# --- Frame normalization & ASCII mirroring helpers ---
def normalize_frame_to_width(frame: List[str], width: int) -> List[str]:
    """Pad each line with spaces to exactly `width` characters."""
    return [line + (" " * (width - len(line))) for line in frame]

def normalize_frames(frames: List[List[str]]) -> List[List[str]]:
    """Pad every frame so all lines across all frames share the same width."""
    max_w = 0
    for fr in frames:
        max_w = max(max_w, max((len(line) for line in fr), default=0))
    return [normalize_frame_to_width(fr, max_w) for fr in frames]

_MIRROR_PAIRS = {
    '/': '\\', '\\': '/',
    '(': ')', ')': '(',
    '[': ']', ']': '[',
    '{': '}', '}': '{',
    '<': '>', '>': '<'
}
def _mirror_char(ch: str) -> str:
    return _MIRROR_PAIRS.get(ch, ch)

def mirror_frame(frame: List[str]) -> List[str]:
    # reverse each line and swap asymmetrical glyphs; spaces preserved by normalization
    return ["".join(_mirror_char(c) for c in line[::-1]) for line in frame]

def mirror_frames(frames: List[List[str]]) -> List[List[str]]:
    # ensure constant width first, then mirror each frame
    normed = normalize_frames(frames)
    return [mirror_frame(fr) for fr in normed]


# ---------- Core classes ----------
class AgentColorScheme:
    def __init__(self, color_map: Dict[str, str]):
        self.color_map = color_map

    def get_color(self, ch: str) -> str:
        return self.color_map.get(ch, "")

class Actor:
    def __init__(self, x: int, y: int, frames: List[List[str]], color_map=None):
        self._x = x
        self._y = y
        self._ascii_art = frames
        self.color_map = color_map or {}
        self._visible = False
        self._current_frame = 0

    def spawn(self) -> None:
        self._visible = True

    def despawn(self) -> None:
        self._visible = False

    def move(self, new_x: int, new_y: int) -> None:
        self._x = new_x
        self._y = new_y

    def advance_frame(self) -> None:
        if self._ascii_art:
            self._current_frame = (self._current_frame + 1) % len(self._ascii_art)

    def update(self) -> List[str]:
        if not self._visible or not self._ascii_art:
            return []
        self.advance_frame()
        return self._draw()

    def _apply_color(self, ch: str) -> str:
        if isinstance(self.color_map, AgentColorScheme):
            color = self.color_map.get_color(ch)
        else:
            color = self.color_map.get(ch, "")
        if color:
            return f"{color}{ch}{COLOR_CODES['reset']}"
        return ch

    def _draw(self) -> List[str]:
        if not self._ascii_art:
            return []
        frame = self._ascii_art[self._current_frame]
        buffer: List[str] = []
        for i, line in enumerate(frame):
            colored_line = "".join(self._apply_color(ch) for ch in line)
            buffer.append(f"\033[{self._y + i};{self._x}H{colored_line}")
        return buffer

    def _clear(self) -> List[str]:
        if not self._ascii_art:
            return []
        frame = self._ascii_art[self._current_frame]
        buffer: List[str] = []
        for i, line in enumerate(frame):
            buffer.append(f"\033[{self._y + i};{self._x}H{' ' * len(line)}")
        return buffer

    # Helpers for derived classes
    @staticmethod
    def frame_size(frames: List[List[str]]) -> Tuple[int, int]:
        """Returns (max_width, max_height) over all frames."""
        if not frames:
            return (0, 0)
        max_w = 0
        max_h = 0
        for fr in frames:
            h = len(fr)
            w = max((len(line) for line in fr), default=0)
            max_w = max(max_w, w)
            max_h = max(max_h, h)
        return (max_w, max_h)


class Background(Actor):
    """Static background that does not get cleared each frame."""
    def __init__(self, frames: List[List[str]], color_map=None):
        super().__init__(0, 0, frames, color_map=color_map)

    def _clear(self) -> List[str]:
        return []


class StaticProp(Actor):
    """Immovable background prop; drawn every frame, no per-frame clearing."""
    def __init__(self, x: int, y: int, lines: List[str], color_map=None):
        # Wrap the single frame as a list of frames
        frames = [lines] if lines else []
        super().__init__(x, y, frames, color_map=color_map)

    def _clear(self) -> List[str]:
        return []  # never clears; acts like part of the background

    def _draw(self) -> List[str]:
        if not self._ascii_art:
            return []
        # Clip against current terminal size to avoid wrap
        term_w, term_h = shutil.get_terminal_size()
        frame = self._ascii_art[self._current_frame]
        out: List[str] = []
        for i, raw_line in enumerate(frame):
            y = self._y + i
            if y < 1 or y > term_h:
                continue
            # Horizontal clipping
            start_x = max(1, self._x)
            end_x   = min(term_w, self._x + len(raw_line) - 1)
            if end_x < start_x:
                continue
            left_clip = start_x - self._x
            right_len = end_x - start_x + 1
            visible = raw_line[left_clip:left_clip + right_len]
            colored = "".join(self._apply_color(ch) for ch in visible)
            out.append(f"\033[{y};{start_x}H{colored}")
        return out


class MovingActor(Actor):
    """Moves horizontally from one edge to the other, then can be removed."""
    def __init__(self, x: int, y: int, frames: List[List[str]], vx: int, bounds: Tuple[int, int], color_map=None):
        # Normalize frames so clearing uses a constant width
        frames = normalize_frames(frames)
        super().__init__(x, y, frames, color_map=color_map)
        self.vx = vx  # characters per frame (sign indicates direction)
        self.bounds_w, self.bounds_h = bounds
        self._w, self._h = Actor.frame_size(frames)

    def _draw(self) -> List[str]:
        if not self._ascii_art:
            return []
        frame = self._ascii_art[self._current_frame]
        out: List[str] = []
        term_w, term_h = self.bounds_w, self.bounds_h

        for i, raw_line in enumerate(frame):
            y = self._y + i
            if y < 1 or y > term_h:
                continue

            # Visible horizontal slice
            start_x = max(1, self._x)
            end_x   = min(term_w, self._x + len(raw_line) - 1)
            if end_x < start_x:
                continue

            left_clip = start_x - self._x
            right_len = end_x - start_x + 1
            segment = raw_line[left_clip:left_clip + right_len]

            # Emit only non-space *runs* (so spaces are transparent)
            run_start = None
            for idx, ch in enumerate(segment + "\0"):  # sentinel to flush last run
                if ch != ' ' and ch != "\0" and run_start is None:
                    run_start = idx
                elif (ch == ' ' or ch == "\0") and run_start is not None:
                    run = segment[run_start:idx]
                    colored = "".join(self._apply_color(c) for c in run)
                    x = start_x + run_start
                    out.append(f"\033[{y};{x}H{colored}")
                    run_start = None
        return out
   
    def _clear(self) -> List[str]:
        # Clear a solid rectangle using max w/h, but clip to terminal width to avoid wrap
        buf: List[str] = []
        for i in range(self._h):
            start_x = max(1, self._x)
            end_x = min(self.bounds_w, self._x + self._w - 1)
            if end_x >= start_x:
                span = end_x - start_x + 1
                buf.append(f"\033[{self._y + i};{start_x}H" + (" " * span))
        return buf

    def update(self) -> List[str]:
        if not self._visible or not self._ascii_art:
            return []
        # Erase current
        buf = self._clear()
        # Move strictly horizontally
        self._x += self.vx
        # Advance animation frame
        self.advance_frame()
        # Draw new
        buf += self._draw()
        return buf

    def is_offscreen(self) -> bool:
        # Completely out of view horizontally
        return (self._x > self.bounds_w) or (self._x + self._w < 1)


class Scene:
    def __init__(self):
        self.entities: List[Actor] = []

    def add_entity(self, entity: Actor) -> None:
        self.entities.append(entity)
        entity.spawn()

    def remove_entity(self, entity: Actor) -> None:
        entity.despawn()
        if entity in self.entities:
            self.entities.remove(entity)

    def update(self) -> None:
        full_buffer: List[str] = []

        movers = [e for e in self.entities if isinstance(e, MovingActor)]
        backgrounds = [e for e in self.entities if isinstance(e, Background) or type(e).__name__ == "StaticProp"]

        # 1) Clear movers at their *old* positions
        for m in movers:
            full_buffer.extend(m._clear())

        # 2) Draw background/props (these should not clear)
        for b in backgrounds:
            full_buffer.extend(b.update())

        # 3) Move + draw movers (with transparent spaces)
        for m in list(movers):
            # move + advance
            m._x += m.vx
            m.advance_frame()
            full_buffer.extend(m._draw())
            if m.is_offscreen():
                self.remove_entity(m)

        print_buffered(full_buffer)

def _movers(scene: "Scene") -> list:
    return [e for e in scene.entities if isinstance(e, MovingActor)]




def _rect_intersect(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> bool:
    """
    Axis-aligned rectangle intersection.
    Rect = (x, y, w, h) with 1-based console coords.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw - 1, ay + ah - 1
    bx2, by2 = bx + bw - 1, by + bh - 1
    return not (ax2 < bx or bx2 < ax or ay2 < by or by2 < ay)

def _occupied_rects(scene: "Scene") -> list[Tuple[int,int,int,int]]:
    """Current rectangles of all MovingActors (could also include static props if you want to avoid those)."""
    rects = []
    for e in scene.entities:
        if isinstance(e, MovingActor):
            rects.append((e._x, e._y, e._w, e._h))
        # If you also want to avoid palm overlap, include StaticProp here:
        # elif isinstance(e, StaticProp):
        #     w = max((len(line) for line in e._ascii_art[0]), default=0) if e._ascii_art else 0
        #     h = len(e._ascii_art[0]) if e._ascii_art else 0
        #     rects.append((e._x, e._y, w, h))
    return rects

def spawn_random_agent_nonoverlap(scene: Scene, term_w: int, term_h: int, speed_range: Tuple[int, int] = SPEED_RANGE) -> bool:
    """
    Spawn one random agent with:
      - Random side (left/right)
      - Mirroring if moving left
      - Non-overlapping (x,y,w,h) vs current movers
    """
    name, base_frames = _load_random_agent_frames()
    if not base_frames:
        return False

    # Precompute two variants: facing right (normalized) and facing left (mirrored+normalized)
    frames_right = normalize_frames(base_frames)
    frames_left  = mirror_frames(base_frames)

    attempts = 24  # total tries across sides/y positions
    occ = _occupied_rects(scene)

    for _ in range(attempts):
        from_left = bool(random.getrandbits(1))
        use_frames = frames_right if from_left else frames_left
        fw, fh = Actor.frame_size(use_frames)
        if fw <= 0 or fh <= 0:
            return False

      
            
        # Choose Y first (non-overlapping vertically helps reduce retries)
        sun_w, sun_h = _sun_size()
        reserved_top = (1 + sun_h + 10) if sun_h > 0 else 1  # rows 1..(sun_h+10) are reserved
        frame_w, frame_h = Actor.frame_size(use_frames)

        max_y = max(1, term_h - frame_h + 1)
        min_y = min(max_y, max(1, reserved_top))  # clamp
        if min_y > max_y:
            # if terminal is tiny, fall back to topmost allowed
            y = max_y
        else:
            y = random.randint(min_y, max_y)

        # Choose X + velocity based on side
        if from_left:
            x  = 1
            vx = random.randint(*speed_range)      # →
        else:
            x  = max(1, term_w - fw)
            vx = -random.randint(*speed_range)     # ←

        start_rect = (x, y, fw, fh)

        # Check collision with current movers
        if any(_rect_intersect(start_rect, r) for r in occ):
            # Try a new Y next time
            continue

        # Looks free: spawn it
        color_scheme = generate_color_scheme(abs(hash(name)) % 7)
        mover = MovingActor(x, y, use_frames, vx=vx, bounds=(term_w, term_h), color_map=color_scheme)
        scene.add_entity(mover)
        return True

    # No room right now
    return False
# ---------- Convenience builders ----------
def generate_color_scheme(index: int) -> AgentColorScheme:
    colors = [
        COLOR_CODES['red'], COLOR_CODES['green'], COLOR_CODES['yellow'],
        COLOR_CODES['blue'], COLOR_CODES['magenta'], COLOR_CODES['cyan'],
        COLOR_CODES['white']
    ]
    symbols = ['/', '\\', 'o', '|', '_', '~', '-', '=']
    return AgentColorScheme({s: colors[(index + i) % len(colors)] for i, s in enumerate(symbols)})

def _make_background(width: int, height: int) -> Background:
    bg_frames = [["." * width for _ in range(height)]]
    bg_color_map = {
        '.': COLOR_CODES['magenta'],
        ' ': COLOR_CODES['reset']
    }
    return Background(bg_frames, color_map=bg_color_map)

def _available_agent_dirs() -> List[str]:
    if not os.path.isdir(BASE_ANIMS):
        return []
    return [
        d for d in os.listdir(BASE_ANIMS)
        if os.path.isdir(os.path.join(BASE_ANIMS, d)) and d not in ('bye', 'background')
    ]

def _load_random_agent_frames() -> Tuple[str, List[List[str]]]:
    # Pick a random dinosaur folder with at least one frame
    dirs = _available_agent_dirs()
    random.shuffle(dirs)
    for d in dirs:
        frames = load_animation_frames_from_dir(os.path.join(BASE_ANIMS, d))
        if frames:
            return d, frames
    return "", []


def spawn_random_agent(scene: Scene, term_w: int, term_h: int, speed_range: tuple[int, int] = (1, 2)) -> bool:
    """
    Spawn ONE random agent.
    - Random side: left (x=1, vx>0) or right (x=term_w - frame_w, vx<0)
    - Random y within visible range: 1 .. term_h - frame_h (clamped)
    Returns True if spawned, False otherwise.
    """
    name, frames = _load_random_agent_frames()
    if not frames:
        return False

    # Base size from original frames (may change if mirrored)
    frame_w, frame_h = Actor.frame_size(frames)
    if frame_w <= 0 or frame_h <= 0:
        return False

    
    # Reserve rows below the sun so dinos don't clip into it
    sun_w, sun_h = _sun_size()
    reserved_top = (1 + sun_h + 10) if sun_h > 0 else 1

    # Random vertical placement (respect reserved top band)
    max_y = max(1, term_h - frame_h + 1)
    min_y = min(max_y, max(1, reserved_top))
    if min_y > max_y:
        y = max_y
    else:
        y = random.randint(min_y, max_y)

    # Random edge + direction
    from_left = bool(random.getrandbits(1))
    if from_left:
        x = 1
        vx = random.randint(*speed_range)          # move right
        use_frames = normalize_frames(frames)      # keep original, normalized
        # width may change after normalization
        frame_w, frame_h = Actor.frame_size(use_frames)
    else:
        # mirror so it faces left, then recompute size and start at right edge
        use_frames = mirror_frames(frames)         # mirror includes normalization
        frame_w, frame_h = Actor.frame_size(use_frames)
        x = max(1, term_w - frame_w)
        vx = -random.randint(*speed_range)         # move left

    # Stable but varied color scheme seeded by the name
    color_scheme = generate_color_scheme(abs(hash(name)) % 7)

    mover = MovingActor(x, y, use_frames, vx=vx, bounds=(term_w, term_h), color_map=color_scheme)
    scene.add_entity(mover)
    return True


def add_palm_prop(scene: Scene, term_w: int, term_h: int) -> None:
    """
    Place the palm near the bottom-left-ish area, relative to terminal size.
    Tweak offsets to taste.
    """
    lines = load_ascii_block(PALM_PATH)
    if not lines:
        return
    # Measure palm size
    palm_w = max((len(line) for line in lines), default=0)
    palm_h = len(lines)

    x = max(1, term_w // 8)
    y = max(1, term_h - palm_h + 1)

    palm = StaticProp(x, y, lines)
    scene.add_entity(palm)


def add_sun_prop(scene: Scene, term_w: int, term_h: int) -> None:
    """
    Place the sun at the top-right corner (1-based console coords).
    """
    lines = load_ascii_block(SUN_PATH)
    if not lines:
        return
    sun_w = max((len(line) for line in lines), default=0)
    x = max(1, term_w - sun_w + 1)
    y = 1 
    sun = StaticProp(x, y, lines)
    scene.add_entity(sun)

def add_background_props(scene: Scene, term_w: int, term_h: int) -> None:
    """
    Adds all static background props, positioned for the given terminal size.
    Controlled by DRAW_PROPS.
    """
    if not DRAW_PROPS:
        return
    add_palm_prop(scene, term_w, term_h)
    add_sun_prop(scene, term_w, term_h)


def create_scene() -> Scene:
    scene = Scene()
    curr_width, curr_height = shutil.get_terminal_size()

    if DEBUG_BACKGROUND:
        scene.add_entity(_make_background(curr_width, curr_height))

    add_background_props(scene, curr_width, curr_height)  # props gated by DRAW_PROPS
    spawn_random_agent(scene, curr_width, curr_height)
    return scene


# ---------- Main ----------
if __name__ == '__main__':
    clear_console()
    prev_width, prev_height = shutil.get_terminal_size()
    random.seed()  # seed from system entropy

    scene = create_scene()
    
    try:
        hide_cursor()
        next_spawn = time.perf_counter()  # spawn as soon as the loop starts

            
        while True:
            curr_width, curr_height = shutil.get_terminal_size()

     # Rebuild on resize
            if (curr_width, curr_height) != (prev_width, prev_height):
                prev_width, prev_height = curr_width, curr_height
                clear_console()
                scene = Scene()
                if DEBUG_BACKGROUND:
                    scene.add_entity(_make_background(curr_width, curr_height))
                add_background_props(scene, curr_width, curr_height)  # props recalculated per size
                next_spawn = time.perf_counter()  # reset spawn timer after resize

        # Spawn new movers on a cooldown, up to MAX_CONCURRENT
            now = time.perf_counter()
            current_movers = _movers(scene)
            if len(current_movers) < MAX_CONCURRENT and now >= next_spawn:
                if spawn_random_agent_nonoverlap(scene, curr_width, curr_height, SPEED_RANGE):
                    next_spawn = now + random.uniform(*SPAWN_INTERVAL_RANGE)
                else:
                    next_spawn = now + 0.5

        # ALWAYS update & sleep each tick
            scene.update()
            time.sleep(0.10)



    except KeyboardInterrupt:
        pass
    finally:
        clear_console()
        show_cursor()
        display_ascii_file(BYE_PATH)
