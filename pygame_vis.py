"""Pygame live visualisation — map, spike raster, scrollable cognitive dashboard.

Launch with:  python -m genesis.main --gui

Keyboard controls:
  TAB    — cycle focused agent  (neural raster + vision circle + cogmap)
  M      — toggle cognitive-map overlay
  ESC    — quit
  Scroll — scroll sidebar up / down
"""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

try:
    import pygame

    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

if TYPE_CHECKING:
    from genesis.agent.agent import ConsciousAgent
    from genesis.environment.sandbox import Sandbox

# ── colour palette ───────────────────────────────────────────────
BG = (10, 11, 18)
GRID_BG = (20, 22, 32)
GRID_LINE = (26, 28, 40)
OBSTACLE_C = (44, 46, 58)
OBSTACLE_HI = (56, 58, 70)
CRYSTAL_C = (0, 240, 195)
CRYSTAL_DIM = (0, 140, 115)
CRYSTAL_GLOW = (0, 180, 150)
AGENT_COLS = [(255, 95, 95), (95, 145, 255), (95, 255, 130), (255, 225, 70)]
AGENT_GLOW = [
    (255, 60, 60, 45),
    (60, 110, 255, 45),
    (60, 255, 100, 45),
    (255, 200, 40, 45),
]
DEAD_C = (90, 45, 45)

TEXT_C = (218, 220, 230)
TEXT_BRIGHT = (245, 245, 255)
MUTED_C = (135, 138, 152)
DIM_C = (78, 80, 95)
ACCENT_C = (100, 180, 255)

PANEL_BG = (18, 20, 32)
PANEL_EDGE = (36, 38, 52)
PANEL_HEAD = (26, 28, 42)

BAR_BG = (32, 34, 48)
BAR_BORDER = (48, 50, 65)
BAR_FG_E = (50, 210, 110)
BAR_FG_I = (80, 150, 255)
BAR_PHI = (235, 175, 50)
BAR_CURIO = (180, 110, 255)
BAR_BIND = (80, 200, 220)
BAR_CONF = (210, 190, 70)
BAR_EMP = (255, 130, 80)
BAR_IDENT = (160, 220, 255)
BAR_PRED = (255, 100, 120)

HEATMAP_COLS = [
    np.array([12, 12, 30], dtype=np.float32),
    np.array([25, 70, 155], dtype=np.float32),
    np.array([35, 180, 200], dtype=np.float32),
    np.array([220, 220, 55], dtype=np.float32),
    np.array([255, 75, 45], dtype=np.float32),
]

EMOTION_COLS = [
    (255, 70, 70),
    (180, 110, 255),
    (70, 230, 130),
    (255, 170, 50),
    (90, 140, 220),
    (255, 255, 90),
]
EMOTION_NAMES_SHORT = ["FEA", "CUR", "CON", "FRU", "LON", "SUR"]

GOAL_COLS = [
    (255, 80, 80),
    (70, 210, 110),
    (180, 110, 255),
    (90, 190, 255),
    (255, 210, 70),
    (255, 150, 50),
]
GOAL_NAMES_SHORT = ["SURVIVE", "FORAGE", "EXPLORE", "SOCIAL", "LEARN", "CREATE"]

CP_COLS = [
    (255, 140, 70),
    (70, 210, 110),
    (90, 190, 255),
    (210, 190, 70),
    (180, 110, 255),
]
CP_NAMES = ["sens", "motr", "socl", "lang", "self"]

CELL = 5
SIDE_W = 440
RASTER_H = 110
SIDE_PAD = 14
MIN_FPS = 30

NIGHT_TINT = (8, 10, 35)
RAIN_TINT = (20, 30, 60)

# Biome terrain colours (subtle, dark tones to not overpower glows)
BIOME_COLS = {
    0: (14, 20, 38),    # ocean — deep blue-black
    1: (16, 28, 22),    # wetlands — dark teal-green
    2: GRID_BG,         # grasslands — default grid bg
    3: (30, 24, 16),    # desert — warm dark brown
    4: (28, 26, 34),    # mountains — cool grey-purple
}
BIOME_OBS_COLS = {
    0: (30, 38, 56),    # ocean obstacles
    1: (34, 50, 42),    # wetlands obstacles
    2: OBSTACLE_C,      # grasslands obstacles
    3: (58, 48, 36),    # desert obstacles (sandstone)
    4: (56, 54, 64),    # mountain obstacles (granite)
}

VISION_COL = (255, 255, 255, 22)
SPARK_LEN = 60  # datapoints in sparklines


def _lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class PygameVisualiser:
    """Real-time graphical front-end for the simulation."""

    def __init__(self, sandbox: "Sandbox", num_agents: int) -> None:
        if not HAS_PYGAME:
            raise RuntimeError("pygame is not installed")

        pygame.init()
        self.map_w = sandbox.width * CELL
        self.map_h = sandbox.height * CELL
        self.win_w = self.map_w + SIDE_W
        self.win_h = max(self.map_h + RASTER_H, 900)
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption("Project Genesis")
        self.clock = pygame.time.Clock()

        # Fonts — try nicer system font, fall back to default
        for family in ("segoeui", "consolas", None):
            try:
                self.font_title = pygame.font.SysFont(family, 19, bold=True)
                self.font = pygame.font.SysFont(family, 13)
                self.font_sm = pygame.font.SysFont(family, 11)
                self.font_bold = pygame.font.SysFont(family, 12, bold=True)
                self.font_tiny = pygame.font.SysFont(family, 10)
                self.font_val = pygame.font.SysFont("consolas", 10)
                break
            except Exception:
                continue

        self.num_agents = num_agents
        self._running = True
        self._show_cogmap = False
        self._focused = 0  # TAB-switchable focused agent
        self._frame = 0
        self._scroll_y = 0  # sidebar scroll offset (negative = scrolled down)
        self._sidebar_content_h = 0  # last measured sidebar content height

        # Pre-render terrain with biome colours, grid, and obstacles
        self._obstacle_surf = pygame.Surface((self.map_w, self.map_h))
        self._obstacle_surf.fill(GRID_BG)
        # Paint biome base colours
        for by in range(sandbox.height):
            for bx in range(sandbox.width):
                biome_id = sandbox.biome_map.biome_at(bx, by)
                col = BIOME_COLS.get(biome_id, GRID_BG)
                if col != GRID_BG:
                    pygame.draw.rect(self._obstacle_surf, col,
                                     (bx * CELL, by * CELL, CELL, CELL))
        # Grid lines on top
        for gx in range(0, self.map_w, CELL * 5):
            pygame.draw.line(self._obstacle_surf, GRID_LINE, (gx, 0), (gx, self.map_h), 1)
        for gy in range(0, self.map_h, CELL * 5):
            pygame.draw.line(self._obstacle_surf, GRID_LINE, (0, gy), (self.map_w, gy), 1)
        # Obstacles drawn last
        for ox, oy in sandbox.obstacles:
            px, py = ox * CELL, oy * CELL
            # Tint obstacle by biome
            biome_id = sandbox.biome_map.biome_at(ox, oy)
            obs_col = BIOME_OBS_COLS.get(biome_id, OBSTACLE_C)
            pygame.draw.rect(self._obstacle_surf, obs_col, (px, py, CELL, CELL))
            hi = tuple(min(255, c + 12) for c in obs_col)
            pygame.draw.line(self._obstacle_surf, hi, (px, py), (px + CELL - 1, py), 1)
            pygame.draw.line(self._obstacle_surf, hi, (px, py), (px, py + CELL - 1), 1)

        # Alpha surfaces
        self._glow_surf = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)
        self._overlay_surf = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)
        self._night_surf = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)

        # Pre-build glow sprites
        self._crystal_glow = self._make_glow_circle(18, CRYSTAL_GLOW, 55)
        self._agent_glows = [self._make_glow_circle(26, c[:3], c[3]) for c in AGENT_GLOW]

        # Pre-compute ocean tile positions for shimmer animation
        self._ocean_tiles: list[tuple[int, int]] = []
        for by in range(sandbox.height):
            for bx in range(sandbox.width):
                if sandbox.biome_map.biome_at(bx, by) == 0 and (bx, by) not in sandbox.obstacles:
                    self._ocean_tiles.append((bx * CELL, by * CELL))

        # Sparkline history buffers per agent
        self._spark_pred: dict[int, deque] = {}
        self._spark_phi: dict[int, deque] = {}
        self._spark_energy: dict[int, deque] = {}

    @staticmethod
    def _make_glow_circle(radius: int, color: tuple, alpha: int) -> pygame.Surface:
        size = radius * 2
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        for r in range(radius, 0, -1):
            frac = r / radius
            a = int(alpha * (1 - frac) ** 0.4 * frac ** 0.2)
            a = max(0, min(255, a))
            pygame.draw.circle(surf, (*color[:3], a), (radius, radius), r)
        return surf

    # ─── public API ───────────────────────────────────────────────

    def alive(self) -> bool:
        return self._running

    def handle_events(self) -> bool:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self._running = False
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    self._running = False
                    return False
                if ev.key == pygame.K_m:
                    self._show_cogmap = not self._show_cogmap
                if ev.key == pygame.K_TAB:
                    self._focused = (self._focused + 1) % max(1, self.num_agents)
            if ev.type == pygame.MOUSEWHEEL:
                # Scroll sidebar if mouse is over sidebar area
                mx, _ = pygame.mouse.get_pos()
                if mx > self.map_w:
                    self._scroll_y += ev.y * 28
                    max_scroll = 0
                    min_scroll = min(0, self.win_h - 58 - self._sidebar_content_h)
                    self._scroll_y = max(min_scroll, min(max_scroll, self._scroll_y))
        return True

    def render(self, sandbox: "Sandbox", agents: list["ConsciousAgent"], tick: int) -> None:
        self._frame += 1
        self.screen.fill(BG)

        # Update sparklines
        for a in agents:
            aid = a.agent_id
            if aid not in self._spark_pred:
                self._spark_pred[aid] = deque(maxlen=SPARK_LEN)
                self._spark_phi[aid] = deque(maxlen=SPARK_LEN)
                self._spark_energy[aid] = deque(maxlen=SPARK_LEN)
            if tick % 3 == 0:  # sample every 3rd tick to reduce noise
                self._spark_pred[aid].append(a.prediction_engine.average_error)
                self._spark_phi[aid].append(a.phi_calculator.get_consciousness_assessment()["phi"])
                self._spark_energy[aid].append(a.body.energy / a.config.agent.max_energy)

        # Ensure focused agent index is valid
        if self._focused >= len(agents):
            self._focused = 0

        # Map
        self._draw_map(sandbox, agents)

        # Day/night/rain tint
        self._draw_atmosphere(sandbox)

        # Vision circle for focused agent
        if agents and agents[self._focused].alive:
            self._draw_vision_circle(agents[self._focused])

        # Cognitive map overlay
        if self._show_cogmap and agents:
            self._draw_cogmap_overlay(agents[self._focused])

        # Cultural lines
        self._draw_cultural_lines(agents)

        # Map border
        pygame.draw.rect(self.screen, PANEL_EDGE, (0, 0, self.map_w, self.map_h), 1)

        # Bottom status bar (overlaid on map)
        self._draw_bottom_status_bar(sandbox, agents, tick)

        # Spike raster (below map)
        if agents:
            self._draw_spike_raster(agents[self._focused], 0, self.map_h)

        # Sidebar (scrollable)
        self._draw_sidebar(agents, tick, sandbox)

        # Divider
        pygame.draw.line(self.screen, PANEL_EDGE, (self.map_w, 0), (self.map_w, self.win_h), 1)

        pygame.display.flip()
        self.clock.tick(MIN_FPS)

    def quit(self) -> None:
        pygame.quit()

    # ─── map ──────────────────────────────────────────────────────

    def _draw_map(self, sandbox: "Sandbox", agents: list["ConsciousAgent"]) -> None:
        self.screen.blit(self._obstacle_surf, (0, 0))
        self._glow_surf.fill((0, 0, 0, 0))

        # Ocean shimmer — subtle animated highlights on water tiles
        if self._ocean_tiles and (self._frame % 3 == 0):
            shimmer_surf = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)
            t = self._frame * 0.015
            for px, py in self._ocean_tiles:
                # Two-wave interference pattern
                wave = math.sin(px * 0.07 + t) + math.sin(py * 0.09 - t * 0.7)
                if wave > 0.8:
                    a = int((wave - 0.8) * 40)
                    pygame.draw.rect(shimmer_surf, (60, 90, 160, a),
                                     (px, py, CELL, CELL))
            self.screen.blit(shimmer_surf, (0, 0))

        pulse = 0.85 + 0.15 * math.sin(self._frame * 0.08)

        # Crystal glow + draw
        for c in sandbox.crystals:
            if c.consumed or c.is_expired:
                continue
            gx, gy = c.position.grid_pos()
            cx = gx * CELL + CELL // 2
            cy = gy * CELL + CELL // 2
            fresh = c.freshness > 0.5
            base_col = CRYSTAL_C if fresh else CRYSTAL_DIM

            # Pulsing brightness
            col = _lerp_color(base_col, (255, 255, 255), _clamp((pulse - 0.85) * 2.0 * c.freshness))

            glow = self._crystal_glow
            self._glow_surf.blit(glow, (cx - glow.get_width() // 2, cy - glow.get_height() // 2))

            # Diamond shape — size pulses slightly
            hs = int((CELL // 2) * (0.9 + 0.1 * pulse))
            pts = [(cx, cy - hs), (cx + hs, cy), (cx, cy + hs), (cx - hs, cy)]
            pygame.draw.polygon(self.screen, col, pts)
            if fresh:
                pygame.draw.polygon(self.screen, TEXT_C, pts, 1)

        # Shelters — small house/triangle icon
        shelter_cols = [AGENT_COLS[i % len(AGENT_COLS)] for i in range(10)]
        for (sx, sy), owner_id in sandbox.shelters.items():
            px = sx * CELL + CELL // 2
            py = sy * CELL + CELL // 2
            sc = shelter_cols[owner_id % len(shelter_cols)]
            dim_c = tuple(max(0, c - 60) for c in sc)
            # Triangle roof
            hs = CELL // 2 - 1
            roof = [(px, py - hs), (px + hs, py + 1), (px - hs, py + 1)]
            pygame.draw.polygon(self.screen, dim_c, roof)
            # Base
            pygame.draw.rect(self.screen, dim_c,
                             (px - hs + 1, py + 1, hs * 2 - 2, hs - 1))
            pygame.draw.polygon(self.screen, sc, roof, 1)

        # Agent glows
        for a in agents:
            if not a.alive:
                continue
            gx, gy = a.body.position.grid_pos()
            cx = gx * CELL + CELL // 2
            cy = gy * CELL + CELL // 2
            glow = self._agent_glows[a.agent_id % len(self._agent_glows)]
            self._glow_surf.blit(glow, (cx - glow.get_width() // 2, cy - glow.get_height() // 2))

        # Predators — red X marks
        for pred in sandbox.predators.predators:
            gx, gy = pred.position.grid_pos()
            cx = gx * CELL + CELL // 2
            cy = gy * CELL + CELL // 2
            pred_r = CELL // 2 + 1
            pred_col = (255, 40, 40) if not pred.scared_ticks else (255, 150, 50)
            pygame.draw.line(self.screen, pred_col,
                             (cx - pred_r, cy - pred_r), (cx + pred_r, cy + pred_r), 2)
            pygame.draw.line(self.screen, pred_col,
                             (cx + pred_r, cy - pred_r), (cx - pred_r, cy + pred_r), 2)
            # Detection range ring (faint)
            det_r = int(pred.detection_range * CELL)
            det_surf = pygame.Surface((det_r * 2, det_r * 2), pygame.SRCALPHA)
            pygame.draw.circle(det_surf, (255, 40, 40, 15), (det_r, det_r), det_r)
            self.screen.blit(det_surf, (cx - det_r, cy - det_r))

        self.screen.blit(self._glow_surf, (0, 0))

        # Draw agents on top
        for a in agents:
            gx, gy = a.body.position.grid_pos()
            cx = gx * CELL + CELL // 2
            cy = gy * CELL + CELL // 2
            col = AGENT_COLS[a.agent_id % len(AGENT_COLS)]

            if not a.alive:
                sz = CELL // 2
                pygame.draw.line(self.screen, DEAD_C, (cx - sz, cy - sz), (cx + sz, cy + sz), 2)
                pygame.draw.line(self.screen, DEAD_C, (cx + sz, cy - sz), (cx - sz, cy + sz), 2)
                continue

            r = CELL // 2 + 2

            # Energy arc
            e_frac = a.body.energy / a.config.agent.max_energy
            arc_col = BAR_FG_E if e_frac > 0.3 else (255, 80, 60)
            end_angle = e_frac * 2 * math.pi
            if end_angle > 0.1:
                arc_rect = pygame.Rect(cx - r - 2, cy - r - 2, (r + 2) * 2, (r + 2) * 2)
                pygame.draw.arc(self.screen, arc_col, arc_rect, math.pi / 2, math.pi / 2 + end_angle, 2)

            # Goal ring
            goal_col = GOAL_COLS[a.goal_system.active_goal % len(GOAL_COLS)]
            pygame.draw.circle(self.screen, goal_col, (cx, cy), r + 1, 1)

            # Body
            pygame.draw.circle(self.screen, col, (cx, cy), r)
            hi = _lerp_color(col, (255, 255, 255), 0.3)
            pygame.draw.circle(self.screen, hi, (cx - 1, cy - 1), max(1, r // 2))

            # Focused ring
            if a.agent_id == self._focused:
                pygame.draw.circle(self.screen, TEXT_C, (cx, cy), r + 4, 1)

            # Direction arrow
            vx, vy = a.body.velocity.x, a.body.velocity.y
            spd = math.hypot(vx, vy)
            if spd > 0.1:
                dx, dy = vx / spd, vy / spd
                tip_x = cx + int(dx * (CELL + 4))
                tip_y = cy + int(dy * (CELL + 4))
                pygame.draw.line(self.screen, col, (cx, cy), (tip_x, tip_y), 2)
                px, py_a = -dy * 3, dx * 3
                pygame.draw.polygon(
                    self.screen,
                    col,
                    [
                        (tip_x, tip_y),
                        (int(tip_x - dx * 4 + px), int(tip_y - dy * 4 + py_a)),
                        (int(tip_x - dx * 4 - px), int(tip_y - dy * 4 - py_a)),
                    ],
                )

            # Agent ID label
            lbl = self.font_tiny.render(str(a.agent_id), True, (0, 0, 0))
            self.screen.blit(lbl, (cx - lbl.get_width() // 2, cy - lbl.get_height() // 2))

            # ── Status indicators above agent ──
            tag_y = cy - r - 16

            # Consciousness phase dot (colour encodes composite score)
            cs = a.phi_calculator.phi_history[-1] if a.phi_calculator.phi_history else 0.0
            # blend red→yellow→green→cyan as composite rises
            if cs < 0.3:
                phase_col = _lerp_color((180, 60, 60), (230, 200, 50), _clamp(cs / 0.3))
            elif cs < 0.6:
                phase_col = _lerp_color((230, 200, 50), (50, 210, 110), _clamp((cs - 0.3) / 0.3))
            else:
                phase_col = _lerp_color((50, 210, 110), (60, 200, 255), _clamp((cs - 0.6) / 0.4))
            pygame.draw.circle(self.screen, phase_col, (cx, tag_y + 4), 3)

            # Ego badge (small star if ego has emerged)
            if a.self_model.has_ego:
                sx, sy_s = cx + 6, tag_y + 4
                star_r = 3
                star_pts = []
                for si in range(10):
                    sa = math.pi / 2 + si * math.pi / 5
                    sr = star_r if si % 2 == 0 else star_r * 0.45
                    star_pts.append((sx + int(math.cos(sa) * sr),
                                     sy_s - int(math.sin(sa) * sr)))
                pygame.draw.polygon(self.screen, BAR_PHI, star_pts)

            # Dominant emotion colour indicator
            emo_idx_map = {"fear": 0, "curiosity": 1, "contentment": 2,
                           "frustration": 3, "loneliness": 4, "surprise": 5}
            dom = a.emotions.get_dominant()
            ei = emo_idx_map.get(dom, 1)
            pygame.draw.rect(self.screen, EMOTION_COLS[ei],
                             (cx - 5, tag_y - 3, 4, 4), border_radius=1)

            # Dreaming indicator
            if a.dream_engine.stats.is_dreaming:
                zz = self.font_tiny.render("Zz", True, (180, 140, 255))
                self.screen.blit(zz, (cx + 9, tag_y - 6))

    # ─── atmosphere ───────────────────────────────────────────────

    def _draw_atmosphere(self, sandbox: "Sandbox") -> None:
        light = sandbox.day_cycle.light_level
        weather = sandbox.weather
        self._night_surf.fill((0, 0, 0, 0))

        # Weather-specific tints
        w = weather.current_weather
        if w == 2:  # STORM
            tint = (30, 20, 50)
            darkness = max(60, int((1.0 - light) * 100))
        elif w == 4:  # FOG
            tint = (40, 40, 50)
            darkness = 50
        elif w == 3:  # DROUGHT
            tint = (40, 30, 10)
            darkness = 20
        elif sandbox.day_cycle.is_raining or w == 1:
            tint = RAIN_TINT
            darkness = int((1.0 - light) * 80)
        else:
            tint = NIGHT_TINT
            darkness = int((1.0 - light) * 80)

        if light >= 0.95 and w == 0:
            return
        self._night_surf.fill((*tint, max(0, darkness)))
        self.screen.blit(self._night_surf, (0, 0))

        # ── Weather particles ──
        if w == 1 or w == 2:  # RAIN or STORM
            rain_surf = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)
            count = 90 if w == 1 else 200
            length = 6 if w == 1 else 10
            alpha = 55 if w == 1 else 80
            for i in range(count):
                rx = int((math.sin(i * 7.13 + self._frame * 0.023) * 0.5 + 0.5) * self.map_w) % self.map_w
                ry = int((i * 17.7 + self._frame * 3.5) % (self.map_h + length)) - length
                wind = 2 if w == 2 else 0
                pygame.draw.line(rain_surf, (140, 160, 220, alpha),
                                 (rx, ry), (rx + wind, ry + length), 1)
            self.screen.blit(rain_surf, (0, 0))

            # Storm: occasional lightning flash
            if w == 2 and (self._frame % 180) < 2:
                flash = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)
                flash.fill((200, 210, 255, 35))
                self.screen.blit(flash, (0, 0))

        elif w == 4:  # FOG wisps
            fog_surf = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)
            for i in range(12):
                fx = int((math.sin(i * 4.7 + self._frame * 0.004) * 0.5 + 0.5) * self.map_w)
                fy = int((math.cos(i * 6.3 + self._frame * 0.003) * 0.5 + 0.5) * self.map_h)
                fr = 30 + int(math.sin(i * 0.7) * 15)
                pygame.draw.circle(fog_surf, (180, 180, 195, 18), (fx, fy), fr)
            self.screen.blit(fog_surf, (0, 0))

        elif w == 3:  # DROUGHT dust
            dust_surf = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)
            for i in range(25):
                dx = int((math.sin(i * 5.7 + self._frame * 0.008) * 0.5 + 0.5) * self.map_w)
                dy = int((i * 47.3 + self._frame * 0.7) % self.map_h)
                pygame.draw.circle(dust_surf, (200, 170, 100, 30), (dx, dy), 2)
            self.screen.blit(dust_surf, (0, 0))

    # ─── bottom status bar ────────────────────────────────────────

    def _draw_bottom_status_bar(self, sandbox: "Sandbox",
                                agents: list["ConsciousAgent"],
                                tick: int) -> None:
        """Semi-transparent info bar overlaid at the bottom of the map."""
        from genesis.environment.weather import WEATHER_NAMES, SEASON_NAMES

        bar_h = 22
        bar_y = self.map_h - bar_h
        bar_surf = pygame.Surface((self.map_w, bar_h), pygame.SRCALPHA)
        bar_surf.fill((8, 10, 20, 170))
        pygame.draw.line(bar_surf, (50, 55, 75, 200), (0, 0), (self.map_w, 0), 1)

        x = 8
        fps_val = self.clock.get_fps()
        fps_col = (100, 210, 130) if fps_val >= 25 else (
            (230, 190, 60) if fps_val >= 15 else (230, 80, 70))
        fps_lbl = self.font_tiny.render(f"FPS {fps_val:.0f}", True, fps_col)
        bar_surf.blit(fps_lbl, (x, 5))
        x += fps_lbl.get_width() + 14

        tick_lbl = self.font_tiny.render(f"Tick {tick:,}", True, MUTED_C)
        bar_surf.blit(tick_lbl, (x, 5))
        x += tick_lbl.get_width() + 14

        # Day / night
        dn = "NIGHT" if sandbox.day_cycle.is_night else "DAY"
        light = sandbox.day_cycle.light_level
        dn_col = (80, 100, 180) if sandbox.day_cycle.is_night else (240, 210, 100)
        sun_moon = self.font_tiny.render(f"{dn} {light:.0%}", True, dn_col)
        bar_surf.blit(sun_moon, (x, 5))
        x += sun_moon.get_width() + 14

        # Season + weather
        ws = sandbox.weather
        season = SEASON_NAMES[ws.season].upper()
        weather = WEATHER_NAMES[ws.current_weather].upper()
        sw_lbl = self.font_tiny.render(f"{season} / {weather}", True, ACCENT_C)
        bar_surf.blit(sw_lbl, (x, 5))
        x += sw_lbl.get_width() + 14

        # Counts
        alive = sum(1 for a in agents if a.alive)
        n_crystals = sum(1 for c in sandbox.crystals if not c.consumed and not c.is_expired)
        n_preds = len(sandbox.predators.predators)
        n_shelters = len(sandbox.shelters)
        counts = f"Alive {alive}  Crystals {n_crystals}  Pred {n_preds}  Shelters {n_shelters}"
        ct_lbl = self.font_tiny.render(counts, True, MUTED_C)
        bar_surf.blit(ct_lbl, (x, 5))

        self.screen.blit(bar_surf, (0, bar_y))

    # ─── vision circle ────────────────────────────────────────────

    def _draw_vision_circle(self, agent: "ConsciousAgent") -> None:
        gx, gy = agent.body.position.grid_pos()
        cx = gx * CELL + CELL // 2
        cy = gy * CELL + CELL // 2
        vr = agent.config.agent.vision_range * CELL
        vis_surf = pygame.Surface((vr * 2, vr * 2), pygame.SRCALPHA)
        pygame.draw.circle(vis_surf, VISION_COL, (vr, vr), vr)
        self.screen.blit(vis_surf, (cx - vr, cy - vr))
        col = AGENT_COLS[agent.agent_id % len(AGENT_COLS)]
        dim_col = (*col[:3], 30)
        vis_ring = pygame.Surface((vr * 2 + 4, vr * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(vis_ring, dim_col, (vr + 2, vr + 2), vr, 1)
        self.screen.blit(vis_ring, (cx - vr - 2, cy - vr - 2))

    # ─── cognitive map overlay ────────────────────────────────────

    def _draw_cogmap_overlay(self, agent: "ConsciousAgent") -> None:
        self._overlay_surf.fill((0, 0, 0, 0))
        cmap = agent.cognitive_map
        cs = cmap.cell_size
        for (kx, ky), cell in cmap.cells.items():
            score = cell.crystal_score - cell.danger_score
            if abs(score) < 0.05:
                continue
            alpha = int(min(100, abs(score) * 50))
            colour = (0, 220, 130, alpha) if score > 0 else (255, 50, 50, alpha)
            pygame.draw.rect(
                self._overlay_surf, colour, (kx * cs * CELL, ky * cs * CELL, cs * CELL, cs * CELL)
            )
        self.screen.blit(self._overlay_surf, (0, 0))
        lbl = self.font_tiny.render(f" [M] Cognitive Map: Agent {self._focused} ", True, TEXT_C)
        r = lbl.get_rect(topleft=(4, 4))
        r.inflate_ip(6, 2)
        s = pygame.Surface(r.size, pygame.SRCALPHA)
        s.fill((*BG, 190))
        self.screen.blit(s, r.topleft)
        self.screen.blit(lbl, (4, 4))

    # ─── cultural transmission lines ──────────────────────────────

    def _draw_cultural_lines(self, agents: list["ConsciousAgent"]) -> None:
        for a in agents:
            if not a.alive:
                continue
            if not a.culture.can_teach(a.body.ticks_alive, a.self_model.model_accuracy):
                continue
            for b in agents:
                if b.agent_id == a.agent_id or not b.alive:
                    continue
                dist = a.body.position.distance_to(b.body.position)
                if dist < a.config.agent.vision_range:
                    ax, ay = a.body.position.grid_pos()
                    bx, by = b.body.position.grid_pos()
                    p1 = (ax * CELL + CELL // 2, ay * CELL + CELL // 2)
                    p2 = (bx * CELL + CELL // 2, by * CELL + CELL // 2)
                    self._dashed_line(p1, p2, (255, 220, 60), 1, 6)

    def _dashed_line(self, p1, p2, color, width, dash):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = max(1, math.hypot(dx, dy))
        steps = int(dist / dash)
        for i in range(0, steps, 2):
            t1, t2 = i / steps, min(1.0, (i + 1) / steps)
            s = (int(p1[0] + dx * t1), int(p1[1] + dy * t1))
            e = (int(p1[0] + dx * t2), int(p1[1] + dy * t2))
            pygame.draw.line(self.screen, color, s, e, width)

    # ─── spike raster (replaces old heatmap) ──────────────────────

    def _draw_spike_raster(self, agent: "ConsciousAgent", x_off: int, y_off: int) -> None:
        """Scrolling spike raster — each column is one timestep, each row a neuron.
        Gives a beautiful real-time view of neural activity patterns."""
        bg_rect = pygame.Rect(x_off, y_off, self.map_w, RASTER_H)
        pygame.draw.rect(self.screen, (12, 12, 22), bg_rect)

        history = agent.brain.spike_history  # list of np.ndarray [num_neurons]
        if not history:
            lbl = self.font_tiny.render("Waiting for neural data...", True, DIM_C)
            self.screen.blit(lbl, (x_off + 6, y_off + RASTER_H // 2 - 5))
            pygame.draw.rect(self.screen, PANEL_EDGE, bg_rect, 1)
            return

        n_neurons = len(history[0])
        n_steps = len(history)
        max_cols = min(n_steps, self.map_w)

        # Row height — fit all neurons in raster height minus margins
        row_h_f = (RASTER_H - 18) / n_neurons
        col_w = max(1, self.map_w // max(1, max_cols))

        # Draw spikes — iterate recent history (right = now)
        start = max(0, n_steps - max_cols)
        ns = agent.brain.config.sensory_neurons
        nm = agent.brain.config.motor_neurons
        n_inter_start = ns
        n_motor_start = n_neurons - nm

        for t_idx in range(start, n_steps):
            col_idx = t_idx - start
            px = x_off + col_idx * col_w
            spikes = history[t_idx]
            for ni in range(n_neurons):
                if spikes[ni] > 0.5:
                    py = y_off + 14 + int(ni * row_h_f)
                    # Colour by neuron type
                    if ni < ns:
                        c = (60, 150, 255)
                    elif ni >= n_motor_start:
                        c = (255, 120, 70)
                    else:
                        c = (100, 220, 160)
                    if col_w <= 2:
                        self.screen.set_at((min(px, self.map_w - 1), min(py, y_off + RASTER_H - 4)), c)
                    else:
                        pygame.draw.rect(self.screen, c, (px, py, max(1, col_w - 1), max(1, int(row_h_f))))

        # Section labels
        labels = [
            (y_off + 14, "SENS", (60, 150, 255)),
            (y_off + 14 + int(n_inter_start * row_h_f), "INTER", (100, 220, 160)),
            (y_off + 14 + int(n_motor_start * row_h_f), "MOTOR", (255, 120, 70)),
        ]
        for ly, text, lc in labels:
            lbl = self.font_tiny.render(text, True, lc)
            # Draw on semi-transparent background so it's readable
            bg = pygame.Surface((lbl.get_width() + 4, lbl.get_height()), pygame.SRCALPHA)
            bg.fill((12, 12, 22, 200))
            self.screen.blit(bg, (x_off + 2, ly))
            self.screen.blit(lbl, (x_off + 4, ly))

        # Title
        title_str = f"Spike Raster — Agent {agent.agent_id}"
        title_lbl = self.font_tiny.render(title_str, True, MUTED_C)
        self.screen.blit(title_lbl, (x_off + self.map_w - title_lbl.get_width() - 6, y_off + 2))

        # Time arrow
        arrow_y = y_off + RASTER_H - 4
        pygame.draw.line(self.screen, DIM_C, (x_off + 4, arrow_y), (x_off + self.map_w - 4, arrow_y), 1)
        # arrowhead
        aw = x_off + self.map_w - 4
        pygame.draw.polygon(self.screen, DIM_C, [(aw, arrow_y), (aw - 5, arrow_y - 3), (aw - 5, arrow_y + 3)])
        t_lbl = self.font_tiny.render("time →", True, DIM_C)
        self.screen.blit(t_lbl, (x_off + self.map_w - 48, y_off + RASTER_H - 15))

        pygame.draw.rect(self.screen, PANEL_EDGE, bg_rect, 1)

    # ─── sidebar ──────────────────────────────────────────────────

    def _draw_sidebar(self, agents: list["ConsciousAgent"], tick: int, sandbox: "Sandbox") -> None:
        sidebar_x = self.map_w
        x0 = sidebar_x + SIDE_PAD
        pw = SIDE_W - SIDE_PAD * 2

        # ── Fixed title bar ──
        pygame.draw.rect(self.screen, PANEL_HEAD, (sidebar_x, 0, SIDE_W, 52))
        pygame.draw.line(self.screen, ACCENT_C, (sidebar_x, 52), (sidebar_x + SIDE_W, 52), 2)

        title = self.font_title.render("PROJECT GENESIS", True, TEXT_BRIGHT)
        self.screen.blit(title, (x0, 6))

        dn = "NIGHT" if sandbox.day_cycle.is_night else "DAY"
        rain = "  RAIN" if sandbox.day_cycle.is_raining else ""
        ws = sandbox.weather.get_summary()
        season = ws["season"].upper()[:3]
        weather_str = ws["weather"].upper()[:5]
        temp = ws["temperature"]
        preds = len(sandbox.predators.predators)
        dn_col = (100, 130, 220) if sandbox.day_cycle.is_night else (230, 210, 80)
        info = self.font_sm.render(
            f"Tick {tick:,}  |  {dn}{rain}  |  {season} {weather_str}  |  "
            f"{temp:.0%}°  |  ☠{preds}  |  ♦{len(sandbox.crystals)}",
            True, MUTED_C
        )
        self.screen.blit(info, (x0, 28))

        # Keybinds
        keys = self.font_tiny.render(f"[TAB] focus  [M] map  Agent {self._focused}", True, DIM_C)
        kr = keys.get_rect(right=sidebar_x + SIDE_W - SIDE_PAD, centery=40)
        self.screen.blit(keys, kr)

        # ── Scrollable content area ──
        content_top = 58
        clip_h = self.win_h - content_top
        clip_rect = pygame.Rect(sidebar_x, content_top, SIDE_W, clip_h)

        # Render to offscreen surface so we can scroll
        # First pass: measure height
        content_h = self._measure_sidebar_content(agents, pw)
        self._sidebar_content_h = content_h

        # Clamp scroll
        max_scroll = 0
        min_scroll = min(0, clip_h - content_h - 10)
        self._scroll_y = max(min_scroll, min(max_scroll, self._scroll_y))

        # Create content surface
        surf = pygame.Surface((SIDE_W, max(content_h + 10, clip_h)), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))

        y = 0
        for a in agents:
            y = self._draw_agent_panel(a, SIDE_PAD, y, pw, surf)
            y += 10

        # Blit scrolled content
        self.screen.set_clip(clip_rect)
        self.screen.blit(surf, (sidebar_x, content_top + self._scroll_y))
        self.screen.set_clip(None)

        # Scroll indicator
        if content_h > clip_h:
            scroll_track_h = clip_h - 8
            thumb_h = max(20, int(scroll_track_h * clip_h / content_h))
            scroll_frac = -self._scroll_y / max(1, content_h - clip_h) if content_h > clip_h else 0
            thumb_y = content_top + 4 + int(scroll_frac * (scroll_track_h - thumb_h))
            track_x = sidebar_x + SIDE_W - 5
            pygame.draw.rect(self.screen, (30, 32, 45), (track_x, content_top + 4, 3, scroll_track_h), border_radius=2)
            pygame.draw.rect(self.screen, (70, 72, 90), (track_x, thumb_y, 3, thumb_h), border_radius=2)

    def _measure_sidebar_content(self, agents: list, pw: int) -> int:
        """Estimate total sidebar content height."""
        h = 0
        for a in agents:
            h += 540 if a.alive else 45
            h += 10
        return h

    # ─── agent panel ──────────────────────────────────────────────

    def _draw_agent_panel(self, agent: "ConsciousAgent", x0: int, y: int, pw: int, surf: pygame.Surface) -> int:
        col = AGENT_COLS[agent.agent_id % len(AGENT_COLS)]
        ix = x0 + 10
        cw = pw - 20
        bw = cw - 100

        # Dead agent — compact
        if not agent.alive:
            card_h = 35
            card = pygame.Rect(x0, y, pw, card_h)
            pygame.draw.rect(surf, PANEL_BG, card, border_radius=6)
            pygame.draw.rect(surf, PANEL_EDGE, card, 1, border_radius=6)
            pygame.draw.rect(surf, DEAD_C, (x0, y + 4, 3, card_h - 8), border_radius=2)
            hdr = self.font_bold.render(f"AGENT {agent.agent_id}  —  DEAD", True, DEAD_C)
            surf.blit(hdr, (ix, y + 10))
            return y + card_h

        # ── Card background ──
        panel_h = 540
        card = pygame.Rect(x0, y, pw, panel_h)
        pygame.draw.rect(surf, PANEL_BG, card, border_radius=6)
        pygame.draw.rect(surf, PANEL_EDGE, card, 1, border_radius=6)
        # Focused highlight
        if agent.agent_id == self._focused:
            pygame.draw.rect(surf, (*col, 25), card, border_radius=6)
            pygame.draw.rect(surf, col, card, 1, border_radius=6)
        # Left accent stripe
        pygame.draw.rect(surf, col, (x0, y + 4, 3, panel_h - 8), border_radius=2)

        y += 8

        # ── Header ──
        hdr = self.font_bold.render(f"AGENT {agent.agent_id}", True, col)
        surf.blit(hdr, (ix, y))

        assess = agent.phi_calculator.get_consciousness_assessment(
            self_model_accuracy=agent.self_model.model_accuracy,
            attention_accuracy=agent.attention_schema.schema_accuracy,
            metacognitive_confidence=agent.inner_speech.confidence,
            binding_coherence=agent.binding.coherence,
            empowerment=agent.empowerment.empowerment,
            narrative_identity=agent.narrative.identity_strength,
            curiosity_level=agent.curiosity.curiosity_level,
        )
        phase_short = assess["phase"].split("\u2014")[0].split("—")[0].strip()
        badge = self.font_tiny.render(phase_short, True, BAR_PHI)
        br = badge.get_rect(right=x0 + pw - 10, centery=y + 7)
        bg_r = br.inflate(8, 4)
        pygame.draw.rect(surf, (50, 45, 25), bg_r, border_radius=3)
        pygame.draw.rect(surf, BAR_PHI, bg_r, 1, border_radius=3)
        surf.blit(badge, br)

        # Dream indicator
        if agent.dream_engine.stats.is_dreaming:
            dream_badge = self.font_tiny.render("DREAMING", True, (180, 140, 255))
            db_r = dream_badge.get_rect(right=bg_r.left - 6, centery=y + 7)
            db_bg = db_r.inflate(6, 4)
            pygame.draw.rect(surf, (35, 28, 55), db_bg, border_radius=3)
            pygame.draw.rect(surf, (140, 100, 220), db_bg, 1, border_radius=3)
            surf.blit(dream_badge, db_r)

        y += 20

        # ── Vitals ──
        e_frac = agent.body.energy / agent.config.agent.max_energy
        i_frac = agent.body.integrity / agent.config.agent.max_integrity
        y = self._fancy_bar(ix, y, "Energy", e_frac, BAR_FG_E, bw, surf)
        y = self._fancy_bar(ix, y, "Integrity", i_frac, BAR_FG_I, bw, surf)

        # Pain/pleasure micro-indicators
        pain = agent.body.pain_signal
        pleasure = agent.body.pleasure_signal
        if pain > 0.01 or pleasure > 0.01:
            pp_parts = []
            if pain > 0.01:
                pp_parts.append(f"pain:{pain:.2f}")
            if pleasure > 0.01:
                pp_parts.append(f"pleasure:{pleasure:.2f}")
            pp_col = (255, 90, 90) if pain > pleasure else (90, 255, 130)
            pp_lbl = self.font_tiny.render("  ".join(pp_parts), True, pp_col)
            surf.blit(pp_lbl, (ix, y))
            y += 11

        # ── Consciousness ──
        y = self._section_hdr(ix, y, "CONSCIOUSNESS", ACCENT_C, cw, surf)
        y = self._fancy_bar(ix, y, "Phi", min(1.0, assess["phi"] * 5), BAR_PHI, bw, surf)

        # Phi sparkline
        if agent.agent_id in self._spark_phi and len(self._spark_phi[agent.agent_id]) > 2:
            self._draw_sparkline(surf, ix + 66, y - 2, bw, 14, self._spark_phi[agent.agent_id], BAR_PHI, scale=0.2)
            y += 16
        
        cs_val = assess["composite_score"]
        self._kv(ix, y, "Composite", f"{cs_val:.3f}", surf=surf)
        y += 13

        ego = "EGO" if agent.self_model.has_ego else "no ego"
        ego_col = BAR_PHI if agent.self_model.has_ego else DIM_C
        self._kv(ix, y, "Self", f"{agent.self_model.model_accuracy:.0%}", surf=surf)
        el = self.font_tiny.render(ego, True, ego_col)
        surf.blit(el, (ix + 130, y + 1))
        y += 13

        ws = agent.workspace.get_broadcast_summary()
        self._kv(ix, y, "WS", ws["current_source"], val_col=ACCENT_C, surf=surf)
        y += 13

        # ── Broadcast distribution mini-chart ──
        y = self._draw_broadcast_dist(agent, ix, y, cw, surf)

        # ── Emotions ──
        y = self._draw_emotions(agent, ix, y, cw, surf)

        # ── Goals ──
        y = self._section_hdr(ix, y, "GOALS", GOAL_COLS[0], cw, surf)
        y = self._draw_goals(agent, ix, y, cw, surf)

        # ── Cognition ──
        y = self._section_hdr(ix, y, "COGNITION", BAR_CURIO, cw, surf)
        y = self._fancy_bar(ix, y, "Curiosity", agent.curiosity.curiosity_level, BAR_CURIO, bw, surf)
        y = self._fancy_bar(ix, y, "Confid.", agent.inner_speech.confidence, BAR_CONF, bw, surf)
        y = self._fancy_bar(ix, y, "Binding", agent.binding.binding_strength, BAR_BIND, bw, surf)

        # Prediction error with sparkline
        y = self._fancy_bar(ix, y, "Pred.Err", _clamp(agent.prediction_engine.average_error), BAR_PRED, bw, surf)
        if agent.agent_id in self._spark_pred and len(self._spark_pred[agent.agent_id]) > 2:
            self._draw_sparkline(surf, ix + 66, y - 2, bw, 14, self._spark_pred[agent.agent_id], BAR_PRED)
            y += 16

        cmap_s = agent.cognitive_map.get_summary()
        self._kv(ix, y, "Map", f"{cmap_s['coverage']:.0%} ({cmap_s['explored_cells']}/{cmap_s['total_cells']})", surf=surf)
        y += 13

        # Attention
        attn = agent.attention_schema
        self._kv(ix, y, "Attn", f"{attn.current_focus} ({attn.focus_duration}t)", val_col=ACCENT_C, surf=surf)
        acc_lbl = self.font_tiny.render(f"acc:{attn.schema_accuracy:.0%}", True, DIM_C)
        surf.blit(acc_lbl, (ix + cw - acc_lbl.get_width(), y + 1))
        y += 13

        # Critical periods
        y = self._draw_crit_periods(agent, ix, y, cw, surf)

        cf = agent.counterfactual.get_summary()
        cult = agent.culture.get_summary()
        self._text(f"CF {cf['total_regrets']} regrets  Cult {cult['teachings_received']} recv", ix, y, DIM_C, surf)
        y += 13

        # ── Higher Cognition ──
        y = self._section_hdr(ix, y, "HIGHER COGNITION", BAR_EMP, cw, surf)
        y = self._fancy_bar(ix, y, "Empower.", agent.empowerment.empowerment, BAR_EMP, bw, surf)
        y = self._fancy_bar(ix, y, "Identity", agent.narrative.identity_strength, BAR_IDENT, bw, surf)

        n_concepts = len(agent.abstraction.active_concepts)
        stage = agent.critical_periods.get_developmental_stage(agent.tick_count)
        self._kv(ix, y, "Concepts", f"{n_concepts} active", surf=surf)
        sl = self.font_tiny.render(stage, True, DIM_C)
        sr = sl.get_rect(right=ix + cw, centery=y + 6)
        surf.blit(sl, sr)
        y += 13

        # Memory
        mem_lt = agent.episodic_memory.long_term_count
        dreams = agent.dream_engine.stats.total_dream_cycles
        self._kv(ix, y, "Memory", f"{mem_lt} episodes  {dreams} dreams", val_col=DIM_C, surf=surf)
        y += 13

        syn = agent.brain.get_active_connections()
        lr = agent.critical_periods.modulate_snn_learning_rate(1.0, agent.tick_count)
        self._kv(ix, y, "Syn", f"{syn:,}  LR {lr:.1f}x", val_col=DIM_C, surf=surf)
        y += 14

        # ── Mood & Personality ──
        y = self._section_hdr(ix, y, "MOOD", (180, 140, 255), cw, surf)
        emo_s = agent.emotions.get_summary()
        mood_v = emo_s.get("mood_valence", 0.0)
        mood_a = emo_s.get("mood_arousal", 0.0)
        mood_col = (90, 255, 130) if mood_v > 0.05 else (255, 90, 90) if mood_v < -0.05 else DIM_C
        self._kv(ix, y, "Mood", f"V:{mood_v:+.2f}  A:{mood_a:.2f}", val_col=mood_col, surf=surf)
        y += 13
        pers = emo_s.get("personality", {})
        if pers:
            traits = f"B:{pers.get('boldness',0):.1f} C:{pers.get('curiosity_trait',0):.1f} S:{pers.get('sociability',0):.1f}"
            self._text(traits, ix, y, DIM_C, surf)
            y += 12
        bonds = emo_s.get("bonds", 0)
        strongest = emo_s.get("strongest_bond", 0.0) or 0.0
        if bonds > 0:
            self._kv(ix, y, "Bonds", f"{bonds} (str:{strongest:.2f})", val_col=DIM_C, surf=surf)
            y += 13

        # ── Tools & Cooperation ──
        tool_s = agent.body.tools.get_summary()
        coop_s = agent.cooperation.get_summary()
        if tool_s["tool_count"] > 0:
            tool_names = ", ".join(tool_s["tools"])
            self._kv(ix, y, "Tools", f"{tool_s['tool_count']}: {tool_names}", val_col=ACCENT_C, surf=surf)
            y += 13
        else:
            self._kv(ix, y, "Tools", "none", val_col=DIM_C, surf=surf)
            y += 13
        if coop_s["partners"] > 0:
            self._kv(ix, y, "Coop", f"{coop_s['partners']} partners  trust:{coop_s.get('avg_trust',0):.2f}", val_col=DIM_C, surf=surf)
            y += 13

        # Language
        lang_s = agent.communication.get_language_summary()
        vocab = lang_s.get("vocabulary_size", 0)
        named = lang_s.get("named_entities", 0)
        if vocab > 0 or named > 0:
            self._kv(ix, y, "Lang", f"vocab:{vocab}  named:{named}", val_col=DIM_C, surf=surf)
            y += 13

        return y

    # ─── broadcast distribution ───────────────────────────────────

    def _draw_broadcast_dist(self, agent: "ConsciousAgent", x0: int, y: int, w: int, surf: pygame.Surface) -> int:
        bc = agent.workspace.broadcast_counts
        if not bc:
            return y
        total = sum(bc.values())
        if total == 0:
            return y

        # Top 5 modules by broadcast count
        sorted_bc = sorted(bc.items(), key=lambda kv: kv[1], reverse=True)[:5]
        bar_h = 5
        for name, count in sorted_bc:
            frac = count / total
            name_short = name[:8]
            nl = self.font_tiny.render(name_short, True, DIM_C)
            surf.blit(nl, (x0, y))
            bx = x0 + 60
            bw = w - 60
            pygame.draw.rect(surf, BAR_BG, (bx, y + 2, bw, bar_h), border_radius=2)
            fill = max(0, int(frac * bw))
            if fill > 1:
                pygame.draw.rect(surf, ACCENT_C, (bx, y + 2, fill, bar_h), border_radius=2)
            pct = self.font_tiny.render(f"{frac:.0%}", True, DIM_C)
            surf.blit(pct, (bx + bw + 4, y))
            y += bar_h + 6
        return y + 2

    # ─── emotion display ──────────────────────────────────────────

    def _draw_emotions(self, agent: "ConsciousAgent", x0: int, y: int, w: int, surf: pygame.Surface) -> int:
        emo = agent.emotions.get_summary()
        cx, cy = x0 + 38, y + 30
        r_max = 26

        # Background circle
        pygame.draw.circle(surf, (28, 30, 42), (cx, cy), r_max + 3)
        pygame.draw.circle(surf, (42, 44, 58), (cx, cy), r_max + 3, 1)

        emo_names = ["fear", "curiosity", "contentment", "frustration", "loneliness", "surprise"]

        # Filled polygon
        points = []
        for i, name in enumerate(emo_names):
            angle = math.pi / 2 - i * (2 * math.pi / 6)
            val = emo.get(name, 0.0)
            r = max(2, int(val * r_max))
            points.append((cx + int(math.cos(angle) * r), cy - int(math.sin(angle) * r)))

        if len(points) >= 3:
            poly = pygame.Surface((r_max * 2 + 10, r_max * 2 + 10), pygame.SRCALPHA)
            shifted = [(p[0] - cx + r_max + 5, p[1] - cy + r_max + 5) for p in points]
            pygame.draw.polygon(poly, (140, 120, 200, 45), shifted)
            pygame.draw.polygon(poly, (180, 160, 255, 100), shifted, 1)
            surf.blit(poly, (cx - r_max - 5, cy - r_max - 5))

        # Spokes + dots + labels
        for i, name in enumerate(emo_names):
            angle = math.pi / 2 - i * (2 * math.pi / 6)
            val = emo.get(name, 0.0)
            c = EMOTION_COLS[i]
            ex = cx + int(math.cos(angle) * r_max)
            ey = cy - int(math.sin(angle) * r_max)
            pygame.draw.line(surf, (38, 40, 52), (cx, cy), (ex, ey), 1)
            if val > 0.05:
                r = int(val * r_max)
                pygame.draw.circle(surf, c, (cx + int(math.cos(angle) * r), cy - int(math.sin(angle) * r)), 3)
            lx = cx + int(math.cos(angle) * (r_max + 15))
            ly = cy - int(math.sin(angle) * (r_max + 13))
            lbl = self.font_tiny.render(EMOTION_NAMES_SHORT[i], True, c)
            surf.blit(lbl, (lx - lbl.get_width() // 2, ly - lbl.get_height() // 2))

        # Dominant + valence
        dom = emo.get("dominant", "?")
        v = emo.get("valence", 0)
        v_col = BAR_FG_E if v > 0 else (255, 80, 80) if v < -0.1 else DIM_C
        dl = self.font_sm.render(dom, True, TEXT_C)
        surf.blit(dl, (x0 + 88, y + 12))
        vl = self.font_tiny.render(f"v={v:+.2f}  a={emo.get('arousal', 0):.2f}", True, v_col)
        surf.blit(vl, (x0 + 88, y + 26))

        return y + 62

    # ─── goal indicator ───────────────────────────────────────────

    def _draw_goals(self, agent: "ConsciousAgent", x0: int, y: int, w: int, surf: pygame.Surface) -> int:
        g = agent.goal_system
        active = g.active_goal
        gc = GOAL_COLS[active % len(GOAL_COLS)]

        lbl = self.font_sm.render(GOAL_NAMES_SHORT[active], True, gc)
        surf.blit(lbl, (x0, y))

        from genesis.cognition.goals import SUBGOAL_NAMES

        sn = SUBGOAL_NAMES[g.active_subgoal.subgoal_type]
        if sn != "none":
            sl = self.font_tiny.render(f"> {sn}", True, DIM_C)
            surf.blit(sl, (x0 + 62, y + 1))

        y += 15
        n_goals = len(g.goals)
        bw = (w - (n_goals - 1) * 3) // n_goals
        for i, goal in enumerate(g.goals):
            bx = x0 + i * (bw + 3)
            c = GOAL_COLS[i]
            frac = min(1.0, goal.current_priority / 3.0)
            pygame.draw.rect(surf, BAR_BG, (bx, y, bw, 7), border_radius=2)
            fill = max(0, int(frac * bw))
            if fill > 0:
                pygame.draw.rect(surf, c, (bx, y, fill, 7), border_radius=2)
            if i == active:
                pygame.draw.rect(surf, c, (bx - 1, y - 1, bw + 2, 9), 1, border_radius=2)
            t = self.font_tiny.render(GOAL_NAMES_SHORT[i][0], True, c if i == active else DIM_C)
            surf.blit(t, (bx + bw // 2 - 3, y + 8))
        return y + 20

    # ─── critical periods ─────────────────────────────────────────

    def _draw_crit_periods(self, agent: "ConsciousAgent", x0: int, y: int, w: int, surf: pygame.Surface) -> int:
        cp = agent.critical_periods
        tick = agent.tick_count
        sw = (w - 5 * 2) // 5
        for i, (domain, window) in enumerate(cp.windows.items()):
            x = x0 + i * (sw + 2)
            o = window.openness(tick)
            c = CP_COLS[i]
            pygame.draw.rect(surf, BAR_BG, (x, y, sw, 6), border_radius=2)
            if o > 0:
                pygame.draw.rect(surf, c, (x, y, max(1, int(o * sw)), 6), border_radius=2)
            elif tick > window.close_tick:
                pygame.draw.rect(surf, (38, 38, 48), (x, y, sw, 6), border_radius=2)
            t = self.font_tiny.render(CP_NAMES[i], True, c if o > 0 else DIM_C)
            surf.blit(t, (x, y + 7))
        return y + 19

    # ─── sparkline ────────────────────────────────────────────────

    def _draw_sparkline(self, surf: pygame.Surface, x: int, y: int, w: int, h: int, data: deque, color: tuple, scale: float = 1.0) -> None:
        """Draw a mini sparkline graph."""
        n = len(data)
        if n < 2:
            return
        max_val = max(max(data), scale, 0.001)
        points = []
        for i, v in enumerate(data):
            px = x + int(i * w / (n - 1))
            py = y + h - int((_clamp(v / max_val) * (h - 2)))
            points.append((px, py))
        if len(points) >= 2:
            # Filled area beneath
            fill_pts = list(points) + [(points[-1][0], y + h), (points[0][0], y + h)]
            fill_surf = pygame.Surface((w + 2, h + 2), pygame.SRCALPHA)
            shifted_fill = [(p[0] - x + 1, p[1] - y + 1) for p in fill_pts]
            pygame.draw.polygon(fill_surf, (*color, 20), shifted_fill)
            surf.blit(fill_surf, (x - 1, y - 1))
            # Line
            pygame.draw.lines(surf, (*color, 120), False, points, 1)
            # Current value dot
            pygame.draw.circle(surf, color, points[-1], 2)

    # ─── helpers ──────────────────────────────────────────────────

    def _text(self, msg: str, x: int, y: int, colour: tuple = MUTED_C, surf: pygame.Surface | None = None) -> None:
        target = surf if surf is not None else self.screen
        target.blit(self.font_sm.render(msg, True, colour), (x, y))

    def _kv(self, x: int, y: int, label: str, value: str, val_col: tuple = TEXT_C, surf: pygame.Surface | None = None) -> None:
        target = surf if surf is not None else self.screen
        l = self.font_sm.render(f"{label}: ", True, MUTED_C)
        v = self.font_sm.render(value, True, val_col)
        target.blit(l, (x, y))
        target.blit(v, (x + l.get_width(), y))

    def _section_hdr(self, x: int, y: int, title: str, color: tuple, w: int, surf: pygame.Surface | None = None) -> int:
        target = surf if surf is not None else self.screen
        y += 2
        pygame.draw.line(target, color, (x, y + 5), (x + 28, y + 5), 1)
        lbl = self.font_tiny.render(title, True, color)
        target.blit(lbl, (x + 32, y))
        lw = lbl.get_width()
        pygame.draw.line(target, (42, 44, 56), (x + 34 + lw + 4, y + 5), (x + w, y + 5), 1)
        return y + 14

    def _fancy_bar(self, x: int, y: int, label: str, frac: float, colour: tuple, width: int, surf: pygame.Surface | None = None) -> int:
        target = surf if surf is not None else self.screen
        frac = _clamp(frac)
        lw = 66
        bar_w = width

        target.blit(self.font_sm.render(label, True, MUTED_C), (x, y))

        bx = x + lw
        bh = 10
        pygame.draw.rect(target, BAR_BG, (bx, y + 1, bar_w, bh), border_radius=3)

        fill = max(0, int(frac * bar_w))
        if fill > 2:
            pygame.draw.rect(target, colour, (bx, y + 1, fill, bh), border_radius=3)
            hi = _lerp_color(colour, (255, 255, 255), 0.18)
            pygame.draw.rect(target, hi, (bx, y + 1, fill, bh // 2), border_radius=3)

        pygame.draw.rect(target, BAR_BORDER, (bx, y + 1, bar_w, bh), 1, border_radius=3)

        pct = self.font_val.render(f"{frac:.0%}", True, TEXT_C)
        target.blit(pct, (bx + bar_w + 4, y + 1))

        return y + bh + 4

    def _bar(self, x: int, y: int, label: str, frac: float, colour: tuple, width: int = 150) -> int:
        return self._fancy_bar(x, y, label, frac, colour, width)
