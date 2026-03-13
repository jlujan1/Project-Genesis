"""3D OpenGL renderer for Project Genesis.

Uses PyOpenGL + Pygame for a full 3D view of the simulation world:
- Height-mapped terrain mesh with biome-coloured vertices
- 3D agent capsules with direction arrows & glow
- Rotating crystal prisms, shelter models, predator shapes
- Dynamic day/night lighting & weather atmosphere
- Orbit camera with mouse drag & scroll zoom
- Translucent HUD overlay for stats
"""

from __future__ import annotations

import math
import ctypes
from typing import TYPE_CHECKING

import numpy as np
import pygame
from pygame.locals import (
    DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_ESCAPE, K_TAB, K_m,
    MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION, MOUSEWHEEL,
    K_SPACE, K_r, K_PLUS, K_MINUS, K_EQUALS, K_f, K_h,
)

from OpenGL.GL import *
from OpenGL.GLU import *

from genesis.environment.resources import (
    BIOME_OCEAN, BIOME_WETLANDS, BIOME_GRASSLANDS, BIOME_DESERT, BIOME_MOUNTAINS,
)

if TYPE_CHECKING:
    from genesis.agent.agent import ConsciousAgent
    from genesis.environment.sandbox import Sandbox

# ─── colour palettes ──────────────────────────────────────────────

BIOME_COLOURS = {
    BIOME_OCEAN:      (0.05, 0.14, 0.32),
    BIOME_WETLANDS:   (0.08, 0.28, 0.16),
    BIOME_GRASSLANDS: (0.16, 0.32, 0.12),
    BIOME_DESERT:     (0.52, 0.42, 0.24),
    BIOME_MOUNTAINS:  (0.38, 0.35, 0.42),
}

AGENT_COLOURS = [
    (1.0, 0.30, 0.30),   # red
    (0.30, 0.55, 1.0),   # blue
    (0.20, 0.90, 0.45),  # green
    (1.0, 0.85, 0.15),   # yellow
    (0.85, 0.40, 1.0),   # purple
    (1.0, 0.60, 0.15),   # orange
]

CRYSTAL_COL = (0.0, 1.0, 0.80)
CRYSTAL_DIM_COL = (0.0, 0.50, 0.40)
OBSTACLE_COL = (0.28, 0.26, 0.32)
SHELTER_COL = (0.65, 0.55, 0.35)
PREDATOR_COL = (1.0, 0.12, 0.12)
PREDATOR_SCARED_COL = (1.0, 0.55, 0.15)

SKY_DAY = (0.48, 0.68, 0.92)
SKY_NIGHT = (0.01, 0.01, 0.05)
SKY_STORM = (0.12, 0.10, 0.18)
SKY_FOG = (0.50, 0.50, 0.53)

WATER_COL = (0.06, 0.22, 0.44, 0.50)

# ─── geometry helpers ─────────────────────────────────────────────

def _sphere_vertices(radius: float, slices: int = 10, stacks: int = 8):
    """Generate sphere vertex list for drawing with GL_TRIANGLE_STRIP."""
    verts = []
    for i in range(stacks):
        lat0 = math.pi * (-0.5 + i / stacks)
        lat1 = math.pi * (-0.5 + (i + 1) / stacks)
        z0, zr0 = math.sin(lat0) * radius, math.cos(lat0) * radius
        z1, zr1 = math.sin(lat1) * radius, math.cos(lat1) * radius
        for j in range(slices + 1):
            lng = 2 * math.pi * j / slices
            x, y = math.cos(lng), math.sin(lng)
            verts.append((x * zr1, y * zr1, z1, x, y, math.sin(lat1)))
            verts.append((x * zr0, y * zr0, z0, x, y, math.sin(lat0)))
    return verts


def _make_octahedron(scale: float = 1.0):
    """Return faces of an octahedron (crystal shape)."""
    top = (0, 0, scale)
    bot = (0, 0, -scale)
    pts = [(scale, 0, 0), (0, scale, 0), (-scale, 0, 0), (0, -scale, 0)]
    faces = []
    for i in range(4):
        j = (i + 1) % 4
        faces.append((top, pts[i], pts[j]))
        faces.append((bot, pts[j], pts[i]))
    return faces


# ─── main 3D visualiser ──────────────────────────────────────────

class Visualiser3D:
    """Full 3D OpenGL renderer with orbit camera and HUD."""

    def __init__(self, sandbox: "Sandbox", num_agents: int) -> None:
        self.sandbox = sandbox
        self.num_agents = num_agents
        self.world_w = sandbox.width
        self.world_h = sandbox.height

        # Window setup
        self.win_w, self.win_h = 1600, 1000
        pygame.init()
        pygame.display.set_caption("Project Genesis — 3D")
        # Request MSAA antialiasing
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        try:
            self.screen = pygame.display.set_mode(
                (self.win_w, self.win_h), DOUBLEBUF | OPENGL
            )
        except pygame.error:
            # Fallback without MSAA
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 0)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 0)
            self.screen = pygame.display.set_mode(
                (self.win_w, self.win_h), DOUBLEBUF | OPENGL
            )
        self.clock = pygame.time.Clock()
        self._fps = 0.0

        # Camera state (orbit camera)
        self.cam_yaw = -45.0       # degrees around Y
        self.cam_pitch = 55.0      # degrees above horizon
        self.cam_dist = max(120.0, max(self.world_w, self.world_h) * 0.6)  # scale with world
        self.cam_target = [self.world_w / 2, 0.0, self.world_h / 2]
        self._dragging = False
        self._last_mouse = (0, 0)

        # Focus / UI state
        self._focused = 0
        self._show_cogmap = False
        self._show_hud = True
        self._follow_cam = False
        self.time_scale = 1       # 1 = normal, 2/4/8 = fast-forward
        self._frame = 0

        # Precompute terrain mesh
        self._terrain_list = None
        self._water_list = None
        self._obstacle_list = None
        self._build_terrain_mesh()
        self._build_obstacle_mesh()

        # Precompute sphere display list
        self._sphere_list = self._build_sphere_list(0.35, 18, 14)
        self._crystal_list = self._build_crystal_list()

        # HUD font
        self._hud_font = pygame.font.SysFont("Consolas", 14)
        self._hud_font_big = pygame.font.SysFont("Consolas", 18, bold=True)
        self._hud_font_label = pygame.font.SysFont("Consolas", 12)

        # Text rendering cache {(text, colour_tuple, font_id): (gl_data, w, h)}
        self._text_cache: dict[tuple, tuple] = {}
        self._text_cache_frame = 0

        # Movement trail history per agent: agent_id -> deque of (x, y, z)
        from collections import deque
        self._trails: dict[int, deque] = {}
        self._trail_len = 40

        # Minimap surface
        self._minimap_size = 160
        self._minimap_tex = None
        self._build_minimap()

        # OpenGL init
        self._init_gl()

    # ─── OpenGL setup ─────────────────────────────────────────

    def _init_gl(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)  # Fill light
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_NORMALIZE)

        # Enable MSAA if available
        try:
            glEnable(GL_MULTISAMPLE)
        except Exception:
            pass

        # Line smoothing
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # Point smoothing
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

        # Subtle fog for depth perception
        glEnable(GL_FOG)
        glFogi(GL_FOG_MODE, GL_LINEAR)
        glFogf(GL_FOG_START, 150.0)
        glFogf(GL_FOG_END, 400.0)

        # Fill light — cool blue from opposite side
        glLightfv(GL_LIGHT1, GL_POSITION, [-0.5, 0.3, -0.8, 0.0])
        glLightfv(GL_LIGHT1, GL_AMBIENT, [0.0, 0.0, 0.0, 1.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.08, 0.10, 0.15, 1.0])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])

    def _set_lighting(self, light_level: float, weather: int) -> None:
        """Configure lighting based on day/night and weather."""
        # Sun direction — sweeps across sky
        sun_angle = light_level * math.pi
        sun_dir = [math.cos(sun_angle), max(0.3, light_level), math.sin(sun_angle), 0.0]
        glLightfv(GL_LIGHT0, GL_POSITION, sun_dir)

        # Ambient scales with light level
        amb = 0.15 + 0.25 * light_level
        diff = 0.3 + 0.7 * light_level
        if weather == 2:  # storm
            amb *= 0.6
            diff *= 0.5
        elif weather == 4:  # fog
            amb *= 0.8
            diff *= 0.7

        glLightfv(GL_LIGHT0, GL_AMBIENT, [amb, amb, amb * 1.05, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [diff, diff * 0.95, diff * 0.9, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.4 * light_level, 0.4 * light_level, 0.35 * light_level, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.15, 0.15, 0.15, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 20.0)

    def _get_sky_colour(self, light_level: float, weather: int):
        """Blend sky colour based on conditions."""
        if weather == 2:
            target = SKY_STORM
        elif weather == 4:
            target = SKY_FOG
        else:
            target = tuple(
                SKY_NIGHT[i] + (SKY_DAY[i] - SKY_NIGHT[i]) * light_level
                for i in range(3)
            )
        return target

    def _build_minimap(self) -> None:
        """Build a small minimap texture from biome data."""
        sz = self._minimap_size
        surf = pygame.Surface((sz, sz), pygame.SRCALPHA)
        bm_colours_255 = {
            k: tuple(int(c * 255) for c in v) for k, v in BIOME_COLOURS.items()
        }
        sx = self.world_w / sz
        sy = self.world_h / sz
        for py in range(sz):
            for px in range(sz):
                wx = min(int(px * sx), self.world_w - 1)
                wy = min(int(py * sy), self.world_h - 1)
                biome = self.sandbox.biome_map.biome_at(wx, wy)
                c = bm_colours_255.get(biome, (40, 80, 30))
                if (wx, wy) in self.sandbox.obstacles:
                    c = (60, 55, 70)
                surf.set_at((px, py), (*c, 200))
        self._minimap_surf = surf
        self._minimap_data = pygame.image.tostring(self._minimap_surf, "RGBA", True)

    def _draw_minimap(self, agents: list["ConsciousAgent"]) -> None:
        """Draw minimap in bottom-left corner of HUD."""
        sz = self._minimap_size
        mx, my = 12, self.win_h - sz - 12

        # Border
        glColor4f(0.15, 0.15, 0.20, 0.75)
        glBegin(GL_QUADS)
        glVertex2f(mx - 3, my - 3)
        glVertex2f(mx + sz + 3, my - 3)
        glVertex2f(mx + sz + 3, my + sz + 3)
        glVertex2f(mx - 3, my + sz + 3)
        glEnd()

        # Minimap image
        glRasterPos2f(mx, my + sz)
        glDrawPixels(sz, sz, GL_RGBA, GL_UNSIGNED_BYTE, self._minimap_data)

        # Agent dots
        scx = sz / self.world_w
        scy = sz / self.world_h
        glPointSize(6.0)
        glBegin(GL_POINTS)
        for agent in agents:
            if not agent.alive:
                continue
            col = AGENT_COLOURS[agent.agent_id % len(AGENT_COLOURS)]
            glColor3f(*col)
            glVertex2f(mx + agent.body.position.x * scx,
                       my + agent.body.position.y * scy)
        glEnd()

        # Predator dots
        glColor3f(1.0, 0.1, 0.1)
        glPointSize(4.0)
        glBegin(GL_POINTS)
        for pred in self.sandbox.predators.predators:
            if not pred.alive:
                continue
            glVertex2f(mx + pred.position.x * scx,
                       my + pred.position.y * scy)
        glEnd()
        glPointSize(1.0)

        # Label
        self._render_text("MAP", mx + 2, my - 16, (160, 165, 180),
                          self._hud_font_label)

    def _draw_stars(self, light_level: float) -> None:
        """Draw stars in the night sky (only visible at low light)."""
        if light_level > 0.4:
            return
        alpha = max(0.0, (0.4 - light_level) / 0.4)

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_FOG)
        glPointSize(2.0)
        glBegin(GL_POINTS)
        cx, cy, cz = self.cam_target
        for i in range(150):
            lat = math.sin(i * 2.399) * 0.4 + 0.5
            lng = (i * 137.508) % 360.0
            r = 190.0
            cla = math.radians(lat * 90)
            clg = math.radians(lng)
            star_x = cx + r * math.cos(cla) * math.cos(clg)
            star_y = cy + r * math.sin(cla)
            star_z = cz + r * math.cos(cla) * math.sin(clg)
            bright = 0.5 + 0.5 * math.sin(i * 3.7 + self._frame * 0.015)
            glColor4f(0.9, 0.93, 1.0, alpha * bright * 0.7)
            glVertex3f(star_x, star_y, star_z)
        glEnd()
        glPointSize(1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_FOG)
        glEnable(GL_LIGHTING)

    def _draw_sun_moon(self, light_level: float, phase: float) -> None:
        """Draw a sun or moon disc in the sky."""
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_FOG)

        cx, cy, cz = self.cam_target
        sky_r = 180.0

        # Sun angle tracks the day cycle
        sun_angle = phase * 2 * math.pi
        body_x = cx + sky_r * math.cos(sun_angle) * 0.6
        body_y = cy + sky_r * abs(math.sin(sun_angle)) * 0.8 + 20.0
        body_z = cz + sky_r * math.sin(sun_angle) * 0.3

        if not (phase > 0.5):  # daytime — sun
            disc_r = 6.0
            # Sun glow halo
            glow_r = disc_r * 2.5
            glBegin(GL_TRIANGLE_FAN)
            glColor4f(1.0, 0.95, 0.7, 0.15)
            glVertex3f(body_x, body_y, body_z)
            glColor4f(1.0, 0.9, 0.5, 0.0)
            for i in range(25):
                a = 2 * math.pi * i / 24
                glVertex3f(body_x + math.cos(a) * glow_r,
                           body_y + math.sin(a) * glow_r * 0.6,
                           body_z + math.sin(a) * glow_r * 0.3)
            glEnd()
            # Sun disc
            glBegin(GL_TRIANGLE_FAN)
            glColor4f(1.0, 0.95, 0.75, 0.9)
            glVertex3f(body_x, body_y, body_z)
            glColor4f(1.0, 0.85, 0.4, 0.7)
            for i in range(25):
                a = 2 * math.pi * i / 24
                glVertex3f(body_x + math.cos(a) * disc_r,
                           body_y + math.sin(a) * disc_r * 0.6,
                           body_z + math.sin(a) * disc_r * 0.3)
            glEnd()
        else:  # nighttime — moon
            disc_r = 4.0
            # Moon glow
            glow_r = disc_r * 2.0
            glBegin(GL_TRIANGLE_FAN)
            glColor4f(0.7, 0.75, 0.9, 0.08)
            glVertex3f(body_x, body_y, body_z)
            glColor4f(0.5, 0.55, 0.7, 0.0)
            for i in range(25):
                a = 2 * math.pi * i / 24
                glVertex3f(body_x + math.cos(a) * glow_r,
                           body_y + math.sin(a) * glow_r * 0.6,
                           body_z + math.sin(a) * glow_r * 0.3)
            glEnd()
            # Moon disc
            glBegin(GL_TRIANGLE_FAN)
            glColor4f(0.85, 0.88, 0.95, 0.85)
            glVertex3f(body_x, body_y, body_z)
            glColor4f(0.7, 0.72, 0.82, 0.6)
            for i in range(25):
                a = 2 * math.pi * i / 24
                glVertex3f(body_x + math.cos(a) * disc_r,
                           body_y + math.sin(a) * disc_r * 0.6,
                           body_z + math.sin(a) * disc_r * 0.3)
            glEnd()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_FOG)
        glEnable(GL_LIGHTING)

    def _draw_ambient_particles(self, light_level: float,
                                weather: int) -> None:
        """Floating dust motes by day, fireflies at night."""
        if weather in (1, 2, 4):  # skip during rain/storm/fog
            return

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        cx, cz = self.cam_target[0], self.cam_target[2]
        t = self._frame

        if light_level > 0.5:
            # Daytime floating pollen / dust motes
            glColor4f(1.0, 0.95, 0.7, 0.12)
            glPointSize(2.0)
            glBegin(GL_POINTS)
            for i in range(30):
                px = cx + math.sin(i * 4.3 + t * 0.002) * 25
                pz = cz + math.cos(i * 5.9 + t * 0.003) * 25
                py = 1.5 + math.sin(i * 1.7 + t * 0.015) * 2.0
                glVertex3f(px, py, pz)
            glEnd()
            glPointSize(1.0)
        else:
            # Nighttime fireflies
            n_flies = 25
            glPointSize(3.5)
            glBegin(GL_POINTS)
            for i in range(n_flies):
                # Slow, drifting, blinking
                px = cx + math.sin(i * 7.1 + t * 0.004) * 20
                pz = cz + math.cos(i * 9.3 + t * 0.003) * 20
                py = 0.3 + abs(math.sin(i * 2.1 + t * 0.008)) * 2.5
                blink = max(0.0, math.sin(i * 5.3 + t * 0.03))
                if blink > 0.3:
                    alpha = (blink - 0.3) * 0.7
                    glColor4f(0.4, 1.0, 0.3, alpha)
                    glVertex3f(px, py, pz)
            glEnd()
            glPointSize(1.0)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def _draw_cogmap_3d(self, agent: "ConsciousAgent") -> None:
        """Draw semi-transparent overlay on terrain showing cognitive map."""
        cmap = agent.cognitive_map
        cs = cmap.cell_size
        if not cmap.cells:
            return

        glDisable(GL_LIGHTING)
        glDepthMask(GL_FALSE)
        glEnable(GL_BLEND)

        for (cx, cy), cell in cmap.cells.items():
            crystal = cell.crystal_score
            danger = cell.danger_score
            if crystal < 0.05 and danger < 0.05:
                continue
            wx = cx * cs
            wz = cy * cs
            # Clamp inside world
            if wx < 0 or wz < 0 or wx >= self.world_w or wz >= self.world_h:
                continue
            ey = self._elev(min(wx, self.world_w - 1),
                            min(wz, self.world_h - 1)) + 0.15

            # Green for crystal, red for danger, blend if both
            if crystal > danger:
                r, g, b = 0.1, 0.9, 0.2
                alpha = min(0.45, crystal * 0.15)
            else:
                r, g, b = 0.9, 0.15, 0.1
                alpha = min(0.45, danger * 0.15)

            glColor4f(r, g, b, alpha)
            glBegin(GL_QUADS)
            glVertex3f(wx, ey, wz)
            glVertex3f(wx + cs, ey, wz)
            glVertex3f(wx + cs, ey, wz + cs)
            glVertex3f(wx, ey, wz + cs)
            glEnd()

        glDepthMask(GL_TRUE)
        glEnable(GL_LIGHTING)

    def _draw_action_sparkles(self, agents: list["ConsciousAgent"]) -> None:
        """Draw sparkle particles for active agent actions."""
        # Action indices: 5=collect, 9=build, 12=craft, 6=emit_sound, 13=share
        ACTION_SPARKLE = {
            5: (0.2, 1.0, 0.4),   # collect – green
            9: (0.9, 0.7, 0.2),   # build – gold
            12: (0.6, 0.3, 1.0),  # craft – purple
            6: (0.3, 0.8, 1.0),   # emit_sound – cyan
            13: (1.0, 0.5, 0.8),  # share – pink
        }
        t = self._frame
        glDisable(GL_LIGHTING)
        glPointSize(3.0)
        for agent in agents:
            if not agent.alive:
                continue
            act = agent.body.last_action
            if act not in ACTION_SPARKLE:
                continue
            col = ACTION_SPARKLE[act]
            pos = agent.body.position
            gx, gz = pos.x, pos.y
            ix = int(min(max(gx, 0), self.world_w - 1))
            iz = int(min(max(gz, 0), self.world_h - 1))
            gy = self._elev(ix, iz) + 0.6

            glBegin(GL_POINTS)
            for i in range(8):
                angle = (t * 0.1 + i * math.pi / 4)
                r = 0.3 + 0.2 * math.sin(t * 0.08 + i)
                dy = 0.3 * ((t * 0.05 + i * 0.7) % 1.5)
                alpha = max(0.0, 0.7 - dy * 0.3)
                glColor4f(col[0], col[1], col[2], alpha)
                px = gx + math.cos(angle) * r
                pz = gz + math.sin(angle) * r
                glVertex3f(px, gy + dy, pz)
            glEnd()

        glPointSize(1.0)
        glEnable(GL_LIGHTING)

    # ─── terrain mesh ─────────────────────────────────────────

    def _elev(self, x: int, y: int) -> float:
        """Get elevation at grid cell, scaled for 3D."""
        raw = self.sandbox.heightmap.elevation_at(x, y)
        return raw * 1.0  # taller hills for bigger world

    def _terrain_colour(self, x: int, y: int, elev: float):
        """Get vertex colour from biome + elevation shading."""
        biome = self.sandbox.biome_map.biome_at(x, y)
        base = BIOME_COLOURS.get(biome, BIOME_COLOURS[BIOME_GRASSLANDS])
        # Elevation-based shading: valleys darker, peaks lighter
        shade = 0.80 + 0.20 * max(-1.0, min(1.0, elev / 2.0))
        # Add subtle colour variation for visual interest
        noise = math.sin(x * 0.7 + y * 1.1) * 0.03
        return (
            min(1.0, max(0.0, base[0] * shade + noise)),
            min(1.0, max(0.0, base[1] * shade + noise * 0.7)),
            min(1.0, max(0.0, base[2] * shade - noise * 0.5)),
        )

    def _build_terrain_mesh(self) -> None:
        """Pre-build a display list for the terrain as a triangle mesh."""
        w, h = self.world_w, self.world_h
        # Use stride for large worlds to keep vertex count manageable
        step = 2 if w * h > 40000 else 1
        self._terrain_step = step
        self._terrain_list = glGenLists(1)
        glNewList(self._terrain_list, GL_COMPILE)
        glBegin(GL_TRIANGLES)
        for y in range(0, h - step, step):
            for x in range(0, w - step, step):
                e00 = self._elev(x, y)
                e10 = self._elev(x + step, y)
                e01 = self._elev(x, y + step)
                e11 = self._elev(x + step, y + step)
                c00 = self._terrain_colour(x, y, e00)
                c10 = self._terrain_colour(x + step, y, e10)
                c01 = self._terrain_colour(x, y + step, e01)
                c11 = self._terrain_colour(x + step, y + step, e11)

                # Compute face normal for triangle 1
                v1 = (step, 0, e10 - e00)
                v2 = (0, step, e01 - e00)
                n1 = (v1[1]*v2[2] - v1[2]*v2[1],
                       v1[2]*v2[0] - v1[0]*v2[2],
                       v1[0]*v2[1] - v1[1]*v2[0])

                # Triangle 1: (0,0) (1,0) (0,1)
                glNormal3f(*n1)
                glColor3f(*c00); glVertex3f(x,      e00, y)
                glColor3f(*c10); glVertex3f(x+step,  e10, y)
                glColor3f(*c01); glVertex3f(x,      e01, y+step)

                # Triangle 2: (1,0) (1,1) (0,1)
                v3 = (-step, 0, e01 - e11)
                v4 = (0, -step, e10 - e11)
                n2 = (v3[1]*v4[2] - v3[2]*v4[1],
                       v3[2]*v4[0] - v3[0]*v4[2],
                       v3[0]*v4[1] - v3[1]*v4[0])
                glNormal3f(*n2)
                glColor3f(*c10); glVertex3f(x+step,  e10, y)
                glColor3f(*c11); glVertex3f(x+step,  e11, y+step)
                glColor3f(*c01); glVertex3f(x,      e01, y+step)
        glEnd()
        glEndList()

        # Water plane for ocean areas
        step_w = step
        self._water_list = glGenLists(1)
        glNewList(self._water_list, GL_COMPILE)
        glBegin(GL_QUADS)
        glNormal3f(0, 1, 0)
        water_y = -0.3  # water surface level
        for y in range(0, h, step_w):
            for x in range(0, w, step_w):
                biome = self.sandbox.biome_map.biome_at(x, y)
                if biome == BIOME_OCEAN or biome == BIOME_WETLANDS:
                    glColor4f(*WATER_COL)
                    glVertex3f(x, water_y, y)
                    glVertex3f(x+step_w, water_y, y)
                    glVertex3f(x+step_w, water_y, y+step_w)
                    glVertex3f(x, water_y, y+step_w)
        glEnd()
        glEndList()

        # River display list
        self._river_list = glGenLists(1)
        glNewList(self._river_list, GL_COMPILE)
        glBegin(GL_QUADS)
        glNormal3f(0, 1, 0)
        river_col = (0.2, 0.45, 0.7, 0.65)
        for (rx, ry) in self.sandbox.rivers:
            e = self._elev(min(rx, w-1), min(ry, h-1))
            wy = max(e, -0.25) + 0.05
            glColor4f(*river_col)
            glVertex3f(rx, wy, ry)
            glVertex3f(rx+1, wy, ry)
            glVertex3f(rx+1, wy, ry+1)
            glVertex3f(rx, wy, ry+1)
        glEnd()
        glEndList()

    def _build_obstacle_mesh(self) -> None:
        """Pre-build obstacle cubes."""
        self._obstacle_list = glGenLists(1)
        glNewList(self._obstacle_list, GL_COMPILE)
        for (ox, oy) in self.sandbox.obstacles:
            e = self._elev(ox, oy)
            self._draw_cube(ox + 0.5, e, oy + 0.5, 0.48, 1.2, OBSTACLE_COL)
        glEndList()

    def _draw_cube(self, cx, cy, cz, half_w, height, col):
        """Draw a simple cube at position."""
        x0, x1 = cx - half_w, cx + half_w
        z0, z1 = cz - half_w, cz + half_w
        y0, y1 = cy, cy + height
        glColor3f(*col)
        glBegin(GL_QUADS)
        # Top
        glNormal3f(0, 1, 0)
        c = (col[0]*1.2, col[1]*1.2, col[2]*1.2)
        glColor3f(*[min(1, v) for v in c])
        glVertex3f(x0, y1, z0); glVertex3f(x1, y1, z0)
        glVertex3f(x1, y1, z1); glVertex3f(x0, y1, z1)
        # Front
        glNormal3f(0, 0, -1); glColor3f(*col)
        glVertex3f(x0, y0, z0); glVertex3f(x1, y0, z0)
        glVertex3f(x1, y1, z0); glVertex3f(x0, y1, z0)
        # Back
        glNormal3f(0, 0, 1)
        glVertex3f(x0, y0, z1); glVertex3f(x0, y1, z1)
        glVertex3f(x1, y1, z1); glVertex3f(x1, y0, z1)
        # Left
        glNormal3f(-1, 0, 0)
        glVertex3f(x0, y0, z0); glVertex3f(x0, y1, z0)
        glVertex3f(x0, y1, z1); glVertex3f(x0, y0, z1)
        # Right
        glNormal3f(1, 0, 0)
        glVertex3f(x1, y0, z0); glVertex3f(x1, y0, z1)
        glVertex3f(x1, y1, z1); glVertex3f(x1, y1, z0)
        glEnd()

    # ─── display lists ────────────────────────────────────────

    def _build_sphere_list(self, radius: float, slices: int, stacks: int) -> int:
        """Create a display list for a unit sphere."""
        dl = glGenLists(1)
        glNewList(dl, GL_COMPILE)
        verts = _sphere_vertices(radius, slices, stacks)
        glBegin(GL_TRIANGLE_STRIP)
        for vx, vy, vz, nx, ny, nz in verts:
            glNormal3f(nx, ny, nz)
            glVertex3f(vx, vy, vz)
        glEnd()
        glEndList()
        return dl

    def _build_crystal_list(self) -> int:
        """Crystal as an octahedron display list."""
        dl = glGenLists(1)
        glNewList(dl, GL_COMPILE)
        faces = _make_octahedron(0.32)
        glBegin(GL_TRIANGLES)
        for v0, v1, v2 in faces:
            # compute face normal
            e1 = (v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2])
            e2 = (v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2])
            n = (e1[1]*e2[2]-e1[2]*e2[1],
                 e1[2]*e2[0]-e1[0]*e2[2],
                 e1[0]*e2[1]-e1[1]*e2[0])
            glNormal3f(*n)
            glVertex3f(*v0); glVertex3f(*v1); glVertex3f(*v2)
        glEnd()
        glEndList()
        return dl

    # ─── drawing entities ─────────────────────────────────────

    def _draw_agents(self, agents: list["ConsciousAgent"]) -> None:
        from collections import deque as _deque
        for agent in agents:
            if not agent.alive:
                continue
            col = AGENT_COLOURS[agent.agent_id % len(AGENT_COLOURS)]
            pos = agent.body.position
            gx, gz = pos.x, pos.y
            gy = self._elev(int(min(max(gx, 0), self.world_w-1)),
                            int(min(max(gz, 0), self.world_h-1)))

            # ── Track movement trail ──
            aid = agent.agent_id
            if aid not in self._trails:
                self._trails[aid] = _deque(maxlen=self._trail_len)
            self._trails[aid].append((gx, gy, gz))

            # ── Draw movement trail ──
            trail = self._trails[aid]
            if len(trail) > 2:
                glDisable(GL_LIGHTING)
                glLineWidth(2.0)
                glBegin(GL_LINE_STRIP)
                for ti, (tx, ty, tz) in enumerate(trail):
                    alpha = (ti / len(trail)) * 0.5
                    glColor4f(col[0], col[1], col[2], alpha)
                    glVertex3f(tx, ty + 0.1, tz)
                glEnd()
                glLineWidth(1.0)
                glEnable(GL_LIGHTING)

            # ── Consciousness aura ──
            comp = 0.0
            try:
                assess = agent.phi_calculator.get_consciousness_assessment(
                    self_model_accuracy=agent.self_model.model_accuracy,
                    attention_accuracy=agent.attention_schema.schema_accuracy,
                    metacognitive_confidence=agent.inner_speech.confidence,
                    binding_coherence=agent.binding.coherence,
                    empowerment=agent.empowerment.empowerment,
                    narrative_identity=agent.narrative.identity_strength,
                    curiosity_level=agent.curiosity.curiosity_level,
                )
                comp = assess["composite_score"]
            except Exception:
                pass

            if comp > 0.1:
                # Pulsing aura ring at head height
                aura_r = 0.8 + comp * 0.6
                pulse = 0.8 + 0.2 * math.sin(self._frame * 0.06)
                aura_r *= pulse
                # Colour: red→yellow→green→cyan by composite
                if comp < 0.3:
                    ac = (0.9, 0.3 + comp * 2, 0.2)
                elif comp < 0.6:
                    t = (comp - 0.3) / 0.3
                    ac = (0.9 - t * 0.7, 0.8 + t * 0.1, 0.2 + t * 0.3)
                else:
                    t = (comp - 0.6) / 0.4
                    ac = (0.2 - t * 0.1, 0.85 + t * 0.15, 0.5 + t * 0.5)
                aura_alpha = 0.08 + comp * 0.12

                glPushMatrix()
                glTranslatef(gx, gy + 0.7, gz)
                glDisable(GL_LIGHTING)
                # Filled aura disc
                glColor4f(ac[0], ac[1], ac[2], aura_alpha)
                glBegin(GL_TRIANGLE_FAN)
                glVertex3f(0, 0, 0)
                for i in range(25):
                    a = 2 * math.pi * i / 24
                    glVertex3f(math.cos(a) * aura_r, 0.05, math.sin(a) * aura_r)
                glEnd()
                # Bright ring edge
                glColor4f(ac[0], ac[1], ac[2], aura_alpha * 2.5)
                glLineWidth(1.5)
                glBegin(GL_LINE_LOOP)
                for i in range(25):
                    a = 2 * math.pi * i / 24
                    glVertex3f(math.cos(a) * aura_r, 0.05, math.sin(a) * aura_r)
                glEnd()
                glLineWidth(1.0)
                glEnable(GL_LIGHTING)
                glPopMatrix()

            # Body sphere
            glPushMatrix()
            glTranslatef(gx, gy + 0.45, gz)
            glColor3f(*col)
            glCallList(self._sphere_list)
            glPopMatrix()

            # Head (smaller sphere above)
            glPushMatrix()
            glTranslatef(gx, gy + 1.0, gz)
            glColor3f(min(1, col[0]*1.1), min(1, col[1]*1.1), min(1, col[2]*1.1))
            glScalef(0.6, 0.6, 0.6)
            glCallList(self._sphere_list)
            glPopMatrix()

            # Eyes (two small white dots on head)
            vel = agent.body.velocity
            if vel.magnitude() > 0.05:
                norm = vel.normalized()
            else:
                norm_x, norm_y = 1.0, 0.0

            glDisable(GL_LIGHTING)
            glColor3f(1.0, 1.0, 1.0)
            glPointSize(4.0)
            glBegin(GL_POINTS)
            if vel.magnitude() > 0.05:
                fx, fz = norm.x * 0.15, norm.y * 0.15
                sx, sz = -norm.y * 0.08, norm.x * 0.08
            else:
                fx, fz = 0.15, 0.0
                sx, sz = 0.0, 0.08
            glVertex3f(gx + fx + sx, gy + 1.05, gz + fz + sz)
            glVertex3f(gx + fx - sx, gy + 1.05, gz + fz - sz)
            glEnd()
            glPointSize(1.0)
            glEnable(GL_LIGHTING)

            # Direction arrow
            if vel.magnitude() > 0.05:
                norm = vel.normalized()
                glPushMatrix()
                glTranslatef(gx, gy + 0.5, gz)
                glDisable(GL_LIGHTING)
                glColor3f(1, 1, 1)
                glLineWidth(2.0)
                glBegin(GL_LINES)
                glVertex3f(0, 0, 0)
                glVertex3f(norm.x * 0.8, 0, norm.y * 0.8)
                glEnd()
                glLineWidth(1.0)
                glEnable(GL_LIGHTING)
                glPopMatrix()

            # Glow ring on ground
            glPushMatrix()
            glTranslatef(gx, gy + 0.02, gz)
            glDisable(GL_LIGHTING)
            glColor4f(col[0], col[1], col[2], 0.35)
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, 0, 0)
            for i in range(21):
                a = 2 * math.pi * i / 20
                glVertex3f(math.cos(a) * 0.7, 0, math.sin(a) * 0.7)
            glEnd()
            glEnable(GL_LIGHTING)
            glPopMatrix()

            # ── Floating label above agent ──
            # Project 3D position to 2D screen coordinates
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            viewport = glGetIntegerv(GL_VIEWPORT)
            try:
                sx, sy, sz = gluProject(gx, gy + 1.7, gz, modelview, projection, viewport)
                if 0 < sz < 1:  # visible
                    screen_y = self.win_h - int(sy)
                    screen_x = int(sx)
                    # Agent ID + composite score
                    label = f"A{aid}"
                    if comp > 0:
                        label += f" [{comp:.2f}]"
                    # Ego badge
                    if agent.self_model.has_ego:
                        label += " *"
                    # Dreaming
                    if agent.dream_engine.stats.is_dreaming:
                        label += " Zz"
                    # Store for HUD pass (draw text in 2D)
                    if not hasattr(self, '_floating_labels'):
                        self._floating_labels = []
                    col_255 = tuple(int(c * 255) for c in col)
                    self._floating_labels.append((screen_x, screen_y, label, col_255))
            except Exception:
                pass

    def _draw_crystals(self, sandbox: "Sandbox") -> None:
        t = self._frame * 0.06
        for crystal in sandbox.crystals:
            if crystal.consumed or crystal.is_expired:
                continue
            pos = crystal.position
            gx, gz = pos.x, pos.y
            ix = int(min(max(gx, 0), self.world_w - 1))
            iz = int(min(max(gz, 0), self.world_h - 1))
            gy = self._elev(ix, iz)
            f = crystal.freshness

            col = tuple(
                CRYSTAL_DIM_COL[i] + (CRYSTAL_COL[i] - CRYSTAL_DIM_COL[i]) * f
                for i in range(3)
            )
            # Hover & rotate
            hover = 0.3 + 0.12 * math.sin(t + gx * 0.5 + gz * 0.3)
            spin = (self._frame * 2.0 + gx * 37 + gz * 53) % 360

            glPushMatrix()
            glTranslatef(gx + 0.5, gy + hover, gz + 0.5)
            glRotatef(spin, 0, 1, 0)

            # Glow
            glDisable(GL_LIGHTING)
            glColor4f(col[0], col[1], col[2], 0.30 + 0.15 * f)
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, 0, 0)
            r_glow = 0.55 + 0.15 * f
            for i in range(17):
                a = 2 * math.pi * i / 16
                glVertex3f(math.cos(a)*r_glow, 0, math.sin(a)*r_glow)
            glEnd()
            # Vertical light pillar
            pillar_h = 1.5 + 1.0 * f
            pillar_w = 0.06 + 0.04 * f
            glBegin(GL_QUADS)
            # Two crossing planes for the beam
            for ang in (0, math.pi / 2):
                dx = math.cos(ang) * pillar_w
                dz = math.sin(ang) * pillar_w
                # Bottom bright
                glColor4f(col[0], col[1], col[2], 0.15 + 0.10 * f)
                glVertex3f(-dx, -0.2, -dz)
                glVertex3f(dx, -0.2, dz)
                # Top fade out
                glColor4f(col[0], col[1], col[2], 0.0)
                glVertex3f(dx, pillar_h, dz)
                glVertex3f(-dx, pillar_h, -dz)
            glEnd()
            glEnable(GL_LIGHTING)

            glColor3f(*col)
            glCallList(self._crystal_list)
            glPopMatrix()

    def _draw_shelters(self, sandbox: "Sandbox") -> None:
        for (sx, sy), owner in sandbox.shelters.items():
            e = self._elev(sx, sy)
            col = AGENT_COLOURS[owner % len(AGENT_COLOURS)]
            # Base walls
            self._draw_cube(sx + 0.5, e, sy + 0.5, 0.35, 0.5,
                            (col[0]*0.5, col[1]*0.5, col[2]*0.5))
            # Roof (pyramid)
            glColor3f(col[0]*0.7, col[1]*0.7, col[2]*0.7)
            cx, cz = sx + 0.5, sy + 0.5
            roof_y = e + 0.5
            peak_y = e + 0.9
            glBegin(GL_TRIANGLES)
            pts = [(cx-0.4, cz-0.4), (cx+0.4, cz-0.4),
                   (cx+0.4, cz+0.4), (cx-0.4, cz+0.4)]
            for i in range(4):
                j = (i + 1) % 4
                p1, p2 = pts[i], pts[j]
                mx = (p1[0]+p2[0])/2 - cx
                mz = (p1[1]+p2[1])/2 - cz
                length = math.sqrt(mx*mx + mz*mz) or 1
                glNormal3f(mx/length, 0.5, mz/length)
                glVertex3f(p1[0], roof_y, p1[1])
                glVertex3f(p2[0], roof_y, p2[1])
                glVertex3f(cx, peak_y, cz)
            glEnd()

            # Door — small dark rectangle on front face
            door_w, door_h = 0.12, 0.28
            glColor3f(col[0]*0.25, col[1]*0.25, col[2]*0.25)
            glBegin(GL_QUADS)
            glNormal3f(0, 0, -1)
            glVertex3f(cx - door_w, e, cz - 0.35)
            glVertex3f(cx + door_w, e, cz - 0.35)
            glVertex3f(cx + door_w, e + door_h, cz - 0.35)
            glVertex3f(cx - door_w, e + door_h, cz - 0.35)
            glEnd()

            # Warm glow from door/windows
            glDisable(GL_LIGHTING)
            pulse = 0.7 + 0.3 * math.sin(self._frame * 0.03 + sx * 3.1 + sy * 7.7)
            glow_r = 1.2 * pulse
            glColor4f(1.0, 0.75, 0.3, 0.06 * pulse)
            glBegin(GL_TRIANGLE_FAN)
            gy = e + 0.25
            glVertex3f(cx, gy, cz)
            for gi in range(17):
                a = 2 * math.pi * gi / 16
                glVertex3f(cx + math.cos(a) * glow_r, gy + 0.02,
                           cz + math.sin(a) * glow_r)
            glEnd()
            glEnable(GL_LIGHTING)

    # ─── civilisation visuals ─────────────────────────────────

    def _draw_structures(self, sandbox: "Sandbox") -> None:
        """Draw civilisation structures (farms, granaries, workshops, etc.)."""
        civ = getattr(sandbox, 'civilization', None)
        if civ is None:
            return
        from genesis.cognition.civilization import BuildingType
        for (sx, sy), struct in civ.structures.items():
            e = self._elev(sx, sy)
            bt = struct.building_type
            if bt == BuildingType.SHELTER:
                continue  # shelters drawn by _draw_shelters
            elif bt == BuildingType.FARM:
                # Flat brown tilled patch with green row lines
                glColor3f(0.45, 0.30, 0.15)
                glBegin(GL_QUADS)
                glNormal3f(0, 1, 0)
                glVertex3f(sx, e + 0.02, sy)
                glVertex3f(sx + 1, e + 0.02, sy)
                glVertex3f(sx + 1, e + 0.02, sy + 1)
                glVertex3f(sx, e + 0.02, sy + 1)
                glEnd()
                glDisable(GL_LIGHTING)
                glColor3f(0.3, 0.55, 0.2)
                glLineWidth(1.5)
                glBegin(GL_LINES)
                for r in range(4):
                    rz = sy + 0.15 + r * 0.22
                    glVertex3f(sx + 0.1, e + 0.03, rz)
                    glVertex3f(sx + 0.9, e + 0.03, rz)
                glEnd()
                glLineWidth(1.0)
                glEnable(GL_LIGHTING)
            elif bt == BuildingType.GRANARY:
                # Larger brown cube with pointed roof
                self._draw_cube(sx + 0.5, e, sy + 0.5, 0.4, 0.6,
                                (0.55, 0.40, 0.20))
                glColor3f(0.65, 0.50, 0.25)
                cx, cz = sx + 0.5, sy + 0.5
                roof_y = e + 0.6
                peak_y = e + 1.05
                glBegin(GL_TRIANGLES)
                pts = [(cx-0.45, cz-0.45), (cx+0.45, cz-0.45),
                       (cx+0.45, cz+0.45), (cx-0.45, cz+0.45)]
                for i in range(4):
                    j = (i + 1) % 4
                    p1, p2 = pts[i], pts[j]
                    glNormal3f(0, 0.5, 0)
                    glVertex3f(p1[0], roof_y, p1[1])
                    glVertex3f(p2[0], roof_y, p2[1])
                    glVertex3f(cx, peak_y, cz)
                glEnd()
            elif bt == BuildingType.WORKSHOP:
                # Grey cube with chimney
                self._draw_cube(sx + 0.5, e, sy + 0.5, 0.35, 0.55,
                                (0.45, 0.45, 0.50))
                self._draw_cube(sx + 0.8, e + 0.55, sy + 0.3, 0.08, 0.35,
                                (0.35, 0.35, 0.35))
            elif bt == BuildingType.WALL:
                # Stone wall segment
                self._draw_cube(sx + 0.5, e, sy + 0.5, 0.15, 0.7,
                                (0.50, 0.48, 0.42))
            elif bt == BuildingType.MONUMENT:
                # Tall stone pillar
                self._draw_cube(sx + 0.5, e, sy + 0.5, 0.18, 1.4,
                                (0.75, 0.72, 0.65))
                # Capstone
                self._draw_cube(sx + 0.5, e + 1.4, sy + 0.5, 0.25, 0.12,
                                (0.85, 0.82, 0.70))
            elif bt == BuildingType.LIBRARY:
                # Cube with dome (sphere on top)
                self._draw_cube(sx + 0.5, e, sy + 0.5, 0.38, 0.55,
                                (0.55, 0.45, 0.35))
                glColor3f(0.60, 0.55, 0.45)
                glPushMatrix()
                glTranslatef(sx + 0.5, e + 0.55, sy + 0.5)
                glScalef(0.35, 0.25, 0.35)
                glCallList(self._sphere_list)
                glPopMatrix()

    def _draw_crops(self, sandbox: "Sandbox") -> None:
        """Draw growing crops as small plants coloured by growth stage."""
        civ = getattr(sandbox, 'civilization', None)
        if civ is None:
            return
        for crop in civ.crops:
            ix = int(min(max(crop.x, 0), self.world_w - 1))
            iz = int(min(max(crop.y, 0), self.world_h - 1))
            e = self._elev(ix, iz)
            g = crop.growth
            # Interpolate green → golden-yellow
            r = 0.15 + 0.65 * g
            gr = 0.55 - 0.15 * g
            b = 0.10
            h = 0.08 + 0.30 * g  # plant height grows
            cx, cz = crop.x + 0.5, crop.y + 0.5
            glDisable(GL_LIGHTING)
            glColor3f(r, gr, b)
            glLineWidth(2.0)
            glBegin(GL_LINES)
            # Stem
            glVertex3f(cx, e, cz)
            glVertex3f(cx, e + h, cz)
            # Leaves
            glVertex3f(cx - 0.08, e + h * 0.6, cz)
            glVertex3f(cx + 0.08, e + h * 0.85, cz)
            glVertex3f(cx, e + h * 0.5, cz - 0.08)
            glVertex3f(cx, e + h * 0.8, cz + 0.08)
            glEnd()
            if crop.is_mature:
                # Golden dot at top for ripe crop
                glColor3f(0.9, 0.8, 0.2)
                glPointSize(4.0)
                glBegin(GL_POINTS)
                glVertex3f(cx, e + h + 0.04, cz)
                glEnd()
            glLineWidth(1.0)
            glEnable(GL_LIGHTING)

    def _draw_berry_bushes(self, sandbox: "Sandbox") -> None:
        """Draw berry bushes as green spherical shrubs with red berries."""
        t = self._frame
        for bush in sandbox.berry_bushes:
            pos = bush.position
            gx, gz = pos.x, pos.y
            ix = int(min(max(gx, 0), self.world_w - 1))
            iz = int(min(max(gz, 0), self.world_h - 1))
            gy = self._elev(ix, iz)

            # Bush canopy (green sphere)
            glPushMatrix()
            glTranslatef(gx + 0.5, gy + 0.4, gz + 0.5)
            glColor3f(0.2, 0.55, 0.15)
            glScalef(0.5, 0.45, 0.5)
            glCallList(self._sphere_list)
            glPopMatrix()

            # Trunk
            glDisable(GL_LIGHTING)
            glColor3f(0.4, 0.25, 0.1)
            glLineWidth(2.0)
            glBegin(GL_LINES)
            glVertex3f(gx + 0.5, gy, gz + 0.5)
            glVertex3f(gx + 0.5, gy + 0.3, gz + 0.5)
            glEnd()
            glLineWidth(1.0)

            # Berry dots
            if bush.berries > 0:
                glPointSize(4.0)
                glBegin(GL_POINTS)
                for i in range(bush.berries):
                    angle = i * 2.094 + gx * 3.7  # spread around
                    br = 0.3
                    bx = gx + 0.5 + math.cos(angle) * br
                    bz = gz + 0.5 + math.sin(angle) * br
                    by = gy + 0.35 + 0.1 * math.sin(i * 1.5)
                    glColor3f(0.9, 0.15, 0.1)
                    glVertex3f(bx, by, bz)
                glEnd()
                glPointSize(1.0)
            glEnable(GL_LIGHTING)

    def _draw_fungi(self, sandbox: "Sandbox") -> None:
        """Draw glowing fungi as small luminous caps."""
        t = self._frame
        light = sandbox.day_cycle.light_level
        for fungus in sandbox.fungi:
            if fungus.consumed:
                continue
            pos = fungus.position
            gx, gz = pos.x, pos.y
            ix = int(min(max(gx, 0), self.world_w - 1))
            iz = int(min(max(gz, 0), self.world_h - 1))
            gy = self._elev(ix, iz)

            # Glow intensity stronger at night
            glow = 0.3 + 0.7 * (1.0 - light)
            pulse = 0.8 + 0.2 * math.sin(fungus.glow_phase * 3.0 + gx)

            glDisable(GL_LIGHTING)
            # Mushroom cap (small triangle fan)
            cap_r = 0.25
            cap_h = gy + 0.3
            glColor4f(0.3 * pulse, 0.9 * glow * pulse, 0.5 * glow * pulse, 0.85)
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(gx + 0.5, cap_h + 0.1, gz + 0.5)
            for i in range(9):
                a = 2 * math.pi * i / 8
                glVertex3f(gx + 0.5 + math.cos(a) * cap_r, cap_h,
                           gz + 0.5 + math.sin(a) * cap_r)
            glEnd()

            # Stem
            glColor3f(0.7, 0.75, 0.6)
            glLineWidth(2.0)
            glBegin(GL_LINES)
            glVertex3f(gx + 0.5, gy, gz + 0.5)
            glVertex3f(gx + 0.5, cap_h, gz + 0.5)
            glEnd()
            glLineWidth(1.0)

            # Night glow aura
            if light < 0.6:
                aura_r = 0.6 * glow * pulse
                glColor4f(0.2, 0.8, 0.4, 0.15 * glow)
                glBegin(GL_TRIANGLE_FAN)
                glVertex3f(gx + 0.5, gy + 0.02, gz + 0.5)
                for i in range(13):
                    a = 2 * math.pi * i / 12
                    glVertex3f(gx + 0.5 + math.cos(a) * aura_r, gy + 0.02,
                               gz + 0.5 + math.sin(a) * aura_r)
                glEnd()
            glEnable(GL_LIGHTING)

    def _draw_ruins(self, sandbox: "Sandbox") -> None:
        """Draw ancient ruins as stone pillars and arches."""
        for ruin in sandbox.ruins:
            pos = ruin.position
            gx, gz = pos.x, pos.y
            ix = int(min(max(gx, 0), self.world_w - 1))
            iz = int(min(max(gz, 0), self.world_h - 1))
            gy = self._elev(ix, iz)

            stone_col = (0.55, 0.50, 0.42)
            dark_stone = (0.38, 0.35, 0.28)

            # Central broken pillar
            self._draw_cube(gx + 0.5, gy, gz + 0.5, 0.35, 2.0, stone_col)

            # Surrounding shorter pillars
            for i, (dx, dz) in enumerate([(-1.5, -1.5), (1.5, -1.5),
                                           (-1.5, 1.5), (1.5, 1.5)]):
                h = 1.0 + 0.5 * ((i * 37 + int(gx * 13)) % 3) / 2.0
                self._draw_cube(gx + dx, gy, gz + dz, 0.25, h, dark_stone)

            # Connecting lintel (arch between two pillars)
            glDisable(GL_LIGHTING)
            glColor3f(*stone_col)
            glLineWidth(3.0)
            glBegin(GL_LINES)
            glVertex3f(gx - 1.5, gy + 1.2, gz - 1.5)
            glVertex3f(gx + 1.5, gy + 1.2, gz - 1.5)
            glVertex3f(gx - 1.5, gy + 1.0, gz + 1.5)
            glVertex3f(gx + 1.5, gy + 1.0, gz + 1.5)
            glEnd()
            glLineWidth(1.0)

            # Mysterious glow at base
            glColor4f(0.6, 0.5, 0.9, 0.12)
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(gx + 0.5, gy + 0.03, gz + 0.5)
            for i in range(17):
                a = 2 * math.pi * i / 16
                glVertex3f(gx + 0.5 + math.cos(a) * ruin.radius,
                           gy + 0.03,
                           gz + 0.5 + math.sin(a) * ruin.radius)
            glEnd()
            glEnable(GL_LIGHTING)

    def _draw_predators(self, sandbox: "Sandbox") -> None:
        for pred in sandbox.predators.predators:
            if not pred.alive:
                continue
            gx, gz = pred.position.x, pred.position.y
            ix = int(min(max(gx, 0), self.world_w - 1))
            iz = int(min(max(gz, 0), self.world_h - 1))
            gy = self._elev(ix, iz)
            scared = pred.scared_ticks > 0
            col = PREDATOR_SCARED_COL if scared else PREDATOR_COL

            # Pulsing danger aura on ground
            glPushMatrix()
            glTranslatef(gx, gy + 0.03, gz)
            glDisable(GL_LIGHTING)
            pulse = 0.6 + 0.4 * math.sin(self._frame * 0.08)
            aura_r = 1.0 + 0.3 * pulse
            glColor4f(col[0], col[1], col[2], 0.10 * pulse)
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, 0, 0)
            for i in range(21):
                a = 2 * math.pi * i / 20
                glVertex3f(math.cos(a) * aura_r, 0, math.sin(a) * aura_r)
            glEnd()
            glEnable(GL_LIGHTING)
            glPopMatrix()

            # Body — spiky sphere
            glPushMatrix()
            glTranslatef(gx, gy + 0.4, gz)
            glColor3f(*col)
            glCallList(self._sphere_list)

            # Spikes — more of them, longer
            glDisable(GL_LIGHTING)
            glColor3f(*col)
            glLineWidth(2.0)
            glBegin(GL_LINES)
            for i in range(8):
                a = math.pi * i / 4
                dx, dz = math.cos(a) * 0.55, math.sin(a) * 0.55
                glVertex3f(0, 0, 0)
                glVertex3f(dx, 0.4, dz)
            # Top spike
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0.55, 0)
            glEnd()
            glLineWidth(1.0)

            # Glowing eyes — two bright red/orange dots
            glPointSize(5.0)
            glColor3f(1.0, 0.3, 0.0)
            glBegin(GL_POINTS)
            glVertex3f(0.12, 0.08, -0.25)
            glVertex3f(-0.12, 0.08, -0.25)
            glEnd()
            glPointSize(1.0)
            glEnable(GL_LIGHTING)

            # Detection range ring — pulsing
            glDisable(GL_LIGHTING)
            glColor4f(col[0], col[1], col[2], 0.06 + 0.04 * pulse)
            glLineWidth(1.5)
            glBegin(GL_LINE_LOOP)
            dr = pred.detection_range
            for i in range(32):
                a = 2 * math.pi * i / 32
                glVertex3f(math.cos(a)*dr, -0.3, math.sin(a)*dr)
            glEnd()
            glLineWidth(1.0)
            glEnable(GL_LIGHTING)
            glPopMatrix()

    def _draw_wildlife(self, sandbox: "Sandbox") -> None:
        """Draw passive wildlife — fish as small blue shapes, birds as black specks."""
        if not hasattr(sandbox, 'wildlife'):
            return
        glDisable(GL_LIGHTING)
        for creature in sandbox.wildlife:
            gx, gz = creature.position.x, creature.position.y
            ix = int(min(max(gx, 0), self.world_w - 1))
            iz = int(min(max(gz, 0), self.world_h - 1))
            gy = self._elev(ix, iz)
            if creature.kind == "fish":
                # Small blue diamond at water level
                glColor3f(0.2, 0.4, 0.9)
                wy = gy - 0.05
                glBegin(GL_TRIANGLES)
                glVertex3f(gx - 0.15, wy, gz)
                glVertex3f(gx + 0.25, wy, gz)
                glVertex3f(gx + 0.05, wy + 0.1, gz)
                glEnd()
            elif creature.kind == "bird":
                # V-shaped wing silhouette in the air
                by = gy + 2.5 + 0.3 * math.sin(self._frame * 0.1 + gx)
                glColor3f(0.15, 0.12, 0.10)
                glLineWidth(1.5)
                glBegin(GL_LINES)
                glVertex3f(gx - 0.3, by, gz)
                glVertex3f(gx, by + 0.1, gz)
                glVertex3f(gx, by + 0.1, gz)
                glVertex3f(gx + 0.3, by, gz)
                glEnd()
                glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def _draw_weather_particles(self, sandbox: "Sandbox") -> None:
        """Particle effects for weather."""
        weather = sandbox.weather.current_weather
        if weather == 0:  # clear
            return

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

        cx, cz = self.cam_target[0], self.cam_target[2]
        spread = 35.0
        t = self._frame

        if weather == 1 or weather == 2:  # rain / storm
            glColor4f(0.5, 0.6, 0.9, 0.25 if weather == 1 else 0.45)
            glLineWidth(1.5 if weather == 1 else 2.5)
            glBegin(GL_LINES)
            count = 150 if weather == 1 else 250
            for i in range(count):
                px = cx + math.sin(i * 7.13 + t * 0.01) * spread
                pz = cz + math.cos(i * 11.37 + t * 0.01) * spread
                py = ((i * 3.7 + t * 0.5) % 14.0) - 1.0
                length = 0.7 if weather == 1 else 1.2
                wind = 0.05 if weather == 1 else 0.15
                glVertex3f(px, py, pz)
                glVertex3f(px + wind, py - length, pz)
            glEnd()
            glLineWidth(1.0)

            # Storm lightning flashes
            if weather == 2 and (t % 200) < 3:
                glColor4f(0.8, 0.85, 1.0, 0.15)
                glBegin(GL_QUADS)
                glVertex2f(0, 0)
                glVertex2f(self.win_w, 0)
                glVertex2f(self.win_w, self.win_h)
                glVertex2f(0, self.win_h)
                glEnd()

        elif weather == 3:  # drought — heat shimmer / floating dust
            glColor4f(0.8, 0.65, 0.3, 0.12)
            glPointSize(2.0)
            glBegin(GL_POINTS)
            for i in range(60):
                px = cx + math.sin(i * 5.7 + t * 0.003) * spread
                pz = cz + math.cos(i * 8.3 + t * 0.003) * spread
                py = 0.5 + math.sin(i * 2.1 + t * 0.02) * 1.5
                glVertex3f(px, py, pz)
            glEnd()
            glPointSize(1.0)

        elif weather == 4:  # fog — floating wisps
            glColor4f(0.6, 0.6, 0.65, 0.04)
            for i in range(20):
                px = cx + math.sin(i * 4.7 + t * 0.002) * spread * 0.8
                pz = cz + math.cos(i * 6.3 + t * 0.002) * spread * 0.8
                py = 1.0 + math.sin(i * 1.3 + t * 0.01) * 2.0
                r = 2.0 + math.sin(i * 0.7) * 1.0
                glBegin(GL_TRIANGLE_FAN)
                glVertex3f(px, py, pz)
                for j in range(13):
                    a = 2 * math.pi * j / 12
                    glVertex3f(px + math.cos(a) * r, py + 0.1, pz + math.sin(a) * r)
                glEnd()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    # ─── HUD overlay ──────────────────────────────────────────

    def _draw_hud(self, agents: list["ConsciousAgent"], tick: int,
                  sandbox: "Sandbox") -> None:
        """Draw 2D HUD overlay on top of the 3D scene."""
        # Switch to 2D orthographic
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.win_w, self.win_h, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        # Top bar background
        glColor4f(0.0, 0.0, 0.0, 0.6)
        glBegin(GL_QUADS)
        glVertex2f(0, 0); glVertex2f(self.win_w, 0)
        glVertex2f(self.win_w, 68); glVertex2f(0, 68)
        glEnd()

        # Top bar text
        ws = sandbox.weather.get_summary()
        dn = "NIGHT" if sandbox.day_cycle.is_night else "DAY"
        preds = len(sandbox.predators.predators)
        alive = sum(1 for a in agents if a.alive)
        wildlife_n = len(sandbox.wildlife) if hasattr(sandbox, 'wildlife') else 0
        ts_label = f"  |  Speed: {self.time_scale}x" if self.time_scale > 1 else ""
        info = (f"Tick {tick:,}  |  {dn}  |  "
                f"{ws['season'].upper()} {ws['weather'].upper()}  |  "
                f"Temp: {ws['temperature']:.0%}  |  "
                f"Predators: {preds}  |  Wildlife: {wildlife_n}  |  "
                f"Crystals: {len(sandbox.crystals)}  |  "
                f"Alive: {alive}/{len(agents)}{ts_label}")
        self._render_text(info, 12, 8, (220, 225, 235), self._hud_font)

        # Civilisation info line
        civ = getattr(sandbox, 'civilization', None)
        if civ is not None:
            from genesis.cognition.civilization import EPOCH_NAMES, EPOCH_POP_CAP
            epoch_name = EPOCH_NAMES[int(civ.epoch)]
            pop_cap = EPOCH_POP_CAP[int(civ.epoch)]
            civ_info = (f"Epoch: {epoch_name}  |  "
                        f"Techs: {len(civ.discovered_techs)}  |  "
                        f"Structures: {len(civ.structures)}  |  "
                        f"Crops: {len(civ.crops)}  |  "
                        f"Pop: {alive}/{pop_cap}")
            self._render_text(civ_info, 12, 28, (180, 200, 140), self._hud_font)

        # Controls hint
        follow_str = " [ON]" if self._follow_cam else ""
        hint = (f"Drag: orbit | Scroll: zoom | Tab/RClick: focus | "
                f"R: reset | Space: cogmap | F: follow{follow_str} | "
                f"H: HUD | +/-: speed")
        self._render_text(hint, 12, 48, (120, 125, 140), self._hud_font)

        # FPS counter (top right)
        fps_str = f"FPS: {self._fps:.0f}"
        self._render_text(fps_str, self.win_w - 90, 8, (100, 105, 120),
                          self._hud_font)

        # Agent panels (right side)
        panel_x = self.win_w - 320
        panel_y = 78
        for agent in agents:
            panel_y = self._draw_agent_hud(agent, panel_x, panel_y, sandbox)

        # Floating labels above agents (collected during _draw_agents)
        if hasattr(self, '_floating_labels'):
            for lx, ly, label, lcol in self._floating_labels:
                self._render_text(label, lx - len(label) * 3, ly - 14, lcol,
                                  self._hud_font_label)
            self._floating_labels.clear()

        # Minimap (bottom left)
        self._draw_minimap(agents)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def _draw_agent_hud(self, agent: "ConsciousAgent", x: int, y: int,
                        sandbox: "Sandbox") -> int:
        """Draw a compact agent status panel. Returns next y."""
        col = AGENT_COLOURS[agent.agent_id % len(AGENT_COLOURS)]
        col_255 = tuple(int(c * 255) for c in col)

        # Panel background
        ph = 30 if not agent.alive else 190
        glColor4f(0.0, 0.0, 0.0, 0.55)
        glBegin(GL_QUADS)
        glVertex2f(x, y); glVertex2f(x + 310, y)
        glVertex2f(x + 310, y + ph); glVertex2f(x, y + ph)
        glEnd()
        # Accent stripe
        glColor4f(col[0], col[1], col[2], 0.8)
        glBegin(GL_QUADS)
        glVertex2f(x, y); glVertex2f(x + 3, y)
        glVertex2f(x + 3, y + ph); glVertex2f(x, y + ph)
        glEnd()

        if not agent.alive:
            self._render_text(f"AGENT {agent.agent_id} -- DEAD",
                              x + 10, y + 8, (180, 90, 90), self._hud_font_big)
            return y + ph + 6

        assess = agent.phi_calculator.get_consciousness_assessment(
            self_model_accuracy=agent.self_model.model_accuracy,
            attention_accuracy=agent.attention_schema.schema_accuracy,
            metacognitive_confidence=agent.inner_speech.confidence,
            binding_coherence=agent.binding.coherence,
            empowerment=agent.empowerment.empowerment,
            narrative_identity=agent.narrative.identity_strength,
            curiosity_level=agent.curiosity.curiosity_level,
        )
        phase = assess["phase"].split("--")[0].split("\u2014")[0].strip()
        comp = assess["composite_score"]

        # Header with ego badge
        ego_str = " *" if agent.self_model.has_ego else ""
        self._render_text(f"AGENT {agent.agent_id}{ego_str}", x + 10, y + 4, col_255,
                          self._hud_font_big)
        self._render_text(phase, x + 200, y + 6, (235, 175, 50), self._hud_font)

        # Dream indicator
        if agent.dream_engine.stats.is_dreaming:
            self._render_text("Zz", x + 180, y + 4, (180, 140, 255), self._hud_font)

        ly = y + 24
        e_pct = agent.body.energy / agent.config.agent.max_energy
        i_pct = agent.body.integrity / agent.config.agent.max_integrity
        self._draw_hud_bar(x + 10, ly, 140, "Energy", e_pct, (50, 210, 110))
        self._draw_hud_bar(x + 160, ly, 140, "Integrity", i_pct, (80, 150, 255))
        ly += 18

        self._render_text(
            f"Composite: {comp:.3f}  Phi: {assess['phi']:.4f}",
            x + 10, ly, (210, 210, 220), self._hud_font)
        ly += 16

        # Consciousness sub-metrics bars
        self._draw_hud_bar(x + 10, ly, 140, "Curiosity",
                           agent.curiosity.curiosity_level, (180, 110, 255))
        self._draw_hud_bar(x + 160, ly, 140, "Binding",
                           agent.binding.binding_strength, (80, 200, 220))
        ly += 16
        self._draw_hud_bar(x + 10, ly, 140, "Empower.",
                           agent.empowerment.empowerment, (255, 130, 80))
        self._draw_hud_bar(x + 160, ly, 140, "Identity",
                           agent.narrative.identity_strength, (160, 220, 255))
        ly += 18

        emo_s = agent.emotions.get_summary()
        mood_v = emo_s.get("mood_valence", 0.0)
        mood_col = (90, 255, 130) if mood_v > 0.05 else (
            (255, 90, 90) if mood_v < -0.05 else (160, 160, 170))
        self._render_text(
            f"Mood: {mood_v:+.2f}  Emo: {emo_s['dominant']}  "
            f"Goal: {agent.goal_system.get_summary().get('active_goal', '?')}",
            x + 10, ly, mood_col, self._hud_font)
        ly += 16

        # Attention
        attn = agent.attention_schema
        self._render_text(
            f"Attn: {attn.current_focus} ({attn.focus_duration}t)  "
            f"Acc: {attn.schema_accuracy:.0%}",
            x + 10, ly, (160, 180, 210), self._hud_font)
        ly += 16

        tool_s = agent.body.tools.get_summary()
        coop_s = agent.cooperation.get_summary()
        tool_str = ", ".join(tool_s["tools"]) if tool_s["tool_count"] > 0 else "none"
        self._render_text(
            f"Tools: {tool_str}  Coop: {coop_s['partners']}p",
            x + 10, ly, (160, 165, 180), self._hud_font)
        ly += 16

        lang_s = agent.communication.get_language_summary()
        vocab = lang_s.get("vocabulary_size", 0)
        named = lang_s.get("named_entities", 0)
        self._render_text(
            f"Vocab: {vocab}  Named: {named}  "
            f"Bonds: {emo_s.get('bonds', 0)}",
            x + 10, ly, (140, 145, 160), self._hud_font)
        ly += 16

        return ly + 8

    def _draw_hud_bar(self, x, y, w, label, fraction, col):
        """Draw a horizontal bar on the HUD."""
        fraction = max(0.0, min(1.0, fraction))
        # Background
        glColor4f(0.15, 0.15, 0.2, 0.8)
        glBegin(GL_QUADS)
        glVertex2f(x, y); glVertex2f(x + w, y)
        glVertex2f(x + w, y + 12); glVertex2f(x, y + 12)
        glEnd()
        # Fill
        fw = int(fraction * w)
        if fw > 0:
            glColor4f(col[0]/255, col[1]/255, col[2]/255, 0.85)
            glBegin(GL_QUADS)
            glVertex2f(x, y); glVertex2f(x + fw, y)
            glVertex2f(x + fw, y + 12); glVertex2f(x, y + 12)
            glEnd()
        self._render_text(f"{label}: {fraction:.0%}", x + 3, y, (220, 220, 230),
                          self._hud_font)

    def _render_text(self, text: str, x: int, y: int, colour, font) -> None:
        """Render text string onto the OpenGL viewport using Pygame fonts.
        Uses a cache to avoid re-rendering identical text each frame."""
        font_id = id(font)
        key = (text, colour, font_id)
        if key not in self._text_cache:
            surf = font.render(text, True, colour)
            data = pygame.image.tostring(surf, "RGBA", True)
            w, h = surf.get_size()
            self._text_cache[key] = (data, w, h)
        data, w, h = self._text_cache[key]
        glRasterPos2f(x, y + h)
        glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, data)

    def _flush_text_cache(self) -> None:
        """Periodically flush the text cache to avoid unbounded growth."""
        self._text_cache_frame += 1
        if self._text_cache_frame % 300 == 0:
            self._text_cache.clear()

    # ─── camera ───────────────────────────────────────────────

    def _setup_camera(self) -> None:
        """Set perspective projection and orbit camera view."""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(50.0, self.win_w / self.win_h, 0.5, 400.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Orbit camera position from spherical coords
        yaw_r = math.radians(self.cam_yaw)
        pitch_r = math.radians(self.cam_pitch)
        cam_x = self.cam_target[0] + self.cam_dist * math.cos(pitch_r) * math.sin(yaw_r)
        cam_y = self.cam_target[1] + self.cam_dist * math.sin(pitch_r)
        cam_z = self.cam_target[2] + self.cam_dist * math.cos(pitch_r) * math.cos(yaw_r)

        gluLookAt(cam_x, cam_y, cam_z,
                  self.cam_target[0], self.cam_target[1], self.cam_target[2],
                  0, 1, 0)

    # ─── main render ──────────────────────────────────────────

    def render(self, sandbox: "Sandbox", agents: list["ConsciousAgent"],
               tick: int) -> None:
        """Render one frame of the 3D scene."""
        self._frame += 1
        self._flush_text_cache()

        # Sky colour
        light = sandbox.day_cycle.light_level
        weather = sandbox.weather.current_weather
        sky = self._get_sky_colour(light, weather)
        glClearColor(*sky, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Fog colour matches sky
        glFogfv(GL_FOG_COLOR, [*sky, 1.0])
        # Denser fog for fog weather
        if weather == 4:
            glFogf(GL_FOG_START, 60.0)
            glFogf(GL_FOG_END, 160.0)
        else:
            glFogf(GL_FOG_START, 150.0)
            glFogf(GL_FOG_END, 400.0)

        # Store for right-click picking
        self._last_agents = agents

        # Camera follow focused agent
        alive_agents = [a for a in agents if a.alive]
        if alive_agents:
            f_idx = self._focused % len(agents)
            if agents[f_idx].alive:
                target_agent = agents[f_idx]
            else:
                target_agent = alive_agents[0]
            tx, tz = target_agent.body.position.x, target_agent.body.position.y
            ty = self._elev(int(min(max(tx, 0), self.world_w-1)),
                            int(min(max(tz, 0), self.world_h-1)))
            # Tighter tracking when follow-cam enabled
            alpha = 0.5 if self._follow_cam else 0.08
            for i, v in enumerate([tx, ty, tz]):
                self.cam_target[i] += (v - self.cam_target[i]) * alpha

        self._setup_camera()
        self._set_lighting(light, weather)

        # Celestial bodies
        self._draw_stars(light)
        self._draw_sun_moon(light, sandbox.day_cycle.phase)

        # Draw world
        glCallList(self._terrain_list)
        glCallList(self._obstacle_list)

        # Animated water with blending
        glDepthMask(GL_FALSE)
        self._draw_water(sandbox, light)
        glDepthMask(GL_TRUE)

        # Entities
        self._draw_shelters(sandbox)
        self._draw_structures(sandbox)
        self._draw_crops(sandbox)
        self._draw_crystals(sandbox)
        self._draw_berry_bushes(sandbox)
        self._draw_fungi(sandbox)
        self._draw_ruins(sandbox)
        self._draw_predators(sandbox)
        self._draw_wildlife(sandbox)
        self._draw_agents(agents)
        self._draw_action_sparkles(agents)
        self._draw_weather_particles(sandbox)
        self._draw_ambient_particles(light, sandbox.weather.current_weather)

        # 3D cognitive map overlay
        if self._show_cogmap and agents:
            self._draw_cogmap_3d(agents[self._focused % len(agents)])

        # HUD
        if self._show_hud:
            self._draw_hud(agents, tick, sandbox)

        # Track FPS
        self._fps = self.clock.get_fps()

        pygame.display.flip()
        self.clock.tick(60)

    def _draw_water(self, sandbox: "Sandbox", light_level: float) -> None:
        """Draw animated water with wave vertex displacement."""
        t = self._frame * 0.04
        wave_tint = 0.03 * math.sin(t * 0.5)
        base_alpha = WATER_COL[3] * (0.6 + 0.4 * light_level)
        step = getattr(self, '_terrain_step', 1)

        # Cache water cells on first call
        if not hasattr(self, '_water_cells'):
            self._water_cells = []
            bm = sandbox.biome_map
            w, h = self.world_w, self.world_h
            for y in range(0, h, step):
                for x in range(0, w, step):
                    if bm.biome_at(x, y) in (BIOME_OCEAN, BIOME_WETLANDS):
                        self._water_cells.append((x, y))

        glNormal3f(0, 1, 0)
        glBegin(GL_QUADS)
        water_y_base = -0.3
        for (x, y) in self._water_cells:
            for vx, vz in ((x, y), (x+step, y), (x+step, y+step), (x, y+step)):
                wave = 0.06 * math.sin(vx * 0.4 + t) + \
                       0.04 * math.sin(vz * 0.5 - t * 0.7)
                vy = water_y_base + wave
                brightness = 0.5 + wave * 3.0
                glColor4f(
                    WATER_COL[0] + wave_tint + brightness * 0.03,
                    WATER_COL[1] + wave_tint * 0.5 + brightness * 0.04,
                    WATER_COL[2] - wave_tint + brightness * 0.02,
                    base_alpha
                )
                glVertex3f(vx, vy, vz)
        glEnd()

        # Draw rivers (use pre-built display list, shimmer only near camera)
        if sandbox.rivers:
            glCallList(self._river_list)

        # Specular highlights on water surface (sun sparkle)
        if light_level > 0.4:
            glDisable(GL_LIGHTING)
            glPointSize(2.0)
            glBegin(GL_POINTS)
            cx, cz_c = self.cam_target[0], self.cam_target[2]
            w, h = self.world_w, self.world_h
            bm = sandbox.biome_map
            for i in range(40):
                sx_w = cx + math.sin(i * 13.7 + t * 0.3) * 30
                sz_w = cz_c + math.cos(i * 17.1 + t * 0.2) * 30
                bx = int(min(max(sx_w, 0), w - 1))
                bz = int(min(max(sz_w, 0), h - 1))
                if bm.biome_at(bx, bz) in (BIOME_OCEAN, BIOME_WETLANDS):
                    sparkle = max(0.0, math.sin(i * 7.3 + t * 1.5))
                    if sparkle > 0.6:
                        glColor4f(1.0, 1.0, 0.95, (sparkle - 0.6) * light_level)
                        glVertex3f(sx_w, water_y_base + 0.1, sz_w)
            glEnd()
            glPointSize(1.0)
            glEnable(GL_LIGHTING)

    # ─── event handling ───────────────────────────────────────

    def handle_events(self) -> bool:
        """Process input events. Returns False to quit."""
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    return False
                elif event.key == K_TAB:
                    self._focused = (self._focused + 1) % self.num_agents
                elif event.key == K_m or event.key == K_SPACE:
                    self._show_cogmap = not self._show_cogmap
                elif event.key == K_r:
                    # Reset camera
                    self.cam_yaw = -45.0
                    self.cam_pitch = 55.0
                    self.cam_dist = 120.0
                elif event.key == K_f:
                    self._follow_cam = not self._follow_cam
                elif event.key == K_h:
                    self._show_hud = not self._show_hud
                elif event.key == K_PLUS or event.key == K_EQUALS:
                    self.time_scale = min(8, self.time_scale * 2)
                elif event.key == K_MINUS:
                    self.time_scale = max(1, self.time_scale // 2)
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # left click — drag
                    self._dragging = True
                    self._last_mouse = event.pos
                elif event.button == 3:  # right click — pick agent
                    self._pick_agent(event.pos)
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self._dragging = False
            elif event.type == MOUSEMOTION:
                if self._dragging:
                    dx = event.pos[0] - self._last_mouse[0]
                    dy = event.pos[1] - self._last_mouse[1]
                    self.cam_yaw += dx * 0.4
                    self.cam_pitch = max(10.0, min(85.0,
                                                    self.cam_pitch + dy * 0.3))
                    self._last_mouse = event.pos
            elif event.type == MOUSEWHEEL:
                self.cam_dist = max(20.0, min(300.0,
                                               self.cam_dist - event.y * 5.0))
        return True

    def _pick_agent(self, screen_pos: tuple[int, int]) -> None:
        """Select the agent closest to the screen click position.

        Uses a 2D projection approximation: projects each agent's world
        position to screen space via the current modelview/projection
        matrices and picks the nearest one.
        """
        try:
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            viewport = glGetIntegerv(GL_VIEWPORT)
        except Exception:
            return

        mx, my = screen_pos
        my = viewport[3] - my  # flip Y

        best_dist = float("inf")
        best_idx = self._focused
        for i, agent in enumerate(self._last_agents):
            gx = agent.body.position.x
            gz = agent.body.position.y
            ix = int(min(max(gx, 0), self.world_w - 1))
            iz = int(min(max(gz, 0), self.world_h - 1))
            gy = self._elev(ix, iz) + 0.5

            try:
                sx, sy, _sz = gluProject(gx, gy, gz, modelview, projection, viewport)
                d = (sx - mx) ** 2 + (sy - my) ** 2
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            except Exception:
                continue

        self._focused = best_idx

    def quit(self) -> None:
        pygame.quit()
