# =============================================================================
# Tri-Quarter Framework: Radial Dual Triangular Lattice Graphs with Exact
# Bijective Dualities and Equivariant Encodings via the Inversive Hexagonal
# Dihedral Symmetry Group T_24
#
# Simulation 01: Visualizing Random Connections in the Radial Dual Triangular
# Lattice Graph
#
# Author: Nathan O. Schmidt
# Affiliation: Cold Hammer Research & Development LLC, Eagle, Idaho, USA
# Email: nate.o.schmidt@coldhammer.net
# Date: September 28, 2025
#
# Description:
# This Python script dynamically visualizes random adjacent paths in the outer
# zone of the radial dual triangular lattice graph Lambda_r (with admissible
# inversion radius r = sqrt(1) and truncation radius R = 4), with their inversions
# mirrored in the inner zone via the circle inversion map iota_r. The script
# animates the connections, updating every 5 seconds, to demonstrate the Escher
# reflective duality in action.
#
# Requirements:
# - Python 3.x
# - Pygame library (install via: pip install pygame)
#
# Usage:
# Run the script with: python simulation_01_visualize_random_connections.py
#
# Source code is freely available at:
# https://github.com/nathanoschmidt/tri-quarter-toolbox/
# (MIT License; see repository LICENSE for details)
#
# =============================================================================

import pygame
import sys
import math
import random

# Initialize Pygame library
pygame.init()

# Set screen dimensions and window title
WIDTH, HEIGHT = 1200, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))  # Create the display window
pygame.display.set_caption(
    "Tri-Quarter Framework Simulation: "
    "Visualizing Random Connections in the "
    "Radial Dual Triangular Lattice Graph"
)

# Define colors used in the figure
WHITE = (255, 255, 255)  # Background
BLACK = (0, 0, 0)        # Axes and text
GRAY = (128, 128, 128)   # Dotted lines
RED = (255, 0, 0)        # Outer zone vertices and paths
GREEN = (0, 180, 0)      # Boundary zone vertices and circle
BLUE = (0, 0, 255)       # Inner zone vertices and paths

# Sector colors with opacity for wedges
color0 = (255, 0, 0)
color1 = (255, 255, 0)
color2 = (0, 255, 0)
color3 = (2, 192, 230)
color4 = (2, 67, 230)
color5 = (188, 2, 230)
sector_colors = [color0, color1, color2, color3, color4, color5]

# Fonts for labels
font_str = "Arial"
font_large = pygame.font.SysFont(font_str, 48)
font_largedium = pygame.font.SysFont(font_str, 36)
font_medium = pygame.font.SysFont(font_str, 28)
font_small = pygame.font.SysFont(font_str, 24)
font_tiny = pygame.font.SysFont(font_str, 22)

# Center of the screen for origin
center_x, center_y = WIDTH // 2, HEIGHT // 2

# Scaling factor reduced to make drawing smaller and fit better: 120 -> 100
scale = 100  # Adjust this to zoom in/out

# Define inversion radius r = sqrt(1), r_sq = 1 for boundary
# (aligned with sector boundaries)
r = math.sqrt(1)
r_sq = 1

# Truncation radius R = 4 for generating finite vertices
R = 4

# Basis vectors for triangular lattice
omega1 = (1, 0)
omega2 = (0.5, math.sqrt(3) / 2)

# Deltas for finding nearest neighbors in lattice coordinates
deltas = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, -1), (-1, 1)]

# Function to compute Cartesian position from lattice coordinates (m, n)
def compute_pos(m, n):
    x = (m * omega1[0]) + (n * omega2[0])
    y = (m * omega1[1]) + (n * omega2[1])
    return x, y

# Function to compute squared norm (integer) for exact comparisons
def compute_norm_sq(m, n):
    return (m * m) + (m * n) + (n * n)

# Generate lattice vertices for outer zone and boundary zone
outer_vertices = []  # List of (m, n) for outer vertices
boundary_vertices = []  # List of positions for boundary vertices
max_intnorm = 0  # Track max squared norm for scaling inner radii
min_intnorm_outer = float('inf')  # Track min squared norm for outer
for m in range(-20, 21):  # Range large enough to cover R=4
    for n in range(-20, 21):
        if m == 0 and n == 0:
            continue  # Exclude origin
        intnorm = compute_norm_sq(m, n)
        norm = math.sqrt(intnorm)
        if norm > R:
            continue  # Truncate beyond R
        if intnorm == r_sq:
            pos = compute_pos(m, n)
            boundary_vertices.append(
                {'pos': pos, 'angle': math.atan2(pos[1], pos[0])}
            )
        elif intnorm > r_sq:
            outer_vertices.append((m, n))  # Store outer lattice coords
            max_intnorm = max(max_intnorm, intnorm)
            min_intnorm_outer = min(min_intnorm_outer, intnorm)

# Sort boundary vertices by angle for hexagon drawing
boundary_vertices.sort(key = lambda p: p['angle'])

# Generate inner vertices by inverting outer vertices
inner_vertices = []  # List of inner vertices with positions and radii
# Min distance for inner scaling (adjusted for r_sq=1)
min_dist_prime = r_sq / math.sqrt(max_intnorm)
# Max distance for inner scaling (adjusted for r_sq=1)
max_dist_prime = r_sq / math.sqrt(min_intnorm_outer)

for m, n in outer_vertices:
    pos = compute_pos(m, n)
    intnorm = compute_norm_sq(m, n)
    xprime = r_sq * pos[0] / intnorm  # Invert x (r_sq=1)
    yprime = r_sq * pos[1] / intnorm  # Invert y (r_sq=1)
    dist_prime = math.sqrt(xprime**2 + yprime**2)
    # Scale radius based on distance (smaller near origin, adjusted for new norms)
    rad_blue = 1 + (dist_prime - min_dist_prime) / (
        max_dist_prime - min_dist_prime
    ) if max_dist_prime > min_dist_prime else 1
    rad_blue *= 0.75
    inner_vertices.append(
        {'pos': (xprime, yprime), 'rad': rad_blue, 'orig_mn': (m,n)}
    )

# Build adjacency list for outer graph
outer_set = set(outer_vertices)  # Set for quick lookup
neighbors = {}  # Dict of neighbors for each outer vertex
for v in outer_vertices:
    neigh = []
    m, n = v
    for dm, dn in deltas:
        mm, nn = m + dm, n + dn
        if (mm, nn) in outer_set:
            neigh.append((mm, nn))  # Add adjacent if in outer
    neighbors[v] = neigh

# Convert Cartesian to screen coordinates (invert y for Pygame)
def to_screen(x, y):
    return int(center_x + x * scale), int(center_y - y * scale)  # y inverted

# Dashed line
def draw_dashed_line(start, end, color, dash_length = 10, thickness = 2):
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    dist = math.sqrt(dx**2 + dy**2)
    if dist == 0:
        return
    ux = dx / dist
    uy = dy / dist
    current_x, current_y = sx, sy
    while dist > 0:
        step = min(dash_length, dist)
        next_x = current_x + (ux * step)
        next_y = current_y + (uy * step)
        pygame.draw.line(screen, color, (current_x, current_y),
                         (next_x, next_y), thickness)
        current_x = next_x + (ux * dash_length)
        current_y = next_y + (uy * dash_length)
        dist -= 2 * dash_length

# Function to draw dashed circle (approximated with segments)
def draw_dashed_circle(center, radius, color, dash_length = 10):
    num_segments = 100  # Number of segments for smooth circle
    angle_step = 2 * math.pi / num_segments
    for i in range(num_segments):
        if i % 2 == 0:  # Draw every other segment for dash effect
            theta1 = i * angle_step
            theta2 = (i + 1) * angle_step
            x1 = center[0] + radius * math.cos(theta1)
            y1 = center[1] + radius * math.sin(theta1)
            x2 = center[0] + radius * math.cos(theta2)
            y2 = center[1] + radius * math.sin(theta2)
            pygame.draw.line(screen, color, (x1, y1), (x2, y2), 2)

# Draw the background elements: sectors, rays, axes, labels
def draw_background():
    screen.fill(WHITE)  # Clear screen with white

    # Draw transparent sector wedges
    for k in range(6):
        points = [(center_x, center_y)]  # Start from center
        wedge_radius = 10 * scale
         # Incremental points for polygon
        for angle in range(k * 60, ((k + 1) * 60) + 1, 5):
            rad = math.radians(angle)
            px = center_x + (wedge_radius * math.cos(rad))
            py = center_y - (wedge_radius * math.sin(rad))  # Inverted y
            points.append((px, py))
        col = sector_colors[k]
        col_alpha = (*col, 51)  # 0.2 opacity (255*0.2=51)
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)  # Alpha surface
        pygame.draw.polygon(s, col_alpha, points)
        screen.blit(s, (0, 0))

    # Draw dashed radial rays
    for k in range(6):
        angle = math.radians(k * 60)
        rad_len = 4 if k in [0, 3] else 4.6  # Vary length as in figure
        end_x = rad_len * math.cos(angle)
        end_y = rad_len * math.sin(angle)
        start_screen = to_screen(0, 0)
        end_screen = to_screen(end_x, end_y)
        draw_dashed_line(start_screen, end_screen, BLACK, dash_length = 10,
                         thickness = 2)

    # Draw real and imaginary axes
    pygame.draw.line(screen, BLACK, to_screen(-4.2, 0), to_screen(4.2, 0), 3)
    pygame.draw.line(screen, BLACK, to_screen(0,-4), to_screen(0, 4), 3)

    # Add little black arrows at endpoints
    arrow_size = 10
    # Real axis right
    rx, ry = to_screen(4.2, 0)
    arrow_points = [(rx, ry - arrow_size//2), (rx + arrow_size, ry),
                    (rx, ry + arrow_size//2)]
    pygame.draw.polygon(screen, BLACK, arrow_points)
    # Real axis left
    lx, ly = to_screen(-4.2, 0)
    arrow_points = [(lx, ly - arrow_size//2), (lx - arrow_size, ly),
                    (lx, ly + arrow_size//2)]
    pygame.draw.polygon(screen, BLACK, arrow_points)
    # Imag axis up (positive imag)
    ux, uy = to_screen(0, 3.9)
    arrow_points = [(ux - arrow_size//2, uy), (ux, uy - arrow_size),
                    (ux + arrow_size//2, uy)]
    pygame.draw.polygon(screen, BLACK, arrow_points)
    # Imag axis down (negative imag)
    dx, dy = to_screen(0, -3.9)
    arrow_points = [(dx - arrow_size//2, dy), (dx, dy + arrow_size),
                    (dx + arrow_size//2, dy)]
    pygame.draw.polygon(screen, BLACK, arrow_points)

    # Draw white box with black border in top right for parameters
    box_x, box_y = to_screen(2.2, 4.57)
    box_width, box_height = 390, 100
    pygame.draw.rect(screen, WHITE, (box_x, box_y, box_width, box_height))
    pygame.draw.rect(screen, BLACK, (box_x, box_y, box_width, box_height), 1)

    # Draw truncation radius parameter value
    trunc_radius_x, trunc_radius_y = 5.0, 4.4
    text = font_medium.render('R = 4', True, BLACK)
    screen.blit(text, to_screen(trunc_radius_x, trunc_radius_y))

    # Draw inversion radius parameter value
    inver_radius_x, inver_radius_y = 5.08, 4.1
    text = font_medium.render('r = \u221A' + str(r_sq), True, BLACK)
    screen.blit(text, to_screen(inver_radius_x, inver_radius_y))

    # Outer zone vertex
    red_zone_vertex_x, red_zone_vertex_y = 2.3, 4.5
    text = font_medium.render('\u25CF Outer Zone', True, RED)
    screen.blit(text, to_screen(red_zone_vertex_x, red_zone_vertex_y))

    # Boundary zone vertex
    green_zone_vertex_x, green_zone_vertex_y = 2.3, 4.2
    text = font_medium.render('\u25CF Boundary Zone', True, GREEN)
    screen.blit(text, to_screen(green_zone_vertex_x, green_zone_vertex_y))

    # Inner zone vertex
    blue_zone_vertex_x, blue_zone_vertex_y = 2.3, 3.9
    text = font_medium.render('\u25CF Inner Zone', True, BLUE)
    screen.blit(text, to_screen(blue_zone_vertex_x, blue_zone_vertex_y))

    # Draw dashed boundary circle
    circle_center = to_screen(0, 0)
    circle_radius = int(r * scale)
    draw_dashed_circle(circle_center, circle_radius, GREEN)

    # Dashed green hexagon for boundary
    if boundary_vertices:
        for i in range(len(boundary_vertices)):
            pos1 = boundary_vertices[i]['pos']
            pos2 = boundary_vertices[(i + 1) % len(boundary_vertices)]['pos']
            p1 = to_screen(*pos1)
            p2 = to_screen(*pos2)
            draw_dashed_line(p1, p2, GREEN, dash_length = 5, thickness = 2)

    # Render zone labels with Unicode and subscripts
    # Inner zone Lambda_{-,r}
    pos_x, pos_y = to_screen(1.1, 0.5)
    text_main = font_largedium.render('\u039B', True, BLUE)
    screen.blit(text_main, (pos_x, pos_y))
    text_sub = font_tiny.render('-,\u221A' + str(r_sq), True, BLUE)
    screen.blit(text_sub, (pos_x + text_main.get_width(), pos_y + 20))
    text_sup = font_tiny.render('4', True, BLUE)
    screen.blit(text_sup, (pos_x + text_main.get_width(), pos_y + 1))
    # Draw inner zone pointer line segment
    # (because Lambda_{-,r} is too big to fit inside the circle)
    ptr_line_start = (pos_x, pos_y + 20)
    ptr_line_end = (pos_x - 50, pos_y + 20)
    draw_dashed_line(ptr_line_start, ptr_line_end, BLUE, dash_length = 5,
                     thickness = 2)

    # Boundary zone Lambda_{T,r}
    pos_x, pos_y = to_screen(0.9, 0.9)
    text_main = font_largedium.render('\u039B', True, GREEN)
    screen.blit(text_main, (pos_x, pos_y))
    text_sub = font_tiny.render('T,\u221A' + str(r_sq), True, GREEN)
    screen.blit(text_sub, (pos_x + text_main.get_width(), pos_y + 20))
    text_sup = font_tiny.render('4', True, GREEN)
    screen.blit(text_sup, (pos_x + text_main.get_width(), pos_y + 1))

    # Outer zone Lambda_{+,r}
    pos_x, pos_y = to_screen(1.5, 1.5)
    text_main = font_largedium.render('\u039B', True, RED)
    screen.blit(text_main, (pos_x, pos_y))
    text_sub = font_tiny.render('+,\u221A' + str(r_sq), True, RED)
    screen.blit(text_sub, (pos_x + text_main.get_width(), pos_y + 20))
    text_sup = font_tiny.render('4', True, RED)
    screen.blit(text_sup, (pos_x + text_main.get_width(), pos_y + 1))

    # Render angular sector labels
    ang_sect_radius = 4.3
    for k in [0, 2, 3, 5]:
        angle = (k * 60) + 30
        px, py = to_screen(
            ang_sect_radius * math.cos(math.radians(angle)),
            ang_sect_radius * math.sin(math.radians(angle))
        )
        text_main = font_large.render('S', True, BLACK)
        screen.blit(text_main, (px - 20, py - 20))
        text_sub = font_tiny.render(str(k), True, BLACK)
        screen.blit(text_sub, (px + 5, py + 17))

    # Special position for angular sector S1 to avoid overlap
    ang_sect_radius -= 0.4
    px, py = to_screen(
        ang_sect_radius * math.cos(math.radians(83)),
        ang_sect_radius * math.sin(math.radians(83))
    )
    text_main = font_large.render('S', True, BLACK)
    screen.blit(text_main, (px - 20, py - 20))
    text_sub = font_tiny.render('1', True, BLACK)
    screen.blit(text_sub, (px + 5, py + 17))

    # Special position for angular sector S4 to avoid overlap
    px, py = to_screen(
        ang_sect_radius * math.cos(math.radians(277)),
        ang_sect_radius * math.sin(math.radians(277))
    )
    py -= 20
    text_main = font_large.render('S', True, BLACK)
    screen.blit(text_main, (px - 20, py - 20))
    text_sub = font_tiny.render('4', True, BLACK)
    screen.blit(text_sub, (px + 5, py + 17))

    # Render quadrant phase pair labels
    q1_x, q1_y = 2.6, 3.5
    text = font_medium.render('Quadrant I: (0, \u03C0/2)', True, BLACK)
    screen.blit(text, to_screen(q1_x, q1_y))
    text_phi = font_tiny.render('\u03C6', True, BLACK)
    screen.blit(
        text_phi,
        (to_screen(q1_x, q1_y)[0] + 270, to_screen(q1_x, q1_y)[1] + 10)
    )

    q2_x, q2_y = -5.4, 3.5
    text = font_medium.render('Quadrant II: (\u03C0, \u03C0/2)', True, BLACK)
    screen.blit(text, to_screen(q2_x, q2_y))
    text_phi = font_tiny.render('\u03C6', True, BLACK)
    screen.blit(
        text_phi,
        (to_screen(q2_x, q2_y)[0] + 276, to_screen(q2_x, q2_y)[1] + 10)
    )

    q3_x, q3_y = -5.4, -3.1
    text = font_medium.render('Quadrant III: (\u03C0, 3\u03C0/2)', True, BLACK)
    screen.blit(text, to_screen(q3_x, q3_y))
    text_phi = font_tiny.render('\u03C6', True, BLACK)
    screen.blit(
        text_phi,
        (to_screen(q3_x, q3_y)[0] + 302, to_screen(q3_x, q3_y)[1] + 10)
    )

    q4_x, q4_y = 2.2, -3.1
    text = font_medium.render('Quadrant IV: (0, 3\u03C0/2)', True, BLACK)
    screen.blit(text, to_screen(q4_x, q4_y))
    text_phi = font_tiny.render('\u03C6', True, BLACK)
    screen.blit(
        text_phi,
        (to_screen(q4_x, q4_y)[0] + 307, to_screen(q4_x, q4_y)[1] + 10)
    )

    # Render axis phase pair labels
    east_x, east_y = 4.45, 0.15
    text = font_small.render('East: (0, 0)', True, BLACK)
    screen.blit(text, to_screen(east_x, east_y))
    text_phi = font_tiny.render('\u03C6', True, BLACK)
    screen.blit(
        text_phi,
        (to_screen(east_x, east_y)[0] + 132, to_screen(east_x, east_y)[1] + 10)
    )

    north_x, north_y = -1, 4.4
    text = font_small.render('North: (\u03C0/2, \u03C0/2)', True, BLACK)
    screen.blit(text, to_screen(north_x, north_y))
    text_phi = font_tiny.render('\u03C6', True, BLACK)
    screen.blit(
        text_phi,
        (to_screen(north_x, north_y)[0] + 190, to_screen(north_x, north_y)[1] + 10)
    )

    west_x, west_y = -5.9, 0.15
    text = font_small.render('West: (\u03C0, 0)', True, BLACK)
    screen.blit(text, to_screen(west_x, west_y))
    text_phi = font_tiny.render('\u03C6', True, BLACK)
    screen.blit(
        text_phi,
        (to_screen(west_x, west_y)[0] + 138, to_screen(west_x, west_y)[1] + 10)
    )

    south_x, south_y = -1.1, -4.05
    text = font_small.render('South: (\u03C0/2, 3\u03C0/2)', True, BLACK)
    screen.blit(text, to_screen(south_x, south_y))
    text_phi = font_tiny.render('\u03C6', True, BLACK)
    screen.blit(
        text_phi,
        (to_screen(south_x, south_y)[0] + 208, to_screen(south_x, south_y)[1] + 10)
    )

    # Render axis labels
    text = font_large.render('\u211D', True, BLACK)
    screen.blit(text, to_screen(3.5, 0.6))
    text = font_large.render('\U0001D540', True, BLACK)
    screen.blit(text, to_screen(-0.3, 4))

def draw_vertices():
    for p in boundary_vertices:
        px, py = to_screen(*p['pos'])
        pygame.draw.circle(screen, GREEN, (px, py), 8)

    for m, n in outer_vertices:
        pos = compute_pos(m, n)
        px, py = to_screen(*pos)
        pygame.draw.circle(screen, RED, (px, py), 8)

    for p in inner_vertices:
        px, py = to_screen(*p['pos'])
        rad = int(p['rad'] * 4)
        pygame.draw.circle(screen, BLUE, (px, py), rad)

def select_random_path():
    num_verts = random.randint(3, 5)
    start = random.choice(outer_vertices)
    path = [start]
    visited = set([start])
    while len(path) < num_verts:
        curr = path[-1]
        avail = [n for n in neighbors[curr] if n not in visited]
        if not avail:
            break
        next_v = random.choice(avail)
        path.append(next_v)
        visited.add(next_v)
    return path

def get_pos(mn):
    return compute_pos(*mn)

def get_inv_pos(mn):
    m, n = mn
    intnorm = compute_norm_sq(m, n)
    pos = get_pos(mn)
    xprime = r_sq * pos[0] / intnorm
    yprime = r_sq * pos[1] / intnorm
    return xprime, yprime

def draw_selected_path(path):
    for i in range(len(path) - 1):
        pos1 = get_pos(path[i])
        pos2 = get_pos(path[i + 1])
        p1 = to_screen(*pos1)
        p2 = to_screen(*pos2)
        pygame.draw.line(screen, RED, p1, p2, 4)

    for i in range(len(path) - 1):
        pos1 = get_inv_pos(path[i])
        pos2 = get_inv_pos(path[i + 1])
        p1 = to_screen(*pos1)
        p2 = to_screen(*pos2)
        pygame.draw.line(screen, BLUE, p1, p2, 3)

    for v in path:
        pos = get_pos(v)
        inv = get_inv_pos(v)
        p1 = to_screen(*pos)
        p2 = to_screen(*inv)
        draw_dashed_line(p1, p2, GRAY, dash_length = 5)

clock = pygame.time.Clock()
current_path = select_random_path()
timer = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    dt = clock.tick(60) / 1000.0
    timer += dt

    if timer >= 5:
        current_path = select_random_path()
        timer = 0

    draw_background()
    draw_vertices()
    draw_selected_path(current_path)
    pygame.display.flip()
