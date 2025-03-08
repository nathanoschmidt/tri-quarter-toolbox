# Tri-Quarter Theorem Animation
# Version: 1.0
# Authors: Nathan O. Schmidt and Grok
# Created Date: March 7, 2025
# Modified Date: March 7, 2025
# License: MIT (see LICENSE file)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Force TkAgg backend for Fedora compatibility
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# Constants (easily modifiable)
RADIUS = 1.0          # Unit circle radius (||x|| = 1 for T)
SPEED = 0.05          # Angular speed (radians per frame)
WINDOW_SIZE = 2.0     # Axis limits (±2 times radius)

# Colors (customizable)
BG_COLOR = 'black'    # Initial background color
AXIS_COLOR = 'white'  # Initial axis/text color
CIRCLE_COLOR = 'green'
POINT_COLOR = 'white' # Initial point color
ORIGIN_VECTOR_COLOR = 'orange'
REAL_VECTOR_COLOR = 'red'
IMAG_VECTOR_COLOR = 'blue'
X_LABEL_COLOR = 'purple'  # Default for black background

# Labels (using LaTeX proof terminology)
ZONE_LABELS = {
    'X_minus': {'text': 'X_{-}', 'pos': (0.5, 0.5), 'color': X_LABEL_COLOR},
    'X_plus': {'text': 'X_{+}', 'pos': (1.2, 1.2), 'color': X_LABEL_COLOR}
}
BOUNDARY_POINTS = {
    0: {'text': '0 rad', 'pos': (1.18, 0.05)},
    np.pi/2: {'text': 'π/2 rad', 'pos': (0.2, 1.05)},
    np.pi: {'text': 'π rad', 'pos': (-1.18, 0.05)},
    3*np.pi/2: {'text': '3π/2 rad', 'pos': (0.23, -1.13)}
}

# Setup figure and axes with space for buttons
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.15)  # Space at bottom for buttons
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# Set the window title
fig.canvas.manager.set_window_title('Tri-Quarter Theorem Animation v1.0 - By Nathan O. Schmidt and Grok')

# Configure axes
real_axis = ax.axhline(0, color=AXIS_COLOR, lw=1, label='Real axis')
imag_axis = ax.axvline(0, color=AXIS_COLOR, lw=1, label='Imaginary axis')
ax.set_xlim(-WINDOW_SIZE, WINDOW_SIZE)
ax.set_ylim(-WINDOW_SIZE, WINDOW_SIZE)
ax.set_xlabel('')  # Remove default x-label
ax.set_ylabel('')  # Remove default y-label
ax.tick_params(colors=AXIS_COLOR)

# Add corrected axis labels
axis_labels = [
    ax.text(WINDOW_SIZE - 0.1, -0.1, 'x_R (+)', color=AXIS_COLOR, ha='right', va='top'),
    ax.text(-WINDOW_SIZE + 0.1, -0.1, 'x_R (-)', color=AXIS_COLOR, ha='left', va='top'),
    ax.text(0.1, WINDOW_SIZE - 0.1, 'x_I (+)', color=AXIS_COLOR, ha='left', va='top'),
    ax.text(0.1, -WINDOW_SIZE + 0.1, 'x_I (-)', color=AXIS_COLOR, ha='left', va='bottom')
]

# Draw unit circle (T)
circle = plt.Circle((0, 0), RADIUS, color=CIRCLE_COLOR, fill=False, lw=2)
ax.add_patch(circle)

# Initialize plot elements
point, = ax.plot([], [], 'o', color=POINT_COLOR, label='Point on T', markersize=8)
vec_x, = ax.plot([], [], color=ORIGIN_VECTOR_COLOR, lw=2, label='||vec{x}||')
vec_x_R, = ax.plot([], [], color=REAL_VECTOR_COLOR, lw=2, label='vec{x}_R')
vec_x_I, = ax.plot([], [], color=IMAG_VECTOR_COLOR, lw=2, label='vec{x}_I')

# Text labels for vectors
vec_x_label = ax.text(0, 0, '', color=ORIGIN_VECTOR_COLOR)
vec_x_R_label = ax.text(0, 0, '', color=REAL_VECTOR_COLOR)
vec_x_I_label = ax.text(0, 0, '', color=IMAG_VECTOR_COLOR)

# Phase angle label (orange, right-justified)
phase_label = ax.text(WINDOW_SIZE - 0.1, WINDOW_SIZE - 0.2, '', color=ORIGIN_VECTOR_COLOR, fontsize=12, ha='right')

# Add zone labels
zone_texts = {}
for zone, info in ZONE_LABELS.items():
    zone_texts[zone] = ax.text(info['pos'][0], info['pos'][1], info['text'],
                               color=info['color'], fontsize=12, ha='center')

# Add boundary point labels and green dots
boundary_texts = {}
for angle, info in BOUNDARY_POINTS.items():
    x, y = RADIUS * np.cos(angle), RADIUS * np.sin(angle)
    boundary_texts[angle] = ax.text(info['pos'][0], info['pos'][1], info['text'],
                                    color=CIRCLE_COLOR, fontsize=10, ha='center')
    ax.plot([x], [y], 'o', color=CIRCLE_COLOR, markersize=8)

# Add green "T" label top-right of unit circle
extra_t_label = ax.text(RADIUS * np.cos(np.pi/4) + 0.1, RADIUS * np.sin(np.pi/4) + 0.1, 'T',
                        color=CIRCLE_COLOR, fontsize=12, ha='center')

# Add legend (before animation)
leg = ax.legend(loc='upper left', facecolor=BG_COLOR, edgecolor=AXIS_COLOR, labelcolor=AXIS_COLOR)

# Animation state
anim_running = True
invert_state = False  # False = black bg, white text; True = white bg, black text
current_frame = 0     # Tracks the effective frame for animation

# Animation update function
def update(frame):
    global anim_running, invert_state, BG_COLOR, AXIS_COLOR, POINT_COLOR, current_frame

    # Only increment current_frame when running
    if anim_running:
        current_frame += 1

    # Use current_frame for theta calculation
    theta_raw = SPEED * current_frame
    theta = theta_raw % (2 * np.pi)  # Modulo 2π

    x_R = RADIUS * np.cos(theta)
    x_I = RADIUS * np.sin(theta)
    point.set_data([x_R], [x_I])

    vec_x.set_data([0, x_R], [0, x_I])
    vec_x_label.set_position((x_R/2, x_I/2))
    vec_x_label.set_text('||vec{x}||')

    vec_x_R_val = x_R
    vec_x_R.set_data([0, vec_x_R_val], [0, 0])
    vec_x_R_label.set_position((vec_x_R_val/2, 0.05))
    vec_x_R_label.set_text('vec{x}_R')

    vec_x_I_val = x_I
    vec_x_I.set_data([0, 0], [0, vec_x_I_val])
    vec_x_I_label.set_position((0.05, vec_x_I_val/2))
    vec_x_I_label.set_text('vec{x}_I')

    phase_label.set_text(f'<vec{{x}}> = {theta:.2f} rad')

    # Apply inversion colors every frame
    new_bg = 'white' if invert_state else 'black'
    new_fg = 'black' if invert_state else 'white'

    BG_COLOR = new_bg
    AXIS_COLOR = new_fg
    POINT_COLOR = new_fg

    fig.patch.set_facecolor(new_bg)
    ax.set_facecolor(new_bg)

    real_axis.set_color(new_fg)
    imag_axis.set_color(new_fg)
    ax.tick_params(colors=new_fg)

    for text in axis_labels:
        text.set_color(new_fg)

    phase_label.set_color(ORIGIN_VECTOR_COLOR)

    for text in boundary_texts.values():
        text.set_color(CIRCLE_COLOR)

    extra_t_label.set_color(CIRCLE_COLOR)

    vec_x_label.set_color(ORIGIN_VECTOR_COLOR)
    vec_x_R_label.set_color(REAL_VECTOR_COLOR)
    vec_x_I_label.set_color(IMAG_VECTOR_COLOR)

    point.set_color(new_fg)

    leg.get_frame().set_facecolor(new_bg)
    leg.get_frame().set_edgecolor(new_fg)
    for text, handle in zip(leg.get_texts(), leg.legend_handles):
        text.set_color(new_fg)
        if text.get_text() in ['Real axis', 'Imaginary axis', 'Point on T']:
            handle.set_color(new_fg)
        elif text.get_text() == '||vec{x}||':
            handle.set_color(ORIGIN_VECTOR_COLOR)
        elif text.get_text() == 'vec{x}_R':
            handle.set_color(REAL_VECTOR_COLOR)
        elif text.get_text() == 'vec{x}_I':
            handle.set_color(IMAG_VECTOR_COLOR)

    fig.suptitle('Tri-Quarter Theorem Animation', color=new_fg, fontsize=16, y=0.98, ha='center')

    return (point, vec_x, vec_x_R, vec_x_I,
            vec_x_label, vec_x_R_label, vec_x_I_label, phase_label)

# Button callback for merged Play/Pause
def toggle_play_pause(event):
    global anim_running
    if anim_running:
        anim_running = False
        play_pause_button.label.set_text('Play')
    else:
        anim_running = True
        play_pause_button.label.set_text('Pause')
    fig.canvas.draw_idle()

# Button callback for invert
def invert(event):
    global invert_state, BG_COLOR, AXIS_COLOR, POINT_COLOR
    invert_state = not invert_state
    new_bg = 'white' if invert_state else 'black'
    new_fg = 'black' if invert_state else 'white'

    BG_COLOR = new_bg
    AXIS_COLOR = new_fg
    POINT_COLOR = new_fg

    fig.patch.set_facecolor(new_bg)
    ax.set_facecolor(new_bg)

    real_axis.set_color(new_fg)
    imag_axis.set_color(new_fg)
    ax.tick_params(colors=new_fg)

    for text in axis_labels:
        text.set_color(new_fg)

    phase_label.set_color(ORIGIN_VECTOR_COLOR)

    for text in boundary_texts.values():
        text.set_color(CIRCLE_COLOR)

    extra_t_label.set_color(CIRCLE_COLOR)

    vec_x_label.set_color(ORIGIN_VECTOR_COLOR)
    vec_x_R_label.set_color(REAL_VECTOR_COLOR)
    vec_x_I_label.set_color(IMAG_VECTOR_COLOR)

    point.set_color(new_fg)

    leg.get_frame().set_facecolor(new_bg)
    leg.get_frame().set_edgecolor(new_fg)
    for text, handle in zip(leg.get_texts(), leg.legend_handles):
        text.set_color(new_fg)
        if text.get_text() in ['Real axis', 'Imaginary axis', 'Point on T']:
            handle.set_color(new_fg)
        elif text.get_text() == '||vec{x}||':
            handle.set_color(ORIGIN_VECTOR_COLOR)
        elif text.get_text() == 'vec{x}_R':
            handle.set_color(REAL_VECTOR_COLOR)
        elif text.get_text() == 'vec{x}_I':
            handle.set_color(IMAG_VECTOR_COLOR)

    fig.suptitle('Tri-Quarter Theorem Animation', color=new_fg, fontsize=16, y=0.98, ha='center')

    fig.canvas.draw_idle()

# Create animation with finite frames and repeat
ani = FuncAnimation(fig, update, frames=range(1000), interval=50, blit=False, repeat=True, cache_frame_data=False)

# Add buttons
ax_play_pause = plt.axes([0.45, 0.05, 0.1, 0.04])  # Centered
ax_invert = plt.axes([0.85, 0.05, 0.1, 0.04])      # Bottom-right
play_pause_button = Button(ax_play_pause, 'Pause', color='gray', hovercolor='lightgray')  # Starts as Pause
invert_button = Button(ax_invert, 'Invert', color='gray', hovercolor='lightgray')
play_pause_button.on_clicked(toggle_play_pause)
invert_button.on_clicked(invert)

# Initial state: running
anim_running = True  # Button starts as "Pause"

# Title at top center
fig.suptitle('Tri-Quarter Theorem Animation', color=AXIS_COLOR, fontsize=16, y=0.98, ha='center')

# Display the plot
plt.show()
