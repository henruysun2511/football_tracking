# analysis/visualize_player_stats.py

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

STUB_PATH = 'stubs/tracks_full.pkl'
if not os.path.exists(STUB_PATH):
    print(f"File {STUB_PATH} not found. Run 'python main.py --mode tracking' first.")
    exit(0)

with open(STUB_PATH, 'rb') as f:
    tracks = pickle.load(f)

player_stats = {}

for frame_num, frame_players in enumerate(tracks['players']):
    for pid, data in frame_players.items():
        if pid not in player_stats:
            player_stats[pid] = {'speeds': [], 'max_dist': 0, 'team': data.get('team', 0)}
        if 'speed' in data:
            player_stats[pid]['speeds'].append(data['speed'])
        if 'distance' in data:
            player_stats[pid]['max_dist'] = max(player_stats[pid]['max_dist'], data['distance'])

valid_players = {pid: s for pid, s in player_stats.items() if len(s['speeds']) > 10}

if not valid_players:
    print("No player speed data found. Run the tracking pipeline first (python main.py --mode tracking)")
    exit(0)

pids = sorted(valid_players.keys())[:14]

max_speeds = [np.max(valid_players[p]['speeds']) if valid_players[p]['speeds'] else 0 for p in pids]
avg_speeds = [np.mean(valid_players[p]['speeds']) if valid_players[p]['speeds'] else 0 for p in pids]
distances = [valid_players[p]['max_dist'] for p in pids]
team_colors = [('#e74c3c' if valid_players[p]['team'] == 1 else '#3498db') for p in pids]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Thong ke toc do & quang duong cau thu", fontsize=13, fontweight='bold')
labels = [f'P{p}' for p in pids]
legend_handles = [mpatches.Patch(color='#e74c3c', label='Team 1'),
                  mpatches.Patch(color='#3498db', label='Team 2')]

axes[0].bar(labels, max_speeds, color=team_colors, edgecolor='white')
axes[0].set_title("Toc do toi da (km/h)"); axes[0].set_ylabel("km/h")
axes[0].tick_params(axis='x', rotation=45); axes[0].legend(handles=legend_handles)

axes[1].bar(labels, avg_speeds, color=team_colors, edgecolor='white')
axes[1].set_title("Toc do trung binh (km/h)"); axes[1].set_ylabel("km/h")
axes[1].tick_params(axis='x', rotation=45); axes[1].legend(handles=legend_handles)

axes[2].bar(labels, distances, color=team_colors, edgecolor='white')
axes[2].set_title("Quang duong di chuyen (m)"); axes[2].set_ylabel("Met")
axes[2].tick_params(axis='x', rotation=45); axes[2].legend(handles=legend_handles)

for ax in axes: ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
os.makedirs("analysis/figures", exist_ok=True)
plt.savefig("analysis/figures/player_stats.png", dpi=150, bbox_inches='tight')
plt.show()
