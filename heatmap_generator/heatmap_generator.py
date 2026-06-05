import cv2
import numpy as np
from pitch_keypoint_detector.pitch_keypoint_detector import SoccerPitchConfig


class HeatmapGenerator:
    def __init__(self, w=630, h=420, sigma=15):
        self.w = w
        self.h = h
        self.sigma = sigma
        self.heatmap_team1 = np.zeros((h, w), dtype=np.float32)
        self.heatmap_team2 = np.zeros((h, w), dtype=np.float32)
        self.config = SoccerPitchConfig()

    def update_from_tracks(self, tracks):
        for obj, obj_tracks in tracks.items():
            if obj == 'ball':
                continue
            for frame_tracks in obj_tracks:
                for tid, data in frame_tracks.items():
                    pos = data.get('position_transformed')
                    if pos is None:
                        continue
                    x, y = self._world_to_heatmap(pos)
                    team = data.get('team', 0)
                    if team == 1:
                        self._add_point(self.heatmap_team1, x, y)
                    elif team == 2:
                        self._add_point(self.heatmap_team2, x, y)

    def _world_to_heatmap(self, pos):
        x = pos[0] / self.config.length * self.w
        y = pos[1] / self.config.width * self.h
        return int(x), int(y)

    def _add_point(self, heatmap, x, y):
        if 0 <= x < self.w and 0 <= y < self.h:
            heatmap[y, x] += 1

    def _apply_gaussian(self, heatmap):
        return cv2.GaussianBlur(heatmap, (0, 0), self.sigma)

    def render_both(self):
        hm = self._apply_gaussian(self.heatmap_team1 + self.heatmap_team2)
        hm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(hm, cv2.COLORMAP_JET)

    def render_team(self, team, colormap=cv2.COLORMAP_JET):
        hm = self.heatmap_team1 if team == 1 else self.heatmap_team2
        hm = self._apply_gaussian(hm)
        hm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(hm, colormap)
