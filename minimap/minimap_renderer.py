import cv2
import numpy as np
from pitch_keypoint_detector.pitch_keypoint_detector import SoccerPitchConfig


class MinimapRenderer:
    def __init__(self, w=350, h=230):
        self.w = w
        self.h = h
        self.config = SoccerPitchConfig()
        self.scale_x = w / self.config.length
        self.scale_y = h / self.config.width

    def _pitch_to_minimap(self, pos):
        x = max(0, min(int(pos[0] * self.scale_x), self.w - 1))
        y = max(0, min(int(pos[1] * self.scale_y), self.h - 1))
        return x, y

    def _draw_pitch(self):
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        img[:] = (30, 120, 30)

        # Đường biên
        cv2.rectangle(img, (0, 0), (self.w - 1, self.h - 1),
                      (255, 255, 255), 1)

        # Đường giữa sân
        cv2.line(img, (self.w // 2, 0), (self.w // 2, self.h),
                 (255, 255, 255), 1)

        # Vòng tròn giữa sân
        cx, cy = self.w // 2, self.h // 2
        r = int(self.config.centre_circle_radius * self.scale_x)
        cv2.circle(img, (cx, cy), r, (255, 255, 255), 1)

        # Chấm giữa sân
        cv2.circle(img, (cx, cy), 2, (255, 255, 255), -1)

        # Khu vực cấm địa trái
        pbw = int(self.config.penalty_box_width * self.scale_y)
        pbl = int(self.config.penalty_box_length * self.scale_x)
        cv2.rectangle(img, (0, self.h // 2 - pbw // 2),
                      (pbl, self.h // 2 + pbw // 2),
                      (255, 255, 255), 1)

        # Khu vực cấm địa phải
        cv2.rectangle(img, (self.w - pbl, self.h // 2 - pbw // 2),
                      (self.w, self.h // 2 + pbw // 2),
                      (255, 255, 255), 1)

        # Khu vực 5m50 trái
        gbw = int(self.config.goal_box_width * self.scale_y)
        gbl = int(self.config.goal_box_length * self.scale_x)
        cv2.rectangle(img, (0, self.h // 2 - gbw // 2),
                      (gbl, self.h // 2 + gbw // 2),
                      (255, 255, 255), 1)

        # Khu vực 5m50 phải
        cv2.rectangle(img, (self.w - gbl, self.h // 2 - gbw // 2),
                      (self.w, self.h // 2 + gbw // 2),
                      (255, 255, 255), 1)

        return img

    def render(self, tracks, frame_num, jersey_numbers=None,
               team_colors=None):
        minimap = self._draw_pitch()

        for obj, color, marker in [
            ('players', None, None),
            ('referees', (255, 255, 0), 'circle'),
        ]:
            for tid, data in tracks[obj][frame_num].items():
                pos = data.get('position_transformed')
                if pos is None:
                    pos = data.get('position_adjusted')
                if pos is None:
                    pos = data.get('position')
                if pos is None:
                    continue
                x, y = self._pitch_to_minimap(pos)
                if obj == 'players':
                    team = data.get('team', 0)
                    if team_colors and team in team_colors:
                        c = team_colors[team]
                    else:
                        c = data.get('team_color', (0, 0, 255))
                    cv2.circle(minimap, (x, y), 4, c, -1)
                    cv2.circle(minimap, (x, y), 4, (255, 255, 255), 1)

                    if jersey_numbers and tid in jersey_numbers:
                        jnum = str(jersey_numbers[tid])
                        cv2.putText(minimap, jnum, (x - 5, y - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.3, (255, 255, 255), 1)
                else:
                    cv2.circle(minimap, (x, y), 3, color, -1)

        if 1 in tracks['ball'][frame_num]:
            pos = tracks['ball'][frame_num][1].get('position_transformed')
            if pos is None:
                pos = tracks['ball'][frame_num][1].get('position_adjusted')
            if pos is None:
                pos = tracks['ball'][frame_num][1].get('position')
            if pos:
                x, y = self._pitch_to_minimap(pos)
                cv2.circle(minimap, (x, y), 3, (0, 255, 255), -1)
                cv2.circle(minimap, (x, y), 3, (255, 255, 255), 1)

        return minimap

    def overlay(self, frame, minimap, pos='bottom_right'):
        h, w = frame.shape[:2]
        mh, mw = minimap.shape[:2]
        border = 10

        if pos == 'bottom_right':
            x1 = w - mw - border
            y1 = h - mh - border
        elif pos == 'top_right':
            x1 = w - mw - border
            y1 = border
        elif pos == 'bottom_left':
            x1 = border
            y1 = h - mh - border
        else:
            x1 = border
            y1 = border

        x2 = x1 + mw
        y2 = y1 + mh

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1 - 2, y1 - 2),
                      (x2 + 2, y2 + 2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        frame[y1:y2, x1:x2] = minimap
        return frame
