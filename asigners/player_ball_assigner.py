import numpy as np
from utils import get_center, measure_distance


class PlayerBallAssigner:
    def __init__(self, max_distance=70):
        self.max_distance = max_distance

    def assign_ball_to_player(self, players, ball_bbox):
        ball_pos = get_center(ball_bbox)
        min_dist = float('inf')
        assigned = -1
        for pid, data in players.items():
            bbox = data['bbox']
            # Khoảng cách từ bóng đến chân trái/phải
            d_left  = measure_distance((bbox[0], bbox[-1]), ball_pos)
            d_right = measure_distance((bbox[2], bbox[-1]), ball_pos)
            dist = min(d_left, d_right)
            if dist < self.max_distance and dist < min_dist:
                min_dist = dist
                assigned = pid
        return assigned