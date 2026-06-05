import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        img_2d = image.reshape(-1, 3)
        km = KMeans(n_clusters=2, init='k-means++',
                    n_init=10, random_state=42)
        km.fit(img_2d)
        return km

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        top  = crop[:crop.shape[0]//2, :]    # nửa trên (vùng áo)
        km   = self.get_clustering_model(top)

        labels = km.labels_.reshape(top.shape[:2])
        corners = [labels[0,0], labels[0,-1],
                   labels[-1,0], labels[-1,-1]]
        bg  = max(set(corners), key=corners.count)
        pl  = 1 - bg
        return km.cluster_centers_[pl]

    def assign_team_color(self, frame, player_detections):
        colors = []
        for _, data in player_detections.items():
            colors.append(self.get_player_color(frame, data['bbox']))

        km = KMeans(n_clusters=2, init='k-means++',
                    n_init=10, random_state=42)
        km.fit(colors)
        self.km = km
        self.team_colors[1] = km.cluster_centers_[0]
        self.team_colors[2] = km.cluster_centers_[1]

    def get_player_team(self, frame, bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        color = self.get_player_color(frame, bbox)
        team  = self.km.predict(color.reshape(1,-1))[0] + 1
        self.player_team_dict[player_id] = team
        return team