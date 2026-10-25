class TrackGraph:
    def __init__(self):
        self.global_points3D = {}       # point3D_id -> np.ndarray([X, Y, Z])
        self.kp_to_point3D = {}         # (img_idx, kp_idx) -> point3D_id
        self.point_counter = 0

    def add_point3D(self, pt3d, observations):
        """
        pt3d: np.ndarray([X, Y, Z])
        observations: list of (img_idx, kp_idx) tuples observing this point
        """
        p3d_id = self.point_counter
        self.global_points3D[p3d_id] = pt3d
        for img_idx, kp_idx in observations:
            self.kp_to_point3D[(img_idx, kp_idx)] = p3d_id
        self.point_counter += 1
        return p3d_id

    def get_point3D_id(self, img_idx, kp_idx):
        return self.kp_to_point3D.get((img_idx, kp_idx), None)

    def link_keypoint_to_track(self, img_idx, kp_idx, p3d_id):
        self.kp_to_point3D[(img_idx, kp_idx)] = p3d_id