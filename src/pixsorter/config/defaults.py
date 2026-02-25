class SortConfig:
    def __init__(self):
        self.input_dir: str = ""
        self.output_dir: str = ""
        self.mode: str = "copy"   # copy|move
        self.min_group: int = 2

        # Candidate search (CLIP)
        self.clip_model: str = "vit_l_14"  # vit_l_14|vit_b_32
        self.knn_k: int = 18
        self.clip_sim_thresh: float = 0.86

        # Verification (ORB+homography)
        self.orb_nfeatures: int = 2500
        self.orb_match_thresh: float = 0.75
        self.min_inliers: int = 22
        self.max_side_verify: int = 900

        # Best-of selection
        self.best_of: bool = True
        self.top_k: int = 3
        self.diverse: bool = True
        self.diverse_inliers: int = 40

        # Eyes-open preference
        self.use_eyes: bool = True
        self.eyes_weight: float = 2.2

        # Output folder behavior
        self.output_create_new: bool = False
        self.output_new_name: str = "SortedOutput"