

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults


class AIGym(BaseSolution):
    

    def __init__(self, **kwargs):
        
        kwargs["model"] = kwargs.get("model", "yolo11n-pose.pt")
        super().__init__(**kwargs)
        self.count = []
        self.angle = []
        self.stage = []


        self.initial_stage = None
        self.up_angle = float(self.CFG["up_angle"])
        self.down_angle = float(self.CFG["down_angle"])
        self.kpts = self.CFG["kpts"]

    def process(self, im0):
        
        annotator = SolutionAnnotator(im0, line_width=self.line_width)

        self.extract_tracks(im0)
        tracks = self.tracks[0]

        if tracks.boxes.id is not None:
            if len(tracks) > len(self.count):
                new_human = len(tracks) - len(self.count)
                self.angle += [0] * new_human
                self.count += [0] * new_human
                self.stage += ["-"] * new_human


            for ind, k in enumerate(reversed(tracks.keypoints.data)):

                kpts = [k[int(self.kpts[i])].cpu() for i in range(3)]
                self.angle[ind] = annotator.estimate_pose_angle(*kpts)
                annotator.draw_specific_kpts(k, self.kpts, radius=self.line_width * 3)


                if self.angle[ind] < self.down_angle:
                    if self.stage[ind] == "up":
                        self.count[ind] += 1
                    self.stage[ind] = "down"
                elif self.angle[ind] > self.up_angle:
                    self.stage[ind] = "up"


                if self.show_labels:
                    annotator.plot_angle_and_count_and_stage(
                        angle_text=self.angle[ind],
                        count_text=self.count[ind],
                        stage_text=self.stage[ind],
                        center_kpt=k[int(self.kpts[1])],
                    )
        plot_im = annotator.result()
        self.display_output(plot_im)


        return SolutionResults(
            plot_im=plot_im,
            workout_count=self.count,
            workout_stage=self.stage,
            workout_angle=self.angle,
            total_tracks=len(self.track_ids),
        )
