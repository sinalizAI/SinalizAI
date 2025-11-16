

import numpy as np

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class RegionCounter(BaseSolution):
    

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        self.region_template = {
            "name": "Default Region",
            "polygon": None,
            "counts": 0,
            "dragging": False,
            "region_color": (255, 255, 255),
            "text_color": (0, 0, 0),
        }
        self.region_counts = {}
        self.counting_regions = []

    def add_region(self, name, polygon_points, region_color, text_color):
        
        region = self.region_template.copy()
        region.update(
            {
                "name": name,
                "polygon": self.Polygon(polygon_points),
                "region_color": region_color,
                "text_color": text_color,
            }
        )
        self.counting_regions.append(region)

    def process(self, im0):
        
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)


        if not isinstance(self.region, dict):
            self.region = {"Region#01": self.region or self.initialize_region()}


        for idx, (region_name, reg_pts) in enumerate(self.region.items(), start=1):
            color = colors(idx, True)
            annotator.draw_region(reg_pts, color, self.line_width * 2)
            self.add_region(region_name, reg_pts, color, annotator.get_txt_color())


        for region in self.counting_regions:
            if "prepared_polygon" not in region:
                region["prepared_polygon"] = self.prep(region["polygon"])


        boxes_np = np.array([((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in self.boxes], dtype=np.float32)
        points = [self.Point(pt) for pt in boxes_np]


        if points:
            for point, cls, track_id, box, conf in zip(points, self.clss, self.track_ids, self.boxes, self.confs):
                annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(track_id, True))

                for region in self.counting_regions:
                    if region["prepared_polygon"].contains(point):
                        region["counts"] += 1
                        self.region_counts[region["name"]] = region["counts"]


        for region in self.counting_regions:
            annotator.text_label(
                region["polygon"].bounds,
                label=str(region["counts"]),
                color=region["region_color"],
                txt_color=region["text_color"],
                margin=self.line_width * 4,
            )
            region["counts"] = 0
        plot_im = annotator.result()
        self.display_output(plot_im)

        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids), region_counts=self.region_counts)
