

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class ObjectCounter(BaseSolution):
    

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        self.in_count = 0
        self.out_count = 0
        self.counted_ids = []
        self.classwise_counts = {}
        self.region_initialized = False

        self.show_in = self.CFG["show_in"]
        self.show_out = self.CFG["show_out"]
        self.margin = self.line_width * 2

    def count_objects(self, current_centroid, track_id, prev_position, cls):
        
        if prev_position is None or track_id in self.counted_ids:
            return

        if len(self.region) == 2:
            line = self.LineString(self.region)
            if line.intersects(self.LineString([prev_position, current_centroid])):

                if abs(self.region[0][0] - self.region[1][0]) < abs(self.region[0][1] - self.region[1][1]):

                    if current_centroid[0] > prev_position[0]:
                        self.in_count += 1
                        self.classwise_counts[self.names[cls]]["IN"] += 1
                    else:
                        self.out_count += 1
                        self.classwise_counts[self.names[cls]]["OUT"] += 1

                elif current_centroid[1] > prev_position[1]:
                    self.in_count += 1
                    self.classwise_counts[self.names[cls]]["IN"] += 1
                else:
                    self.out_count += 1
                    self.classwise_counts[self.names[cls]]["OUT"] += 1
                self.counted_ids.append(track_id)

        elif len(self.region) > 2:
            polygon = self.Polygon(self.region)
            if polygon.contains(self.Point(current_centroid)):

                region_width = max(p[0] for p in self.region) - min(p[0] for p in self.region)
                region_height = max(p[1] for p in self.region) - min(p[1] for p in self.region)

                if (
                    region_width < region_height
                    and current_centroid[0] > prev_position[0]
                    or region_width >= region_height
                    and current_centroid[1] > prev_position[1]
                ):
                    self.in_count += 1
                    self.classwise_counts[self.names[cls]]["IN"] += 1
                else:
                    self.out_count += 1
                    self.classwise_counts[self.names[cls]]["OUT"] += 1
                self.counted_ids.append(track_id)

    def store_classwise_counts(self, cls):
        
        if self.names[cls] not in self.classwise_counts:
            self.classwise_counts[self.names[cls]] = {"IN": 0, "OUT": 0}

    def display_counts(self, plot_im):
        
        labels_dict = {
            str.capitalize(key): f"{'IN ' + str(value['IN']) if self.show_in else ''} "
            f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
            for key, value in self.classwise_counts.items()
            if value["IN"] != 0 or value["OUT"] != 0
        }
        if labels_dict:
            self.annotator.display_analytics(plot_im, labels_dict, (104, 31, 17), (255, 255, 255), self.margin)

    def process(self, im0):
        
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        self.extract_tracks(im0)
        self.annotator = SolutionAnnotator(im0, line_width=self.line_width)

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )


        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):

            self.annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(cls, True))
            self.store_tracking_history(track_id, box)
            self.store_classwise_counts(cls)

            current_centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

            prev_position = None
            if len(self.track_history[track_id]) > 1:
                prev_position = self.track_history[track_id][-2]
            self.count_objects(current_centroid, track_id, prev_position, cls)

        plot_im = self.annotator.result()
        self.display_counts(plot_im)
        self.display_output(plot_im)


        return SolutionResults(
            plot_im=plot_im,
            in_count=self.in_count,
            out_count=self.out_count,
            classwise_count=self.classwise_counts,
            total_tracks=len(self.track_ids),
        )
