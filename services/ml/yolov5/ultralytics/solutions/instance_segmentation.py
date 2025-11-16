

from ultralytics.engine.results import Results
from ultralytics.solutions.solutions import BaseSolution, SolutionResults


class InstanceSegmentation(BaseSolution):
    

    def __init__(self, **kwargs):
        
        kwargs["model"] = kwargs.get("model", "yolo11n-seg.pt")
        super().__init__(**kwargs)

        self.show_conf = self.CFG.get("show_conf", True)
        self.show_labels = self.CFG.get("show_labels", True)
        self.show_boxes = self.CFG.get("show_boxes", True)

    def process(self, im0):
        
        self.extract_tracks(im0)


        if self.masks is None:
            self.LOGGER.warning("No masks detected! Ensure you're using a supported Ultralytics segmentation model.")
            plot_im = im0
        else:
            results = Results(im0, path=None, names=self.names, boxes=self.track_data.data, masks=self.masks.data)
            plot_im = results.plot(
                line_width=self.line_width,
                boxes=self.show_boxes,
                conf=self.show_conf,
                labels=self.show_labels,
                color_mode="instance",
            )

        self.display_output(plot_im)


        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
