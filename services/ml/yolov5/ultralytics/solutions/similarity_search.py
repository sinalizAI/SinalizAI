

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ultralytics.data.utils import IMG_FORMATS
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.torch_utils import select_device

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class VisualAISearch(BaseSolution):
    

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        check_requirements(["open-clip-torch", "faiss-cpu"])
        import faiss
        import open_clip

        self.faiss = faiss
        self.open_clip = open_clip

        self.faiss_index = "faiss.index"
        self.data_path_npy = "paths.npy"
        self.model_name = "ViT-B-32-quickgelu"
        self.data_dir = Path(self.CFG["data"])
        self.device = select_device(self.CFG["device"])

        if not self.data_dir.exists():
            from ultralytics.utils import ASSETS_URL

            self.LOGGER.warning(f"{self.data_dir} not found. Downloading images.zip from {ASSETS_URL}/images.zip")
            from ultralytics.utils.downloads import safe_download

            safe_download(url=f"{ASSETS_URL}/images.zip", unzip=True, retry=3)
            self.data_dir = Path("images")

        self.clip_model, _, self.preprocess = self.open_clip.create_model_and_transforms(
            self.model_name, pretrained="openai"
        )
        self.clip_model = self.clip_model.to(self.device).eval()
        self.tokenizer = self.open_clip.get_tokenizer(self.model_name)

        self.index = None
        self.image_paths = []

        self.load_or_build_index()

    def extract_image_feature(self, path):
        
        image = Image.open(path)
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.clip_model.encode_image(tensor).cpu().numpy()

    def extract_text_feature(self, text):
        
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            return self.clip_model.encode_text(tokens).cpu().numpy()

    def load_or_build_index(self):
        

        if Path(self.faiss_index).exists() and Path(self.data_path_npy).exists():
            self.LOGGER.info("Loading existing FAISS index...")
            self.index = self.faiss.read_index(self.faiss_index)
            self.image_paths = np.load(self.data_path_npy)
            return


        self.LOGGER.info("Building FAISS index from images...")
        vectors = []


        for file in self.data_dir.iterdir():

            if file.suffix.lower().lstrip(".") not in IMG_FORMATS:
                continue
            try:

                vectors.append(self.extract_image_feature(file))
                self.image_paths.append(file.name)
            except Exception as e:
                self.LOGGER.warning(f"Skipping {file.name}: {e}")


        if not vectors:
            raise RuntimeError("No image embeddings could be generated.")

        vectors = np.vstack(vectors).astype("float32")
        self.faiss.normalize_L2(vectors)

        self.index = self.faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.faiss.write_index(self.index, self.faiss_index)
        np.save(self.data_path_npy, np.array(self.image_paths))

        self.LOGGER.info(f"Indexed {len(self.image_paths)} images.")

    def search(self, query, k=30, similarity_thresh=0.1):
        
        text_feat = self.extract_text_feature(query).astype("float32")
        self.faiss.normalize_L2(text_feat)

        D, index = self.index.search(text_feat, k)
        results = [
            (self.image_paths[i], float(D[0][idx])) for idx, i in enumerate(index[0]) if D[0][idx] >= similarity_thresh
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        self.LOGGER.info("\nRanked Results:")
        for name, score in results:
            self.LOGGER.info(f"  - {name} | Similarity: {score:.4f}")

        return [r[0] for r in results]

    def __call__(self, query):
        
        return self.search(query)


class SearchApp:
    

    def __init__(self, data="images", device=None):
        
        check_requirements("flask")
        from flask import Flask, render_template, request

        self.render_template = render_template
        self.request = request
        self.searcher = VisualAISearch(data=data, device=device)
        self.app = Flask(
            __name__,
            template_folder="templates",
            static_folder=Path(data).resolve(),
            static_url_path="/images",
        )
        self.app.add_url_rule("/", view_func=self.index, methods=["GET", "POST"])

    def index(self):
        
        results = []
        if self.request.method == "POST":
            query = self.request.form.get("query", "").strip()
            results = self.searcher(query)
        return self.render_template("similarity-search.html", results=results)

    def run(self, debug=False):
        
        self.app.run(debug=debug)
