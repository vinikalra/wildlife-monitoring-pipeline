"""Image classification: sorts camera trap images into bird / animal / empty.

Uses a torchvision ResNet50 trained on ImageNet-1k. ImageNet has no
"empty/no-animal" class of its own, so a prediction is only trusted above
CONFIDENCE_THRESHOLD; anything below that (or anything that isn't a known
bird/mammal class) is treated as empty/unidentified and rejected.

The BIRD_CLASSES set is the exact list of the 59 bird categories in
ImageNet-1k. ANIMAL_KEYWORDS is a heuristic substring match over the
remaining class names and is intentionally easy to extend -- swap in a
fine-tuned Snapshot-Serengeti-style model here later for real species labels.
"""
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision.models import ResNet50_Weights, resnet50

BIRD_CLASSES = {
    "cock", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
    "indigo bunting", "robin", "bulbul", "jay", "magpie", "chickadee",
    "water ouzel", "kite", "bald eagle", "vulture", "great grey owl",
    "black grouse", "ptarmigan", "ruffed grouse", "prairie chicken", "peacock",
    "quail", "partridge", "African grey", "macaw", "sulphur-crested cockatoo",
    "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar",
    "toucan", "drake", "red-breasted merganser", "goose", "black swan",
    "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",
    "American egret", "bittern", "crane", "limpkin", "European gallinule",
    "American coot", "bustard", "ruddy turnstone", "red-backed sandpiper",
    "redshank", "dowitcher", "oystercatcher", "pelican", "king penguin",
    "albatross",
}

ANIMAL_KEYWORDS = (
    "elephant", "zebra", "lion", "cheetah", "leopard", "tiger", "hyena",
    "fox", "wolf", "jackal", "dingo", "hartebeest", "impala", "gazelle",
    "warthog", "hippopotamus", "baboon", "gorilla", "chimpanzee", "monkey",
    "antelope", "buffalo", "boar", "hog", "ram", "bighorn", "ibex", "otter",
    "badger", "skunk", "mongoose", "weasel", "mink", "polecat", "ferret",
    "hare", "rabbit", "porcupine", "hedgehog", "armadillo", "sloth", "koala",
    "wombat", "kangaroo", "llama", "camel", "ox", "bison", "cat", "dog",
    "wolf", "hyena", "civet", "genet", "mongoose", "deer", "gazelle",
    "wildebeest", "giraffe", "rhinoceros", "panther", "lynx", "cougar",
)


@dataclass
class ClassificationResult:
    category: str  # "bird" | "animal" | "empty"
    label: str
    confidence: float


class WildlifeClassifier:
    def __init__(self, confidence_threshold: float = 0.3, device: str | None = None):
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=weights).to(self.device).eval()
        self.categories = weights.meta["categories"]
        self.transform = weights.transforms()

    def classify(self, image_path: Path) -> ClassificationResult:
        try:
            image = Image.open(image_path).convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError(f"Not a readable image: {image_path}") from exc

        with torch.no_grad():
            batch = self.transform(image).unsqueeze(0).to(self.device)
            probs = torch.nn.functional.softmax(self.model(batch), dim=1)[0]
            confidence, index = probs.max(dim=0)

        label = self.categories[index.item()]
        confidence = confidence.item()

        if confidence < self.confidence_threshold:
            return ClassificationResult("empty", label, confidence)
        if label in BIRD_CLASSES:
            return ClassificationResult("bird", label, confidence)
        if any(keyword in label.lower() for keyword in ANIMAL_KEYWORDS):
            return ClassificationResult("animal", label, confidence)
        return ClassificationResult("empty", label, confidence)
