import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
              "open-images-v6",
              splits=('train', 'test'),
              label_types=["segmentations"],
              classes=["Cat", "Dog"],
              max_samples=2000,
              num_samples=2000,
          )