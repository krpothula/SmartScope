import math
import shutil

import yaml
from Smartscope.core.models import AtlasModel, SquareModel, DisplayManager
from pathlib import Path

mag_level_factory = {'atlas': AtlasModel, 'square': SquareModel}

EXPORT_DIRECTORY = '/mnt/data/training_data/'

class YoloFormat:
    bounding_boxes = []
    labels = []
    yolo_formatted = []

    def __init__(self, image:Path, shape_x, shape_y, training_set_name='hole_finder'):
        self.image = image
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.training_set_name = training_set_name

    def append(self,bounding_box, label):
        self.bounding_boxes.append(bounding_box)
        self.labels.append(label)

    def convert(self):
        for bounding_box, label in zip(self.bounding_boxes, self.labels):
            x, y, x1, y1 = bounding_box
            x_center = ((x + x1) / 2) / self.shape_x
            y_center = ((y + y1) / 2) / self.shape_y
            width = (x1 - x) / self.shape_x
            height = (y1 - y) / self.shape_y

            self.yolo_formatted.append([label, x_center, y_center, width, height])

    def export(self, output_dir=None):
        if output_dir is None:
            output_dir = Path(EXPORT_DIRECTORY, self.training_set_name)
        else:
            output_dir = Path(output_dir, self.training_set_name)
        with open(self.image.name.with_suffix('.txt'), 'w') as file:
            for target in self.yolo_formatted:
                file.write(f"{target[0]} {target[1]} {target[2]} {target[3]} {target[4]}\n")
        shutil.copy(self.image, output_dir / self.image.name)



EXPORT_FORMATS = {
    'yolo': YoloFormat
}


def get_bounding_box(x, y, radius):
    return [x - radius, y - radius, x + radius, y + radius]


def generate_training_data(instance, export_type: str = 'yolo'):
    query = instance.base_target_query(manager='display').all()

    training_data = EXPORT_FORMATS[export_type](image=query.png, shape_x=query.shape_x, shape_y=query.shape_y)
    for target in query:
        finder = target.finders.first()
        x, y = finder.x, finder.y

        label = target.classifiers.filter(method_name=finder.method_name).values_list('label', flat=True)
        if len(label) == 0:
            label = target.classifiers.filter(method_name='Micrographs curation').values_list('label', flat=True)
        if instance.targets_prefix == 'square':
            radius = math.sqrt(target.area) // 2
        else:
            radius = target.radius

        coordinates = get_bounding_box(x, y, radius)

        training_data.append(bounding_box=coordinates, label=label[0] if len(label) > 0 else 0)

    return training_data


def export_training_data(data, instance, directory='/mnt/data/tmp/'):
    image = f'{instance.pk}.mrc'
    grid_type = instance.grid_id.meshMaterial.name
    detection_type = instance.targets_prefix
    shape_x = instance.shape_x
    shape_y = instance.shape_y
    output_dir = Path(directory, detection_type)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'metadata.yaml', 'a+') as file:
        file.write(yaml.dump([dict(image=image, shape_x=shape_x, shape_y=shape_y,
                   grid_type=grid_type, targets=data)], default_flow_style=None))

    shutil.copy(instance.mrc, output_dir / f'{instance.pk}.mrc')


def add_to_training_set(mag_level: str, id: str, output_directory='/mnt/data/training_data/'):
    instance = mag_level_factory[mag_level].objects.get(pk=id)
    data = generate_training_data(instance=instance)
    export_training_data(data, instance, directory=output_directory)
