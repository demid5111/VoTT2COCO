import json
from pathlib import Path

from tqdm import tqdm
import cv2
import numpy as np
from glob import glob


class VOTItem:
    def __init__(self, path, index, img_dir, loaded_dict):
        self.item_dict = loaded_dict or json.load(open(path, 'r'))

        self.index = index
        self.width = self.item_dict['asset']['size']['width']
        self.height = self.item_dict['asset']['size']['height']
        self.name = self.item_dict['asset']['name']

        self.bbox, self.categories = self.__read_bboxes()
        self.masks, self.areas = self.__read_masks()

        self.image_path = path.split('annotations')[0] + \
                          f'{img_dir}/{self.name}'
        if path.split('annotations')[0] == path:
            # this is the situation when there is everything in one folder (VoTT v2.2.0)
            self.image_path = str(Path(path).parent / img_dir / self.name)

    def __read_bboxes(self):
        bboxes_list = []
        categories_list = []

        for data in self.item_dict['regions']:
            bboxes_list.append([
                int(data['boundingBox']['left']),
                int(data['boundingBox']['top']),
                int(data['boundingBox']['width']),
                int(data['boundingBox']['height']),
            ])

            categories_list.append(data['tags'][0])

        return bboxes_list, categories_list

    def __read_masks(self):
        masks_list = []
        areas = []

        for data in self.item_dict['regions']:
            mask = []
            area = 0

            for point in data['points']:
                mask.append(int(point['x']))
                mask.append(int(point['y']))

            masks_list.append([mask])
            areas.append(cv2.contourArea(np.array([
                [[mask[i], mask[i + 1]] for i in range(0, len(mask), 2)]
            ])))

        return masks_list, areas


class VOTTReader:
    def __init__(self, config):
        self.config = config
        self.global_index = 0
        self.categories = []
        self.items = []

    def parse_files(self):
        source_dataset_config = self.config['dataset']['source']
        directory = source_dataset_config['path']
        key = source_dataset_config['anno_cat']
        img_dir = source_dataset_config['img_cat']
        files_list = glob(f'{directory}**/{key}/*.json')

        is_exported_format = source_dataset_config.get('is_exported', False)
        # for the exported format there is only one .json file containing all the annotations
        if is_exported_format:
            print(f'[LOGS] Parsing {len(files_list)} VoTT json files')
            for path in tqdm(files_list):
                exported_annotation_json = json.load(open(path, 'r'))
                for asset_id, asset_dict in exported_annotation_json['assets'].items():
                    item = VOTItem(path=path, index=self.global_index, img_dir=img_dir, loaded_dict=asset_dict)
                    self.global_index += 1
                    self.items.append(item)

        else:
            print(f'[LOGS] Parsing {len(files_list)} VoTT json files')
            for path in tqdm(files_list):
                item = VOTItem(path=path, index=self.global_index, img_dir=img_dir, loaded_dict=None)
                self.global_index += 1
                self.items.append(item)

        for item in self.items:
            for cat in set(item.categories):
                if cat not in self.categories:
                    self.categories.append(cat)
