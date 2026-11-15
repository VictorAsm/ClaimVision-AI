import os
from PIL import Image
import numpy as np
import pandas as pd
import shutil

from sklearn.model_selection import train_test_split


class Preprocesor:
    def __init__(self, original_raw_path, original_processed_path, retrain_raw_path, retrain_processed_path, no_classify_path):
        self.original_raw_path = original_raw_path
        self.original_processed_path = original_processed_path

        self.retrain_raw_path = retrain_raw_path
        self.retrain_processed_path = retrain_processed_path

        self.no_classify_path = no_classify_path

    def __resize_with_aspect_ratio(self, image_path, size=512):
        imagen = Image.open(image_path).convert('RGB')
        image_arr = np.array(imagen)
        prom_color = image_arr.mean(axis=(0, 1)).astype(int)
        background = tuple(prom_color)
        width, height = imagen.size

        if width > height:
            new_height = int(size * (height / width))
            new_width = size
        else:
            new_width = int(size * (width / height))
            new_height = size

        resized_image = imagen.resize((new_width, new_height), Image.LANCZOS)
        square_image = Image.new('RGB', (size, size), background)
        pos = ((size - new_width) // 2, (size - new_height) // 2)
        square_image.paste(resized_image, pos)

        return square_image

    def __split_val_data_and_move(self):
        origin_path = F"{self.no_classify_path}/images"
        csv_path = F"{self.no_classify_path}/csv/no_classify.csv"
        df = pd.read_csv(csv_path)
        df_shuffled = df.sample(frac=1).reset_index(drop=True)

        val_set, dev_set = train_test_split(df_shuffled, test_size=0.8, random_state=42)

        for index, row in val_set.iterrows():
            image_name = row['image_path']
            destiny = F"{self.retrain_raw_path}/car-images/val"
            shutil.copy(F"{origin_path}/{image_name}", destiny)

        for index, row in dev_set.iterrows():
            image_name = row['image_path']
            destiny = F"{self.retrain_raw_path}/car-images/images"
            shutil.copy(F"{origin_path}/{image_name}", destiny)

        val_set.to_csv(F"{self.retrain_raw_path}/csv/val.csv", index=False)
        dev_set.to_csv(F"{self.retrain_raw_path}/csv/dev.csv", index=False)

    def __split_dev_data(self, use_path):
        train_csv_path = F"{use_path}"

        df = pd.read_csv(train_csv_path)
        test_set, train_set = train_test_split(df, test_size=0.8, random_state=42)
        return test_set, train_set

    def __order_retrain_images(self, name_folder):
        origin_path = F"{self.no_classify_path}/images"
        destiny_path = F"{self.retrain_raw_path}/{name_folder}"

        imagenes = [f for f in os.listdir(origin_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        np.random.shuffle(imagenes)
        total_imagenes = len(imagenes)

        val_ratio = 0.20
        test_ratio = 0.05

        val_count = int(total_imagenes * val_ratio)
        test_count = int((total_imagenes - val_count) * (test_ratio / (1 - val_ratio)))
        train_count = total_imagenes - val_count - test_count

        for i, imagen in enumerate(imagenes):
            origen_imagen = os.path.join(origin_path, imagen)

            if i < val_count:
                destiny = F"{destiny_path}/val/{imagen}"
            elif i < val_count + test_count:
                destiny = F"{destiny_path}/test/{imagen}"
            else:
                destiny = F"{destiny_path}/train/{imagen}"

            shutil.move(origen_imagen, destiny)

    def __move_items(self, raw_path, processed_path, folder_name, csv_name, dataset, processed_folder):
        raw_path_folder = F"{raw_path}/{folder_name}/"
        processed_path_folder = F"{processed_path}/{csv_name}/"

        i = 0
        for index, row in dataset.iterrows():
            image_path = row['image_path']
            source_path = os.path.join(raw_path_folder, image_path)
            print(source_path)
            new_image = self.__resize_with_aspect_ratio(source_path, 512)
            print("new_image")
            print("processed_path_folder", processed_path_folder)
            destination_path = os.path.join(processed_path_folder, F"{i}.jpg")
            dataset.at[index, 'image_path'] = destination_path
            new_image.save(destination_path)

            i = i + 1

        path_to_csv = F"{processed_folder}/csv/{csv_name}.csv"
        dataset.to_csv(path_to_csv, index=False)

    def __process_images(self, use_raw, use_process, image_folder):
        processed_path_parent = F"{use_process}/{image_folder}"

        raw_path_parent = F"{use_raw}/{image_folder}"
        dev_csv_path = F"{use_raw}/csv/dev.csv"
        val_csv_path = F"{use_raw}/csv/val.csv"

        test_set, train_set = self.__split_dev_data(dev_csv_path)
        val_set = pd.read_csv(val_csv_path)

        self.__move_items(raw_path_parent, processed_path_parent, 'val', 'val', val_set, use_process)
        self.__move_items(raw_path_parent, processed_path_parent, 'images', 'test', test_set, use_process)
        self.__move_items(raw_path_parent, processed_path_parent, 'images', 'train', train_set, use_process)

    def process_data(self, method, name_folder):
        if method == 'scratch':
            use_raw = self.original_raw_path
            use_process = self.original_processed_path
        else:
            use_raw = self.retrain_raw_path
            use_process = self.retrain_processed_path
            self.__split_val_data_and_move()

        self.__process_images(use_raw, use_process, name_folder)