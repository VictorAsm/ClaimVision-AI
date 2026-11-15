import os
import json
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.optim import SGD
from torchvision.transforms import v2 as T
from ml.preprocessing.cocoDataloader import COCODataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

class ObjectDetector:
    def __init__(self, model_path, labels_path, model_checkpoints_path, batch_size, save_path, threshold):
        self.__device = torch.device('cpu')
        self.__model = None
        self.__classes = None
        self.__optimizer = None
        self.__threshold = threshold
        self.__save_path = save_path
        self.__batch_size = batch_size
        self.__model_path = model_path
        self.__labels_path = labels_path
        self.__model_checkpoints_path = model_checkpoints_path
        self.__transform = transforms.Compose([transforms.ToTensor()])

        if os.path.exists(self.__labels_path):
            self.__load_labels(self.__labels_path)
        else:
            print("No se encontraron los labels.")

        if os.path.exists(self.__model_path):
            self.__load_model(self.__model_path)
        else:
            print("No se encontró un modelo en la ruta especificada. Por favor, entrene uno.")

    def __load_labels(self, labels_path):
        try:
            with open(labels_path, 'r') as archivo:
                json_data = json.load(archivo)
                class_mappings = {item['id']: item['name'] for item in json_data['classes']}
                self.__classes = class_mappings
            print("Labels cargadas exitosamente - Torch")
        except Exception as e:
            print(e)

    def __load_model(self, model_path):
        try:
            self.__model_path = model_path
            self.__create_model()
            self.__model.load_state_dict(torch.load(self.__model_path, weights_only=True))
            self.__model.eval()
            print("Modelo cargado con éxito - Torch")
        except Exception as e:
            print(e)

    def __create_model(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        num_classes = len(self.__classes) + 1
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        self.__model = model
        self.__optimizer = optimizer

    def __create_loader(self, path_json, path_root):
        dataset = COCODataset(
            json_file=path_json,
            root_dir=path_root,
            transforms=self.__transform
        )

        return DataLoader(dataset, batch_size=self.__batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    def __preprocess_image(self, image, target_size=(1024, 1024)):
        image = image.resize(target_size)
        image_tensor = self.__transform(image).unsqueeze(0)
        return image_tensor

    def __plot_detections(self, image, outputs, image_name = datetime.now().strftime("%Y%m%d_%H%M%S")):
        image = image.permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=(10, 10))
        plt.imshow(image)

        for i in range(len(outputs['boxes'])):
            score = outputs['scores'][i].item()
            if score > self.__threshold:
                bbox = outputs['boxes'][i].cpu().numpy()
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                         linewidth=2, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)
                plt.text(bbox[0], bbox[1], f"{outputs['labels'][i].item()} {score:.2f}",
                         fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5))

        plt.axis('off')
        plt.savefig(F"{self.__save_path}/detected/{image_name}.jpg", dpi=300, bbox_inches='tight')

    def __prepare_data(self, train_json, train_root, val_json, val_root):
        train_loader = self.__create_loader(train_json, train_root)
        val_loader = self.__create_loader(val_json, val_root)

        return train_loader, val_loader

    def __format_response(self, predictions):
        output_predictions = []
        for i in range(len(predictions[0]['scores'])):
            score = predictions[0]['scores'][i].item()
            if score > self.__threshold:
                box = predictions[0]['boxes'][i].tolist()
                label_id = predictions[0]['labels'][i].item()
                label_name = self.__classes[label_id]
                output_predictions.append({'label': label_name, 'score': score, 'box': box})

        return output_predictions

    def train(self, data_path, epochs=10):

        train_loader, val_loader = self.__prepare_data(
            train_json=F"{data_path}/json/train.json",
            train_root=F"{data_path}/car-parts/train",
            val_json=F"{data_path}/json/test.json",
            val_root=F"{data_path}/car-parts/test"
        )

        if self.__model is None:
            self.__create_model()

        self.__model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for images, targets in train_loader:
                images = [img.to(self.__device) for img in images]
                targets = [{k: v.to(self.__device) for k, v in t.items()} for t in targets]

                loss_dict = self.__model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                self.__optimizer.zero_grad()
                losses.backward()
                self.__optimizer.step()

                epoch_loss += losses.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}")

    def save_model(self, path):
        torch.save(self.__model.state_dict(), path)

    def predict(self, image, image_name):
        if self.__model is None:
            return "No hay un modelo cargado, por favor, cargue uno"

        self.__model.eval()
        with torch.no_grad():
            predict_image = self.__preprocess_image(image).to(self.__device)
            outputs = self.__model(predict_image)
            # self.__plot_detections(predict_image[0], outputs[0], image_name=image_name)

        output_predictions = self.__format_response(outputs)

        return output_predictions