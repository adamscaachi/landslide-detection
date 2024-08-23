import torch
import numpy as np
import matplotlib.pyplot as plt

class Evaluator:

    def __init__(self, model, state_dict, device, val_loader, test_loader):
        self.model = model.to(device)
        self.state_dict = state_dict
        self.model.load_state_dict(torch.load(state_dict, weights_only=True))
        self.device = device
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.precision = None
        self.recall = None
        self.f1 = None
        self.iou = None
        self.optimal_threshold = None
        self.best_iou = -np.inf
        self.find_optimal_threshold()
        self.evaluate(val_loader, "Validation", self.optimal_threshold)
        self.evaluate(test_loader, "Testing", self.optimal_threshold)
        self.visualise_predictions()

    def print_metrics(self, mode, threshold):
        print (f"\n{mode}\n"
               f"Model: {self.state_dict}" f" (threshold = {threshold:.2f})\n"
               f"Precision = {self.precision:.4f}\n"
               f"Recall = {self.recall:.4f}\n"
               f"F1 Score = {self.f1:.4f}\n"
               f"IoU = {self.iou:.4f}")

    def initialise_confusion_matrix(self):
        self.TP_total = 0
        self.FP_total = 0
        self.TN_total = 0
        self.FN_total = 0 

    def evaluate(self, loader, mode, threshold):
        self.initialise_confusion_matrix()
        for images, masks in loader:
            images, masks = images.to(self.device), masks.to(self.device)  
            for image, mask in zip(images, masks):
                prediction = self.predict_mask(image.unsqueeze(0), threshold)
                TP, FP, TN, FN = self.calculate_confusion_matrix(prediction, mask.cpu().numpy())
                self.TP_total += TP
                self.FP_total += FP
                self.TN_total += TN
                self.FN_total += FN
        self.precision = self.calculate_precision()
        self.recall = self.calculate_recall()
        self.f1 = self.calculate_f1()
        self.iou = self.calculate_iou()
        if mode:
            self.print_metrics(mode, threshold)

    def predict_mask(self, image, threshold):
        with torch.no_grad():
            output = self.model(image)
        output = torch.sigmoid(output).squeeze().cpu().numpy()
        prediction = (output > threshold).astype(np.uint8)
        return prediction

    def calculate_confusion_matrix(self, prediction, truth):
        TP = np.sum((prediction == 1) & (truth == 1))
        FP = np.sum((prediction == 1) & (truth == 0))
        TN = np.sum((prediction == 0) & (truth == 0))
        FN = np.sum((prediction == 0) & (truth == 1))
        return TP, FP, TN, FN

    def calculate_precision(self):
        return self.TP_total / (self.TP_total + self.FP_total + 1e-14)

    def calculate_recall(self):
        return self.TP_total / (self.TP_total + self.FN_total + 1e-14)

    def calculate_f1(self):
        return 2 * (self.precision * self.recall) / (self.precision + self.recall + 1e-14)

    def calculate_iou(self):
        return self.TP_total / (self.TP_total + self.FP_total + self.FN_total + 1e-14)

    def find_optimal_threshold(self):
        thresholds = np.arange(0.05, 1, 0.05)
        for threshold in thresholds:
            self.evaluate(self.val_loader, 0, threshold)
            if self.iou > self.best_iou:
                self.best_iou = self.iou
                self.optimal_threshold = threshold

    def visualise_predictions(self):
        images, masks = next(iter(self.test_loader))
        indices = [29, 32, 59]
        fig, axs = plt.subplots(3, 3, figsize=(6, 6))
        for i, idx in enumerate(indices):
            image = images[idx].to(self.device)
            mask = masks[idx].to(self.device)
            prediction = self.predict_mask(image.unsqueeze(0), self.optimal_threshold)
            image = image[:3].permute(1, 2, 0).cpu().numpy()
            mask = mask.squeeze().cpu().numpy()
            prediction = prediction.squeeze()
            axs[i, 0].imshow(image)
            axs[0, 0].set_title("Input (RGB)")
            axs[i, 0].axis('off')
            axs[i, 1].imshow(prediction, cmap='gray')
            axs[0, 1].set_title("Prediction")
            axs[i, 1].axis('off')
            axs[i, 2].imshow(mask, cmap='gray')
            axs[0, 2].set_title("Target")
            axs[i, 2].axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.savefig('predictions.png', bbox_inches='tight', dpi=300)