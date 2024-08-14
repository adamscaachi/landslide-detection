import torch
import numpy as np

class Evaluator:

    def __init__(self, model, state_dict, device, val_loader, test_loader, threshold):
        self.model = model.to(device)
        self.state_dict = state_dict
        self.model.load_state_dict(torch.load(state_dict, weights_only=True))
        self.device = device
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.threshold = threshold
        self.precision = None
        self.recall = None
        self.f1 = None
        self.iou = None

    def print_metrics(self, mode):
        print (f"\n{mode}\n"
               f"Model: {self.state_dict}" f" (threshold = {self.threshold})\n"
               f"Precision = {self.precision:.2f}\n"
               f"Recall = {self.recall:.2f}\n"
               f"F1 Score = {self.f1:.2f}\n"
               f"IOU = {self.iou:.2f}")

    def initialise_confusion_matrix(self):
        self.TP_total = 0
        self.FP_total = 0
        self.TN_total = 0
        self.FN_total = 0 

    def evaluate(self):
        for loader, mode in [(self.val_loader, "Validation"), (self.test_loader, "Testing")]:
            self.initialise_confusion_matrix()
            for images, masks in loader:
                images, masks = images.to(self.device), masks.to(self.device)  
                for image, mask in zip(images, masks):
                    prediction = self.predict_mask(image.unsqueeze(0), self.threshold)
                    TP, FP, TN, FN = self.calculate_confusion_matrix(prediction, mask.cpu().numpy())
                    self.TP_total += TP
                    self.FP_total += FP
                    self.TN_total += TN
                    self.FN_total += FN
            self.precision = self.calculate_precision()
            self.recall = self.calculate_recall()
            self.f1 = self.calculate_f1()
            self.iou = self.calculate_iou()
            self.print_metrics(mode)

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