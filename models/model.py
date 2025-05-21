import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from models.erfnet import ERFNetModel
from models.loss import CrossEntropyBayesRiskLoss
from utils.utils import save_preds

from torchmetrics.classification.f_beta import F1Score



class SemanticNetwork(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.network = ERFNetModel(hparams["model"]["num_classes"], hparams["model"]["pre-trained"])
        self.optimizer = self.configure_optimizers()
        self.loss = CrossEntropyBayesRiskLoss(weights=[1.0, 10.0, 10.0])
        self.num_classes = hparams["model"]["num_classes"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.f1_score = F1Score(num_classes=self.num_classes, average=None, mdmc_reduce="global", task="multiclass").to(
            device)
        self.running_f1 = torch.zeros(self.num_classes, device=device)
        self.running_f1_filtered = torch.zeros(self.num_classes, device=device)

        self.conf_mat = torch.zeros((self.num_classes, self.num_classes)).cuda()
        self.filtered_conf_mat = torch.zeros((self.num_classes, self.num_classes)).cuda()

    def kl_div_coefficient(self) -> float:
        return np.minimum(1.0, self.current_epoch / self.hparams["model"]["loss"]["kl_div_anneal_epochs"])

    def getLoss(self, z: torch.Tensor, targets):
        loss = self.loss(z, targets, self.kl_div_coefficient())
        return loss

    def forward(self, x: torch.Tensor):
        y = self.network.forward(x)
        return nn.Softplus()(y)

    def get_evidencial_prediction(self, evidence: torch.Tensor):
        ones = torch.ones_like(evidence, device=evidence.device)
        S = torch.sum(evidence + 1, dim=1, keepdim=True)
        prob = (evidence + 1) / S
        epistemic_unc = (torch.sum(ones, dim=1, keepdim=True) / S).squeeze(1)
        aleatoric_unc = torch.zeros_like(epistemic_unc, device=evidence.device)
        return (prob, epistemic_unc, aleatoric_unc)

    def training_step(self, batch, batch_idx):
        y = self.forward(batch["image"])
        loss = self.getLoss(y, batch["semantics"][:, 0])
        self.log("train:loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = self.forward(batch["image"])
        loss = self.getLoss(y, batch["semantics"][:, 0])
        self.log("val:loss", loss, prog_bar=True)

        y, epistemic_unc, aleatoric_unc = self.get_evidencial_prediction(y)
        flatted_sem = torch.argmax(batch["semantics"][:, 0], 1)
        preds = torch.argmax(y[:, :self.num_classes], dim=1)
        f1 = self.f1_score(preds, flatted_sem)
        self.running_f1 += f1

    def validation_epoch_end(self, outputs):
        n_batches = self.trainer.num_val_batches[0]
        avg_f1 = self.running_f1 / n_batches

        self.logger.experiment.add_scalars(
            "Validation/F1-Score",
            {f"class_{i}": avg_f1[i] for i in range(len(avg_f1))},
            self.current_epoch
        )
        mean_f1 = avg_f1.mean()
        self.log("val:f1_score", mean_f1, prog_bar=True)
        self.running_f1 *= 0.0

    def test_step(self, batch, batch_idx):
        y = self.forward(batch["image"])
        prob, epistemic_unc, _ = self.get_evidencial_prediction(y)
        semantic = torch.argmax(prob[:, :self.num_classes], dim=1, keepdim=True)

        flatted_sem = torch.argmax(batch["semantics"][:, 0], 1)
        f1 = self.f1_score(semantic.squeeze(1), flatted_sem)
        self.running_f1 += f1
        self.update("conf", semantic, flatted_sem)
        filtered_pred = self.filter_preds(semantic, epistemic_unc)
        f1_filtered = self.f1_score(filtered_pred.squeeze(1), flatted_sem)
        self.running_f1_filtered += f1_filtered
        self.update("filtered_conf", filtered_pred, flatted_sem)

        save_preds(semantic, filtered_pred, epistemic_unc, batch["image"], batch["name"])

    def update(self, mat_name, semantic, flatted_sem):
        if mat_name == "conf":
            self.conf_mat[0, 0] += ((semantic == 0) * (flatted_sem == 0)).sum()
            self.conf_mat[0, 1] += ((semantic == 0) * (flatted_sem == 1)).sum()
            self.conf_mat[0, 2] += ((semantic == 0) * (flatted_sem == 2)).sum()
            self.conf_mat[1, 0] += ((semantic == 1) * (flatted_sem == 0)).sum()
            self.conf_mat[1, 1] += ((semantic == 1) * (flatted_sem == 1)).sum()
            self.conf_mat[1, 2] += ((semantic == 1) * (flatted_sem == 2)).sum()
            self.conf_mat[2, 0] += ((semantic == 2) * (flatted_sem == 0)).sum()
            self.conf_mat[2, 1] += ((semantic == 2) * (flatted_sem == 1)).sum()
            self.conf_mat[2, 2] += ((semantic == 2) * (flatted_sem == 2)).sum()
        elif mat_name == "filtered_conf":
            self.filtered_conf_mat[0, 0] += ((semantic == 0) * (flatted_sem == 0)).sum()
            self.filtered_conf_mat[0, 1] += ((semantic == 0) * (flatted_sem == 1)).sum()
            self.filtered_conf_mat[0, 2] += ((semantic == 0) * (flatted_sem == 2)).sum()
            self.filtered_conf_mat[1, 0] += ((semantic == 1) * (flatted_sem == 0)).sum()
            self.filtered_conf_mat[1, 1] += ((semantic == 1) * (flatted_sem == 1)).sum()
            self.filtered_conf_mat[1, 2] += ((semantic == 1) * (flatted_sem == 2)).sum()
            self.filtered_conf_mat[2, 0] += ((semantic == 2) * (flatted_sem == 0)).sum()
            self.filtered_conf_mat[2, 1] += ((semantic == 2) * (flatted_sem == 1)).sum()
            self.filtered_conf_mat[2, 2] += ((semantic == 2) * (flatted_sem == 2)).sum()
        else:
            print(f"No matrix to update with name {mat_name}.")

    def filter_preds(self, semantic, epistemic_unc):
        # epistemic_unc is N X H X W, where N is the batch size
        semantic = semantic.squeeze(1)
        for batch_id in range(epistemic_unc.shape[0]):
            current_sem = np.array(semantic[batch_id].detach().cpu())
            current_epistemic = np.array(epistemic_unc[batch_id].detach().cpu())

            max_unc = current_epistemic.max()
            min_unc = current_epistemic.min()
            th = (max_unc - min_unc) * 0.6 + min_unc
            current_epistemic = current_epistemic >= th

            plant_components = cv2.connectedComponentsWithStats((current_sem == 1).astype(np.uint8))
            contours = cv2.dilate((plant_components[1] != 0).astype(np.uint8), (3, 3), 5) - cv2.erode(
                (plant_components[1] != 0).astype(np.uint8), (3, 3), 5
            )

            for comp in range(plant_components[0]):
                mask = (plant_components[1] == comp) - (plant_components[1] == comp) * contours
                mask_clean = plant_components[1] == comp
                area = mask_clean.sum()

                # threshold of uncertain area over total area
                area_th = 0.3
                unc = (current_epistemic * mask).sum()
                if unc / area >= area_th:
                    semantic[batch_id, mask_clean] = 2

        return torch.tensor(semantic).unsqueeze(1)

    def test_epoch_end(self, outputs):
        n_batches = self.trainer.num_test_batches[0]

        avg_f1 = 100 * self.running_f1 / n_batches
        avg_f1_filtered = 100 * self.running_f1_filtered / n_batches

        print(
            f"F1-score before post-processing:\n"
            + "\n".join([f"Class {i}: {avg_f1[i]:.2f}" for i in range(len(avg_f1))])
            + f"\nMean F1: {avg_f1.mean():.2f}\n"
        )
        print(
            f"F1-score after post-processing:\n"
            + "\n".join([f"Class {i}: {avg_f1_filtered[i]:.2f}" for i in range(len(avg_f1_filtered))])
            + f"\nMean F1: {avg_f1_filtered.mean():.2f}\n"
        )
        print(
            f"Per-class improvement after post-processing:\n"
            + "\n".join([f"Class {i}: {avg_f1_filtered[i] - avg_f1[i]:.2f}" for i in range(len(avg_f1))])
            + f"\nMean Improvement: {(avg_f1_filtered.mean() - avg_f1.mean()):.2f}"
        )

        self.running_f1 *= 0.0
        self.running_f1_filtered *= 0.0

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.hparams["model"]["lr"])
        return [self.optimizer]