import torch
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
from torch.utils.data import Subset, random_split

# Pfade zu den COCO-Dateien
annotation_path = "Pfad_zur_Ausgabedatei.json"
image_path = "Pfad_zum_gesamten_Datensatz"

# Transform zur Konvertierung der Bilder in Tensoren
transform = ToTensor()

# Gesamten COCO-Datensatz laden
full_dataset = CocoDetection(root=image_path, annFile=annotation_path, transform=transform)

# Aufteilung der Daten in Trainings-, Test- und Validierungssätze (60% - 20% - 20%)
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# Berechnung der Größen der Trainings-, Test- und Validierungssätze basierend auf den Verhältnissen
num_samples = len(full_dataset)
train_size = int(train_ratio * num_samples)
val_size = int(val_ratio * num_samples)
test_size = num_samples - train_size - val_size

# Aufteilung der Indizes für Trainings-, Test- und Validierungssätze
train_indices, val_indices, test_indices = random_split(
    range(num_samples), [train_size, val_size, test_size])

# Erstellen der Subset-Datasets basierend auf den aufgeteilten Indizes
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)
test_dataset = Subset(full_dataset, test_indices)

# Erstellen der DataLoader für das Training, die Validierung und den Test
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Überprüfung der Größen der Trainings-, Validierungs- und Testsets
print("Anzahl der Trainingsdaten:", len(train_dataset))
print("Anzahl der Validierungsdaten:", len(val_dataset))
print("Anzahl der Testdaten:", len(test_dataset))
