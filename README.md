# Firearm Detection

## Descrizione del Progetto

Questo progetto implementa sistemi di rilevamento automatico di pistole in immagini utilizzando tecniche di computer vision e machine learning. Sono stati sviluppati e confrontati due diversi approcci: Faster R-CNN e YOLOv11, con diverse configurazioni di data augmentation e regolarizzazione.

## Struttura del Progetto

```
.
├── Code/                # Notebook di sviluppo e training
├── Dataset/             # Dataset organizzati per i diversi modelli
│   ├── fasterrcnn/      # Dataset formattato per Faster R-CNN
│   └── yolo/            # Dataset formattato per YOLOv11
├── Demo/                # Notebook dimostrativi, video dimostrativo e immagini di esempio
├── Models/              # Modelli addestrati salvati
└── Performance/         # Grafici e metriche di performance
```

## Modelli Implementati

- **Faster R-CNN**: Implementazione basata su torchvision con diverse configurazioni:
  - Modello base con fasterrcnn_resnet50_fpn
  - Modello con fasterrcnn_resnet50_fpn e implementazione di dropout e data augmentation per ridurre l'overfitting
  - Modello usando fasterrcnn_resnet50_fpn_v2 con dropout e data augmentation per ridurre l'overfitting
- **YOLOv11**: Implementazione basata sulla libreria ultralytics per la rilevazione di oggetti real-time

## Requisiti

Per eseguire il codice è necessario installare le seguenti dipendenze:

```
pip install -r requirements.txt
```

Le principali dipendenze includono:

- PyTorch 2.2.0 (CUDA 11.8)
- Torchvision 0.17.0
- Ultralytics (per YOLOv11)
- OpenCV
- Albumentations (per data augmentation)
- Matplotlib, NumPy, Pandas (per analisi dati)
- Roboflow (per gestione dataset)

## Workflow di Sviluppo

1. **Preparazione Dataset**:

   - `create_dataset.ipynb` - Preparazione dataset per Faster R-CNN
   - `create_dataset_yolo.ipynb` - Preparazione dataset per YOLOv11

2. **Training**:

   - `fine_tuning.ipynb` - Training modello Faster R-CNN base
   - `fine_tuning_dropout_dataaugmented.ipynb` - Training con dropout e data augmentation
   - `fine_tuning_dropout_dataaugmented_v2.ipynb` - Training con dropout e data augmentation con fasterrcnn_resnet50_fpn_v2
   - `fine_tuning_yolo.ipynb` - Training modello YOLOv11

3. **Inferenza**:

   - `inference_fasterrcnn.ipynb` - Inferenza con modello Faster R-CNN
   - `inference_fasterrcnn_dropout_dataaugmented.ipynb` - Inferenza con modello migliorato
   - `inference_fasterrcnn_dropout_dataaugmented_v2.ipynb` - Inferenza con modello migliorato con fasterrcnn_resnet50_fpn_v2
   - `inference_yolo.ipynb` - Inferenza con modello YOLOv11

## Demo

Nella cartella `Demo/` sono disponibili notebook per dimostrare il funzionamento dei modelli su immagini di test. Ogni notebook è configurato per caricare un modello pre-addestrato specifico:

- `inference_demo_fasterrcnn.ipynb`
- `inference_demo_fasterrcnn_dropout_data_augmented.ipynb`
- `inference_demo_fasterrcnn_dropout_data_augmented_v2.ipynb`
- `inference_demo_yolo.ipynb`

## Performance

Le performance dei diversi modelli sono documentate nella cartella `Performance/`, con grafici che mostrano l'andamento delle metriche durante il training.

## Uso

Per eseguire una demo:

1. Clonare il repository
2. Installare le dipendenze: `pip install -r requirements.txt`
3. Aprire uno dei notebook nella cartella `Demo/` con Jupyter
4. Eseguire il notebook per testare il modello su immagini di esempio
