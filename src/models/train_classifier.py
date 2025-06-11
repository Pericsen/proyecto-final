import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, BertForSequenceClassification, BertTokenizer, BertModel
from torch.optim import AdamW
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import re
import nltk
from nltk.corpus import stopwords
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Pre-download NLTK data
try:
    nltk.data.find('corpora/stopwords')
    print("Stopwords already downloaded")
except LookupError:
    print("Downloading stopwords...")
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
    print("Punkt already downloaded")
except LookupError:
    print("Downloading punkt...")
    nltk.download('punkt', quiet=True)

# Load stopwords once
from nltk.corpus import stopwords
spanish_stop = stopwords.words('spanish')

class MultiOutputBertClassifier(nn.Module):
    """Multi-output BERT classifier for area and priority prediction"""
    
    def __init__(self, bert_model_name, num_area_classes, num_priority_classes, dropout=0.3):
        super(MultiOutputBertClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Separate classification heads
        self.area_classifier = nn.Linear(self.bert.config.hidden_size, num_area_classes)
        self.priority_classifier = nn.Linear(self.bert.config.hidden_size, num_priority_classes)
        
    def forward(self, input_ids, attention_mask, area_labels=None, priority_labels=None):
        # Get BERT representation
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Get logits for both tasks
        area_logits = self.area_classifier(pooled_output)
        priority_logits = self.priority_classifier(pooled_output)
        
        loss = None
        if area_labels is not None and priority_labels is not None:
            # Calculate losses for both tasks
            loss_fn = nn.CrossEntropyLoss()
            area_loss = loss_fn(area_logits, area_labels)
            priority_loss = loss_fn(priority_logits, priority_labels)
            
            # Combined loss (you can adjust weights if needed)
            loss = area_loss + priority_loss
        
        return {
            'loss': loss,
            'area_logits': area_logits,
            'priority_logits': priority_logits
        }

class ComplaintClassifier:
    def __init__(self, model_name="VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis", num_area_labels=5, num_priority_labels=3, use_multioutput=True):
        """
        Initialize the complaint classifier with multi-output support
        
        Args:
            model_name: The name of the pre-trained model to use
            num_area_labels: Number of area categories to classify
            num_priority_labels: Number of priority levels to classify
            use_multioutput: Whether to use multi-output model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = 256
        self.use_multioutput = use_multioutput
        
        if use_multioutput:
            # Use custom multi-output model
            self.model = MultiOutputBertClassifier(
                bert_model_name=model_name,
                num_area_classes=num_area_labels,
                num_priority_classes=num_priority_labels
            ).to(self.device)
        else:
            # Use standard single-output model (for backward compatibility)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=num_area_labels,
                ignore_mismatched_sizes=True
            ).to(self.device)

    def load_and_preprocess_data(self, data_path, text_column, area_label_column, priority_label_column=None, test_size=0.2):
        """
        Load and preprocess the official complaint dataset with both area and priority labels
        
        Args:
            data_path: Path to the dataset file (CSV)
            text_column: Name of column containing complaint text
            area_label_column: Name of column containing area category
            priority_label_column: Name of column containing priority level
            test_size: Fraction of data to use for testing
        
        Returns:
            Processed train and validation datasets
        """
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Basic cleaning
        df[text_column] = df[text_column].fillna("")
        df[text_column] = df[text_column].apply(self._clean_text)
        
        # Convert area labels to numerical values
        self.area_label_encoder = {label: i for i, label in enumerate(df[area_label_column].unique())}
        self.area_label_decoder = {i: label for label, i in self.area_label_encoder.items()}
        df['area_label_id'] = df[area_label_column].map(self.area_label_encoder)
        
        # Convert priority labels to numerical values if using multi-output
        if self.use_multioutput and priority_label_column:
            self.priority_label_encoder = {label: i for i, label in enumerate(df[priority_label_column].unique())}
            self.priority_label_decoder = {i: label for label, i in self.priority_label_encoder.items()}
            df['priority_label_id'] = df[priority_label_column].map(self.priority_label_encoder)
        else:
            self.priority_label_encoder = None
            self.priority_label_decoder = None
        
        # Split data
        if self.use_multioutput and priority_label_column:
            # Stratify by both area and priority
            df['combined_stratify'] = df['area_label_id'].astype(str) + '_' + df['priority_label_id'].astype(str)
            train_df, val_df = train_test_split(df, test_size=test_size, stratify=df['combined_stratify'], random_state=42)
        else:
            train_df, val_df = train_test_split(df, test_size=test_size, stratify=df['area_label_id'], random_state=42)
        
        # Create PyTorch datasets
        train_dataset = ComplaintDataset(
            texts=train_df[text_column].tolist(),
            area_labels=train_df['area_label_id'].tolist(),
            priority_labels=train_df['priority_label_id'].tolist() if self.use_multioutput and priority_label_column else None,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            use_multioutput=self.use_multioutput
        )
        
        val_dataset = ComplaintDataset(
            texts=val_df[text_column].tolist(),
            area_labels=val_df['area_label_id'].tolist(),
            priority_labels=val_df['priority_label_id'].tolist() if self.use_multioutput and priority_label_column else None,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            use_multioutput=self.use_multioutput
        )
        
        print(f"Data loaded with {len(train_df)} training samples and {len(val_df)} validation samples")
        print(f"Area label mapping: {self.area_label_encoder}")
        if self.use_multioutput and priority_label_column:
            print(f"Priority label mapping: {self.priority_label_encoder}")
        
        return train_dataset, val_dataset
    
    def _clean_text(self, text):
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def apply_domain_adaptation(self, train_dataset, social_media_examples=None):
        """
        Apply domain adaptation techniques to help bridge the gap between
        third-party descriptions and direct consumer complaints
        """
        # If we have real social media examples, we can use them for fine-tuning
        if social_media_examples:
            pass
        
        def convert_to_consumer_style(text):
            """Convert third-party descriptions to more consumer-like language"""
            text = re.sub(r"el vecino (solicita)", "Yo solicito", text)
            text = re.sub(r"la vecina(solicita|indica)", "Yo solicito", text) 
            text = re.sub(r"el vecino (reclama)", "Yo reclamo", text)
            text = re.sub(r"la vecina(reclama|indica)", "Yo reclamo", text) 
            text = re.sub(r"vecino (reclama|reitera|solicita)", "solicito", text)
            text = re.sub(r"vecina (reclama|reitera|solicita)", "solicito", text)
            return text
        
        return train_dataset
    
    def train(self, train_dataset, val_dataset, batch_size=32, epochs=3, learning_rate=2e-5):
        """Optimized training with multi-output support"""
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

        # Freeze early layers to speed up training
        if hasattr(self.model, 'bert'):
            for param in self.model.bert.embeddings.parameters():
                param.requires_grad = False

            # Only fine-tune the last 4 transformer layers
            for i in range(8):
                for param in self.model.bert.encoder.layer[i].parameters():
                    param.requires_grad = False

        # Optimizer with correct learning rate
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # Setup mixed precision training
        scaler = GradScaler(device="cuda")

        # Calculate steps with gradient accumulation
        accumulation_steps = 4
        total_steps = (len(train_loader) // accumulation_steps) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}/{epochs}")
            running_loss = 0.0
            running_area_loss = 0.0
            running_priority_loss = 0.0

            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                if self.use_multioutput:
                    area_labels = batch['area_labels'].to(self.device)
                    priority_labels = batch['priority_labels'].to(self.device)
                    
                    # Forward pass with mixed precision
                    with autocast(device_type="cuda"):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            area_labels=area_labels,
                            priority_labels=priority_labels
                        )
                        loss = outputs['loss'] / accumulation_steps
                else:
                    labels = batch['labels'].to(self.device)
                    
                    with autocast(device_type="cuda"):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss / accumulation_steps

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                running_loss += loss.item() * accumulation_steps

                # Update weights every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

            # Print epoch results
            avg_train_loss = running_loss / len(train_loader)
            print(f"Average training loss: {avg_train_loss:.4f}")

            # Evaluate on validation set
            val_report = self.evaluate(val_loader)
            print(f"Validation Report:\n{val_report}")
    
    def evaluate(self, data_loader):
        """Evaluate the model on the provided data loader"""
        self.model.eval()
        
        all_area_preds = []
        all_area_labels = []
        all_priority_preds = []
        all_priority_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                if self.use_multioutput:
                    area_labels = batch['area_labels'].to(self.device)
                    priority_labels = batch['priority_labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    area_preds = torch.argmax(outputs['area_logits'], dim=1).cpu().numpy()
                    priority_preds = torch.argmax(outputs['priority_logits'], dim=1).cpu().numpy()
                    
                    all_area_preds.extend(area_preds)
                    all_area_labels.extend(area_labels.cpu().numpy())
                    all_priority_preds.extend(priority_preds)
                    all_priority_labels.extend(priority_labels.cpu().numpy())
                else:
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                    all_area_preds.extend(preds)
                    all_area_labels.extend(labels.cpu().numpy())
        
        # Convert numeric predictions back to original category labels
        if self.use_multioutput:
            decoded_area_preds = [self.area_label_decoder[p] for p in all_area_preds]
            decoded_area_labels = [self.area_label_decoder[l] for l in all_area_labels]
            decoded_priority_preds = [self.priority_label_decoder[p] for p in all_priority_preds]
            decoded_priority_labels = [self.priority_label_decoder[l] for l in all_priority_labels]
            
            # Calculate classification reports for both tasks
            area_report = classification_report(decoded_area_labels, decoded_area_preds)
            priority_report = classification_report(decoded_priority_labels, decoded_priority_preds)
            
            report = f"AREA CLASSIFICATION:\n{area_report}\n\nPRIORITY CLASSIFICATION:\n{priority_report}"
        else:
            decoded_preds = [self.area_label_decoder[p] for p in all_area_preds]
            decoded_labels = [self.area_label_decoder[l] for l in all_area_labels]
            report = classification_report(decoded_labels, decoded_preds)
        
        return report
    
    def predict(self, texts):
        """
        Predict both area and priority for new texts
        
        Args:
            texts: List of text content to classify
        
        Returns:
            Predicted categories and confidence scores
        """
        self.model.eval()
        
        # Clean texts
        cleaned_texts = [self._clean_text(text) for text in texts]
        
        # Tokenize
        encoded = self.tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        if self.use_multioutput:
            # Get predicted classes and confidences for both tasks
            area_probs = torch.nn.functional.softmax(outputs['area_logits'], dim=1)
            area_confidences, area_predictions = torch.max(area_probs, dim=1)
            
            priority_probs = torch.nn.functional.softmax(outputs['priority_logits'], dim=1)
            priority_confidences, priority_predictions = torch.max(priority_probs, dim=1)
            
            # Convert to original category labels
            results = []
            for i, (area_pred, area_conf, priority_pred, priority_conf) in enumerate(zip(
                area_predictions.cpu().numpy(), area_confidences.cpu().numpy(),
                priority_predictions.cpu().numpy(), priority_confidences.cpu().numpy()
            )):
                area_category = self.area_label_decoder[area_pred]
                priority_category = self.priority_label_decoder[priority_pred]
                results.append({
                    "text": texts[i],
                    "predicted_area": area_category,
                    "area_confidence": float(area_conf),
                    "predicted_priority": priority_category,
                    "priority_confidence": float(priority_conf)
                })
        else:
            # Single output (area only)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            
            results = []
            for i, (pred, conf) in enumerate(zip(predictions.cpu().numpy(), confidences.cpu().numpy())):
                category = self.area_label_decoder[pred]
                results.append({
                    "text": texts[i],
                    "predicted_area": category,
                    "area_confidence": float(conf)
                })
        
        return results
    
    def save_model(self, output_dir):
        """Save the trained model and tokenizer"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if self.use_multioutput:
            # Save custom multi-output model
            # Save the underlying BERT model components
            self.model.bert.save_pretrained(output_dir)
            
            # Save the full model state dict
            torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
            
            # Save model config for multi-output
            config = {
                "model_type": "multioutput_bert",
                "num_area_classes": len(self.area_label_encoder),
                "num_priority_classes": len(self.priority_label_encoder),
                "dropout": 0.3,
                "bert_model_name": "VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis"
            }
            
            with open(f"{output_dir}/config.json", "w") as f:
                import json
                json.dump(config, f, indent=2)
        else:
            # Save standard transformers model
            self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label mappings
        with open(f"{output_dir}/label_mappings.json", "w") as f:
            import json
            mappings = {
                "area_label_encoder": self.area_label_encoder,
                "area_label_decoder": self.area_label_decoder,
                "use_multioutput": self.use_multioutput
            }
            if self.use_multioutput:
                mappings["priority_label_encoder"] = self.priority_label_encoder
                mappings["priority_label_decoder"] = self.priority_label_decoder
            
            json.dump(mappings, f, indent=2)
        
        print(f"Model saved to {output_dir}")
    
    def load_model(self, model_dir):
        """Load a trained model and tokenizer"""
        # Load label mappings first to determine model type
        with open(f"{model_dir}/label_mappings.json", "r") as f:
            import json
            mappings = json.load(f)
            self.area_label_encoder = mappings["area_label_encoder"]
            self.area_label_decoder = mappings["area_label_decoder"]
            self.use_multioutput = mappings.get("use_multioutput", False)
            
            if self.use_multioutput:
                self.priority_label_encoder = mappings["priority_label_encoder"]
                self.priority_label_decoder = mappings["priority_label_decoder"]
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load model based on type
        if self.use_multioutput:
            # Load custom multi-output model
            self.model = MultiOutputBertClassifier(
                bert_model_name='VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis',
                num_area_classes=len(self.area_label_encoder),
                num_priority_classes=len(self.priority_label_encoder)
            ).to(self.device)
            
            # Load state dict
            state_dict_path = f"{model_dir}/pytorch_model.bin"
            if os.path.exists(state_dict_path):
                state_dict = torch.load(state_dict_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            else:
                raise FileNotFoundError(f"Model weights not found at {state_dict_path}")
        else:
            # Load standard transformers model
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        
        print(f"Model loaded from {model_dir}")


class ComplaintDataset(Dataset):
    def __init__(self, texts, area_labels, priority_labels=None, tokenizer=None, max_length=512, use_multioutput=False):
        """
        Dataset for complaint classification with optional multi-output support
        
        Args:
            texts: List of text content
            area_labels: List of corresponding area labels
            priority_labels: List of corresponding priority labels (optional)
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            use_multioutput: Whether to use multi-output format
        """
        self.texts = texts
        self.area_labels = area_labels
        self.priority_labels = priority_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_multioutput = use_multioutput
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        area_label = self.area_labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Remove batch dimension added by tokenizer
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        if self.use_multioutput and self.priority_labels is not None:
            priority_label = self.priority_labels[idx]
            encoding['area_labels'] = torch.tensor(area_label)
            encoding['priority_labels'] = torch.tensor(priority_label)
        else:
            encoding['labels'] = torch.tensor(area_label)
        
        return encoding


# Example usage
def main():
    # Initialize the classifier with multi-output support
    classifier = ComplaintClassifier(
        model_name="VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis",
        use_multioutput=True
    )
    
    train_dataset, val_dataset = classifier.load_and_preprocess_data(
        data_path="../../data/processed/train/train_classifier.csv",
        text_column="observaciones",  # Column with complaint text
        area_label_column="areaServicioDescripcion",  # Column with complaint category
        priority_label_column="prioridadDescripcion",  # Column with priority level
        test_size=0.2
    )
    
    # Apply domain adaptation techniques
    enhanced_train_dataset = classifier.apply_domain_adaptation(train_dataset)
    
    # Train the model
    classifier.train(
        train_dataset=enhanced_train_dataset,
        val_dataset=val_dataset,
        batch_size=16,
        epochs=3
    )
    
    # Save the trained model
    classifier.save_model("../../data/models/classifier/classifier_model")
    
    # Make predictions on social media text examples
    social_media_examples = [
        'Hace un año y medio tengo hecho un reclamo x vereda y pared de mi casa rotas x los árboles. Quien se va a hacer cargo si alguien se tropieza? Saquen los árboles y arreglen la vereda. Exp 15397/2023. Incidente número 7275781',
        'Hace 7 meses estoy reclamando por un Ficus (especie ilegal para estar en la vía pública) que me está rompiendo todos los caños de mi casa y nadie hace nada.',
        'Una basura el barrio, calles rotas, basura por todos lados, inseguridad cada vez más violenta, asco SAN ISIDRO ES UNA MIERDA',
        'Basura por todos lados',
        'Está gestión no es mejor!! Venga a recorrer Boulogne la mugre , veredas todas rotas.',
        'Por favor arreglar la mitad de la calle Uruguay que le pertenece al partido de San Isidro y está llena de pozos'
    ]
    
    results = classifier.predict(social_media_examples)
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Predicted area: {result['predicted_area']} (confidence: {result['area_confidence']:.4f})")
        if 'predicted_priority' in result:
            print(f"Predicted priority: {result['predicted_priority']} (confidence: {result['priority_confidence']:.4f})")
        print("---")

if __name__ == "__main__":
    main()