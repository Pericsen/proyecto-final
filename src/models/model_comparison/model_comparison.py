import pandas as pd
import numpy as np
import re
import os
import dill as pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import warnings
warnings.filterwarnings('ignore')

# BERT/Transformers imports
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from torch.utils.data import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Transformers/PyTorch not available")
    TRANSFORMERS_AVAILABLE = False

# Model-specific imports
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available")
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("LightGBM not available")
    LIGHTGBM_AVAILABLE = False

# Bayesian optimization imports
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# Download NLTK required resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Load Spanish stopwords
spanish_stopwords = set(nltk.corpus.stopwords.words('spanish'))

# BERT Dataset class
class BERTDataset(Dataset):
    """Dataset class for BERT model"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTClassifier(BaseEstimator, TransformerMixin):
    """BERT classifier wrapper for sklearn compatibility"""
    
    def __init__(self, model_name="VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis", 
                 max_length=128, batch_size=16, learning_rate=2e-5, epochs=3):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, X, y):
        """Train the BERT model"""
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        self.model.to(self.device)
        
        # Create dataset
        train_dataset = BERTDataset(
            texts=X.tolist() if hasattr(X, 'tolist') else list(X),
            labels=y.tolist() if hasattr(y, 'tolist') else list(y),
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        # Training arguments - compatible with different transformers versions
        try:
            # Try new parameter name first
            training_args = TrainingArguments(
                output_dir='./bert_temp',
                num_train_epochs=self.epochs,
                per_device_train_batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                logging_steps=50,
                save_steps=1000,
                eval_strategy="no",
                save_strategy="no",
                load_best_model_at_end=False,
                report_to=None,
                disable_tqdm=True
            )
        except TypeError:
            # Fallback to old parameter name
            training_args = TrainingArguments(
                output_dir='./bert_temp',
                num_train_epochs=self.epochs,
                per_device_train_batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                logging_steps=50,
                save_steps=1000,
                evaluation_strategy="no",
                save_strategy="no",
                load_best_model_at_end=False,
                report_to=None,
                disable_tqdm=True
            )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        # Train the model
        print(f"Training BERT on {len(train_dataset)} samples...")
        trainer.train()
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not fitted yet")
        
        self.model.eval()
        predictions = []
        
        texts = X.tolist() if hasattr(X, 'tolist') else list(X)
        
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i+self.batch_size]
                
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(preds.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not fitted yet")
        
        self.model.eval()
        probabilities = []
        
        texts = X.tolist() if hasattr(X, 'tolist') else list(X)
        
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i+self.batch_size]
                
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def transform(self, X):
        """Transform method for pipeline compatibility"""
        return X

class ComplaintFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features that indicate valid complaints"""
    
    def __init__(self):
        self.location_words = [
            'calle', 'avenida', 'plaza', 'barrio', 'esquina', 'cuadra', 
            'vereda', 'manzana', 'casa', 'edificio', 'parque', 'puente',
            'zona', 'área', 'sector', 'distrito', 'pasaje', 'boulevard',
            'paseo', 'camino', 'carretera', 'autopista', 'ruta'
        ]
        self.infrastructure_words = [
            'poste', 'luz', 'agua', 'cloaca', 'alcantarilla', 'bache',
            'semáforo', 'señal', 'tránsito', 'basura', 'contenedor',
            'árbol', 'banco', 'asiento', 'parada', 'vereda', 'acera',
            'asfalto', 'pavimento', 'drenaje', 'desagüe', 'alumbrado',
            'luminaria', 'cámara', 'seguridad', 'incendio', 'escape',
            'fuga', 'pérdida', 'rotura', 'ruptura', 'caída', 'derrumbe'
        ]
        self.official_words = [
            'reclamo', 'expediente', 'exp', 'solicitud', 'número', 
            'trámite', 'registro', 'denuncia', 'municipalidad', 'queja',
            'referencia', 'caso', 'incidente', 'gestión', 'ticket', 
            'soporte', 'atención', 'servicio', 'seguimiento'
        ]
        self.action_verbs = [
            'arreglar', 'solucionar', 'reparar', 'reemplazar', 'instalar',
            'remover', 'limpiar', 'atender', 'revisar', 'inspeccionar', 
            'verificar', 'gestionar', 'modificar', 'actualizar', 'cambiar'
        ]
        
        # Named entity patterns
        self.location_patterns = [
            r'\b(?:en|sobre|cerca de|junto a|frente a|detrás de)\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+',
            r'\bcalle\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+',
            r'\bavenida\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+',
            r'\bplaza\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+',
            r'\bbarrio\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+'
        ]
        
        # Compile regex patterns for efficiency
        self.phone_pattern = re.compile(r'\b(?:\+?54)?(?:11|15)?[0-9]{8,10}\b')
        self.expedition_pattern = re.compile(r'\b(?:exp(?:ediente)?\.?\s?(?:n[°º]?\.?\s?)?[0-9-]+\/[0-9]{4})\b', re.IGNORECASE)
        self.incident_pattern = re.compile(r'\b(?:incidente|caso|ticket|reclamo)\s?(?:n[°º]?\.?\s?)?[0-9-]+\b', re.IGNORECASE)
        self.date_pattern = re.compile(r'\b(?:(?:0?[1-9]|[12][0-9]|3[01])[\/-](?:0?[1-9]|1[0-2])[\/-][0-9]{4}|(?:0?[1-9]|1[0-2])[\/-](?:0?[1-9]|[12][0-9]|3[01])[\/-][0-9]{4})\b')
        self.location_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.location_patterns]
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        """Extract complaint features from text"""
        features = np.zeros((len(X), 13))
        
        for i, text in enumerate(X):
            if isinstance(text, str):
                text_lower = text.lower()
                tokens = nltk.word_tokenize(text_lower, language='spanish')
                
                # Length features
                features[i, 0] = len(text)
                features[i, 1] = len(tokens)
                
                # Count types of words
                location_count = sum(1 for word in tokens if word in self.location_words)
                infra_count = sum(1 for word in tokens if word in self.infrastructure_words)
                official_count = sum(1 for word in tokens if word in self.official_words)
                action_count = sum(1 for word in tokens if word in self.action_verbs)
                
                features[i, 2] = location_count
                features[i, 3] = infra_count  
                features[i, 4] = official_count
                features[i, 5] = action_count
                
                # Simple location named entity detection using patterns
                location_entities = 0
                for pattern in self.location_patterns:
                    location_entities += len(re.findall(pattern, text))
                features[i, 6] = location_entities
                
                # Contains specific patterns
                features[i, 7] = 1 if re.search(self.phone_pattern, text) else 0
                features[i, 8] = 1 if re.search(self.expedition_pattern, text) else 0
                features[i, 9] = 1 if re.search(self.incident_pattern, text) else 0
                features[i, 10] = 1 if re.search(self.date_pattern, text) else 0
                
                # Contains question vs imperative sentence
                features[i, 11] = 1 if '?' in text else 0
                features[i, 12] = 1 if '!' in text else 0
        
        return features

def preprocess_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower().strip()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'@(\w+)', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text

def create_base_feature_pipeline():
    """Create the feature extraction pipeline"""
    return FeatureUnion([
        ('text_features', Pipeline([
            ('tfidf', TfidfVectorizer(
                preprocessor=preprocess_text,
                min_df=3, 
                max_df=0.8, 
                ngram_range=(1, 2),
                use_idf=True,
                smooth_idf=True,
                stop_words=list(spanish_stopwords)
            ))
        ])),
        ('complaint_features', ComplaintFeatureExtractor())
    ])

# Model configurations and parameter spaces
MODEL_CONFIGS = {
    'logistic_regression': {
        'class': LogisticRegression,
        'base_params': {
            'random_state': 42,
            'max_iter': 1000,
            'class_weight': 'balanced'
        },
        'param_space': [
            Real(0.001, 100, name='C', prior='log-uniform'),
            Categorical(['l1', 'l2', 'elasticnet'], name='penalty'),
            Real(0.1, 0.9, name='l1_ratio'),
            Categorical(['liblinear', 'saga'], name='solver')
        ],
        'available': True
    }
}

# Add XGBoost if available
if XGBOOST_AVAILABLE:
    MODEL_CONFIGS['xgboost'] = {
        'class': XGBClassifier,
        'base_params': {
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        },
        'param_space': [
            Integer(100, 1000, name='n_estimators'),
            Integer(3, 10, name='max_depth'),
            Real(0.01, 0.3, name='learning_rate'),
            Real(0.5, 1.0, name='subsample'),
            Real(0.5, 1.0, name='colsample_bytree'),
            Integer(10, 200, name='min_child_weight'),
            Real(0.0, 0.5, name='gamma'),
            Real(1.0, 10.0, name='scale_pos_weight')
        ],
        'available': True
    }

# Add LightGBM if available
if LIGHTGBM_AVAILABLE:
    MODEL_CONFIGS['lightgbm'] = {
        'class': LGBMClassifier,
        'base_params': {
            'random_state': 42,
            'verbose': -1,
            'class_weight': 'balanced'
        },
        'param_space': [
            Integer(100, 1000, name='n_estimators'),
            Integer(10, 100, name='num_leaves'),
            Real(0.01, 0.3, name='learning_rate'),
            Real(0.5, 1.0, name='feature_fraction'),
            Real(0.5, 1.0, name='bagging_fraction'),
            Integer(10, 200, name='min_child_samples'),
            Real(0.0, 1.0, name='reg_alpha'),
            Real(0.0, 1.0, name='reg_lambda')
        ],
        'available': True
    }

# Add BERT if available
if TRANSFORMERS_AVAILABLE:
    MODEL_CONFIGS['bert'] = {
        'class': BERTClassifier,
        'base_params': {
            'model_name': "VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis"
        },
        'param_space': [
            Integer(64, 256, name='max_length'),
            Integer(8, 32, name='batch_size'),
            Real(1e-5, 5e-5, name='learning_rate', prior='log-uniform'),
            Integer(2, 5, name='epochs')
        ],
        'available': True,
        'is_bert': True  # Special flag for BERT models
    }

# Global variables for optimization
global_X_val = None
global_y_val = None
global_base_pipeline = None
global_current_model_config = None

def create_objective_function(dimensions, model_config):
    """Create objective function for a specific model"""
    
    @use_named_args(dimensions=dimensions)
    def objective(**params):
        """Objective function for Bayesian optimization"""
        model_class = model_config['class']
        base_params = model_config['base_params'].copy()
        
        # Handle special cases for different models
        if model_class == LogisticRegression:
            # Handle solver compatibility
            if params.get('penalty') == 'elasticnet' and params.get('solver') not in ['saga']:
                params['solver'] = 'saga'
            elif params.get('penalty') == 'l1' and params.get('solver') not in ['liblinear', 'saga']:
                params['solver'] = 'saga'
            
            # Remove l1_ratio if not using elasticnet
            if params.get('penalty') != 'elasticnet':
                params.pop('l1_ratio', None)
        
        # Create classifier with the specified parameters
        classifier = model_class(**{**base_params, **params})
        
        # Handle BERT models differently
        if model_config.get('is_bert', False):
            # For BERT, we do simple train/validation split instead of CV
            # because CV with BERT is too computationally expensive
            try:
                X_train_bert, X_val_bert, y_train_bert, y_val_bert = train_test_split(
                    global_X_val, global_y_val, test_size=0.3, random_state=42, stratify=global_y_val
                )
                
                classifier.fit(X_train_bert, y_train_bert)
                y_pred = classifier.predict(X_val_bert)
                f1 = f1_score(y_val_bert, y_pred)
                
                # Clean up temporary files
                import shutil
                if os.path.exists('./bert_temp'):
                    shutil.rmtree('./bert_temp')
                
                return -f1
                
            except Exception as e:
                print(f"Error in BERT objective function: {e}")
                return 0
        else:
            # Traditional ML models with feature pipeline
            pipeline = Pipeline([
                ('features', global_base_pipeline),
                ('classifier', classifier)
            ])
            
            try:
                # Use cross-validation to evaluate the model
                scores = cross_val_score(
                    pipeline, 
                    global_X_val, 
                    global_y_val, 
                    cv=3,
                    scoring='f1',
                    n_jobs=1  # Avoid nested parallelism issues
                )
                return -np.mean(scores)
            except Exception as e:
                print(f"Error in objective function: {e}")
                return 0  # Return bad score if error
    
    return objective

def optimize_model(model_name, X_val, y_val, n_calls=30):
    """Optimize hyperparameters for a specific model"""
    global global_X_val, global_y_val, global_base_pipeline
    
    if model_name not in MODEL_CONFIGS or not MODEL_CONFIGS[model_name]['available']:
        print(f"Model {model_name} not available")
        return None, None
    
    config = MODEL_CONFIGS[model_name]
    
    # Set global variables
    global_X_val = X_val
    global_y_val = y_val
    global_base_pipeline = create_base_feature_pipeline()
    
    # Reduce optimization calls for BERT due to computational cost
    if config.get('is_bert', False):
        n_calls = min(n_calls, 10)  # Limit BERT optimization
        print(f"Optimizing {model_name.upper()} with {n_calls} iterations (reduced for BERT)...")
    else:
        print(f"Optimizing {model_name.upper()} with {n_calls} iterations...")
    
    # Create objective function with correct dimensions
    dimensions = config['param_space']
    objective_func = create_objective_function(dimensions, config)
    
    try:
        # Run Bayesian optimization
        result = gp_minimize(
            func=objective_func,
            dimensions=dimensions,
            n_calls=n_calls,
            random_state=42,
            acq_func='EI',
            n_jobs=1,
            verbose=False
        )
        
        # Extract best parameters
        best_params = dict(zip([dim.name for dim in dimensions], result.x))
        best_score = -result.fun
        
        # Convert numpy types to native Python types
        for key, value in best_params.items():
            if isinstance(value, np.integer):
                best_params[key] = int(value)
            elif isinstance(value, np.floating):
                best_params[key] = float(value)
        
        print(f"{model_name.upper()} - Best CV F1 score: {best_score:.4f}")
        print(f"{model_name.upper()} - Best parameters: {best_params}")
        
        return best_params, best_score
        
    except Exception as e:
        print(f"Error optimizing {model_name}: {e}")
        return None, None

def train_and_evaluate_model(model_name, X_train, X_test, y_train, y_test, best_params=None):
    """Train and evaluate a model with given parameters"""
    if model_name not in MODEL_CONFIGS or not MODEL_CONFIGS[model_name]['available']:
        return None
    
    config = MODEL_CONFIGS[model_name]
    model_class = config['class']
    base_params = config['base_params'].copy()
    
    # Use optimized parameters if provided
    if best_params:
        # Handle special cases for LogisticRegression
        if model_class == LogisticRegression and best_params:
            if best_params.get('penalty') != 'elasticnet':
                best_params.pop('l1_ratio', None)
        
        params = {**base_params, **best_params}
    else:
        params = base_params
    
    # Handle BERT models differently (no feature pipeline needed)
    if config.get('is_bert', False):
        # BERT doesn't need feature engineering pipeline
        model = model_class(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Clean up temporary files
        import shutil
        if os.path.exists('./bert_temp'):
            shutil.rmtree('./bert_temp')
        
    else:
        # Traditional ML models with feature pipeline
        pipeline = Pipeline([
            ('features', create_base_feature_pipeline()),
            ('classifier', model_class(**params))
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        model = pipeline
    
    # Calculate metrics
    metrics = {
        'f1_score': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    return {
        'model': model,
        'metrics': metrics,
        'best_params': best_params,
        'y_pred': y_pred,
        'y_proba': y_proba
    }

def compare_models(labeled_data_file, n_calls=30, test_size=0.2, models_to_test=None):
    """Compare all available models or specific models"""
    print("="*60)
    print("MODEL COMPARISON FOR COMPLAINT DETECTION")
    print("="*60)
    
    # Load and prepare data
    if not os.path.exists(labeled_data_file):
        raise FileNotFoundError(f"Labeled data file not found: {labeled_data_file}")
    
    labeled_df = pd.read_csv(labeled_data_file)
    labeled_df = labeled_df.dropna(subset=['text', 'is_valid_complaint'])
    
    print(f"Loaded {len(labeled_df)} labeled samples")
    print(f"Class distribution: {labeled_df['is_valid_complaint'].value_counts().to_dict()}")
    
    # Split data
    X = labeled_df['text']
    y = labeled_df['is_valid_complaint']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Further split training data for optimization
    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {len(X_train_opt)} samples")
    print(f"Validation set: {len(X_val_opt)} samples") 
    print(f"Test set: {len(X_test)} samples")
    
    # Store results
    results = {}
    
    # Determine which models to test
    if models_to_test:
        # Filter to only requested models that are available
        available_models = [name for name in models_to_test 
                          if name in MODEL_CONFIGS and MODEL_CONFIGS[name]['available']]
        if not available_models:
            print("❌ None of the requested models are available!")
            return None, None
    else:
        # Test all available models
        available_models = [name for name, config in MODEL_CONFIGS.items() if config['available']]
    
    print(f"\nModels to test: {available_models}")
    
    for model_name in available_models:
        print(f"\n{'-'*40}")
        print(f"Testing {model_name.upper()}")
        print(f"{'-'*40}")
        
        # Optimize hyperparameters
        best_params, best_cv_score = optimize_model(
            model_name, X_train_opt, y_train_opt, n_calls=n_calls
        )
        
        # Train and evaluate on test set
        result = train_and_evaluate_model(
            model_name, X_train, X_test, y_train, y_test, best_params
        )
        
        if result:
            result['cv_score'] = best_cv_score
            results[model_name] = result
            
            # Print results
            metrics = result['metrics']
            print(f"\nTest Set Results for {model_name.upper()}:")
            print(f"F1 Score:  {metrics['f1_score']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
            print(f"CV F1:     {best_cv_score:.4f}")
    
    # Print comparison summary
    print(f"\n{'='*60}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    if results:
        # Create comparison DataFrame
        comparison_data = []
        for model_name, result in results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name.upper(),
                'F1 Score': f"{metrics['f1_score']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'ROC AUC': f"{metrics['roc_auc']:.4f}",
                'CV F1': f"{result['cv_score']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['metrics']['f1_score'])
        best_f1 = results[best_model_name]['metrics']['f1_score']
        
        print(f"\n🏆 WINNER: {best_model_name.upper()} with F1 Score: {best_f1:.4f}")
        
        # Show detailed results for best model
        best_result = results[best_model_name]
        print(f"\nDetailed results for {best_model_name.upper()}:")
        print(f"Best parameters: {best_result['best_params']}")
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, best_result['y_pred'])
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y_test, best_result['y_pred']))
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'model_comparison_results_{timestamp}.json'
        
        # Prepare results for JSON serialization
        json_results = {}
        for model_name, result in results.items():
            json_results[model_name] = {
                'metrics': result['metrics'],
                'best_params': result['best_params'],
                'cv_score': result['cv_score']
            }
        
        json_results['best_model'] = best_model_name
        json_results['comparison_date'] = datetime.now().isoformat()
        json_results['models_tested'] = available_models
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
        # Save best model
        best_model_file = f'best_detector_model_{best_model_name}_{timestamp}.pkl'
        model_metadata = {
            'model': best_result['model'],
            'model_name': best_model_name,
            'best_params': best_result['best_params'],
            'test_metrics': best_result['metrics'],
            'cv_score': best_result['cv_score'],
            'training_date': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        with open(best_model_file, 'wb') as f:
            pickle.dump(model_metadata, f)
        
        print(f"Best model saved to {best_model_file}")
        
        return results, best_model_name
    
    else:
        print("No models were successfully trained!")
        return None, None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare different models for complaint detection')
    parser.add_argument('--labeled', 
                        help='Path to labeled data CSV file', 
                        default='../../../data/processed/train/train_detector.csv')
    parser.add_argument('--n-calls', 
                        type=int, 
                        default=30, 
                        help='Number of optimization iterations per model (default: 30)')
    parser.add_argument('--test-size', 
                        type=float, 
                        default=0.2, 
                        help='Test set proportion (default: 0.2)')
    parser.add_argument('--models', 
                        nargs='+', 
                        choices=['logistic_regression', 'xgboost', 'lightgbm', 'bert'],
                        help='Specific models to test (default: all available)')
    
    args = parser.parse_args()
    
    print("Starting model comparison...")
    print(f"Using {args.n_calls} optimization iterations per model")
    
    results, best_model = compare_models(
        labeled_data_file=args.labeled,
        n_calls=args.n_calls,
        test_size=args.test_size,
        models_to_test=args.models
    )
    
    if best_model:
        print(f"\n✅ Model comparison completed successfully!")
        print(f"🏆 Best model: {best_model.upper()}")
    else:
        print(f"\n❌ Model comparison failed!")