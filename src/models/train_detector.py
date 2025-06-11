import pandas as pd
import numpy as np
import re
import os
import dill as pickle
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import warnings
warnings.filterwarnings('ignore')

# Bayesian optimization imports
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.acquisition import gaussian_ei

# Download NLTK required resources
try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK Punkt tokenizer already downloaded")
except LookupError:
    print("Downloading NLTK Punkt tokenizer...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
    print("NLTK Stopwords already downloaded")
except LookupError:
    print("Downloading NLTK Stopwords...")
    nltk.download('stopwords')

# Load Spanish stopwords
spanish_stopwords = set(nltk.corpus.stopwords.words('spanish'))
    
# Define feature extractors
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
        
        # Named entity patterns (simple regex approach instead of spaCy NER)
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
                # Use NLTK for tokenization instead of spaCy
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
    
    # Convert to lowercase and remove extra spaces
    text = text.lower().strip()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Replace hashtags with just the word
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove mentions but keep names for context
    text = re.sub(r'@(\w+)', r'\1', text)
    
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text

# Global variable to store validation data for optimization
global_X_val = None
global_y_val = None
global_base_pipeline = None

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

# Define the search space for Bayesian optimization
dimensions = [
    Integer(100, 1000, name='n_estimators'),
    Integer(3, 10, name='max_depth'),
    Real(0.01, 0.3, name='learning_rate'),
    Real(0.5, 1.0, name='subsample'),
    Real(0.5, 1.0, name='colsample_bytree'),
    Integer(10, 200, name='min_child_weight'),
    Real(0.0, 0.5, name='gamma'),
    Real(1.0, 10.0, name='scale_pos_weight'),  # For class imbalance
]

@use_named_args(dimensions=dimensions)
def objective(**params):
    """Objective function for Bayesian optimization"""
    from xgboost import XGBClassifier
    
    # Create XGBoost classifier with the specified parameters
    classifier = XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        **params
    )
    
    # Create full pipeline with the classifier
    pipeline = Pipeline([
        ('features', global_base_pipeline),
        ('classifier', classifier)
    ])
    
    # Use cross-validation to evaluate the model
    scores = cross_val_score(
        pipeline, 
        global_X_val, 
        global_y_val, 
        cv=3,  # 3-fold CV for speed
        scoring='f1',  # Use F1 score for binary classification
        n_jobs=-1
    )
    
    # We minimize, so return negative F1 score
    return -np.mean(scores)

def optimize_hyperparameters(X_val, y_val, n_calls=50):
    """Run Bayesian optimization to find best hyperparameters"""
    global global_X_val, global_y_val, global_base_pipeline
    
    # Set global variables for the objective function
    global_X_val = X_val
    global_y_val = y_val
    global_base_pipeline = create_base_feature_pipeline()
    
    print(f"Starting Bayesian optimization with {n_calls} iterations...")
    print(f"Training set size: {len(X_val)} samples")
    
    # Run Bayesian optimization
    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=n_calls,
        random_state=42,
        acq_func='EI',  # Expected Improvement
        n_jobs=-1,
        verbose=True
    )
    
    # Extract best parameters
    best_params = dict(zip([dim.name for dim in dimensions], result.x))
    best_score = -result.fun

    for key, value in best_params.items():
        if isinstance(value, np.integer):
            best_params[key] = int(value)
        elif isinstance(value, np.floating):
            best_params[key] = float(value)
    
    print(f"\nOptimization completed!")
    print(f"Best cross-validation F1 score: {best_score:.4f}")
    print(f"Best parameters: {best_params}")
    
    return best_params, best_score

def build_complaint_detector(hyperparams=None):
    """Build the classification pipeline with optional optimized hyperparameters"""
    from xgboost import XGBClassifier
    
    # Default parameters
    default_params = {
        'n_estimators': 500,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 10,
        'gamma': 0.0,
        'scale_pos_weight': 1.0,
    }
    
    # Use optimized parameters if provided
    if hyperparams:
        params = {**default_params, **hyperparams}
        print(f"Using optimized hyperparameters: {hyperparams}")
    else:
        params = default_params
        print("Using default hyperparameters")
    
    # Define feature extraction pipeline
    features = create_base_feature_pipeline()
    
    classifier = XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        **params
    )
    
    # Full pipeline
    pipeline = Pipeline([
        ('features', features),
        ('classifier', classifier)
    ])
    
    return pipeline

def load_and_prepare_data(comments_file, labeled_data_file=None):
    """Load comments and any labeled data available"""

    # Load comments
    df = pd.read_csv(comments_file)
    df = df.dropna(subset=['comment_text'])
    df = df.drop_duplicates(subset=['comment_text'])
    df = df.reset_index(drop=True)
    
    print(f"Loaded {len(df)} comments from {comments_file}")

    # Make sure we have the right column
    if 'comment_text' in df.columns:
        comment_col = 'comment_text'
    elif 'message' in df.columns:
        comment_col = 'message'
    else:
        raise ValueError("Could not find comments column in data")
    
    # Extract the raw text
    comments = df[comment_col].tolist()
    
    # If we have labeled data, use it for training
    if labeled_data_file and os.path.exists(labeled_data_file):
        labeled_df = pd.read_csv(labeled_data_file)

        labeled_df = labeled_df.dropna(subset=['text', 'is_valid_complaint'])
        
        # Make sure it has the expected columns
        required_cols = ['text', 'is_valid_complaint']
        if not all(col in labeled_df.columns for col in required_cols):
            raise ValueError(f"Labeled data must have columns: {required_cols}")
        
        return df, labeled_df
    else:
        # If no labeled data, return just Facebook data
        return df, None

def train_model_with_labeled_data(labeled_df, model_path="complaint_detector_model.pkl", optimize=True, n_calls=50):
    """Train the model with labeled data and optional hyperparameter optimization"""
    # Split data
    X = labeled_df['text']
    y = labeled_df['is_valid_complaint']
    
    # First split for train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    best_params = None
    
    if optimize and len(X_train) > 50:  # Only optimize if we have enough data
        print("Starting hyperparameter optimization...")
        best_params, best_score = optimize_hyperparameters(X_train, y_train, n_calls=n_calls)
        
        # Save optimization results
        optimization_results = {
            'best_params': best_params,
            'best_cv_score': best_score,
            'optimization_date': datetime.now().isoformat()
        }
        
        with open(f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            import json
            json.dump(optimization_results, f, indent=2)
        
        print(f"Optimization results saved. Best CV F1 score: {best_score:.4f}")
    else:
        print("Skipping optimization (not enough data or optimize=False)")
    
    # Build and train final model with best parameters
    pipeline = build_complaint_detector(best_params)
    
    # Use all training data for final model
    pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    test_f1 = f1_score(y_test, y_pred)
    
    print("\nFinal Model Evaluation on Test Set:")
    print(f"F1 Score: {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Save model with metadata
    model_metadata = {
        'model': pipeline,
        'best_params': best_params,
        'test_f1_score': test_f1,
        'training_date': datetime.now().isoformat(),
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    with open(model_path, 'wb') as file:
        pickle.dump(model_metadata, file)
    
    print(f"\nModel and metadata saved to {model_path}")
    return pipeline

def predict_and_analyze(df, model, comment_col='comment_text', threshold=0.5):
    """Make predictions on Facebook comments"""
    # Extract comments
    comments = df[comment_col].tolist()
    
    # Handle model loaded with metadata
    if isinstance(model, dict) and 'model' in model:
        print(f"Using model trained on {model['training_date']}")
        print(f"Test F1 score during training: {model['test_f1_score']:.4f}")
        if model['best_params']:
            print(f"Optimized parameters: {model['best_params']}")
        model = model['model']
    
    # Get probabilities
    probas = model.predict_proba(comments)
    
    # Add predictions to dataframe
    df['complaint_probability'] = probas[:, 1]
    df['is_valid_complaint'] = (probas[:, 1] >= threshold).astype(int)
    
    # Show some examples of valid complaints
    print("\nTop 5 Most Likely Valid Complaints:")
    valid_complaints = df[df['is_valid_complaint'] == 1].sort_values(
        'complaint_probability', ascending=False
    ).head(5)
    
    for i, (_, row) in enumerate(valid_complaints.iterrows()):
        print(f"\n{i+1}. Probability: {row['complaint_probability']:.3f}")
        print(f"Comment: {row[comment_col]}")
    
    # Show some examples of non-valid complaints/comments
    print("\nTop 5 Most Likely Non-Valid Comments:")
    non_valid = df[df['is_valid_complaint'] == 0].sort_values(
        'complaint_probability', ascending=True
    ).head(5)
    
    for i, (_, row) in enumerate(non_valid.iterrows()):
        print(f"\n{i+1}. Probability: {1 - row['complaint_probability']:.3f} (not valid)")
        print(f"Comment: {row[comment_col]}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'classified_comments_{timestamp}.csv'
    result_path = '../../data/models/detector/results'
    df.to_csv(os.path.join(result_path,result_file), index=False)
    print(f"\nResults saved to {os.path.join(result_path,result_file)}")
    
    # Summary stats
    valid_count = df['is_valid_complaint'].sum()
    total = len(df)
    print(f"\nSummary: Found {valid_count} valid complaints out of {total} comments ({valid_count/total*100:.1f}%)")
    
    return df

# Main execution functions
def bootstrap_label_data(df, sample_size=300, output_file="complaint_labels_template.xlsx"):
    """Create a template file for manual labeling"""
    # If we don't have enough data, use what we have
    sample_size = min(sample_size, len(df))
    
    # Get comment column
    if 'comment_text' in df.columns:
        comment_col = 'comment_text'
    elif 'message' in df.columns:
        comment_col = 'message'
    else:
        raise ValueError("Could not find comments column in Facebook data")
    
    # Take a stratified sample based on basic heuristics
    df['length'] = df[comment_col].apply(lambda x: len(str(x)) if isinstance(x, str) else 0)
    df['has_numbers'] = df[comment_col].apply(
        lambda x: 1 if isinstance(x, str) and bool(re.search(r'\d', x)) else 0
    )
    
    # Define a preliminary strata using length and number presence
    df['strata'] = df['length'].apply(
        lambda x: 0 if x < 50 else (1 if x < 150 else 2)
    ) * 2 + df['has_numbers']
    
    # Sample from each strata
    samples = []
    for strata in df['strata'].unique():
        strata_df = df[df['strata'] == strata]
        strata_size = max(1, int(sample_size * len(strata_df) / len(df)))
        
        if len(strata_df) > strata_size:
            samples.append(strata_df.sample(strata_size, random_state=42))
        else:
            samples.append(strata_df)
    
    # Combine samples and prepare labeling file
    sample_df = pd.concat(samples)
    
    # Make sure we don't exceed desired sample size
    if len(sample_df) > sample_size:
        sample_df = sample_df.sample(sample_size, random_state=42)
    
    # Create labeling template
    labeling_df = pd.DataFrame({
        'text': sample_df[comment_col],
        'is_valid_complaint': ''  # This will be filled by the person labeling
    })

    labeling_df.drop_duplicates(subset=['text'], inplace=True)
    labeling_df.reset_index(drop=True, inplace=True)
    
    # Save template
    labeling_df.to_excel(output_file, index=False)
    print(f"Created labeling template with {len(labeling_df)} samples at {output_file}")
    print("Please fill the 'is_valid_complaint' column with 1 (valid) or 0 (not valid)")
    
    return labeling_df

def run_active_learning(comments_file, labeled_data_file=None, model_path=None, optimize=True, n_calls=50):
    """Main function to run the active learning process"""
    # Load data
    df, labeled_df = load_and_prepare_data(comments_file, labeled_data_file)
    
    # Determine which comment column to use
    if 'comment_text' in df.columns:
        comment_col = 'comment_text'
    elif 'message' in df.columns:
        comment_col = 'message'
    else:
        raise ValueError("Could not find comments column in Facebook data")
    
    # If no labeled data, create a template for labeling
    if labeled_df is None:
        print("\nNo labeled data found. Creating a template for manual labeling...")
        bootstrap_label_data(df)
        print("\nPlease label the data and run this script again with the labeled file.")
        return
    
    # If we have labeled data, train the model
    print("\nTraining model with labeled data...")
    if optimize:
        print(f"Hyperparameter optimization enabled with {n_calls} iterations")
    else:
        print("Using default hyperparameters (optimization disabled)")
    
    model = train_model_with_labeled_data(labeled_df, model_path, optimize=optimize, n_calls=n_calls)
    
    # Make predictions on the full dataset
    print("\nMaking predictions on all comments...")
    result_df = predict_and_analyze(df, model, comment_col)
    
    print("\nProcess complete. You can now:")
    print("1. Review the output file with predictions")
    print("2. Add more labeled examples to improve the model")
    print("3. Adjust the threshold if needed (current: 0.7)")
    if optimize:
        print("4. Check the optimization results file for hyperparameter details")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Social Media Complaint Classifier with Bayesian Optimization')
    parser.add_argument('--raw', help='Path to the social media comments CSV file', default='../../data/raw/facebook/fb_page_comments_20250604_163120.csv')
    parser.add_argument('--labeled', help='Path to labeled data CSV file (if available)', default='../../data/processed/train/train_detector.csv')
    parser.add_argument('--model', help='Path to save/load the model', default='../../data/models/detector/detector_model/complaint_detector_model.pkl')
    parser.add_argument('--optimize', action='store_true', default=True, help='Enable Bayesian optimization')
    parser.add_argument('--no-optimize', dest='optimize', action='store_false', help='Disable Bayesian optimization')
    parser.add_argument('--n-calls', type=int, default=50, help='Number of optimization iterations (default: 50)')
    
    args = parser.parse_args()
    
    run_active_learning(args.raw, args.labeled, args.model, args.optimize, args.n_calls)