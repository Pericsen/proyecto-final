import pandas as pd
import numpy as np
import os
import re
import dill as pickle
import json
from datetime import datetime
import argparse
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

# Add src path for model imports
import sys
from pathlib import Path
src_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_path))

# Try importing models and transformers
try:
    from models.train_classifier import ComplaintClassifier
    CLASSIFIER_AVAILABLE = True
except ImportError:
    print("Warning: ComplaintClassifier not available")
    CLASSIFIER_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import json
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: Transformers not available")
    TRANSFORMERS_AVAILABLE = False

class InferencePipeline:
    """Pipeline for complaint detection and classification with multi-output support"""
    
    def __init__(self, detector_model_path=None, classifier_model_path=None):
        self.detector_model = None
        self.classifier_model = None
        self.detector_model_path = detector_model_path
        self.classifier_model_path = classifier_model_path
        self.use_multioutput = False
        
        # Area mapping for final output
        self.area_mapping = {
            'Arbolado Urbano': 'Arbolado Urbano',
            'Alumbrado': 'Alumbrado', 
            'Higiene Urbana': 'Higiene Urbana',
            'CLIBA': 'Higiene Urbana',  # Map CLIBA to Higiene Urbana
            'Obras Públicas': 'Obras Públicas',
            'Infraestructura Pública': 'Infraestructura Pública'
        }
        
        # Priority mapping for standardization
        self.priority_mapping = {
            'Alta': 'Alta',
            'Media': 'Media', 
            'Baja': 'Baja'
        }
        
    def load_detector_model(self, model_path=None):
        """Load the detector model"""
        if model_path is None:
            model_path = self.detector_model_path
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Detector model not found at {model_path}")
        
        print(f"Loading detector model from {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Handle different model formats
        if isinstance(model_data, dict) and 'model' in model_data:
            self.detector_model = model_data['model']
            print(f"Detector model loaded (trained on {model_data.get('training_date', 'unknown')})")
            print(f"Detector F1 score: {model_data.get('test_f1_score', 'unknown'):.4f}")
        else:
            self.detector_model = model_data
            print("Detector model loaded (legacy format)")
            
    def load_classifier_model(self, model_path=None):
        """Load the classifier model with multi-output support"""
        if model_path is None:
            model_path = self.classifier_model_path
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Classifier model not found at {model_path}")
        
        print(f"Loading classifier model from {model_path}")
        
        # If model_path is a directory, check for BERT/Transformers format
        if os.path.isdir(model_path):
            print(f"Model path is a directory, checking for BERT/Transformers format...")
            
            # Check for required BERT files
            required_files = ['config.json', 'tokenizer_config.json']
            model_files = ['pytorch_model.bin', 'model.safetensors']
            
            has_config = any(os.path.exists(os.path.join(model_path, f)) for f in required_files)
            has_model = any(os.path.exists(os.path.join(model_path, f)) for f in model_files)
            
            # Check for label mappings to determine if multi-output
            label_mappings_path = os.path.join(model_path, 'label_mappings.json')
            if os.path.exists(label_mappings_path):
                with open(label_mappings_path, 'r') as f:
                    mappings = json.load(f)
                    self.use_multioutput = mappings.get("use_multioutput", False)
                    print(f"Multi-output model detected: {self.use_multioutput}")
            
            if has_config and has_model:
                print("Found BERT/Transformers model files")
                
                # Try loading with ComplaintClassifier first
                if CLASSIFIER_AVAILABLE:
                    try:
                        self.classifier_model = ComplaintClassifier(model_name=model_path, use_multioutput=self.use_multioutput)
                        self.classifier_model.load_model(model_path)
                        print("Classifier model loaded (ComplaintClassifier format)")
                        return
                    except Exception as e:
                        print(f"Failed to load with ComplaintClassifier: {e}")
                
                # Fallback to direct transformers loading
                if TRANSFORMERS_AVAILABLE:
                    try:
                        # Load label mappings
                        if os.path.exists(label_mappings_path):
                            with open(label_mappings_path, 'r') as f:
                                mappings = json.load(f)
                                self.area_label_decoder = mappings.get("area_label_decoder", {})
                                if self.use_multioutput:
                                    self.priority_label_decoder = mappings.get("priority_label_decoder", {})
                        else:
                            print("Warning: No label_mappings.json found")
                            self.area_label_decoder = {}
                            self.priority_label_decoder = {}
                        
                        # Load tokenizer
                        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        
                        if self.use_multioutput:
                            # Load custom multi-output model
                            from models.train_classifier import MultiOutputBertClassifier
                            
                            # Use original model name to initialize BERT base
                            original_model_name = "VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis"
                            
                            self.model = MultiOutputBertClassifier(
                                bert_model_name=original_model_name,
                                num_area_classes=len(self.area_label_decoder),
                                num_priority_classes=len(self.priority_label_decoder)
                            ).to(self.device)
                            
                            # Load state dict
                            state_dict_path = os.path.join(model_path, 'pytorch_model.bin')
                            if os.path.exists(state_dict_path):
                                state_dict = torch.load(state_dict_path, map_location=self.device)
                                self.model.load_state_dict(state_dict)
                            else:
                                raise FileNotFoundError(f"Model weights not found at {state_dict_path}")
                        else:
                            # Load standard model
                            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                            self.model.to(self.device)
                        
                        self.model.eval()
                        
                        print("Classifier model loaded (Direct Transformers format)")
                        self.classifier_model = self  # Set self as the model for compatibility
                        return
                    except Exception as e:
                        print(f"Failed to load with direct transformers: {e}")
            
            # Look for pickle files in directory
            pickle_files = [f for f in os.listdir(model_path) if f.endswith('.pkl')]
            if pickle_files:
                pickle_path = os.path.join(model_path, pickle_files[0])
                print(f"Found pickle file: {pickle_path}")
                try:
                    with open(pickle_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    if isinstance(model_data, dict) and 'model' in model_data:
                        self.classifier_model = model_data['model']
                        print(f"Classifier model loaded from pickle (trained on {model_data.get('training_date', 'unknown')})")
                    else:
                        self.classifier_model = model_data
                        print("Classifier model loaded from pickle (legacy format)")
                    return
                except Exception as e:
                    print(f"Failed to load pickle file: {e}")
            
            # List contents to help debug
            contents = os.listdir(model_path)
            print(f"Directory contents: {contents}")
            raise Exception(f"No compatible model files found in directory {model_path}")
        
        else:
            # model_path is a file
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                if isinstance(model_data, dict) and 'model' in model_data:
                    self.classifier_model = model_data['model']
                    print(f"Classifier model loaded (trained on {model_data.get('training_date', 'unknown')})")
                else:
                    self.classifier_model = model_data
                    print("Classifier model loaded (legacy format)")
            except Exception as e:
                raise Exception(f"Failed to load classifier model: {e}")
    
    def predict_with_transformers(self, texts, max_length=512, batch_size=16):
        """Direct prediction with transformers model (multi-output support)"""
        if not hasattr(self, 'model') or not hasattr(self, 'tokenizer'):
            raise ValueError("Transformers model not properly loaded")
        
        area_predictions = []
        area_probabilities = []
        priority_predictions = []
        priority_probabilities = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                if self.use_multioutput:
                    # Multi-output model
                    area_logits = outputs['area_logits']
                    priority_logits = outputs['priority_logits']
                    
                    area_probs = torch.softmax(area_logits, dim=-1)
                    area_preds = torch.argmax(area_logits, dim=-1)
                    
                    priority_probs = torch.softmax(priority_logits, dim=-1)
                    priority_preds = torch.argmax(priority_logits, dim=-1)
                    
                    # Convert to numpy and extend lists
                    area_predictions.extend(area_preds.cpu().numpy())
                    area_probabilities.extend(area_probs.cpu().numpy())
                    priority_predictions.extend(priority_preds.cpu().numpy())
                    priority_probabilities.extend(priority_probs.cpu().numpy())
                else:
                    # Single output model
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    preds = torch.argmax(logits, dim=-1)
                    
                    area_predictions.extend(preds.cpu().numpy())
                    area_probabilities.extend(probs.cpu().numpy())
        
        if self.use_multioutput:
            return (np.array(area_predictions), np.array(area_probabilities), 
                    np.array(priority_predictions), np.array(priority_probabilities))
        else:
            return np.array(area_predictions), np.array(area_probabilities)
    
    def load_data(self, data_path):
        """Load data from predict folder"""
        if os.path.isfile(data_path):
            # Single file
            print(f"Loading data from {data_path}")
            return self._load_single_file(data_path)
        elif os.path.isdir(data_path):
            # Directory with multiple files
            print(f"Loading data from directory {data_path}")
            return self._load_directory(data_path)
        else:
            raise FileNotFoundError(f"Data path not found: {data_path}")
    
    def _load_single_file(self, file_path):
        """Load a single CSV file"""
        df = pd.read_csv(file_path)
        
        # Add platform info if not present
        if 'platform' not in df.columns:
            if 'fb_' in file_path.lower() or 'facebook' in file_path.lower():
                df['platform'] = 'facebook'
            elif 'ig_' in file_path.lower() or 'instagram' in file_path.lower():
                df['platform'] = 'instagram'
            else:
                df['platform'] = 'unknown'
        
        return df
    
    def _load_directory(self, dir_path):
        """Load and combine all CSV files from directory"""
        csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {dir_path}")
        
        dataframes = []
        for file in csv_files:
            file_path = os.path.join(dir_path, file)
            df = self._load_single_file(file_path)
            dataframes.append(df)
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"Loaded {len(csv_files)} files with {len(combined_df)} total comments")
        
        return combined_df
    
    def preprocess_data(self, df):
        """Preprocess data for inference"""
        print(f"Preprocessing {len(df)} comments...")
        
        # Identify text column
        text_columns = ['comment_text', 'text', 'message', 'content']
        text_col = None
        for col in text_columns:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            raise ValueError(f"No text column found. Expected one of: {text_columns}")
        
        print(f"Using '{text_col}' as text column")
        
        # Store original dataframe
        df_original = df.copy()
        
        # Clean data for processing
        df_clean = df.copy()
        df_clean = df_clean.dropna(subset=[text_col])
        df_clean = df_clean[df_clean[text_col].str.strip() != '']
        
        # Remove duplicates but keep track of original indices
        df_clean['original_index'] = df_clean.index
        df_clean = df_clean.drop_duplicates(subset=[text_col], keep='first')
        
        print(f"After cleaning: {len(df_clean)} unique non-empty comments")
        
        return df_original, df_clean, text_col
    
    def run_detector(self, df_clean, text_col, threshold=0.5):
        """Run complaint detection"""
        if self.detector_model is None:
            raise ValueError("Detector model not loaded")
        
        print(f"Running detector on {len(df_clean)} comments...")
        
        # Get text for prediction
        texts = df_clean[text_col].tolist()
        
        # Get predictions
        probabilities = self.detector_model.predict_proba(texts)
        predictions = (probabilities[:, 1] >= threshold).astype(int)
        
        # Add predictions to dataframe
        df_clean['complaint_probability'] = probabilities[:, 1]
        df_clean['is_valid_complaint'] = predictions
        
        # Filter to only valid complaints
        valid_complaints = df_clean[df_clean['is_valid_complaint'] == 1].copy()
        
        print(f"Detector found {len(valid_complaints)} valid complaints out of {len(df_clean)} comments ({len(valid_complaints)/len(df_clean)*100:.1f}%)")
        
        return df_clean, valid_complaints
    
    def run_classifier(self, valid_complaints, text_col):
        """Run complaint classification with multi-output support"""
        if self.classifier_model is None:
            raise ValueError("Classifier model not loaded")
        
        if len(valid_complaints) == 0:
            print("No valid complaints to classify")
            return valid_complaints
        
        print(f"Running classifier on {len(valid_complaints)} valid complaints...")
        
        # Get text for classification
        texts = valid_complaints[text_col].tolist()
        
        # Handle different model types
        if hasattr(self.classifier_model, 'predict') and hasattr(self.classifier_model, 'area_label_decoder'):
            # ComplaintClassifier format with multi-output support
            predictions = self.classifier_model.predict(texts)
            
            # Extract area and priority predictions
            if self.use_multioutput and isinstance(predictions[0], dict):
                area_predictions = [pred['predicted_area'] for pred in predictions]
                area_confidences = [pred['area_confidence'] for pred in predictions]
                priority_predictions = [pred['predicted_priority'] for pred in predictions]
                priority_confidences = [pred['priority_confidence'] for pred in predictions]
                
                class_names = area_predictions
                priority_names = priority_predictions
            else:
                # Single output format
                if isinstance(predictions[0], dict):
                    area_predictions = [pred['predicted_area'] for pred in predictions]
                    area_confidences = [pred['area_confidence'] for pred in predictions]
                    class_names = area_predictions
                    priority_names = [''] * len(area_predictions)
                    priority_confidences = [0.0] * len(area_predictions)
                else:
                    class_names = predictions
                    priority_names = [''] * len(predictions)
                    area_confidences = [1.0] * len(predictions)  # Default confidence
                    priority_confidences = [0.0] * len(predictions)
                    
        elif hasattr(self, 'model') and hasattr(self, 'tokenizer'):
            # Direct transformers format with multi-output support
            if self.use_multioutput:
                area_predictions, area_probabilities, priority_predictions, priority_probabilities = self.predict_with_transformers(texts)
                
                # Convert predictions to class names
                class_names = []
                priority_names = []
                area_confidences = []
                priority_confidences = []
                
                for area_pred, area_probs, priority_pred, priority_probs in zip(
                    area_predictions, area_probabilities, priority_predictions, priority_probabilities
                ):
                    # Area prediction
                    if hasattr(self, 'area_label_decoder') and str(area_pred) in self.area_label_decoder:
                        area_name = self.area_label_decoder[str(area_pred)]
                    elif hasattr(self, 'area_label_decoder') and area_pred in self.area_label_decoder:
                        area_name = self.area_label_decoder[area_pred]
                    else:
                        area_name = f'Area_{area_pred}'
                    
                    # Priority prediction
                    if hasattr(self, 'priority_label_decoder') and str(priority_pred) in self.priority_label_decoder:
                        priority_name = self.priority_label_decoder[str(priority_pred)]
                    elif hasattr(self, 'priority_label_decoder') and priority_pred in self.priority_label_decoder:
                        priority_name = self.priority_label_decoder[priority_pred]
                    else:
                        priority_name = f'Priority_{priority_pred}'
                    
                    class_names.append(area_name)
                    priority_names.append(priority_name)
                    area_confidences.append(float(np.max(area_probs)))
                    priority_confidences.append(float(np.max(priority_probs)))
            else:
                # Single output
                predictions, probabilities = self.predict_with_transformers(texts)
                
                class_names = []
                area_confidences = []
                for pred, probs in zip(predictions, probabilities):
                    if hasattr(self, 'area_label_decoder') and str(pred) in self.area_label_decoder:
                        class_names.append(self.area_label_decoder[str(pred)])
                    elif hasattr(self, 'area_label_decoder') and pred in self.area_label_decoder:
                        class_names.append(self.area_label_decoder[pred])
                    else:
                        class_names.append(f'Class_{pred}')
                    area_confidences.append(float(np.max(probs)))
                
                priority_names = [''] * len(class_names)
                priority_confidences = [0.0] * len(class_names)
                    
        elif hasattr(self.classifier_model, 'predict'):
            # Sklearn pipeline format
            predictions = self.classifier_model.predict(texts)
            probabilities = self.classifier_model.predict_proba(texts)
            
            # For sklearn models, predictions might already be class names
            if isinstance(predictions[0], str):
                class_names = predictions
            else:
                class_names = [f'Class_{pred}' for pred in predictions]
            
            area_confidences = np.max(probabilities, axis=1)
            priority_names = [''] * len(class_names)
            priority_confidences = [0.0] * len(class_names)
        else:
            raise ValueError("Classifier model format not recognized")
        
        # Map to standardized area names
        mapped_areas = [self.area_mapping.get(area, area) for area in class_names]
        
        # Map to standardized priority names
        if self.use_multioutput:
            mapped_priorities = [self.priority_mapping.get(priority, priority) for priority in priority_names]
        else:
            mapped_priorities = [''] * len(mapped_areas)
        
        # Add classifications to dataframe
        valid_complaints['predicted_class'] = class_names
        valid_complaints['area_derivacion'] = mapped_areas
        valid_complaints['classification_confidence'] = area_confidences
        
        if self.use_multioutput:
            valid_complaints['predicted_priority'] = priority_names
            valid_complaints['priority_derivacion'] = mapped_priorities
            valid_complaints['priority_confidence'] = priority_confidences
        
        # Print classification summary
        area_counts = pd.Series(mapped_areas).value_counts()
        print("\nArea classification results:")
        for area, count in area_counts.items():
            print(f"  {area}: {count} complaints")
        
        if self.use_multioutput:
            priority_counts = pd.Series(mapped_priorities).value_counts()
            print("\nPriority classification results:")
            for priority, count in priority_counts.items():
                if priority:  # Only show non-empty priorities
                    print(f"  {priority}: {count} complaints")
        
        return valid_complaints
    
    def merge_results(self, df_original, df_processed, df_classified):
        """Merge all results back to original dataframe"""
        print("Merging results with original data...")
        
        # Start with original dataframe
        df_final = df_original.copy()
        
        # Initialize prediction columns
        df_final['complaint_probability'] = 0.0
        df_final['is_valid_complaint'] = 0
        df_final['area_derivacion'] = ''
        df_final['classification_confidence'] = 0.0
        
        if self.use_multioutput:
            df_final['priority_derivacion'] = ''
            df_final['priority_confidence'] = 0.0
        
        # Merge detector results
        if 'original_index' in df_processed.columns:
            # Map back using original indices
            for _, row in df_processed.iterrows():
                orig_idx = row['original_index']
                if orig_idx in df_final.index:
                    df_final.loc[orig_idx, 'complaint_probability'] = row['complaint_probability']
                    df_final.loc[orig_idx, 'is_valid_complaint'] = row['is_valid_complaint']
        
        # Merge classifier results
        if len(df_classified) > 0 and 'original_index' in df_classified.columns:
            for _, row in df_classified.iterrows():
                orig_idx = row['original_index']
                if orig_idx in df_final.index:
                    df_final.loc[orig_idx, 'area_derivacion'] = row['area_derivacion']
                    df_final.loc[orig_idx, 'classification_confidence'] = row['classification_confidence']
                    
                    if self.use_multioutput:
                        df_final.loc[orig_idx, 'priority_derivacion'] = row.get('priority_derivacion', '')
                        df_final.loc[orig_idx, 'priority_confidence'] = row.get('priority_confidence', 0.0)
        
        return df_final
    
    def create_output(self, df_final, text_col):
        """Create final output dataframe"""
        print("Creating final output...")
        
        # Required columns for output
        output_columns = [
            text_col,
            'is_valid_complaint', 
            'area_derivacion',
            'post_id',
            'comment_id'
        ]
        
        # Optional columns to include if available
        optional_columns = [
            'complaint_probability',
            'classification_confidence', 
            'platform',
            'comment_time',
            'username',
            'comment_likes',
            'post_time'
        ]
        
        # Multi-output specific columns
        if self.use_multioutput:
            optional_columns.extend([
                'priority_derivacion',
                'priority_confidence'
            ])
        
        # Build final columns list
        final_columns = []
        
        # Add required columns
        for col in output_columns:
            if col in df_final.columns:
                final_columns.append(col)
            else:
                print(f"Warning: Required column '{col}' not found")
        
        # Add optional columns if available
        for col in optional_columns:
            if col in df_final.columns:
                final_columns.append(col)
        
        # Create output dataframe
        df_output = df_final[final_columns].copy()
        
        # Rename text column to standard name
        if text_col != 'text':
            df_output = df_output.rename(columns={text_col: 'text'})
        
        print(f"Final output contains {len(df_output)} comments")
        print(f"Valid complaints: {df_output['is_valid_complaint'].sum()}")
        
        return df_output
    
    def save_output(self, df_output, output_path=None):
        """Save output to Excel file"""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f'complaint_predictions_{timestamp}.xlsx'
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Save to Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main results
            df_output.to_excel(writer, sheet_name='Predictions', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': [
                    'Total Comments',
                    'Valid Complaints',
                    'Percentage Valid',
                    'Multi-output Model',
                    'Processing Date'
                ],
                'Value': [
                    len(df_output),
                    df_output['is_valid_complaint'].sum(),
                    f"{df_output['is_valid_complaint'].mean()*100:.1f}%",
                    'Yes' if self.use_multioutput else 'No',
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Area breakdown if classifications exist
            if 'area_derivacion' in df_output.columns:
                area_breakdown = df_output[df_output['is_valid_complaint'] == 1]['area_derivacion'].value_counts().reset_index()
                area_breakdown.columns = ['Area', 'Count']
                area_breakdown.to_excel(writer, sheet_name='Area_Breakdown', index=False)
            
            # Priority breakdown if multi-output
            if self.use_multioutput and 'priority_derivacion' in df_output.columns:
                priority_breakdown = df_output[
                    (df_output['is_valid_complaint'] == 1) & 
                    (df_output['priority_derivacion'] != '')
                ]['priority_derivacion'].value_counts().reset_index()
                priority_breakdown.columns = ['Priority', 'Count']
                priority_breakdown.to_excel(writer, sheet_name='Priority_Breakdown', index=False)
        
        print(f"Results saved to {output_path}")
        return output_path
    
    def run_pipeline(self, data_path, output_path=None, detector_threshold=0.5):
        """Run the complete inference pipeline"""
        print("="*60)
        print("COMPLAINT DETECTION AND CLASSIFICATION PIPELINE")
        if self.use_multioutput:
            print("WITH MULTI-OUTPUT SUPPORT (AREA + PRIORITY)")
        print("="*60)
        
        # Load models
        if self.detector_model is None:
            self.load_detector_model()
        if self.classifier_model is None:
            self.load_classifier_model()
        
        # Load and preprocess data
        df_raw = self.load_data(data_path)
        df_original, df_clean, text_col = self.preprocess_data(df_raw)
        
        # Run detector
        df_processed, valid_complaints = self.run_detector(df_clean, text_col, detector_threshold)
        
        # Run classifier on valid complaints
        df_classified = self.run_classifier(valid_complaints, text_col)
        
        # Merge results
        df_final = self.merge_results(df_original, df_processed, df_classified)
        
        # Create output
        df_output = self.create_output(df_final, text_col)
        
        # Save results
        output_file = self.save_output(df_output, output_path)
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📊 Processed {len(df_original)} comments")
        print(f"🎯 Found {df_output['is_valid_complaint'].sum()} valid complaints")
        if self.use_multioutput:
            priority_classified = df_output[
                (df_output['is_valid_complaint'] == 1) & 
                (df_output.get('priority_derivacion', '') != '')
            ]
            print(f"🏆 Classified {len(priority_classified)} complaints with priority")
        print(f"📁 Results saved to {output_file}")
        
        return df_output, output_file

def main():
    parser = argparse.ArgumentParser(description='Run complaint detection and classification inference')
    parser.add_argument('--data', 
                        help='Path to input data (file or directory)', 
                        default='../../data/processed/predict/fb_page_comments_20250531_171518.csv')
    parser.add_argument('--detector', 
                        help='Path to detector model', 
                        default='../../data/models/detector/detector_model/complaint_detector_model.pkl')
    parser.add_argument('--classifier', 
                        help='Path to classifier model', 
                        default='../../data/models/classifier/classifier_model')
    parser.add_argument('--output', 
                        help='Output file path', 
                        default=None)
    parser.add_argument('--threshold', 
                        type=float, 
                        default=0.5, 
                        help='Detection threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = InferencePipeline(
        detector_model_path=args.detector,
        classifier_model_path=args.classifier
    )
    
    # Run pipeline
    try:
        results, output_file = pipeline.run_pipeline(
            data_path=args.data,
            output_path=args.output,
            detector_threshold=args.threshold
        )
        
        print(f"\n🎉 Success! Check results in {output_file}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()