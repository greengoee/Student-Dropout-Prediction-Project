
# model_training.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("⚠️ LightGBM not available")

class StudentDropoutModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        
    def load_cleaned_data(self):
        """Load the cleaned dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            print("✅ Cleaned dataset loaded successfully!")
            print(f"Dataset Shape: {self.df.shape}")
            
            # Check target distribution
            if 'Target' in self.df.columns:
                target_dist = self.df['Target'].value_counts()
                print(f"Target distribution:\n{target_dist}")
                print(f"Target proportions:\n{target_dist / len(self.df)}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading cleaned data: {e}")
            return False
    
    def prepare_data(self):
        """Prepare data for model training"""
        print("\n=== DATA PREPARATION ===")
        
        # Separate features and target
        if 'Target' not in self.df.columns:
            print("❌ Target column not found!")
            return False
        
        # Create feature set (exclude target)
        feature_columns = [col for col in self.df.columns if col != 'Target']
        self.X = self.df[feature_columns]
        self.y = self.df['Target']
        
        print(f"Features: {len(feature_columns)}")
        print(f"Target classes: {self.y.unique()}")
        print(f"Target distribution:\n{self.y.value_counts()}")
        
        # Handle categorical features
        self.X = self.preprocess_features(self.X)
        
        # Encode target variable
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        print(f"Encoded target: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Split the data with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y_encoded, test_size=0.15, random_state=42, stratify=self.y_encoded
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Testing set: {self.X_test.shape}")
        
        return True
    
    def preprocess_features(self, X):
        """Preprocess features for model training"""
        X_processed = X.copy()
        
        # Handle categorical columns
        categorical_columns = X_processed.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            if X_processed[col].dtype == 'object' or X_processed[col].dtype.name == 'category':
                # Use label encoding for categorical variables
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                print(f"✅ Encoded categorical feature: {col}")
        
        return X_processed
    
    def handle_class_imbalance(self):
        """Handle class imbalance using SMOTE"""
        print("\n⚖️ Handling class imbalance with SMOTE...")
        
        # Check original distribution
        unique, counts = np.unique(self.y_train, return_counts=True)
        print(f"Original class distribution: {dict(zip(unique, counts))}")
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        
        # Check new distribution
        unique, counts = np.unique(self.y_train, return_counts=True)
        print(f"After SMOTE class distribution: {dict(zip(unique, counts))}")
        
        return True
    
    def create_advanced_models(self):
        """Create comprehensive set of advanced models"""
        models = {
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000, n_jobs=-1),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Add advanced models if available
        if XGB_AVAILABLE:
            models['XGBoost'] = XGBClassifier(
                random_state=42, 
                eval_metric='mlogloss', 
                n_jobs=-1,
                use_label_encoder=False
            )
        
        if LGBM_AVAILABLE:
            models['LightGBM'] = LGBMClassifier(random_state=42, n_jobs=-1)
        
        return models
    
    def train_models_with_advanced_tuning(self):
        """Train models with advanced hyperparameter tuning - OPTIMIZED VERSION"""
        print("\n=== ADVANCED MODEL TRAINING WITH HYPERPARAMETER TUNING ===")
        
        models = self.create_advanced_models()
        
        # OPTIMIZED parameter grids for faster convergence
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200],  # Reduced from [200, 300, 500]
                'max_depth': [10, 15, None], # Reduced from [15, 20, 25, None]
                'min_samples_split': [2, 5], # Reduced from [2, 5, 10]
                'min_samples_leaf': [1, 2],  # Reduced from [1, 2, 4]
                'max_features': ['sqrt', 'log2'], # Removed 0.5
            },
            'XGBoost': {
                'n_estimators': [100, 200],  # Reduced from [200, 300, 500]
                'max_depth': [6, 8],         # Reduced from [6, 8, 10]
                'learning_rate': [0.05, 0.1], # Reduced from [0.01, 0.05, 0.1]
                'subsample': [0.8, 0.9],     # Reduced from [0.8, 0.9, 1.0]
            },
            'LightGBM': {
                'n_estimators': [100, 200],  # Reduced from [200, 300, 500]
                'max_depth': [8, 12],        # Reduced from [8, 12, 15]
                'learning_rate': [0.05, 0.1], # Reduced from [0.01, 0.05, 0.1]
                'num_leaves': [31, 50],      # Reduced from [31, 50, 100]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],  # Reduced from [200, 300, 500]
                'learning_rate': [0.05, 0.1], # Reduced from [0.01, 0.05, 0.1]
                'max_depth': [5, 7],         # Reduced from [5, 7, 9]
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10],          # Reduced from [0.1, 1, 10, 100]
                'penalty': ['l2'],           # Simplified to only l2
                'solver': ['liblinear'],     # Simplified to only liblinear
                'max_iter': [1000]           # Reduced from 2000
            }
        }
        
        # Remove SVM from tuning as it's often slow
        if 'SVM' in models:
            print("⚠️ Skipping SVM hyperparameter tuning for speed...")
            del models['SVM']
        
        best_score = 0
        best_model_name = None
        
        for name, model in models.items():
            print(f"\n🎯 Training {name} with hyperparameter tuning...")
            
            try:
                # Get parameter grid for current model
                param_grid = param_grids.get(name, {})
                
                if param_grid:
                    # Use RandomizedSearchCV for faster tuning with fewer iterations
                    from sklearn.model_selection import RandomizedSearchCV
                    
                    # Create pipeline
                    pipeline = Pipeline([
                        ('scaler', RobustScaler()),
                        ('classifier', model)
                    ])
                    
                    # OPTIMIZED: Reduced n_iter and cv for faster execution
                    random_search = RandomizedSearchCV(
                        pipeline, 
                        {f'classifier__{k}': v for k, v in param_grid.items()},
                        n_iter=10,  # Reduced from 20
                        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),  # Reduced from 5
                        scoring='accuracy',
                        random_state=42,
                        n_jobs=-1,
                        verbose=1  # Added to see progress
                    )
                    
                    print(f"   Starting search for {name}...")
                    random_search.fit(self.X_train, self.y_train)
                    
                    # Store the best model
                    self.models[name] = {
                        'pipeline': random_search.best_estimator_,
                        'best_score': random_search.best_score_,
                        'best_params': random_search.best_params_
                    }
                    
                    print(f"✅ {name} - Best CV Accuracy: {random_search.best_score_:.4f}")
                    print(f"   Best parameters: {random_search.best_params_}")
                    
                    # Update best model
                    if random_search.best_score_ > best_score:
                        best_score = random_search.best_score_
                        best_model_name = name
                        self.best_model = random_search.best_estimator_
                        
                else:
                    # Fallback to basic training
                    pipeline = Pipeline([
                        ('scaler', RobustScaler()),
                        ('classifier', model)
                    ])
                    
                    cv_scores = cross_val_score(pipeline, self.X_train, self.y_train, cv=3, scoring='accuracy')  # Reduced cv
                    cv_mean = cv_scores.mean()
                    
                    pipeline.fit(self.X_train, self.y_train)
                    
                    self.models[name] = {
                        'pipeline': pipeline,
                        'best_score': cv_mean,
                        'best_params': {}
                    }
                    
                    print(f"✅ {name} - CV Accuracy: {cv_mean:.4f}")
                    
                    if cv_mean > best_score:
                        best_score = cv_mean
                        best_model_name = name
                        self.best_model = pipeline
                        
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n🏆 Best Model: {best_model_name} with CV Accuracy: {best_score:.4f}")
        return best_model_name
    
    def create_ensemble_model(self):
        """Create ensemble model from top performing models"""
        print("\n🤝 Creating Ensemble Model...")
        
        # Get top 2 models instead of 3 for speed
        top_models = sorted(
            [(name, model_info) for name, model_info in self.models.items()],
            key=lambda x: x[1]['best_score'],
            reverse=True
        )[:2]  # Reduced from 3 to 2
        
        print("Top models for ensemble:")
        for name, model_info in top_models:
            print(f"  - {name}: {model_info['best_score']:.4f}")
        
        # Create voting classifier
        estimators = []
        for name, model_info in top_models:
            # Extract the classifier from the pipeline
            classifier = model_info['pipeline'].named_steps['classifier']
            estimators.append((name, classifier))
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        # Create ensemble pipeline
        ensemble_pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('classifier', voting_clf)
        ])
        
        # Train ensemble with reduced cv
        ensemble_cv = cross_val_score(ensemble_pipeline, self.X_train, self.y_train, cv=3, scoring='accuracy').mean()  # Reduced cv
        ensemble_pipeline.fit(self.X_train, self.y_train)
        ensemble_test = accuracy_score(self.y_test, ensemble_pipeline.predict(self.X_test))
        
        print(f"✅ Ensemble CV Score: {ensemble_cv:.4f}")
        print(f"✅ Ensemble Test Score: {ensemble_test:.4f}")
        
        # Add ensemble to models
        self.models['Ensemble'] = {
            'pipeline': ensemble_pipeline,
            'best_score': ensemble_cv,
            'best_params': {}
        }
        
        # Update best model if ensemble is better
        current_best_score = max(model_info['best_score'] for model_info in self.models.values() if model_info['best_score'] is not None)
        if ensemble_cv > current_best_score:
            self.best_model = ensemble_pipeline
            print("🎯 Ensemble model selected as best!")
        
        return ensemble_pipeline
    
    def evaluate_models_comprehensive(self):
        """Comprehensive model evaluation"""
        print("\n=== COMPREHENSIVE MODEL EVALUATION ===")
        
        results = {}
        
        for name, model_info in self.models.items():
            pipeline = model_info['pipeline']
            
            # Predictions
            y_pred = pipeline.predict(self.X_test)
            y_pred_proba = pipeline.predict_proba(self.X_test)
            
            # Calculate multiple metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_score': model_info['best_score'],
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"\n📊 {name} Performance:")
            print(f"  Test Accuracy:  {accuracy:.4f}")
            print(f"  Test Precision: {precision:.4f}")
            print(f"  Test Recall:    {recall:.4f}")
            print(f"  Test F1-Score:  {f1:.4f}")
            print(f"  CV Accuracy:    {model_info['best_score']:.4f}")
            
            # Show class-wise performance for the best model
            if pipeline == self.best_model:
                print(f"\n🎯 BEST MODEL - {name} - Detailed Classification Report:")
                print(classification_report(self.y_test, y_pred, target_names=self.label_encoder.classes_))
        
        return results
    
    def feature_importance_analysis(self):
        """Analyze feature importance for tree-based models"""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        if self.best_model is None:
            print("❌ No best model found")
            return None
        
        # Check if best model is tree-based
        best_model_name = None
        for name, model_info in self.models.items():
            if model_info['pipeline'] == self.best_model:
                best_model_name = name
                break
        
        if best_model_name in ['Random Forest', 'XGBoost', 'Gradient Boosting', 'LightGBM']:
            try:
                # Extract the classifier from the pipeline
                if hasattr(self.best_model, 'named_steps'):
                    classifier = self.best_model.named_steps['classifier']
                else:
                    classifier = self.best_model
                
                if hasattr(classifier, 'feature_importances_'):
                    feature_importances = classifier.feature_importances_
                    feature_names = self.X.columns
                    
                    # Create importance dataframe
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': feature_importances
                    }).sort_values('importance', ascending=False)
                    
                    print("\n🎯 Top 10 Most Important Features:")
                    for i, row in importance_df.head(10).iterrows():
                        print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
                    
                    # Plot feature importance
                    plt.figure(figsize=(10, 6))
                    top_features = importance_df.head(10)
                    plt.barh(top_features['feature'], top_features['importance'])
                    plt.xlabel('Feature Importance')
                    plt.title(f'Top 10 Feature Importance - {best_model_name}')
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    plt.savefig('visualizations/feature_importance_best_model.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    print("✅ Feature importance plot saved")
                    
                    return importance_df
                    
            except Exception as e:
                print(f"⚠️ Feature importance analysis failed: {e}")
        
        print("⚠️ Feature importance not available for this model type")
        return None
    
    def create_confusion_matrix(self):
        """Create confusion matrix for the best model"""
        print("\n=== CONFUSION MATRIX ===")
        
        if self.best_model is None:
            print("❌ No best model found")
            return
        
        # Get predictions from best model
        y_pred_best = self.best_model.predict(self.X_test)
        
        # Create confusion matrix
        cm = confusion_matrix(self.y_test, y_pred_best)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix - Best Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrix_best_model.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Confusion matrix saved")
        
        return cm
    
    def save_models(self):
        """Save trained models and preprocessing objects"""
        print("\n=== SAVING MODELS ===")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save the best model
        if self.best_model:
            joblib.dump(self.best_model, 'models/best_model.pkl')
            print("✅ Best model saved as 'models/best_model.pkl'")
        
        # Save label encoder
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        print("✅ Label encoder saved")
        
        # Save all models
        for name, model_info in self.models.items():
            filename = f'models/{name.lower().replace(" ", "_")}.pkl'
            joblib.dump(model_info['pipeline'], filename)
            print(f"✅ {name} saved as {filename}")
        
        # Save feature names for future reference
        feature_info = {
            'feature_names': self.X.columns.tolist(),
            'target_names': self.label_encoder.classes_.tolist()
        }
        joblib.dump(feature_info, 'models/feature_info.pkl')
        print("✅ Feature information saved")
    
    def run_complete_training(self):
        """Run complete model training pipeline - OPTIMIZED VERSION"""
        print("🚀 STARTING ADVANCED MODEL TRAINING PIPELINE")
        
        # Step 1: Load cleaned data
        if not self.load_cleaned_data():
            return None
        
        # Step 2: Prepare data
        if not self.prepare_data():
            return None
        
        # Step 3: Handle class imbalance
        self.handle_class_imbalance()
        
        # Step 4: Train models with advanced tuning
        best_model_name = self.train_models_with_advanced_tuning()
        
        # Step 5: Create ensemble model
        ensemble_model = self.create_ensemble_model()
        
        # Step 6: Comprehensive evaluation
        results = self.evaluate_models_comprehensive()
        
        # Step 7: Feature importance analysis
        feature_importance = self.feature_importance_analysis()
        
        # Step 8: Create confusion matrix
        cm = self.create_confusion_matrix()
        
        # Step 9: Save models
        self.save_models()
        
        # Final evaluation
        if self.best_model:
            final_predictions = self.best_model.predict(self.X_test)
            final_accuracy = accuracy_score(self.y_test, final_predictions)
            final_f1 = f1_score(self.y_test, final_predictions, average='weighted')
            
            print(f"\n🎉 FINAL TRAINING RESULTS:")
            print(f"🏆 Best Model: {best_model_name}")
            print(f"📈 Test Accuracy: {final_accuracy:.4f}")
            print(f"🎯 Test F1-Score: {final_f1:.4f}")
            
            if final_accuracy >= 0.85:
                print("🔥 EXCELLENT PERFORMANCE ACHIEVED!")
            elif final_accuracy >= 0.80:
                print("✅ GOOD PERFORMANCE ACHIEVED!")
            else:
                print("⚠️ Performance needs improvement")
            
            # Show sample predictions
            print(f"\n🔍 SAMPLE PREDICTIONS (Best Model):")
            sample_indices = np.random.choice(len(self.X_test), 5, replace=False)  # Reduced from 8
            for i, idx in enumerate(sample_indices):
                actual = self.label_encoder.inverse_transform([self.y_test[idx]])[0]
                predicted = self.label_encoder.inverse_transform(self.best_model.predict([self.X_test.iloc[idx]]))[0]
                proba = self.best_model.predict_proba([self.X_test.iloc[idx]])[0]
                confidence = max(proba)
                correct = "✅" if actual == predicted else "❌"
                print(f"  {correct} Sample {i+1}: Actual={actual}, Predicted={predicted}, Confidence={confidence:.2f}")
        
        return results

# Enhanced prediction function
def predict_student_dropout(new_student_data, model_path='models/best_model.pkl', 
                           encoder_path='models/label_encoder.pkl',
                           feature_info_path='models/feature_info.pkl'):
    """Predict dropout probability for new student data"""
    try:
        # Load model, encoder, and feature info
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        feature_info = joblib.load(feature_info_path)
        
        # Ensure new data has the same features as training
        expected_features = feature_info['feature_names']
        
        # Create a dataframe with the expected features
        if isinstance(new_student_data, dict):
            new_df = pd.DataFrame([new_student_data])
        else:
            new_df = new_student_data.copy()
        
        # Add missing features with default values
        for feature in expected_features:
            if feature not in new_df.columns:
                new_df[feature] = 0  # or appropriate default value
        
        # Reorder columns to match training
        new_df = new_df[expected_features]
        
        # Make prediction
        prediction_encoded = model.predict(new_df)
        prediction_proba = model.predict_proba(new_df)
        
        # Decode prediction
        prediction = label_encoder.inverse_transform(prediction_encoded)
        
        # Create detailed results
        results = {
            'prediction': prediction[0],
            'probabilities': dict(zip(label_encoder.classes_, prediction_proba[0])),
            'confidence': np.max(prediction_proba[0]),
            'risk_level': 'HIGH' if prediction[0] == 'Dropout' and np.max(prediction_proba[0]) > 0.7 else 'MEDIUM' if prediction[0] == 'Dropout' else 'LOW'
        }
        
        return results
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None

if __name__ == "__main__":
    # Initialize and run model training
    model_trainer = StudentDropoutModel('data/cleaned_student_data.csv')
    results = model_trainer.run_complete_training()
    
    if results:
        print("\n" + "="*60)
        print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
    else:
        print("❌ Model training failed!")