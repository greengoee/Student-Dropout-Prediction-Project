# advanced_model_training.py - OPTIMIZED VERSION
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
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

class UltraStudentDropoutModel:
    def __init__(self, data_path, n_iterations=5):  # Reduced iterations
        self.data_path = data_path
        self.n_iterations = n_iterations
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.label_encoder = LabelEncoder()
        self.best_accuracy = 0
        
    def load_cleaned_data(self):
        """Load the cleaned dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            print("✅ Cleaned dataset loaded successfully!")
            print(f"Dataset Shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"❌ Error loading cleaned data: {e}")
            return False
    
    def prepare_data_smart(self):
        """Smart data preparation - FIXED VERSION"""
        print("\n=== SMART DATA PREPARATION ===")
        
        if 'Target' not in self.df.columns:
            print("❌ Target column not found!")
            return False
        
        # Separate features and target FIRST
        feature_columns = [col for col in self.df.columns if col != 'Target']
        self.X = self.df[feature_columns]
        self.y = self.df['Target']
        
        print(f"Features: {len(feature_columns)}")
        print(f"Target classes: {self.y.unique()}")
        
        # Encode target variable
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        print(f"Encoded target: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        # Split data FIRST to avoid data leakage
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y_encoded, test_size=0.2, random_state=42, stratify=self.y_encoded
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Testing set: {self.X_test.shape}")
        
        # Simple preprocessing - no complex feature engineering
        self.X_train = self.smart_preprocessing(self.X_train)
        self.X_test = self.smart_preprocessing(self.X_test)
        
        return True
    
    def smart_preprocessing(self, X):
        """Simple and effective preprocessing"""
        X_processed = X.copy()
        
        # Handle categorical columns
        categorical_columns = X_processed.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            if X_processed[col].dtype == 'object' or X_processed[col].dtype.name == 'category':
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        
        # Fill missing values
        numerical_cols = X_processed.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if X_processed[col].isnull().any():
                X_processed[col].fillna(X_processed[col].median(), inplace=True)
        
        return X_processed
    
    def create_smart_features(self, X_train, X_test):
        """Create only meaningful features - FIXED"""
        print("🔧 Creating smart features...")
        
        # Only create features that make sense
        features_to_create = []
        
        # Success rates
        if all(col in X_train.columns for col in ['Curricular units 1st sem (approved)', 'Curricular units 1st sem (enrolled)']):
            X_train['1st_sem_success_rate'] = X_train['Curricular units 1st sem (approved)'] / (X_train['Curricular units 1st sem (enrolled)'] + 1e-8)
            X_test['1st_sem_success_rate'] = X_test['Curricular units 1st sem (approved)'] / (X_test['Curricular units 1st sem (enrolled)'] + 1e-8)
            features_to_create.append('1st_sem_success_rate')
        
        if all(col in X_train.columns for col in ['Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (enrolled)']):
            X_train['2nd_sem_success_rate'] = X_train['Curricular units 2nd sem (approved)'] / (X_train['Curricular units 2nd sem (enrolled)'] + 1e-8)
            X_test['2nd_sem_success_rate'] = X_test['Curricular units 2nd sem (approved)'] / (X_test['Curricular units 2nd sem (enrolled)'] + 1e-8)
            features_to_create.append('2nd_sem_success_rate')
        
        # Academic potential
        if all(col in X_train.columns for col in ['Admission grade', 'Previous qualification (grade)']):
            X_train['academic_potential'] = (X_train['Admission grade'] + X_train['Previous qualification (grade)']) / 2
            X_test['academic_potential'] = (X_test['Admission grade'] + X_test['Previous qualification (grade)']) / 2
            features_to_create.append('academic_potential')
        
        print(f"✅ Created {len(features_to_create)} smart features: {features_to_create}")
        return X_train, X_test
    
    def handle_imbalance_smart(self):
        """Smart imbalance handling"""
        print("\n⚖️ Handling class imbalance...")
        
        original_counts = np.unique(self.y_train, return_counts=True)
        print(f"Original distribution: {dict(zip(original_counts[0], original_counts[1]))}")
        
        # Use SMOTE - it's reliable and fast
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        
        new_counts = np.unique(self.y_train, return_counts=True)
        print(f"After SMOTE distribution: {dict(zip(new_counts[0], new_counts[1]))}")
        print(f"New training shape: {self.X_train.shape}")
        
        return True
    
    def create_focused_models(self):
        """Create focused set of effective models"""
        models = {
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'XGBoost': XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss') if XGB_AVAILABLE else None,
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        # Remove None models
        models = {k: v for k, v in models.items() if v is not None}
        return models
    
    def smart_hyperparameter_tuning(self, model_name, model):
        """Efficient hyperparameter tuning"""
        print(f"🎯 Tuning {model_name}...")
        
        # Simplified parameter grids
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [6, 8],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8, 0.9]
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['liblinear']
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [5, 7]
            }
        }
        
        param_grid = param_grids.get(model_name, {})
        
        if param_grid:
            # Use fewer iterations and folds for speed
            random_search = RandomizedSearchCV(
                model, param_grid, n_iter=10, cv=3,
                scoring='accuracy', random_state=42, n_jobs=-1,
                verbose=0
            )
            
            random_search.fit(self.X_train, self.y_train)
            best_model = random_search.best_estimator_
            best_score = random_search.best_score_
            
            print(f"✅ Best CV Score: {best_score:.4f}")
        else:
            # Basic training
            best_model = model
            best_model.fit(self.X_train, self.y_train)
            best_score = cross_val_score(best_model, self.X_train, self.y_train, cv=3, scoring='accuracy').mean()
            print(f"✅ CV Score: {best_score:.4f}")
        
        # Test accuracy
        test_accuracy = accuracy_score(self.y_test, best_model.predict(self.X_test))
        test_f1 = f1_score(self.y_test, best_model.predict(self.X_test), average='weighted')
        
        print(f"✅ Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
        
        return best_model, best_score, test_accuracy, test_f1
    
    def evaluate_model_comprehensively(self, model, model_name):
        """Comprehensive model evaluation"""
        print(f"\n📊 Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print(f"🎯 {model_name} Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Detailed report for best model
        if accuracy > self.best_accuracy:
            print(f"\n📈 Detailed Classification Report for {model_name}:")
            print(classification_report(self.y_test, y_pred, target_names=self.label_encoder.classes_))
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            plt.savefig(f'visualizations/confusion_matrix_{model_name.replace(" ", "_")}.png')
            plt.close()
            print("✅ Confusion matrix saved")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'model': model
        }
    
    def create_smart_ensemble(self, top_models):
        """Create smart ensemble"""
        print("\n🤝 Creating Smart Ensemble...")
        
        # Get top 2 models
        estimators = []
        for name, model_info in top_models[:2]:
            estimators.append((name, model_info['model']))
        
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        voting_clf.fit(self.X_train, self.y_train)
        
        # Evaluate ensemble
        ensemble_accuracy = accuracy_score(self.y_test, voting_clf.predict(self.X_test))
        ensemble_f1 = f1_score(self.y_test, voting_clf.predict(self.X_test), average='weighted')
        
        print(f"✅ Ensemble Test Accuracy: {ensemble_accuracy:.4f}")
        print(f"✅ Ensemble Test F1: {ensemble_f1:.4f}")
        
        return voting_clf, ensemble_accuracy, ensemble_f1
    
    def run_smart_training(self):
        """Run optimized training pipeline"""
        print("🚀 STARTING SMART MODEL TRAINING")
        print("🎯 Target: Reliable and Fast Training")
        
        # Step 1: Load data
        if not self.load_cleaned_data():
            return None
        
        # Step 2: Smart data preparation
        if not self.prepare_data_smart():
            return None
        
        # Step 3: Create smart features (AFTER split)
        self.X_train, self.X_test = self.create_smart_features(self.X_train, self.X_test)
        
        # Step 4: Handle imbalance
        self.handle_imbalance_smart()
        
        # Step 5: Train models
        models = self.create_focused_models()
        results = {}
        
        print(f"\n=== TRAINING {len(models)} MODELS ===")
        
        for name, model in models.items():
            try:
                print(f"\n--- Training {name} ---")
                best_model, cv_score, test_accuracy, test_f1 = self.smart_hyperparameter_tuning(name, model)
                
                results[name] = {
                    'model': best_model,
                    'cv_score': cv_score,
                    'test_accuracy': test_accuracy,
                    'test_f1': test_f1
                }
                
                # Update best model
                if test_accuracy > self.best_accuracy:
                    self.best_accuracy = test_accuracy
                    self.best_model = best_model
                    print(f"🎯 New best model: {name} with accuracy {test_accuracy:.4f}")
                    
            except Exception as e:
                print(f"❌ {name} training failed: {e}")
        
        # Step 6: Create ensemble from top models
        if len(results) >= 2:
            top_models = sorted(
                [(name, data) for name, data in results.items()],
                key=lambda x: x[1]['test_accuracy'],
                reverse=True
            )[:2]
            
            ensemble_model, ensemble_accuracy, ensemble_f1 = self.create_smart_ensemble(top_models)
            
            # Add ensemble to results
            results['Ensemble'] = {
                'model': ensemble_model,
                'cv_score': ensemble_accuracy,  # Using test accuracy as proxy
                'test_accuracy': ensemble_accuracy,
                'test_f1': ensemble_f1
            }
            
            # Update best model if ensemble is better
            if ensemble_accuracy > self.best_accuracy:
                self.best_accuracy = ensemble_accuracy
                self.best_model = ensemble_model
                print(f"🎯 Ensemble became best model with accuracy {ensemble_accuracy:.4f}")
        
        # Step 7: Comprehensive evaluation
        print(f"\n=== FINAL EVALUATION ===")
        for name, data in results.items():
            self.evaluate_model_comprehensively(data['model'], name)
        
        # Step 8: Save models
        self.save_models()
        
        # Final results
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"🏆 Best Model Accuracy: {self.best_accuracy:.4f}")
        
        if self.best_accuracy >= 0.85:
            print("✅ EXCELLENT PERFORMANCE!")
        elif self.best_accuracy >= 0.80:
            print("✅ GOOD PERFORMANCE!")
        else:
            print("⚠️ Performance needs improvement")
        
        return {
            'best_model': self.best_model,
            'best_accuracy': self.best_accuracy,
            'all_results': results
        }
    
    def save_models(self):
        """Save trained models"""
        print("\n💾 Saving models...")
        
        os.makedirs('models', exist_ok=True)
        
        # Save best model
        if self.best_model:
            joblib.dump(self.best_model, 'models/best_smart_model.pkl')
            print("✅ Best model saved as 'models/best_smart_model.pkl'")
        
        # Save label encoder
        joblib.dump(self.label_encoder, 'models/smart_label_encoder.pkl')
        print("✅ Label encoder saved")
        
        print("✅ All models saved successfully!")

# Update your main.py to use the smart training
def train_prediction_model(self):
    """Train machine learning model for predictions"""
    print("\n=== MODEL TRAINING ===")
    
    try:
        from advanced_model_training import UltraStudentDropoutModel 
        
        # Use SMART training instead of ULTRA
        model_trainer = UltraStudentDropoutModel('data/cleaned_student_data.csv', n_iterations=3)
        results = model_trainer.run_smart_training()  # Changed to run_smart_training
        
        if results:
            print("✅ Model training completed successfully!")
            return True
        else:
            print("❌ Model training failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        return False