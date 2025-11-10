# main.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent memory issues
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
import csv
import re
warnings.filterwarnings('ignore')

# Import enhanced feature engineering
try:
    from src.feature_engineering import AdvancedFeatureEngineer
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    print("⚠️ Advanced feature engineering not available. Using basic features.")
    FEATURE_ENGINEERING_AVAILABLE = False

class StudentDropoutPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.final_df = None
        
    def clean_quotes_from_data(self, df):
        """Clean quotes from all string columns, especially the target column"""
        df_clean = df.copy()
        
        print("🧹 Cleaning quotes from data...")
        
        # Clean all object/string columns
        for col in df_clean.select_dtypes(include=['object']).columns:
            # Remove quotes and extra whitespace
            df_clean[col] = df_clean[col].astype(str).str.replace('"', '').str.strip()
            print(f"✅ Cleaned quotes from {col}")
            
            # Show unique values for target column
            if col == 'Target':
                print(f"Target values after cleaning: {df_clean[col].unique()}")
        
        return df_clean
    
    def diagnose_data_issues(self, df):
        """Comprehensive data quality diagnosis"""
        print("\n🔍 DATA QUALITY DIAGNOSIS")
        
        # Check target distribution
        if 'Target' in df.columns:
            target_dist = df['Target'].value_counts()
            print(f"Target distribution:\n{target_dist}")
            print(f"Majority class: {target_dist.max()/len(df)*100:.1f}%")
            
            # If one class dominates >90%, accuracy will be high but misleading
            if target_dist.max()/len(df) > 0.9:
                print("⚠️ WARNING: Severe class imbalance - accuracy is misleading!")
            else:
                print("✅ Good class balance")
        
        # Check for constant columns
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        if constant_cols:
            print(f"❌ Constant columns: {constant_cols}")
        else:
            print("✅ No constant columns")
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"❌ Duplicate rows: {duplicates}")
        else:
            print("✅ No duplicate rows")
        
        # Check data types
        print(f"\nData types:\n{df.dtypes.value_counts()}")
        
        # Check correlation with target for numerical columns
        if 'Target' in df.columns:
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            numerical_cols = [col for col in numerical_cols if col != 'Target']
            
            if len(numerical_cols) > 0:
                # Convert target to numerical for correlation
                target_encoded = pd.factorize(df['Target'])[0]
                
                correlations = []
                for col in numerical_cols:
                    if df[col].std() > 0:  # Avoid constant columns
                        corr = np.corrcoef(df[col], target_encoded)[0, 1]
                        correlations.append((col, corr))
                
                correlations.sort(key=lambda x: abs(x[1]), reverse=True)
                print(f"\nTop correlations with target:")
                for col, corr in correlations[:10]:
                    print(f"  {col}: {corr:.3f}")
    
    def manual_csv_parsing(self, lines):
        """Manual CSV parsing for problematic files"""
        try:
            # Parse header
            header_line = lines[0].strip()
            
            # Try different separators
            for sep in [',', ';', '\t', '|']:
                headers = header_line.split(sep)
                if len(headers) > 1:
                    print(f"✅ Found {len(headers)} columns with separator '{sep}'")
                    break
            else:
                # If no separator found, try regex
                headers = re.split(r'[,\t;|]', header_line)
                headers = [h.strip() for h in headers if h.strip()]
                print(f"✅ Found {len(headers)} columns with regex parsing")
            
            # Parse data rows
            data_rows = []
            for line in lines[1:]:
                if line.strip():
                    values = line.strip().split(sep, len(headers)-1)
                    # Pad or truncate to match header length
                    if len(values) < len(headers):
                        values += [''] * (len(headers) - len(values))
                    elif len(values) > len(headers):
                        values = values[:len(headers)]
                    data_rows.append(values)
            
            return pd.DataFrame(data_rows, columns=headers)
            
        except Exception as e:
            print(f"❌ Manual parsing failed: {e}")
            return None

    def find_target_column(self):
        """Find and validate target column"""
        target_candidates = [
            'Target', 'target', 'Dropout', 'dropout', 'Status', 'status',
            'Result', 'result', 'Class', 'class'
        ]
        
        for candidate in target_candidates:
            if candidate in self.df.columns:
                print(f"✅ Found target column: {candidate}")
                
                # Check if target has meaningful values
                unique_values = self.df[candidate].nunique()
                print(f"Target has {unique_values} unique values: {self.df[candidate].unique()}")
                
                if unique_values < 2:
                    print("❌ Target has only 1 unique value - cannot train model!")
                    return False
                    
                # Rename to standard 'Target'
                if candidate != 'Target':
                    self.df = self.df.rename(columns={candidate: 'Target'})
                    print(f"✅ Renamed '{candidate}' to 'Target'")
                    
                return True
        
        print("❌ No target column found! Available columns:")
        for col in self.df.columns:
            print(f"  - {col}")
        return False

    def load_data(self):
        """Load and explore the dataset with robust parsing"""
        try:
            print("🔍 Analyzing file structure...")
            
            # Read raw file to understand structure
            with open(self.data_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            if not lines:
                print("❌ File is empty!")
                return False
            
            # Show first few lines for debugging
            print("First 3 lines of file:")
            for i, line in enumerate(lines[:3]):
                print(f"Line {i}: {repr(line)}")
            
            # Try different parsing strategies
            parsing_strategies = [
                {'delimiter': ',', 'quotechar': '"'},
                {'delimiter': ';', 'quotechar': '"'},
                {'delimiter': '\t', 'quotechar': '"'},
                {'delimiter': ',', 'quotechar': "'"},
            ]
            
            for i, strategy in enumerate(parsing_strategies):
                try:
                    print(f"Trying strategy {i+1}: {strategy}")
                    self.df = pd.read_csv(self.data_path, **strategy, encoding='utf-8')
                    
                    if self.df.shape[1] > 1:  # Successfully parsed multiple columns
                        print(f"✅ Success with strategy {i+1}! Shape: {self.df.shape}")
                        break
                    else:
                        print(f"❌ Strategy {i+1} failed - only 1 column detected")
                        self.df = None
                except Exception as e:
                    print(f"❌ Strategy {i+1} error: {e}")
                    self.df = None
            
            # If all strategies fail, try manual parsing
            if self.df is None or self.df.shape[1] <= 1:
                print("🔄 Trying manual parsing...")
                self.df = self.manual_csv_parsing(lines)
            
            if self.df is None:
                print("❌ All parsing methods failed!")
                return False
            
            print(f"✅ Final dataset shape: {self.df.shape}")
            
            # Clean column names
            self.df.columns = [col.strip().replace('"', '').replace("'", "").replace('\t', '') for col in self.df.columns]
            print(f"📋 Cleaned columns: {list(self.df.columns)}")
            
            # Clean quotes from all data
            self.df = self.clean_quotes_from_data(self.df)
            
            # Run data diagnosis
            self.diagnose_data_issues(self.df)
            
            # Find target column
            if not self.find_target_column():
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def convert_data_types(self, df):
        """Convert columns to proper data types"""
        df_clean = df.copy()
        
        print("\n🔄 Converting data types...")
        
        # Define which columns should be numeric
        numeric_columns = [
            'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
            'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
            'Admission grade', 'Previous qualification (grade)', 'Age at enrollment',
            'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)',
            'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)',
            'Application order', 'Course'
        ]
        
        # Define which columns should be categorical (integer codes)
        categorical_columns = [
            'Marital status', 'Application mode', 'Previous qualification',
            'Displaced', 'Debtor', 'Tuition fees up to date',
            'Gender', 'Scholarship holder', 'International'
        ]
        
        # Convert numeric columns
        for col in numeric_columns:
            if col in df_clean.columns:
                # First clean the column - remove any non-numeric characters
                df_clean[col] = df_clean[col].astype(str).str.replace('[^0-9.-]', '', regex=True)
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                print(f"✅ Converted {col} to numeric")
        
        # Convert categorical columns
        for col in categorical_columns:
            if col in df_clean.columns:
                # Clean and convert to numeric
                df_clean[col] = df_clean[col].astype(str).str.replace('[^0-9.-]', '', regex=True)
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
                print(f"✅ Converted {col} to integer")
        
        return df_clean
    
    def handle_missing_data(self, df):
        """Handle missing values in the dataset"""
        df_clean = df.copy()
        
        # Check for missing values
        missing_values = df_clean.isnull().sum()
        missing_percentage = (missing_values / len(df_clean)) * 100
        
        print("\n=== MISSING VALUES ANALYSIS ===")
        missing_info = pd.DataFrame({
            'Missing Count': missing_values,
            'Missing Percentage': missing_percentage
        })
        
        # Show only columns with missing values
        missing_cols = missing_info[missing_info['Missing Count'] > 0]
        if len(missing_cols) > 0:
            print("Columns with missing values:")
            print(missing_cols)
            
            # Fill numerical missing values with median
            numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    print(f"✅ Filled missing values in {col} with median")
            
            # Fill categorical missing values with mode
            categorical_cols = df_clean.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                    print(f"✅ Filled missing values in {col} with mode")
        else:
            print("✅ No missing values found!")
        
        return df_clean
    
    def detect_and_treat_outliers(self, df):
        """Detect and treat outliers using IQR method on ALL numerical columns"""
        df_outlier_treated = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        outlier_report = {}
        
        print(f"\n=== OUTLIER TREATMENT ON ALL NUMERICAL COLUMNS ===")
        print(f"Processing {len(numerical_cols)} numerical columns...")
        
        for col in numerical_cols:
            try:
                # Skip if all values are the same or constant
                if df[col].nunique() <= 1:
                    outlier_report[col] = {'outlier_count': 0, 'percentage': 0, 'status': 'Skipped (constant)'}
                    continue
                
                # Calculate IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Skip if IQR is 0 (all values are very similar)
                if IQR == 0:
                    outlier_report[col] = {'outlier_count': 0, 'percentage': 0, 'status': 'Skipped (IQR=0)'}
                    continue
                
                # Calculate bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Detect outliers
                outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outliers_mask.sum()
                
                # Cap outliers - handle dtype properly
                if outlier_count > 0:
                    # Convert to float for outlier treatment to avoid dtype issues
                    temp_col = df_outlier_treated[col].astype(float)
                    temp_col = np.where(temp_col < lower_bound, lower_bound, temp_col)
                    temp_col = np.where(temp_col > upper_bound, upper_bound, temp_col)
                    
                    # Convert back to original dtype if possible
                    if df[col].dtype in [np.int64, np.int32]:
                        df_outlier_treated[col] = temp_col.astype(int)
                    else:
                        df_outlier_treated[col] = temp_col
                
                outlier_report[col] = {
                    'outlier_count': outlier_count,
                    'percentage': (outlier_count / len(df)) * 100,
                    'status': 'Treated'
                }
                
                if outlier_count > 0:
                    print(f"✅ Treated {outlier_count} outliers ({outlier_report[col]['percentage']:.2f}%) in {col}")
                else:
                    print(f"✅ No outliers found in {col}")
                    
            except Exception as e:
                print(f"❌ Error processing outliers in {col}: {e}")
                outlier_report[col] = {'outlier_count': 0, 'percentage': 0, 'status': f'Error: {str(e)}'}
        
        # Print summary
        total_outliers = sum([report['outlier_count'] for report in outlier_report.values()])
        print(f"\n📊 OUTLIER TREATMENT SUMMARY:")
        print(f"Total outliers treated: {total_outliers}")
        print(f"Columns processed: {len(numerical_cols)}")
        
        return df_outlier_treated, outlier_report
    
    def enhanced_feature_engineering(self, df):
        """Apply advanced feature engineering"""
        if not FEATURE_ENGINEERING_AVAILABLE:
            print("⚠️ Advanced feature engineering not available. Using basic feature engineering.")
            return self.create_new_features(df)
        
        print("\n=== ADVANCED FEATURE ENGINEERING ===")
        
        try:
            feature_engineer = AdvancedFeatureEngineer()
            
            # Create advanced features
            df_advanced = feature_engineer.create_advanced_features(df)
            
            # Create interaction features
            df_advanced = feature_engineer.create_interaction_features(df_advanced)
            
            # Remove low variance features
            df_advanced = feature_engineer.remove_low_variance_features(df_advanced)
            
            print(f"✅ Advanced feature engineering completed!")
            print(f"📊 Original features: {len(df.columns)}")
            print(f"📈 Enhanced features: {len(df_advanced.columns)}")
            print(f"🎯 New features created: {len(df_advanced.columns) - len(df.columns)}")
            
            return df_advanced
            
        except Exception as e:
            print(f"❌ Advanced feature engineering error: {e}")
            print("🔄 Falling back to basic feature engineering...")
            return self.create_new_features(df)
    
    def create_new_features(self, df):
        """Create basic engineered features (fallback method)"""
        print("\n=== BASIC FEATURE ENGINEERING ===")
        
        df_engineered = df.copy()
        
        # Convert numerical columns to proper numeric types
        numerical_cols = ['Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
                         'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
                         'Admission grade', 'Previous qualification (grade)', 'Age at enrollment']
        
        for col in numerical_cols:
            if col in df_engineered.columns:
                df_engineered[col] = pd.to_numeric(df_engineered[col], errors='coerce')
        
        # Create academic performance ratios
        if all(col in df_engineered.columns for col in ['Curricular units 1st sem (approved)', 'Curricular units 1st sem (enrolled)']):
            df_engineered['1st_sem_success_rate'] = (
                df_engineered['Curricular units 1st sem (approved)'] / 
                df_engineered['Curricular units 1st sem (enrolled)']
            ).replace([np.inf, -np.inf], 0).fillna(0)
            print("✅ Created: 1st_sem_success_rate")
        
        if all(col in df_engineered.columns for col in ['Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (enrolled)']):
            df_engineered['2nd_sem_success_rate'] = (
                df_engineered['Curricular units 2nd sem (approved)'] / 
                df_engineered['Curricular units 2nd sem (enrolled)']
            ).replace([np.inf, -np.inf], 0).fillna(0)
            print("✅ Created: 2nd_sem_success_rate")
        
        # Create overall performance indicator
        if 'Admission grade' in df_engineered.columns and 'Previous qualification (grade)' in df_engineered.columns:
            df_engineered['academic_potential'] = (
                df_engineered['Admission grade'] + df_engineered['Previous qualification (grade)']
            ) / 2
            print("✅ Created: academic_potential")
        
        # Create age categories
        if 'Age at enrollment' in df_engineered.columns:
            df_engineered['age_group'] = pd.cut(
                df_engineered['Age at enrollment'],
                bins=[17, 20, 25, 30, 60],
                labels=['17-20', '21-25', '26-30', '30+']
            )
            print("✅ Created: age_group")
        
        print("✅ Basic feature engineering completed!")
        return df_engineered
    
    def select_relevant_features(self, df):
        """Select only the most relevant features for student dropout prediction"""
        print("\n=== INTELLIGENT FEATURE SELECTION ===")
        
        # Enhanced feature selection including new engineered features
        high_priority_features = [
            # Academic performance (most important)
            'Curricular units 1st sem (approved)',
            'Curricular units 1st sem (grade)',
            'Curricular units 2nd sem (approved)', 
            'Curricular units 2nd sem (grade)',
            'Admission grade',
            'Previous qualification (grade)',
            
            # Engineered academic features
            '1st_sem_success_rate',
            '2nd_sem_success_rate', 
            'academic_potential',
            
            # Socio-economic factors
            'Scholarship holder',
            'Tuition fees up to date',
            'Debtor',
            
            # Demographic factors
            'Age at enrollment',
            'Gender',
            'International',
            'age_group',
            
            # Academic background
            'Previous qualification',
            'Application mode',
            
            # Attendance behavior
            'Daytime/evening attendance',
            'Displaced'
        ]
        
        medium_priority_features = [
            'Curricular units 1st sem (enrolled)',
            'Curricular units 1st sem (evaluations)',
            'Curricular units 2nd sem (enrolled)',
            'Curricular units 2nd sem (evaluations)',
            'Marital status',
            'Application order',
            'Course'
        ]
        
        # Select features that exist in our dataset
        selected_features = []
        
        # First, add high priority features that exist
        high_priority_count = 0
        for feature in high_priority_features:
            if feature in df.columns:
                selected_features.append(feature)
                high_priority_count += 1
                print(f"🎯 HIGH PRIORITY: {feature}")
        
        # Then add medium priority features that exist
        medium_priority_count = 0
        for feature in medium_priority_features:
            if feature in df.columns:
                selected_features.append(feature)
                medium_priority_count += 1
                print(f"📊 MEDIUM PRIORITY: {feature}")
        
        # Always include target if it exists
        if 'Target' in df.columns:
            selected_features.append('Target')
            print("✅ TARGET: Target")
        
        print(f"\n🎯 FINAL FEATURE SELECTION:")
        print(f"Selected {len(selected_features)} features out of {len(df.columns)} total columns")
        print(f"High priority: {high_priority_count}")
        print(f"Medium priority: {medium_priority_count}") 
        print(f"Features removed: {len(df.columns) - len(selected_features)}")
        
        return selected_features
    
    def create_simple_visualizations(self, df):
        """Create simplified visualizations to avoid memory issues"""
        print("\n=== CREATING SIMPLIFIED VISUALIZATIONS ===")
        
        # Create visualizations directory
        os.makedirs('visualizations', exist_ok=True)
        
        try:
            # 1. Simple target distribution
            if 'Target' in df.columns:
                plt.figure(figsize=(10, 6))
                target_counts = df['Target'].value_counts()
                target_counts.plot(kind='bar', color='lightblue')
                plt.title('Target Distribution')
                plt.xlabel('Target Category')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                
                # Add count labels on bars
                for i, count in enumerate(target_counts):
                    plt.text(i, count + 10, str(count), ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig('visualizations/target_distribution.png', dpi=150, bbox_inches='tight')
                plt.close()  # Close figure to free memory
                print("✅ Target distribution saved")
            
            # 2. Simple correlation matrix (only for key numerical features)
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 1:
                # Take only first 8 numerical columns to avoid memory issues
                key_numerical = numerical_cols[:8]
                plt.figure(figsize=(10, 8))
                
                correlation_matrix = df[key_numerical].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                           center=0, fmt='.2f', square=True)
                plt.title('Feature Correlation Matrix (Top 8 Features)')
                plt.tight_layout()
                plt.savefig('visualizations/correlation_matrix.png', dpi=150, bbox_inches='tight')
                plt.close()  # Close figure to free memory
                print("✅ Correlation matrix saved")
            
            # 3. Key feature distributions
            key_features = []
            if 'Curricular units 1st sem (grade)' in df.columns:
                key_features.append('Curricular units 1st sem (grade)')
            if 'Curricular units 2nd sem (grade)' in df.columns:
                key_features.append('Curricular units 2nd sem (grade)')
            if 'Admission grade' in df.columns:
                key_features.append('Admission grade')
            if 'Age at enrollment' in df.columns:
                key_features.append('Age at enrollment')
            
            if len(key_features) > 0:
                n_plots = min(4, len(key_features))
                n_rows = (n_plots + 1) // 2
                fig, axes = plt.subplots(n_rows, 2, figsize=(12, n_rows * 4))
                axes = axes.ravel() if n_plots > 1 else [axes]
                
                for i, col in enumerate(key_features[:n_plots]):
                    df[col].hist(bins=20, ax=axes[i], color='lightgreen', alpha=0.7)
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
                
                # Remove empty subplots
                for i in range(n_plots, len(axes)):
                    fig.delaxes(axes[i])
                
                plt.tight_layout()
                plt.savefig('visualizations/key_features_distribution.png', dpi=150, bbox_inches='tight')
                plt.close()  # Close figure to free memory
                print("✅ Key features distribution saved")
                
        except Exception as e:
            print(f"⚠️ Visualization error (continuing anyway): {e}")
        
        print("✅ Simplified visualizations completed")
    
    def train_prediction_model(self):
        """Train machine learning model for predictions"""
        print("\n=== MODEL TRAINING ===")
        
        try:
            # Import the model training class
            from advanced_model_training import UltraStudentDropoutModel 
            
            # Initialize and train model
            model_trainer = UltraStudentDropoutModel('data/cleaned_student_data.csv', n_iterations=3)
            results = model_trainer.run_smart_training()
            
            if results:
                print("✅ Model training completed successfully!")
                return True
            else:
                print("❌ Model training failed!")
                return False
                
        except Exception as e:
            print(f"❌ Error in model training: {e}")
            return False
    
    def run_enhanced_pipeline(self):
        """Run the enhanced data preprocessing pipeline with advanced features"""
        print("🚀 STARTING ENHANCED DATA PREPROCESSING PIPELINE")
        
        # Step 1: Load data
        if not self.load_data():
            print("❌ Failed to load data. Pipeline stopped.")
            return None
        
        # Step 2: Convert data types
        df_converted = self.convert_data_types(self.df)
        
        # Step 3: Handle missing values
        df_clean = self.handle_missing_data(df_converted)
        
        # Step 4: Enhanced feature engineering
        df_enhanced = self.enhanced_feature_engineering(df_clean)
        
        # Step 5: Treat outliers
        df_outlier_treated, outlier_report = self.detect_and_treat_outliers(df_enhanced)
        
        # Step 6: Intelligent feature selection
        relevant_features = self.select_relevant_features(df_outlier_treated)
        
        if len(relevant_features) == 0:
            print("⚠️ No relevant features found! Using all columns.")
            self.final_df = df_outlier_treated.copy()
        else:
            self.final_df = df_outlier_treated[relevant_features].copy()
        
        # Step 7: Simplified visualizations
        self.create_simple_visualizations(self.final_df)
        
        # Step 8: Save cleaned dataset
        output_path = 'data/cleaned_student_data.csv'
        os.makedirs('data', exist_ok=True)
        self.final_df.to_csv(output_path, index=False)
        
        # Verification
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            verified_df = pd.read_csv(output_path)
            print(f"✅ Enhanced dataset saved as '{output_path}'")
            print(f"✅ File size: {file_size} bytes")
            print(f"✅ Final dataset shape: {verified_df.shape}")
            print(f"✅ Final features: {len(verified_df.columns)}")
            
            # Show sample of data
            print(f"\n📊 DATA SAMPLE:")
            print(verified_df.head(3))
    
        # Summary
        print("\n" + "="*60)
        print("🎉 ENHANCED PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📊 Original dataset shape: {self.df.shape}")
        print(f"✨ Enhanced dataset shape: {self.final_df.shape}")
        print(f"🔧 New features created: {len(self.final_df.columns) - len(self.df.columns)}")
        print(f"🎯 Final feature count: {len(self.final_df.columns)}")
        if FEATURE_ENGINEERING_AVAILABLE:
            print("🚀 Advanced feature engineering: ENABLED")
        else:
            print("⚠️ Advanced feature engineering: NOT AVAILABLE")
        print("="*60)
        
        return self.final_df

    def run_complete_pipeline(self):
        """Run the complete data preprocessing pipeline (legacy method)"""
        return self.run_enhanced_pipeline()

# Main execution
if __name__ == "__main__":
    # Initialize the predictor
    predictor = StudentDropoutPredictor('data/data.csv')
    
    # Run the enhanced pipeline
    cleaned_data = predictor.run_enhanced_pipeline()
    
    if cleaned_data is not None:
        print("\n📁 FINAL CLEANED DATA PREVIEW:")
        print(cleaned_data.head())
        print(f"\n📋 FINAL DATASET INFO:")
        print(f"Shape: {cleaned_data.shape}")
        print(f"Total features: {len(cleaned_data.columns)}")
        
        # Check if Target column exists
        if 'Target' in cleaned_data.columns:
            target_counts = cleaned_data['Target'].value_counts()
            print(f"Target distribution:\n{target_counts}")
            
            # Ask if user wants to train model
            train_model = input("\n🤖 Do you want to train the prediction model? (y/n): ").lower().strip()
            if train_model == 'y':
                predictor.train_prediction_model()
            else:
                print("\n💡 You can train the model later by running: python model_training.py")
        else:
            print("❌ WARNING: Target column not found in final dataset!")
            print("Available columns:", cleaned_data.columns.tolist())
    else:
        print("❌ Pipeline failed to complete")