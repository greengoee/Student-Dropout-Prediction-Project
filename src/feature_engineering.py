# src/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = None
    
    def create_advanced_features(self, df):
        """Create advanced engineered features"""
        df_eng = df.copy()
        
        print("🔧 Creating advanced features...")
        
        # 1. Academic Performance Features
        df_eng = self.create_academic_features(df_eng)
        
        # 2. Behavioral Patterns
        df_eng = self.create_behavioral_features(df_eng)
        
        # 3. Risk Indicators
        df_eng = self.create_risk_features(df_eng)
        
        # 4. Temporal Patterns
        df_eng = self.create_temporal_features(df_eng)
        
        # 5. Demographic Features
        df_eng = self.create_demographic_features(df_eng)
        
        print(f"✅ Created {len([col for col in df_eng.columns if col not in df.columns])} new features")
        return df_eng
    
    def create_academic_features(self, df):
        """Create academic performance features"""
        # Success rate progression
        if all(col in df.columns for col in ['Curricular units 1st sem (approved)', 'Curricular units 1st sem (enrolled)']):
            df['1st_sem_success_rate'] = (
                df['Curricular units 1st sem (approved)'] / 
                df['Curricular units 1st sem (enrolled)']
            ).replace([np.inf, -np.inf], 0).fillna(0)
        
        if all(col in df.columns for col in ['Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (enrolled)']):
            df['2nd_sem_success_rate'] = (
                df['Curricular units 2nd sem (approved)'] / 
                df['Curricular units 2nd sem (enrolled)']
            ).replace([np.inf, -np.inf], 0).fillna(0)
        
        # Performance trend (improvement/decline)
        if all(col in df.columns for col in ['1st_sem_success_rate', '2nd_sem_success_rate']):
            df['performance_trend'] = df['2nd_sem_success_rate'] - df['1st_sem_success_rate']
            df['is_improving'] = (df['performance_trend'] > 0).astype(int)
            df['significant_decline'] = (df['performance_trend'] < -0.2).astype(int)
        
        # Grade consistency
        if all(col in df.columns for col in ['Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']):
            df['grade_consistency'] = (
                df['Curricular units 1st sem (grade)'] - df['Curricular units 2nd sem (grade)']
            ).abs()
            df['grade_drop'] = (df['Curricular units 2nd sem (grade)'] < df['Curricular units 1st sem (grade)']).astype(int)
        
        # Overall academic score
        academic_cols = ['Admission grade', 'Previous qualification (grade)', 
                        'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']
        available_academic = [col for col in academic_cols if col in df.columns]
        if available_academic:
            df['overall_academic_score'] = df[available_academic].mean(axis=1)
            df['academic_volatility'] = df[available_academic].std(axis=1).fillna(0)
        
        # Credit efficiency
        if all(col in df.columns for col in ['Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)']):
            df['credit_efficiency_1st'] = (
                df['Curricular units 1st sem (credited)'] / 
                df['Curricular units 1st sem (enrolled)']
            ).replace([np.inf, -np.inf], 0).fillna(0)
        
        if all(col in df.columns for col in ['Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)']):
            df['credit_efficiency_2nd'] = (
                df['Curricular units 2nd sem (credited)'] / 
                df['Curricular units 2nd sem (enrolled)']
            ).replace([np.inf, -np.inf], 0).fillna(0)
        
        return df
    
    def create_behavioral_features(self, df):
        """Create behavioral pattern features"""
        # Course load management
        if all(col in df.columns for col in ['Curricular units 1st sem (enrolled)', 'Curricular units 2nd sem (enrolled)']):
            df['course_load_change'] = df['Curricular units 2nd sem (enrolled)'] - df['Curricular units 1st sem (enrolled)']
            df['reduced_course_load'] = (df['course_load_change'] < 0).astype(int)
            df['increased_course_load'] = (df['course_load_change'] > 0).astype(int)
            df['load_change_ratio'] = (
                df['Curricular units 2nd sem (enrolled)'] / 
                df['Curricular units 1st sem (enrolled)']
            ).replace([np.inf, -np.inf], 1).fillna(1)
        
        # Evaluation pattern
        if all(col in df.columns for col in ['Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (enrolled)']):
            df['evaluation_ratio_1st'] = (
                df['Curricular units 1st sem (evaluations)'] / 
                df['Curricular units 1st sem (enrolled)']
            ).replace([np.inf, -np.inf], 0).fillna(0)
            df['low_evaluation_1st'] = (df['evaluation_ratio_1st'] < 0.5).astype(int)
        
        if all(col in df.columns for col in ['Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (enrolled)']):
            df['evaluation_ratio_2nd'] = (
                df['Curricular units 2nd sem (evaluations)'] / 
                df['Curricular units 2nd sem (enrolled)']
            ).replace([np.inf, -np.inf], 0).fillna(0)
            df['low_evaluation_2nd'] = (df['evaluation_ratio_2nd'] < 0.5).astype(int)
        
        # Approval efficiency
        if all(col in df.columns for col in ['Curricular units 1st sem (approved)', 'Curricular units 1st sem (evaluations)']):
            df['approval_efficiency_1st'] = (
                df['Curricular units 1st sem (approved)'] / 
                df['Curricular units 1st sem (evaluations)']
            ).replace([np.inf, -np.inf], 0).fillna(0)
        
        if all(col in df.columns for col in ['Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (evaluations)']):
            df['approval_efficiency_2nd'] = (
                df['Curricular units 2nd sem (approved)'] / 
                df['Curricular units 2nd sem (evaluations)']
            ).replace([np.inf, -np.inf], 0).fillna(0)
        
        return df
    
    def create_risk_features(self, df):
        """Create risk indicator features"""
        # Financial risk
        financial_risk = 0
        if 'Debtor' in df.columns:
            financial_risk += df['Debtor'] * 3
        if 'Tuition fees up to date' in df.columns:
            financial_risk += (1 - df['Tuition fees up to date']) * 4
        if 'Scholarship holder' in df.columns:
            financial_risk -= df['Scholarship holder'] * 2
        df['financial_risk_score'] = financial_risk
        
        # Academic risk
        academic_risk = 0
        if '1st_sem_success_rate' in df.columns:
            academic_risk += (df['1st_sem_success_rate'] < 0.3) * 3
            academic_risk += (df['1st_sem_success_rate'] < 0.5) * 2
        if '2nd_sem_success_rate' in df.columns:
            academic_risk += (df['2nd_sem_success_rate'] < 0.3) * 4
            academic_risk += (df['2nd_sem_success_rate'] < 0.5) * 3
        if 'performance_trend' in df.columns:
            academic_risk += (df['performance_trend'] < -0.2) * 3
            academic_risk += (df['performance_trend'] < -0.1) * 2
        if 'grade_drop' in df.columns:
            academic_risk += df['grade_drop'] * 2
        
        df['academic_risk_score'] = academic_risk
        
        # Attendance risk
        attendance_risk = 0
        if 'Daytime/evening attendance' in df.columns:
            # Assuming 1 = daytime (lower risk), 0 = evening (higher risk)
            attendance_risk += (df['Daytime/evening attendance'] == 0) * 2
        if 'International' in df.columns:
            attendance_risk += df['International'] * 1
        
        df['attendance_risk_score'] = attendance_risk
        
        # Overall risk score (weighted combination)
        df['overall_risk_score'] = (
            df['financial_risk_score'] * 0.3 +
            df['academic_risk_score'] * 0.5 +
            df['attendance_risk_score'] * 0.2
        )
        
        # Risk categories
        df['risk_category'] = pd.cut(
            df['overall_risk_score'],
            bins=[-1, 2, 5, 8, 20],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        return df
    
    def create_temporal_features(self, df):
        """Create temporal pattern features"""
        # Age categories with more granularity
        if 'Age at enrollment' in df.columns:
            df['age_category'] = pd.cut(
                df['Age at enrollment'],
                bins=[17, 19, 21, 23, 25, 30, 40, 100],
                labels=['17-19', '20-21', '22-23', '24-25', '26-30', '31-40', '40+']
            )
            df['is_young_student'] = (df['Age at enrollment'] <= 21).astype(int)
            df['is_mature_student'] = (df['Age at enrollment'] >= 25).astype(int)
        
        # Semester intensity
        if all(col in df.columns for col in ['Curricular units 1st sem (approved)', 'Curricular units 1st sem (enrolled)']):
            df['1st_sem_intensity'] = (
                df['Curricular units 1st sem (approved)'] / 
                df['Curricular units 1st sem (enrolled)']
            ).replace([np.inf, -np.inf], 0).fillna(0)
        
        # Academic momentum
        if all(col in df.columns for col in ['1st_sem_success_rate', '2nd_sem_success_rate']):
            df['academic_momentum'] = df['2nd_sem_success_rate'] - df['1st_sem_success_rate']
            df['positive_momentum'] = (df['academic_momentum'] > 0.1).astype(int)
            df['negative_momentum'] = (df['academic_momentum'] < -0.1).astype(int)
        
        return df
    
    def create_demographic_features(self, df):
        """Create demographic-related features"""
        # Gender features
        if 'Gender' in df.columns:
            df['is_female'] = (df['Gender'] == 1).astype(int)
            df['is_male'] = (df['Gender'] == 0).astype(int)
        
        # International student features
        if 'International' in df.columns:
            df['is_international'] = df['International']
        
        # Marital status features
        if 'Marital status' in df.columns:
            df['is_single'] = (df['Marital status'] == 1).astype(int)
            df['is_married'] = (df['Marital status'] == 2).astype(int)
            df['is_divorced'] = (df['Marital status'] == 3).astype(int)
            df['is_widowed'] = (df['Marital status'] == 4).astype(int)
        
        # Previous qualification features
        if 'Previous qualification' in df.columns:
            df['has_previous_qualification'] = (df['Previous qualification'] != 1).astype(int)
        
        # Displacement features
        if 'Displaced' in df.columns:
            df['is_displaced'] = df['Displaced']
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features between important variables"""
        # Academic risk × Financial risk
        if all(col in df.columns for col in ['academic_risk_score', 'financial_risk_score']):
            df['academic_financial_risk'] = df['academic_risk_score'] * df['financial_risk_score']
        
        # Age × Academic performance
        if all(col in df.columns for col in ['Age at enrollment', 'overall_academic_score']):
            df['age_academic_interaction'] = df['Age at enrollment'] * df['overall_academic_score']
        
        # International × Financial risk
        if all(col in df.columns for col in ['International', 'financial_risk_score']):
            df['international_financial_risk'] = df['International'] * df['financial_risk_score']
        
        return df
    
    def select_best_features(self, X, y, k=25):
        """Select best features using multiple methods"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_selection import RFE
            
            print(f"🎯 Selecting top {k} features...")
            
            # Method 1: Random Forest Importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Method 2: ANOVA F-value
            selector_anova = SelectKBest(score_func=f_classif, k=min(k, len(X.columns)))
            selector_anova.fit(X, y)
            anova_scores = pd.DataFrame({
                'feature': X.columns,
                'anova_score': selector_anova.scores_
            }).sort_values('anova_score', ascending=False)
            
            # Method 3: Mutual Information
            selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(k, len(X.columns)))
            selector_mi.fit(X, y)
            mi_scores = pd.DataFrame({
                'feature': X.columns,
                'mi_score': selector_mi.scores_
            }).sort_values('mi_score', ascending=False)
            
            # Combine scores (simple average ranking)
            all_features = set(X.columns)
            ranking_df = pd.DataFrame(index=all_features)
            
            # Add rankings from each method
            ranking_df['rf_rank'] = feature_importance.set_index('feature')['importance'].rank(ascending=False)
            ranking_df['anova_rank'] = anova_scores.set_index('feature')['anova_score'].rank(ascending=False)
            ranking_df['mi_rank'] = mi_scores.set_index('feature')['mi_score'].rank(ascending=False)
            
            # Calculate average rank
            ranking_df['avg_rank'] = ranking_df.mean(axis=1)
            ranking_df = ranking_df.sort_values('avg_rank')
            
            top_features = ranking_df.head(k).index.tolist()
            
            print("🎯 Top 10 Selected Features:")
            for i, feature in enumerate(top_features[:10], 1):
                print(f"  {i}. {feature}")
            
            return top_features
            
        except Exception as e:
            print(f"⚠️ Feature selection error: {e}")
            # Return all features if selection fails
            return X.columns.tolist()
    
    def remove_low_variance_features(self, df, threshold=0.01):
        """Remove features with low variance"""
        from sklearn.feature_selection import VarianceThreshold
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        selector = VarianceThreshold(threshold=threshold)
        
        try:
            selector.fit(df[numerical_cols])
            selected_features = numerical_cols[selector.get_support()]
            removed_features = set(numerical_cols) - set(selected_features)
            
            if removed_features:
                print(f"🗑️ Removed {len(removed_features)} low-variance features")
                for feature in list(removed_features)[:5]:  # Show first 5
                    print(f"   - {feature}")
            
            # Keep non-numerical columns and selected numerical columns
            non_numerical_cols = df.select_dtypes(exclude=[np.number]).columns
            final_columns = list(selected_features) + list(non_numerical_cols)
            return df[final_columns]
            
        except Exception as e:
            print(f"⚠️ Low variance removal error: {e}")
            return df

# Legacy functions for backward compatibility
def create_academic_performance_features(df):
    """
    Create features related to academic performance
    """
    engineer = AdvancedFeatureEngineer()
    return engineer.create_academic_features(df.copy())

def create_demographic_features(df):
    """
    Create demographic-related features
    """
    engineer = AdvancedFeatureEngineer()
    return engineer.create_demographic_features(df.copy())

def create_comprehensive_features(df):
    """
    Create all comprehensive features (legacy function)
    """
    engineer = AdvancedFeatureEngineer()
    return engineer.create_advanced_features(df.copy())