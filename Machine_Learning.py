import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           cohen_kappa_score, confusion_matrix, classification_report,
                           roc_auc_score, roc_curve, auc, matthews_corrcoef)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from skimage.feature import hog, local_binary_pattern
import pandas as pd
import os
import time
import warnings
from collections import Counter
from scipy import stats
import itertools

# Handle GLCM imports with fallbacks
try:
    from skimage.feature import graycomatrix, graycoprops
    GLCM_AVAILABLE = True
except ImportError:
    try:
        from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops
        GLCM_AVAILABLE = True
    except ImportError:
        GLCM_AVAILABLE = False
        print("Warning: GLCM functions not available")

warnings.filterwarnings('ignore')

class ASLFeatureExtractor:
    """Comprehensive ASL Feature Extraction and Analysis Class"""
    
    def __init__(self, dataset_path, img_size=(128, 128), max_samples_per_class=None):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.max_samples_per_class = max_samples_per_class
        self.X = None
        self.y = None
        self.class_names = None
        self.results = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the ASL dataset"""
        print("Loading and preprocessing ASL dataset...")
        print(f"Dataset path: {self.dataset_path}")
        print(f"Target image size: {self.img_size}")
        print(f"Max samples per class: {self.max_samples_per_class}")
        
        X, y, class_names = [], [], []
        
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"Dataset path {self.dataset_path} does not exist!")
        
        # Load images from each class folder
        for label, class_name in enumerate(sorted(os.listdir(self.dataset_path))):
            class_path = os.path.join(self.dataset_path, class_name)
            if not os.path.isdir(class_path):
                continue
                
            class_names.append(class_name)
            count = 0
            
            print(f"Processing class '{class_name}' (label: {label})...")
            
            for img_file in os.listdir(class_path):
                if self.max_samples_per_class and count >= self.max_samples_per_class:
                    break
                    
                # Check if file is an image
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    continue
                    
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                # Preprocessing pipeline
                img = self._preprocess_image(img)
                
                X.append(img)
                y.append(label)
                count += 1
            
            print(f"  Loaded {count} images for class '{class_name}'")
        
        self.X = np.array(X)
        self.y = np.array(y)
        self.class_names = class_names
        
        print(f"\nDataset Summary:")
        print(f"Total images: {len(self.X)}")
        print(f"Number of classes: {len(self.class_names)}")
        print(f"Image shape: {self.X[0].shape}")
        print(f"Classes: {self.class_names}")
        
        # Class distribution analysis
        class_counts = Counter(self.y)
        print(f"\nClass Distribution:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}: {class_counts[i]} images")
        
        return self.X, self.y, self.class_names
    
    def _preprocess_image(self, img):
        """Comprehensive image preprocessing"""
        # Convert to grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size
        img = cv2.resize(img, self.img_size)
        
        # Normalize to [0, 1]
        img = img.astype(np.float64) / 255.0
        
        # Optional: Histogram equalization for better contrast
        img_uint8 = (img * 255).astype(np.uint8)
        img_eq = cv2.equalizeHist(img_uint8)
        img = img_eq.astype(np.float64) / 255.0
        
        return img
    
    def add_noise_to_images(self, noise_factor=0.1):
        """Add Gaussian noise to images for robustness testing"""
        noise = np.random.normal(0, noise_factor, self.X.shape)
        noisy_images = np.clip(self.X + noise, 0, 1)
        return noisy_images
    
    # Feature Extraction Methods
    def extract_hog_features(self, images):
        """Extract HOG (Histogram of Oriented Gradients) features"""
        print("Extracting HOG features...")
        features = []
        
        for i, img in enumerate(images):
            if i % 500 == 0:
                print(f"  Processing image {i}/{len(images)}")
            
            # HOG parameters optimized for hand gestures
            feat = hog(img, 
                      orientations=9,           # Number of orientation bins
                      pixels_per_cell=(8, 8),   # Size of cells
                      cells_per_block=(2, 2),   # Size of blocks
                      block_norm='L2-Hys',      # Block normalization
                      visualize=False,
                      feature_vector=True)
            features.append(feat)
        
        features = np.array(features)
        print(f"HOG features shape: {features.shape}")
        return features
    
    def extract_sift_features(self, images, n_features=100):
        """Extract SIFT (Scale-Invariant Feature Transform) features"""
        print("Extracting SIFT features...")
        
        try:
            sift = cv2.SIFT_create(nfeatures=n_features)
        except AttributeError:
            try:
                sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features)
            except:
                print("SIFT not available, using alternative feature extraction")
                return self.extract_orb_features(images)
        
        features = []
        
        for i, img in enumerate(images):
            if i % 500 == 0:
                print(f"  Processing image {i}/{len(images)}")
            
            img_uint8 = (img * 255).astype(np.uint8)
            keypoints, descriptors = sift.detectAndCompute(img_uint8, None)
            
            if descriptors is not None and len(descriptors) > 0:
                # Use statistical measures of descriptors
                feat = np.concatenate([
                    np.mean(descriptors, axis=0),
                    np.std(descriptors, axis=0),
                    np.max(descriptors, axis=0),
                    np.min(descriptors, axis=0)
                ])
                
                # Ensure consistent feature vector size
                if len(feat) > 512:
                    feat = feat[:512]
                elif len(feat) < 512:
                    feat = np.pad(feat, (0, 512 - len(feat)), 'constant')
            else:
                feat = np.zeros(512)
            
            features.append(feat)
        
        features = np.array(features)
        print(f"SIFT features shape: {features.shape}")
        return features
    
    def extract_glcm_features(self, images):
        """Extract GLCM (Gray-Level Co-occurrence Matrix) features"""
        if not GLCM_AVAILABLE:
            print("GLCM not available, using HOG features instead")
            return self.extract_hog_features(images)
        
        print("Extracting GLCM features...")
        features = []
        
        # GLCM parameters
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        for i, img in enumerate(images):
            if i % 500 == 0:
                print(f"  Processing image {i}/{len(images)}")
            
            img_uint8 = (img * 255).astype(np.uint8)
            # Reduce gray levels to speed up computation
            img_reduced = img_uint8 // 16  # 16 gray levels
            
            try:
                # Calculate GLCM
                glcm = graycomatrix(img_reduced,
                                  distances=distances,
                                  angles=angles,
                                  levels=16,
                                  symmetric=True,
                                  normed=True)
                
                # Extract texture properties
                properties = ['contrast', 'dissimilarity', 'homogeneity', 
                            'energy', 'correlation', 'ASM']
                
                feat_list = []
                for prop in properties:
                    if prop == 'ASM':  # Angular Second Moment
                        prop_values = graycoprops(glcm, 'energy')**2
                    else:
                        prop_values = graycoprops(glcm, prop)
                    
                    feat_list.extend([
                        np.mean(prop_values),
                        np.std(prop_values),
                        np.max(prop_values),
                        np.min(prop_values)
                    ])
                
                feat = np.array(feat_list)
                
            except Exception as e:
                print(f"Error processing GLCM for image {i}: {e}")
                feat = np.zeros(24)  # Default feature vector size
            
            features.append(feat)
        
        features = np.array(features)
        print(f"GLCM features shape: {features.shape}")
        return features
    
    def extract_orb_features(self, images, n_features=100):
        """Extract ORB (Oriented FAST and Rotated BRIEF) features"""
        print("Extracting ORB features...")
        
        orb = cv2.ORB_create(nfeatures=n_features)
        features = []
        
        for i, img in enumerate(images):
            if i % 500 == 0:
                print(f"  Processing image {i}/{len(images)}")
            
            img_uint8 = (img * 255).astype(np.uint8)
            keypoints, descriptors = orb.detectAndCompute(img_uint8, None)
            
            if descriptors is not None and len(descriptors) > 0:
                # Use statistical measures of descriptors
                feat = np.concatenate([
                    np.mean(descriptors, axis=0),
                    np.std(descriptors, axis=0),
                    np.median(descriptors, axis=0)
                ]).astype(np.float64)
                
                # Ensure consistent feature vector size (32 * 3 = 96)
                if len(feat) < 96:
                    feat = np.pad(feat, (0, 96 - len(feat)), 'constant')
                else:
                    feat = feat[:96]
            else:
                feat = np.zeros(96, dtype=np.float64)
            
            features.append(feat)
        
        features = np.array(features)
        print(f"ORB features shape: {features.shape}")
        return features
    
    def calculate_advanced_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Advanced metrics
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['matthews_corr'] = matthews_corrcoef(y_true, y_pred)
        
        # Per-class metrics
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # ROC-AUC (for multiclass)
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                metrics['roc_auc'] = np.nan
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Class-wise error analysis
        class_errors = {}
        cm = confusion_matrix(y_true, y_pred)
        for i in range(len(self.class_names)):
            total_samples = np.sum(cm[i, :])
            correct_predictions = cm[i, i]
            class_errors[self.class_names[i]] = {
                'total_samples': total_samples,
                'correct_predictions': correct_predictions,
                'accuracy': correct_predictions / total_samples if total_samples > 0 else 0,
                'errors': total_samples - correct_predictions
            }
        metrics['class_errors'] = class_errors
        
        return metrics
    
    def evaluate_feature_method(self, feature_extractor, method_name):
        """Comprehensive evaluation of a feature extraction method"""
        print(f"\n{'='*60}")
        print(f"EVALUATING {method_name.upper()} FEATURES")
        print(f"{'='*60}")
        
        # Extract features
        start_time = time.time()
        X_features = feature_extractor(self.X)
        feature_time = time.time() - start_time
        
        # Handle NaN/Inf values
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Test multiple classifiers
        classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=20)
        }
        
        method_results = {
            'feature_extraction_time': feature_time,
            'feature_shape': X_features.shape,
            'classifiers': {}
        }
        
        for clf_name, classifier in classifiers.items():
            print(f"\n--- Testing {clf_name} ---")
            
            # Train classifier
            start_time = time.time()
            classifier.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time
            
            # Predictions
            start_time = time.time()
            y_pred = classifier.predict(X_test_scaled)
            prediction_time = time.time() - start_time
            
            # Prediction probabilities (if available)
            y_pred_proba = None
            if hasattr(classifier, "predict_proba"):
                y_pred_proba = classifier.predict_proba(X_test_scaled)
            
            # Calculate metrics
            metrics = self.calculate_advanced_metrics(y_test, y_pred, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(classifier, X_train_scaled, y_train, 
                                      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
            
            # Store results
            classifier_results = {
                'metrics': metrics,
                'training_time': training_time,
                'prediction_time': prediction_time,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            method_results['classifiers'][clf_name] = classifier_results
            
            # Print results
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
            print(f"Matthews Correlation: {metrics['matthews_corr']:.4f}")
            print(f"ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
            print(f"Training Time: {training_time:.2f}s")
        
        # Test robustness with noisy data
        print(f"\n--- Robustness Testing (Noisy Data) ---")
        noisy_X = self.add_noise_to_images(noise_factor=0.1)
        X_noisy_features = feature_extractor(noisy_X)
        X_noisy_features = np.nan_to_num(X_noisy_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Use best performing classifier for robustness test
        best_clf = max(method_results['classifiers'].items(), 
                      key=lambda x: x[1]['metrics']['accuracy'])
        best_clf_name, best_clf_results = best_clf
        
        # Test on noisy data
        _, X_noisy_test, _, y_noisy_test = train_test_split(
            X_noisy_features, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        X_noisy_test_scaled = scaler.transform(X_noisy_test)
        
        # Re-train best classifier and test on noisy data
        best_classifier = classifiers[best_clf_name]
        best_classifier.fit(X_train_scaled, y_train)
        y_noisy_pred = best_classifier.predict(X_noisy_test_scaled)
        
        noisy_accuracy = accuracy_score(y_noisy_test, y_noisy_pred)
        robustness_score = noisy_accuracy / best_clf_results['metrics']['accuracy']
        
        method_results['robustness'] = {
            'noisy_accuracy': noisy_accuracy,
            'original_accuracy': best_clf_results['metrics']['accuracy'],
            'robustness_score': robustness_score
        }
        
        print(f"Original Accuracy: {best_clf_results['metrics']['accuracy']:.4f}")
        print(f"Noisy Data Accuracy: {noisy_accuracy:.4f}")
        print(f"Robustness Score: {robustness_score:.4f}")
        
        return method_results
    
    def visualize_results(self, results):
        """Create comprehensive visualizations of results"""
        plt.style.use('default')
        
        # 1. Performance comparison heatmap
        performance_data = []
        for method, method_results in results.items():
            for clf_name, clf_results in method_results['classifiers'].items():
                performance_data.append({
                    'Method': method,
                    'Classifier': clf_name,
                    'Accuracy': clf_results['metrics']['accuracy'],
                    'F1-Score': clf_results['metrics']['f1_score'],
                    'Cohen_Kappa': clf_results['metrics']['cohen_kappa'],
                    'ROC_AUC': clf_results['metrics'].get('roc_auc', 0)
                })
        
        df_performance = pd.DataFrame(performance_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Accuracy heatmap
        pivot_acc = df_performance.pivot(index='Method', columns='Classifier', values='Accuracy')
        sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0,0])
        axes[0,0].set_title('Accuracy Comparison')
        
        # F1-Score heatmap
        pivot_f1 = df_performance.pivot(index='Method', columns='Classifier', values='F1-Score')
        sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='YlGnBu', ax=axes[0,1])
        axes[0,1].set_title('F1-Score Comparison')
        
        # Cohen's Kappa heatmap
        pivot_kappa = df_performance.pivot(index='Method', columns='Classifier', values='Cohen_Kappa')
        sns.heatmap(pivot_kappa, annot=True, fmt='.3f', cmap='viridis', ax=axes[1,0])
        axes[1,0].set_title("Cohen's Kappa Comparison")
        
        # Robustness comparison
        robustness_data = [(method, method_results['robustness']['robustness_score']) 
                          for method, method_results in results.items()]
        methods, robustness_scores = zip(*robustness_data)
        
        bars = axes[1,1].bar(methods, robustness_scores, color=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
        axes[1,1].set_title('Robustness to Noise')
        axes[1,1].set_ylabel('Robustness Score')
        axes[1,1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, robustness_scores):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # 2. Confusion matrices for best performing method-classifier combination
        best_combo = None
        best_accuracy = 0
        
        for method, method_results in results.items():
            for clf_name, clf_results in method_results['classifiers'].items():
                if clf_results['metrics']['accuracy'] > best_accuracy:
                    best_accuracy = clf_results['metrics']['accuracy']
                    best_combo = (method, clf_name, clf_results)
        
        if best_combo:
            method_name, clf_name, clf_results = best_combo
            cm = clf_results['metrics']['confusion_matrix']
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title(f'Confusion Matrix - Best Performance\n{method_name} + {clf_name} (Accuracy: {best_accuracy:.3f})')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.show()
        
        # 3. Class-wise performance analysis
        if best_combo:
            class_errors = best_combo[2]['metrics']['class_errors']
            
            class_accuracies = [class_errors[class_name]['accuracy'] for class_name in self.class_names]
            
            plt.figure(figsize=(14, 6))
            bars = plt.bar(self.class_names, class_accuracies, 
                          color=plt.cm.viridis(np.linspace(0, 1, len(self.class_names))))
            plt.title(f'Class-wise Accuracy - {best_combo[0]} + {best_combo[1]}')
            plt.xlabel('ASL Classes')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, acc in zip(bars, class_accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.2f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.show()
    
    def generate_comprehensive_report(self, results):
        """Generate a comprehensive analysis report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE ASL FEATURE EXTRACTION ANALYSIS REPORT")
        print("="*80)
        
        print(f"\nDataset Information:")
        print(f"- Total images: {len(self.X)}")
        print(f"- Number of classes: {len(self.class_names)}")
        print(f"- Image dimensions: {self.img_size}")
        print(f"- Classes: {', '.join(self.class_names)}")
        
        # Find best performing combinations
        all_results = []
        for method, method_results in results.items():
            for clf_name, clf_results in method_results['classifiers'].items():
                all_results.append({
                    'method': method,
                    'classifier': clf_name,
                    'accuracy': clf_results['metrics']['accuracy'],
                    'f1_score': clf_results['metrics']['f1_score'],
                    'cohen_kappa': clf_results['metrics']['cohen_kappa'],
                    'matthews_corr': clf_results['metrics']['matthews_corr'],
                    'roc_auc': clf_results['metrics'].get('roc_auc', 0),
                    'robustness': method_results['robustness']['robustness_score'],
                    'feature_time': method_results['feature_extraction_time'],
                    'training_time': clf_results['training_time']
                })
        
        df_results = pd.DataFrame(all_results)
        
        print(f"\nTOP 5 PERFORMING COMBINATIONS:")
        print("-" * 50)
        top_5 = df_results.nlargest(5, 'accuracy')
        for idx, row in top_5.iterrows():
            print(f"{row['method']} + {row['classifier']}: "
                  f"Accuracy={row['accuracy']:.4f}, "
                  f"F1={row['f1_score']:.4f}, "
                  f"Kappa={row['cohen_kappa']:.4f}")
        
        print(f"\nMETHOD COMPARISON SUMMARY:")
        print("-" * 50)
        method_summary = df_results.groupby('method').agg({
            'accuracy': ['mean', 'std', 'max'],
            'f1_score': ['mean', 'std', 'max'],
            'cohen_kappa': ['mean', 'std', 'max'],
            'robustness': 'first',
            'feature_time': 'first'
        }).round(4)
        
        print(method_summary)
        
        print(f"\nROBUSTNESS RANKING:")
        print("-" * 30)
        robustness_ranking = df_results.groupby('method')['robustness'].first().sort_values(ascending=False)
        for method, score in robustness_ranking.items():
            print(f"{method}: {score:.4f}")
        
        print(f"\nCOMPUTATIONAL EFFICIENCY:")
        print("-" * 30)
        efficiency_data = df_results.groupby('method').agg({
            'feature_time': 'first',
            'training_time': 'mean'
        }).round(4)
        
        for method in efficiency_data.index:
            feat_time = efficiency_data.loc[method, 'feature_time']
            train_time = efficiency_data.loc[method, 'training_time']
            print(f"{method}: Feature Extraction={feat_time:.2f}s, Training={train_time:.2f}s")
        
        return df_results
    
    def run_comprehensive_analysis(self):
        """Run the complete analysis pipeline"""
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Feature extraction methods
        feature_methods = {
            'HOG': self.extract_hog_features,
            'SIFT': self.extract_sift_features,
            'GLCM': self.extract_glcm_features,
            'ORB': self.extract_orb_features
        }
        
        # Evaluate each method
        results = {}
        for method_name, extractor in feature_methods.items():
            results[method_name] = self.evaluate_feature_method(extractor, method_name)
        
        # Generate visualizations
        self.visualize_results(results)
        
        # Generate comprehensive report
        summary_df = self.generate_comprehensive_report(results)
        
        return results, summary_df


def main():
    """Main execution function"""
    # Configuration
    DATASET_PATH = r"C:\Users\vkous\Downloads\DL_PROECT\2\asl_dataset"
    IMG_SIZE = (128, 128)  # Larger size for better feature extraction
    MAX_SAMPLES_PER_CLASS = 150  # Adjust based on your computational resources
    
    print("ASL Dataset - Comprehensive Feature Extraction Analysis")
    print("="*60)
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Max Samples per Class: {MAX_SAMPLES_PER_CLASS}")
    
    # Initialize analyzer
    analyzer = ASLFeatureExtractor(
        dataset_path=DATASET_PATH,
        img_size=IMG_SIZE,
        max_samples_per_class=MAX_SAMPLES_PER_CLASS
    )
    
    try:
        # Run comprehensive analysis
        results, summary_df = analyzer.run_comprehensive_analysis()
        
        # Save results to CSV
        summary_df.to_csv('asl_feature_analysis_results.csv', index=False)
        print(f"\nResults saved to 'asl_feature_analysis_results.csv'")
        
        # Additional analysis: Feature importance for best method
        print(f"\n" + "="*60)
        print("ADDITIONAL INSIGHTS")
        print("="*60)
        
        # Find the best performing combination
        best_row = summary_df.loc[summary_df['accuracy'].idxmax()]
        print(f"\nBest Overall Performance:")
        print(f"Method: {best_row['method']}")
        print(f"Classifier: {best_row['classifier']}")
        print(f"Accuracy: {best_row['accuracy']:.4f}")
        print(f"F1-Score: {best_row['f1_score']:.4f}")
        print(f"Cohen's Kappa: {best_row['cohen_kappa']:.4f}")
        print(f"Robustness Score: {best_row['robustness']:.4f}")
        
        # Performance vs Computational Cost Analysis
        print(f"\nPerformance vs Computational Cost:")
        print("-" * 40)
        for method in ['HOG', 'SIFT', 'GLCM', 'ORB']:
            method_data = summary_df[summary_df['method'] == method]
            if not method_data.empty:
                max_acc = method_data['accuracy'].max()
                avg_feat_time = method_data['feature_time'].iloc[0]
                avg_train_time = method_data['training_time'].mean()
                total_time = avg_feat_time + avg_train_time
                efficiency = max_acc / total_time  # Accuracy per second
                
                print(f"{method:6s}: Max Acc={max_acc:.3f}, "
                      f"Total Time={total_time:.1f}s, "
                      f"Efficiency={efficiency:.4f}")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        print("-" * 20)
        
        # Best for accuracy
        best_acc_row = summary_df.loc[summary_df['accuracy'].idxmax()]
        print(f"• For highest accuracy: {best_acc_row['method']} + {best_acc_row['classifier']}")
        
        # Best for robustness
        best_robust_row = summary_df.loc[summary_df['robustness'].idxmax()]
        print(f"• For best robustness: {best_robust_row['method']} + {best_robust_row['classifier']}")
        
        # Best for speed
        summary_df['total_time'] = summary_df['feature_time'] + summary_df['training_time']
        fastest_row = summary_df.loc[summary_df['total_time'].idxmin()]
        print(f"• For fastest processing: {fastest_row['method']} + {fastest_row['classifier']}")
        
        # Best balance
        summary_df['balance_score'] = (summary_df['accuracy'] * summary_df['robustness']) / summary_df['total_time']
        balanced_row = summary_df.loc[summary_df['balance_score'].idxmax()]
        print(f"• For best balance: {balanced_row['method']} + {balanced_row['classifier']}")
        
        print(f"\nAnalysis Complete! Check the generated visualizations and CSV file for detailed results.")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


# Additional utility functions for extended analysis
def compare_with_baseline(analyzer, results):
    """Compare feature methods with a simple baseline (raw pixels)"""
    print(f"\n" + "="*60)
    print("BASELINE COMPARISON (Raw Pixels)")
    print("="*60)
    
    # Simple pixel-based features
    def extract_pixel_features(images):
        features = []
        for img in images:
            # Downsample and flatten
            downsampled = cv2.resize(img, (32, 32))
            features.append(downsampled.flatten())
        return np.array(features)
    
    baseline_results = analyzer.evaluate_feature_method(extract_pixel_features, "BASELINE")
    
    # Compare with other methods
    print(f"\nBaseline vs Feature Methods (Best Accuracy):")
    print("-" * 50)
    
    baseline_best_acc = max([clf['metrics']['accuracy'] 
                            for clf in baseline_results['classifiers'].values()])
    print(f"Baseline (Raw Pixels): {baseline_best_acc:.4f}")
    
    for method, method_results in results.items():
        method_best_acc = max([clf['metrics']['accuracy'] 
                              for clf in method_results['classifiers'].values()])
        improvement = ((method_best_acc - baseline_best_acc) / baseline_best_acc) * 100
        print(f"{method:15s}: {method_best_acc:.4f} ({improvement:+.1f}% improvement)")


def analyze_class_difficulty(analyzer, results):
    """Analyze which ASL classes are most difficult to classify"""
    print(f"\n" + "="*60)
    print("CLASS DIFFICULTY ANALYSIS")
    print("="*60)
    
    # Find best performing method
    best_method = None
    best_accuracy = 0
    
    for method, method_results in results.items():
        for clf_name, clf_results in method_results['classifiers'].items():
            if clf_results['metrics']['accuracy'] > best_accuracy:
                best_accuracy = clf_results['metrics']['accuracy']
                best_method = (method, clf_name, clf_results)
    
    if best_method:
        class_errors = best_method[2]['metrics']['class_errors']
        
        # Sort classes by difficulty (lowest accuracy first)
        class_difficulty = [(class_name, info['accuracy']) 
                           for class_name, info in class_errors.items()]
        class_difficulty.sort(key=lambda x: x[1])
        
        print(f"Using best method: {best_method[0]} + {best_method[1]}")
        print(f"\nMost Difficult Classes (Lowest Accuracy):")
        print("-" * 40)
        for i, (class_name, accuracy) in enumerate(class_difficulty[:10]):
            print(f"{i+1:2d}. {class_name}: {accuracy:.3f}")
        
        print(f"\nEasiest Classes (Highest Accuracy):")
        print("-" * 40)
        for i, (class_name, accuracy) in enumerate(reversed(class_difficulty[-10:])):
            print(f"{i+1:2d}. {class_name}: {accuracy:.3f}")


def create_feature_extraction_guide():
    """Create a guide for choosing feature extraction methods"""
    guide = """
    
    ═══════════════════════════════════════════════════════════════════
    ASL FEATURE EXTRACTION METHOD SELECTION GUIDE
    ═══════════════════════════════════════════════════════════════════
    
    📊 HOG (Histogram of Oriented Gradients):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ✅ Best for: Hand gesture recognition, shape-based features
    ✅ Strengths: Robust to lighting changes, captures edge information
    ❌ Weaknesses: Sensitive to rotation, computationally moderate
    🎯 Recommended when: You have well-aligned hand gestures
    
    📊 SIFT (Scale-Invariant Feature Transform):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ✅ Best for: Scale and rotation invariant recognition
    ✅ Strengths: Invariant to scale, rotation, illumination
    ❌ Weaknesses: Computationally expensive, may be overkill for ASL
    🎯 Recommended when: Images have varying scales/orientations
    
    📊 GLCM (Gray-Level Co-occurrence Matrix):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ✅ Best for: Texture analysis, subtle pattern differences
    ✅ Strengths: Captures spatial relationships, good for texture
    ❌ Weaknesses: Computationally intensive, may miss global features
    🎯 Recommended when: ASL signs differ primarily in texture patterns
    
    📊 ORB (Oriented FAST and Rotated BRIEF):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ✅ Best for: Real-time applications, keypoint matching
    ✅ Strengths: Fast computation, rotation invariant
    ❌ Weaknesses: May miss fine details, less discriminative
    🎯 Recommended when: Speed is crucial, real-time recognition needed
    
    🏆 GENERAL RECOMMENDATIONS:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • Start with HOG for balanced performance
    • Use SIFT if images have varying conditions
    • Try GLCM for texture-rich datasets
    • Choose ORB for real-time applications
    • Combine multiple methods for best results
    
    ═══════════════════════════════════════════════════════════════════
    """
    print(guide)


if __name__ == "__main__":
    # Run main analysis
    main()
    
    # Print selection guide
    create_feature_extraction_guide()
    
    print(f"\n🎉 Analysis completed successfully!")
    print(f"📁 Check the generated files and visualizations")
    print(f"📊 Review the CSV file for detailed numerical results")
    print(f"🔍 Use the recommendations to choose the best method for your use case")
