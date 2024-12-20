import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, FunctionTransformer, StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, make_scorer, precision_score, recall_score, f1_score, confusion_matrix,\
    ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


""" Load the data """
def load_banknote_data():
    columns = ["variance","skewness","kurtosis","entropy","class"]
    return pd.read_csv("data/data_banknote_authentication.txt", names=columns)

def load_kidney_disease_data():
    return pd.read_csv("data/kidney_disease.csv").drop('id', axis=1)

def features_label_data_banknote():
    data = load_banknote_data()
    return data.drop("class", axis=1), data[["class"]].copy()

def features_label_data_kidney():
    data = load_kidney_disease_data()
    return data.drop("classification", axis=1), data[["classification"]].copy()



""" Clean the data """
def clean_data(X):
    X_cleaned = X.copy()

    # Process object columns only (avoid converting all columns to numeric prematurely)
    object_columns = X_cleaned.select_dtypes(include=['object']).columns
    for col in object_columns:
        # Remove tab characters from string columns and replace '?' with NaN
        X_cleaned[col] = X_cleaned[col].str.replace(r'\t', '', regex=True)
        X_cleaned[col] = X_cleaned[col].replace('?', np.nan)

        # Try converting columns to numeric, catching errors explicitly for non-numeric values
        try:
            X_cleaned[col] = pd.to_numeric(X_cleaned[col])
        except ValueError:
            pass  # Leave non-numeric columns as they are

    return X_cleaned

def map_labels_to_svm_label(X):
    return np.where(X == 0, -1, 1)



""" Transform the data """

""" feature selection """
def feature_selection(X_train, features_to_drop=None):
    X = X_train.copy()
    columns_to_drop = [feature for feature in features_to_drop if feature in X.columns]
    X.drop(columns=columns_to_drop, inplace=True)
    return X

"""Apply this pipeline to numeric_attribute. Give as parameter a list of column you want to drop"""
def numeric_pipeline(features_to_drop):
    return make_pipeline(
        FunctionTransformer(feature_selection,kw_args={'features_to_drop': features_to_drop}),
        #FunctionTransformer(clean_data),
        SimpleImputer(strategy='median'),
        StandardScaler()
    )

"""
Apply this pipeline to column with dtype=object. Give as parameter a list of column you want to drop
"""
def categorical_pipeline(features_to_drop):
    return make_pipeline(
        FunctionTransformer(feature_selection,kw_args={'features_to_drop': features_to_drop}),
        #FunctionTransformer(clean_data),
        SimpleImputer(strategy='most_frequent'),
        OrdinalEncoder(),
        StandardScaler()
    )


""" Setting the preprocessing pipeline. Give as parameter a list of column you want to drop """
def preprocessing_pipeline(*features_to_drop):
    return ColumnTransformer([
        ("numeric_pipeline", numeric_pipeline(list(features_to_drop)), make_column_selector(dtype_exclude=object)),
        ("categorical_pipeline", categorical_pipeline(list(features_to_drop)), make_column_selector(dtype_include=object)),
    ],
        remainder="passthrough")

"""Apply this pipeline to label column"""
def preprocessing_labels_pipeline ():
    return make_pipeline(
        FunctionTransformer(clean_data),
        SimpleImputer(strategy='most_frequent'),
        OrdinalEncoder(),
        FunctionTransformer(map_labels_to_svm_label)
    )

""" Prepare the dataset """
def cross_val(model, data, labels, scoring, n_splits=5):
    results = cross_val_score(model, data, labels, cv=n_splits, scoring=scoring)
    return results


def prepare_dataset(X, y, test_size=0.2, random_state=42):
    # Create training and testing datasets (using stratification to have a good class repartition)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


""" Train The Model """

def perform_cross_validation(model, X_train, y_train, n_splits=5):
    
    # Cross-validation using a Stratified KFold (5 splits by default)
    cv_results = cross_validate(
        model, 
        X_train, 
        y_train, 
        cv=n_splits
    )
    
    return cv_results

def train_and_validate_models(X_train, X_test, y_train, y_test):
    # Test some machine learning models (using default parameters values)
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Ada Boost Classifier' : AdaBoostClassifier()
    }
    
    results = {}
    
    for name, model in models.items():
        # Cross-validation on training set
        cv_results = perform_cross_validation(model, X_train, y_train)
        
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        
        results[name] = {
            'model': model,
            'test_accuracy': accuracy_score(y_test, y_pred),
            'cv_results': cv_results,
            'classification_report': classification_report(y_test, y_pred)
        }
    
    return results

""" Validate The Model """


def visualize_results(results, figsize=(15, 10), adjust_scale=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    model_names = list(results.keys())
    test_accuracies = [results[name]['test_accuracy'] for name in model_names]
    
    if adjust_scale:
        min_acc = min(test_accuracies)
        y_min = max(0, min_acc - 0.1)  # Go 10% below minimum accuracy, but not below 0
        y_max = 1.001  # Slight padding above 1
    else:
        y_min, y_max = 0, 1
    
    bars = ax1.bar(model_names, test_accuracies)
    ax1.set_title('Test Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(y_min, y_max)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    
    for i, v in enumerate(test_accuracies):
        ax1.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    cv_data = []
    for name in model_names:
        cv_data.append(results[name]['cv_results']['test_score'])
    
    ax2.boxplot(cv_data, labels=model_names, vert=True)
    ax2.set_title('Cross-validation Scores Distribution')
    ax2.set_ylabel('CV Accuracy')
    ax2.set_ylim(y_min, y_max)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)


def confusion_matrix_display(model, data, target,labels=None,cv=3):
    y_pred = cross_val_predict(model, data, target, cv=cv)
    
    cm = confusion_matrix(y_true=target, y_pred=y_pred)

    #cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    if labels is None:
        labels = np.unique(target) 

    plt.figure(figsize=(8, 6))

    sns.heatmap(cm, annot=True, cmap="coolwarm", 
                xticklabels=labels, yticklabels=labels, cbar=True,annot_kws={"size": 12})

    plt.title("Normalized Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)

    plt.show()


def plot_class_repartitions(data):
    ax=sns.countplot(data, x=data[data.columns[-1]], hue=data[data.columns[-1]], stat='count')
    for p in ax.patches:
        ax.annotate(f'\n{int(p.get_height())}', (p.get_x()+0.2, p.get_height()), ha='center', va='top', color='white', size=18)


def plot_features(data, figsize=(20, 20)):
    fig, axes = plt.subplots(nrows=len(data.columns)//4, ncols=4, figsize=figsize)
    axes = axes.flatten()

    i = 0
    for column in data.columns:
        if column != "classification" and column != "class" and column != "id":
            sns.histplot(data=data, x=column, hue=data.columns[-1], multiple="stack", ax=axes[i], bins=25)
            axes[i].set_title(column)
            axes[i].set_xlabel("")
            axes[i].set_ylabel("")
            i+=1

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()