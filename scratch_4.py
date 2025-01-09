# Importation des bibliothèques nécessaires
import pandas as pd  # Pour la manipulation de données en format tabulaire
import numpy as np  # Pour les opérations mathématiques
from sklearn.ensemble import IsolationForest  # Pour la détection d'anomalies via l'Isolation Forest
from sklearn.preprocessing import LabelEncoder  # Pour l'encodage des variables catégorielles
from sklearn.feature_extraction.text import \
    TfidfVectorizer  # Pour la transformation du texte en caractéristiques numériques
from sklearn.naive_bayes import MultinomialNB  # Pour la classification des emails en spam ou non-spam
from sklearn.model_selection import \
    train_test_split  # Pour la séparation des données en ensembles d'entraînement et de test
from sklearn.metrics import classification_report  # Pour l'évaluation des performances du classifieur
import seaborn as sns  # Pour la visualisation des résultats (non utilisé dans le code actuel)
import matplotlib.pyplot as plt  # Pour créer des graphiques (non utilisé dans le code actuel)


# Définition de la classe principale du système de sécurité
class SecuritySystem:
    def __init__(self):
        # Initialisation du modèle de détection d'anomalies (Isolation Forest)
        self.isolation_forest = IsolationForest(contamination=0.01, random_state=42)

        # Initialisation du modèle de vectorisation de texte (TF-IDF)
        self.vectorizer = TfidfVectorizer(
            max_features=3000,  # Nombre maximal de caractéristiques
            min_df=2,  # Seules les caractéristiques apparaissant dans au moins 2 documents seront conservées
            stop_words='english',  # Suppression des mots vides (stop words) en anglais
            token_pattern=r'[a-zA-Z0-9]+',  # Expression régulière pour la tokenisation (mots alphanumériques)
            strip_accents='unicode'  # Suppression des accents pour homogénéiser le texte
        )

        # Initialisation du classifieur Naive Bayes pour la classification des emails
        self.spam_classifier = MultinomialNB()

        # Dictionnaire pour stocker les encodeurs de labels des colonnes catégorielles
        self.label_encoders = {}

    # Méthode pour traiter les données réseau
    def process_network_data(self, file_path):
        # Noms des colonnes du fichier CSV réseau
        column_names = [
            "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
            "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
            "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
            "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
            "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
            "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
            "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
            "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
            "label"
        ]

        # Chargement des données à partir du fichier CSV
        network_data = pd.read_csv(file_path, header=None, names=column_names)

        # Encodage des colonnes catégorielles
        categorical_columns = network_data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            network_data[col] = self.label_encoders[col].fit_transform(network_data[col])

        # Sélection des données d'entrée (X) et de la variable cible (label)
        X_network = network_data.drop(columns=["label"]).astype(float)

        # Détection d'anomalies dans les données avec Isolation Forest
        network_data["anomaly"] = self.isolation_forest.fit_predict(X_network)

        # Remplacer les valeurs -1 et 1 par des catégories 'Anomaly' et 'Normal'
        network_data["anomaly"] = network_data["anomaly"].replace({1: "Normal", -1: "Anomaly"})

        # Affichage des résultats de l'analyse
        print("\nNetwork Analysis:")
        print(f"Anomalies detected: {(network_data['anomaly'] == 'Anomaly').sum()}")
        print(f"Normal instances: {(network_data['anomaly'] == 'Normal').sum()}")

        return network_data

    # Méthode pour prétraiter le texte (supprimer les en-têtes d'email)
    def preprocess_text(self, text):
        if pd.isna(text):
            return ""

        # Convertir en texte si ce n'est pas déjà une chaîne
        text = str(text)

        # Retirer les en-têtes d'email (ex. Subject:, From:)
        if "Subject:" in text or "From:" in text:
            parts = text.split("\n\n", 1)  # Garder uniquement le contenu après les en-têtes
            text = parts[-1] if len(parts) > 1 else text

        return text

    # Méthode pour entraîner le classifieur de spam
    def train_spam_classifier(self, email_data_path):
        try:
            # Chargement des données d'email à partir d'un fichier CSV
            emails_df = pd.read_csv(email_data_path)

            # Prétraitement du texte des emails (retirer les en-têtes)
            processed_texts = emails_df['text'].apply(self.preprocess_text)

            # Affichage d'un échantillon de textes pour déboguer
            print("\nSample of processed texts:")
            print(processed_texts.iloc[:2])

            # Vectorisation du texte avec le modèle TF-IDF
            X = self.vectorizer.fit_transform(processed_texts)
            y = emails_df['target']

            # Affichage des statistiques du vectoriseur
            print(f"\nVocabulary size: {len(self.vectorizer.vocabulary_)}")
            print(f"Feature matrix shape: {X.shape}")

            # Division des données en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Entraînement du classifieur Naive Bayes
            self.spam_classifier.fit(X_train, y_train)

            # Prédictions sur l'ensemble de test
            predictions = self.spam_classifier.predict(X_test)

            # Affichage des résultats d'évaluation
            print("\nSpam Classifier Evaluation:")
            print(classification_report(y_test, predictions))

            return True

        except Exception as e:
            print(f"Error training spam classifier: {str(e)}")
            print("Type:", type(e))
            return False

    # Méthode pour classifier un email (spam ou non)
    def classify_email(self, email_text):
        # Vérifier si le classifieur a été entraîné
        if not hasattr(self.spam_classifier, 'classes_'):
            raise ValueError("Spam classifier hasn't been trained yet.")

        # Prétraiter le texte de l'email
        processed_text = self.preprocess_text(email_text)

        # Vectoriser le texte de l'email
        X = self.vectorizer.transform([processed_text])

        # Prédiction de la classe (spam ou non)
        prediction = self.spam_classifier.predict(X)

        # Probabilité que l'email soit du spam
        proba = self.spam_classifier.predict_proba(X)

        return {
            'classification': 'Spam' if prediction[0] == 1 else 'Ham',
            'spam_probability': proba[0][1],
            'text_length': len(processed_text)
        }


# Code principal pour exécuter le système de sécurité
if __name__ == "__main__":
    # Initialisation du système de sécurité
    security_system = SecuritySystem()

    # Traitement des données réseau
    network_file = r"C:\Users\HP\Downloads\archive(1)\nsl-kdd\KDDTrain+.txt"
    network_results = security_system.process_network_data(network_file)

    # Entraînement du classifieur de spam avec les données d'email
    spam_file = r"C:\Users\HP\Downloads\archive(2)\spam_assassin.csv"
    if security_system.train_spam_classifier(spam_file):
        # Test de classification sur un email exemple
        test_email = """
        URGENT: You've won $1,000,000! Send your bank details to claim your prize now!
        """
        result = security_system.classify_email(test_email)

        # Affichage des résultats de la classification
        print(f"\nEmail Classification Results:")
        print(f"Classification: {result['classification']}")
        print(f"Spam Probability: {result['spam_probability']:.2%}")
