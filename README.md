# Projet ECE Dataviz 2025 – Dashboard Marketing Online Retail II

## 1. Contexte du projet

Ce projet a pour objectif de développer :

1. Un notebook d’exploration visuelle complet du jeu de données Online Retail II.  
2. Une application Streamlit interactive permettant aux équipes marketing :
   - de diagnostiquer la rétention par cohortes,
   - d’analyser les segments RFM,
   - d’estimer la CLV (empirique + formule fermée),
   - de simuler des scénarios business (rétention, marge, remises),
   - d’exporter une liste de clients activables.

Les données proviennent du dataset **Online Retail II (UCI)** :  
Transactions e-commerce d’un détaillant UK du **01/12/2009 au 09/12/2011 (~1,07M lignes).**

---

## 2. Objectifs d’analyse

### 2.1 Rétention par cohortes
- Calculer les cohortes mensuelles d’acquisition.  
- Mesurer la rétention M+1, M+2, ..., M+12.  
- Étudier les patterns de décroissance.

### 2.2 Segmentation RFM
- Calculer Recency, Frequency, Monetary.  
- Construire les scores RFM et les segments.  
- Identifier les champions, loyaux, clients en risque, endormis, perdus…

### 2.3 Customer Lifetime Value (CLV)
- CLV empirique basée sur les revenus par âge de cohorte.  
- CLV théorique via formule fermée :  
  `CLV = (m * r) / (1 + d - r)`  
  où m = marge mensuelle moyenne, r = rétention, d = taux d’actualisation.

### 2.4 Simulation de scénarios
- Impact d’une augmentation de la rétention.  
- Impact d’une baisse/hausse de marge.  
- Impact d’une politique de remise.  
- Variation du CA, de la CLV, de la rétention.

---

## 3. Contenu du repository
```
├── app/
│   ├── app.py              # Application Streamlit complète
│   └── utils.py            # Fonctions de chargement/traitement
│
├── data/
│   ├── raw/                # Données brutes
│   └── processed/          # Données retraitées (cohortes, RFM…)
│
├── notebooks/
│   └── 01_exploration.ipynb  # Notebook visuel complet
│
├── docs/
│   └── prez/               # Slides de soutenance (optionnel)
│
├── requirements.txt
├── README.md
└── DATA_DICTIONARY.md      # (optionnel)
```


---

## 4. Fonctionnalités de l’application Streamlit

### 4.1 Filtres globaux
- **Période d’analyse** (sélecteur de dates)
- **Pays**
- **Type client / seuil de commande** (optionnel)
- **Mode retours** : Inclure / Exclure / Neutraliser
- Tous les filtres mis à jour dynamiquement
- Affichage permanent des filtres actifs

### 4.2 Page “Overview”
- KPIs globaux :
  - nombre de clients actifs
  - CA total filtré
  - CLV empirique (baseline)
  - North Star Metric (revenu à M+3)
  - taille des segments RFM
- Définition détaillée de chaque KPI (infobulles ℹ).

### 4.3 Page “Cohortes”
- Heatmap des rétentions par cohortes.
- Courbes de rétention par cohorte sélectionnée.
- Revenu moyen par âge.
- Revenu cumulé par cohorte.

### 4.4 Page “Segmentation RFM”
- Tableau complet des segments RFM :
  - effectifs
  - recency moyenne
  - fréquence moyenne
  - monetary moyen
- Bar chart distribution des segments
- Scatterplot Frequency × Monetary coloré par score RFM
- Aperçu des 50 premiers clients

### 4.5 Page “Scénarios”
- Sliders :
  - marge (%)
  - rétention (r)
  - taux d’actualisation (d)
- Calcul de :
  - CLV empirique
  - CLV formule fermée (baseline + scénario)
  - delta CLV en valeur et en %
- Courbe de sensibilité CLV=f(r)

### 4.6 Page “Export”
- Génération CSV :
  - CustomerID
  - Segment RFM
  - Frequency
  - Monetary
  - CLV (proxy)
- Téléchargement direct

---

## 5. Notebook d’exploration visuelle

Le notebook *01_exploration.ipynb* contient :

### 5.1 Fiche synthétique des données
- volume total
- période couverte
- colonnes importantes
- types
- unités

### 5.2 Qualité des données
- valeurs manquantes
- doublons
- règles d’annulation (InvoiceNo commençant par “C”)
- outliers (quantiles, boxplots)
- granularité temporelle

### 5.3 Visualisations (6-8 graphiques)
- distributions des variables
- saisonnalité des ventes
- évolution du CA par mois
- répartition géographique
- premiers profils RFM
- première heatmap de cohortes

Chaque figure inclut :
- définition de la métrique  
- interprétation claire de l’insight  
- implications business  

---

## 6. Installation & exécution

### 6.1 Cloner le projet

git clone <https://github.com/jadevgrx/PROJET-ECE-DATAVIZ-2025---TD3---Groupe-du-projet-1.git>
cd <PROJET-ECE-DATAVIZ-2025---TD3---Groupe-du-projet-1>


### 6.2 Installer les dépendances

pip install -r requirements.txt


### 6.3 Lancer l’application Streamlit

streamlit run app/app.py


---

## 7. requirements.txt

Un fichier `requirements.txt` est fourni à la racine du projet.


---

## 8. Auteurs & soutenance

Projet réalisé dans le cadre du module **ECE Dataviz 2025**.  

Travail réalisé par Nils Giraud, Jade Vigouroux, Celian Marcus, Arthur Remy, Demian Rembry, Adam Kheloufi.





