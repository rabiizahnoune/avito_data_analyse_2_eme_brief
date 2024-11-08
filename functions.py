import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from typing import Callable

#fonction pour bien traiter les type de ventes
def changer_louer_vendre(row, colone):
    titre_lower = str(row['title']).lower()  # Convert 'titre' to lowercase for case-insensitive matching

    if 'vendre' in titre_lower:
        return 'a vendre'
    elif 'louer' in titre_lower or 'location' in titre_lower:
        return 'a louer'
    elif row[colone] < 50000:
        return 'a louer'
    else:
        return 'a vendre'



#fonction pour le netoyage de colon prix
def nettoyer_prix(prix):
    if isinstance(prix, str):  
        
        prix = prix.replace('\u202f', '').replace(' ', '')
        return float(prix) if prix else None 
    return None  


def changer_nom_ville(ville,mapping_villes):
    return mapping_villes.get(ville,ville)


#fonction pour l etude de ditribution
def analyse_statistique_et_asymetire(dataframe, colone):
    # Calcul des statistiques
    mean = dataframe[colone].mean()
    median = dataframe[colone].median()
    std_dev = dataframe[colone].std()
    skewness = dataframe[colone].skew()  


    print(f'Statistiques descriptives pour la colonne {colone} :')
    print(f'Moyenne : {mean}')
    print(f'Médiane : {median}')
    print(f'Écart-type : {std_dev}')
    print(f'Coefficient d\'asymétrie (skewness) : {skewness}')


    if skewness > 0.5:
        print("La distribution est asymétrique vers la droite (asymétrie positive)")
    elif skewness < -0.5:
        print("La distribution est asymétrique vers la gauche (asymétrie négative)")
    else:
        print("La distribution est probablement symétrique")


    plt.figure(figsize=(10, 6))
    sns.histplot(dataframe[colone], kde=True, color='skyblue', bins=30)
    plt.axvline(mean, color='red', label=f'Moyenne ({mean:.2f})')
    plt.axvline(median, color='green', label=f'Médiane ({median:.2f})')
    plt.title(f"Distribution de {colone}")
    plt.xlabel(colone)
    plt.ylabel('Fréquence')
    plt.legend()
    plt.show()


#fonction pour remplacer les valeurs null
def replace(fonction: Callable[[pd.Series, str], float],x,df:pd.DataFrame,colone:str):
    if pd.isna(x):
        return round(fonction(df[colone]))
    else:
        return x
    


#gestion des valeurs null par le moyenne,median,mode
def remplacer_valuer_null(df: pd.DataFrame, colone: str):
    skewness = df[colone].skew()
    if len(df[colone].unique()) <= 4:  
        mode_value = df[colone].mode()  
        if not mode_value.empty:  
            df[colone] = df[colone].fillna(mode_value.iloc[0])  
        else:
            print(f"Aucun mode calculable pour la colonne {colone}. Les valeurs NaN ne seront pas remplacées.")
    elif skewness>0.5 or skewness <-0.5:
        df[colone] = df[colone].apply(lambda x :replace(pd.Series.median,x,df,colone))

    else:
        df[colone] = df[colone].apply(lambda x :replace(pd.Series.mean,x,df,colone))
        

#anamyse les categories
def analyse_categorique(dataframe, colone):
  
    freq = dataframe[colone].value_counts()

   
    print(f"Distribution des valeurs pour la colonne {colone} :")
    print(freq)

    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=dataframe, x=colone, palette='Set2')
    plt.title(f"Répartition des catégories dans {colone}")
    plt.xlabel(colone)
    plt.ylabel('Fréquence')
    plt.xticks(rotation=45)
    plt.show()





#detecter le valeurs aberrantes avec Z_score:

def detecter_et_plot_zscore(data: pd.DataFrame, colone: str):
    
    data[f'z_score_{colone}'] = (data[colone] - data[colone].mean()) / data[colone].std()
    
   
    data['aberrant'] = np.abs(data[f'z_score_{colone}']) > 3
    
   
    data_normales = data[data['aberrant'] == False]
    data_aberrantes = data[data['aberrant'] == True]
    
  
    plt.figure(figsize=(10, 6))
    plt.scatter(data_normales.index, data_normales[colone], color='blue', label='Valeurs normales')
    plt.scatter(data_aberrantes.index, data_aberrantes[colone], color='red', label='Valeurs aberrantes')
    

    plt.xlabel('Index')
    plt.ylabel(colone)
    plt.title(f'Détection des valeurs aberrantes dans la colonne "{colone}"')
    plt.legend()
    plt.show()
    
    
    return data_aberrantes


#detecter les valeurs aberrantes par frequance
def detecter_aberrantes_par_frequence_nombre(data: pd.DataFrame, colonne: str, seuil_frequence: int):
    
    frequencies = data[colonne].value_counts()

    valeurs_aberrantes = frequencies[frequencies < seuil_frequence].index.tolist()

    data_aberrantes = data[data[colonne].isin(valeurs_aberrantes)]
    
    data_normales = data[~data[colonne].isin(valeurs_aberrantes)]
    
    plt.figure(figsize=(10, 6))

    plt.scatter(data_normales.index, data_normales[colonne], color='blue', label='Données normales', alpha=0.6)

    plt.scatter(data_aberrantes.index, data_aberrantes[colonne], color='red', label='Données aberrantes', alpha=0.6)

    plt.xlabel('Index')
    plt.ylabel(colonne)
    plt.title(f'Données normales vs données aberrantes ({colonne})')

    plt.legend()

    plt.show()

    return data_aberrantes

#detecter les valeurs aberrantes avec le IQR
def detecter_avec_IQR(data:pd.DataFrame,colone:str):
    Q1 = data[colone].quantile(0.25)
    Q3 = data[colone].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print(lower_bound)
    print(upper_bound)
    data_aberrante = data[(data[colone] < lower_bound) | (data[colone]>upper_bound)]
    data_normales = data[(data[colone] >= lower_bound) & (data[colone] <= upper_bound)]
    plt.figure(figsize=(8, 6))
    

    plt.scatter(data_normales.index, data_normales[colone], color='blue', label='Valeurs normales', alpha=0.6)
    
   
    plt.scatter(data_aberrante.index, data_aberrante[colone], color='red', label='Valeurs aberrantes', alpha=0.6)
    
    
    plt.title(f"Distribution des valeurs normales et aberrantes de '{colone}'")
    plt.xlabel('Index')
    plt.ylabel(colone)
    plt.legend()
    plt.show()

    return data_aberrante






