from sklearn.preprocessing import LabelEncoder
import joblib

def preprocess_data(df):
    # Identifier les colonnes catégorielles (type object)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print("Colonnes catégorielles :", categorical_cols)

    # Exclure la colonne cible 'Diagnosis'
    categorical_cols = [col for col in categorical_cols if col != 'Diagnosis']

    # Encoder les colonnes catégorielles
    for col in categorical_cols:
        le_col = LabelEncoder()
        df[col] = le_col.fit_transform(df[col])
        #joblib.dump(le_col, f"encoder_{col}.pkl")

    # Maintenant qu'on a tout encodé :
    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis']

    # Encoder la cible en numérique
    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y

# from sklearn.preprocessing import LabelEncoder

# def preprocess_data(df):
#     # Supposons que la colonne cible s'appelle 'Disease'
#     X = df.drop(columns=['Diagnosis'])
#     y = df['Diagnosis']

#     # Encoder la cible en numérique
#     le = LabelEncoder()
#     y = le.fit_transform(y)

#     return X, y
