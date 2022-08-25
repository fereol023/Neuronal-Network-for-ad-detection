# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 19:49:45 2022

@author: gbeno
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
   
    line = "="*50
    separator = "\n"*2

    EPOCHS = 100
    BATCH_SIZE = 256
    display_graphs = True
    
    target = "class"
    df = pd.read_csv("projetinfo.csv", delimiter = ",", index_col = None)
    
    print(df.head(10))
    print(separator)
    
    # convertir la variable target en numérique binaire
    remplacement = {"ad": 1, "noad": 0}
    df.replace( {"class" : remplacement}, inplace = True)
    
    print(df.head())
    print(separator)
    
    
    """
    print("Matrice de corrélation : avant normalisation")
    corr = df.corr()
    print(corr)
    print(separator)
    
    

    if display_graphs : 
        df.hist(figsize=(20,15), bins=40)
        plt.savefig("Advertisemznt data.png")
        plt.show()
        
        sns.heatmap(corr, linewidth=0.5, annot = True,
                    xtickslabels = corr.columns.values,
                    yticklabels = corr.columns.values)
        plt.title("Matrice de correlation v2")
        plt.show()
        
    
    for item in ["var1", "var2"] :
        median = df[item].median()
        df[item] = df[item].replace[to_replace = 0, value = median]
    """

    ##################
    #
    # Preprocessing
    #
    ##################
    
    variables_du_modele = df.columns.drop(['class'])
    x = df[variables_du_modele]
    y = df['class'] 
    
    # normalisation de tous les prédicteurs
    
    
    from sklearn.preprocessing import StandardScaler
    #from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
    from sklearn.model_selection import train_test_split
    
    scaler = StandardScaler()
    
    scaled_values = scaler.fit_transform(x)
    x = pd.DataFrame(scaled_values, columns=x.columns)
    
    # separation en train_test
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    
    y_train = y_train.values.reshape((-1,1))
    y_test = y_test.values.reshape((-1,1))
    
    
    print(separator)
    print(line)
    print("Taille train set : ", x_train.shape)
    print("Taille test set : ", x_test.shape)
    
    print(line)
    print(separator)
    
    # RN
    
    import time
    t1 = time.time()
    
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    model = Sequential()
    
    nb_entries = 1558
    nb_neurons_hidden_layer_1, nb_neurons_hidden_layer_2 = 64, 16  # décroissant
    
    model.add(Dense(nb_neurons_hidden_layer_1, input_dim = nb_entries, activation = "relu"))
    model.add(Dense(nb_neurons_hidden_layer_2, activation = "softplus"))
    model.add(Dense(1, activation = "sigmoid"))
    
    model.compile(loss="binary_crossentropy",
                  optimizer=Adam(learning_rate=0.01),
                  metrics=["accuracy"])
    
    
    model_file_name = 'adv_pred_best.hdf5'
    
    cb_check = ModelCheckpoint(model_file_name,
                               monitor = 'val_accuracy',
                               mode = 'max',
                               verbose = 1,
                               save_best_only = True)
    
    cb_early = EarlyStopping(monitor = 'val_accuracy',
                                   min_delta = 0.1,
                                   patience = 20)
    
    # callbacks
    callbacks_list = [cb_check, cb_early]
    
    # apprentissage
    history = model.fit(x_train, y_train, 
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE, 
                        verbose = 0)

    t2 = time.time()
    
    # sauvegarde
    precision = history.history["accuracy"][-1]
    
    
    def rapport(X, y, title) :
        global model
        
        print(line)
        print(title)
        
        loss, acc = model.evaluate(X, y, verbose=0)
        print("Précision = ", acc*100.0," %")
        print("Fonction de perte = ", loss)
     
        
    # IMPLEMENTATION
    
    rapport(x_train, y_train, "Train set")
    rapport(x_test, y_test, "Test set")
    
    precisions_vect = history.history['accuracy']
    loss_vect = history.history['loss']
    
    print(line)
    index = np.argmax(precisions_vect)
    print("meilleure epoch = ", index)  
    print("meilleure précision (train set) = ", precisions_vect[index])
    print("loss correspondant = ", loss_vect[index])
    
    # AFFICHAGE DU RESULTAT
    
    print(line)
    print("AFFICHAGE DU RESULTAT")
    
    
    figure, axis = plt.subplots(2, 1)
    epochs_vect = np.arange(0, EPOCHS, 1)
        # accuracy
    axis[0].plot(epochs_vect, precisions_vect)
    axis[0].set_title("Evolution de la precision suivant les 100 epochs")
    
        # loss
    axis[1].plot(epochs_vect, loss_vect, color = 'orange')
    axis[1].set_title("Evolution de la perte suivant les 100 epochs")
    figure.tight_layout()
    plt.show()
    
    