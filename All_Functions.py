#!/usr/bin/env python
# coding: utf-8

# # 1. Les libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns
import datetime

from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


from matplotlib.ticker import AutoMinorLocator
from sklearn import datasets
from sklearn.cluster import KMeans

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Défintion des functions à utiliser

# In[2]:


class DataFrame_and_description:
    
    def Import_data():
        '''This function helps to import data'''
        
        filename=str(input("Please enter the name of a file to read:"))
        raw_data=pd.read_csv(filename, sep='\t')
        
        return raw_data
    
    def Dimension_dataframe(data):
        ''' Cette fonction retourne la dimension de notre dataframe'''
        print("-------------------------------------------------:\n")
        print("Les dimensions de notre dataframe sont:",data.shape)
    
    def Info_data(data):
        '''cette fonction nous donne des indications globales'''
        print("-------------------------------------------------:\n")
        return data.info()
    
    def Valeur_manquante_par_collone(data):
        ''' cette focntion retourne  les valeurs manquantes par colonne '''
        print("-------------------------------------------------:\n")
        return data.isna().mean()
    
    def Doublicate_number(data):
        '''Y a-t-il des lignes en double'''
        print("-------------------------------------------------:\n")
        print('le nombre total des doublons=',data.duplicated().sum())
    
    def  Diff_value_par_column(data):
        '''Combien y a-t-il de valeurs différentes par colonne ?'''
        print("-------------------------------------------------:\n")
        return print('les valeurs differentes par collone:\n',data.nunique())  
    
    def Get_data_description(data):
        '''cette fonction nous donne une idée de la distribution statistique globale de nos données'''
        """ Compte le nombre de ranges dans le fichier EdStatsData.csv"""
        #print(data3.axes)             ##une liste représentant les axes du dataframe.
        nb_ranges = len(data.axes[0])       #Compte le nombre de rangées
        nb_cols = len(data.axes[1])         #Compte le nombre de colonnes 
        print("le nombre de rangées: ", nb_ranges)
        print("le nombre de colonnes : ", nb_cols)
        print('Statistical distribution of data:\n')
        print('--------------------------------------------\n')
        display(data.describe())
        
    def Check_skewness(data):
        ''' This function checks the skewness of our dataframe'''
        print(data.skew()) 
        
    def _interqartile_(data,col):
        Q1=data[col].quantile(0.25)
        Q3=data[col].quantile(0.75)
        IQR=Q3-Q1
        return Q1,Q3,IQR
    
    def _FInding_outliers_using_interquartile(data,Q1,Q3,IQR,col):
        whisker_width = 1.5
        Fare_outliers =data[(data[col] < Q1 - whisker_width*IQR) | (data[col] > Q3 + whisker_width*IQR)]
        return  Fare_outliers
    
    def _FInding_outliers_using_standard_deviation(data,col):
        '''Standard deviation measures the amount of variation and 
        dispersion of a set of values relative to the average value
        of the data, it shows the variability distribution of the data.'''
        fare_mean = data[col].mean()
        fare_std = data[col].std()
        low= fare_mean -(3 * fare_std)
        high= fare_mean + (3 * fare_std)
        fare_outliers = data[(data[col] < low) | (data[col] > high)]
        return high,low,fare_outliers 


# In[3]:


class Dealing_with_defect:
    
    def Drop_column(data,X):
        ''' cette fonction delete les collonnes non pertinentes'''
        return data.drop(columns=X)
    
    def Drop_values_if_greater_than_100g(data,colms):
        for var in colms:
            indexNames = data[ data[var] >= 100 ].index
            data.drop(indexNames , inplace=True)
        return data
    
    
     
    
    
    def Replacing_NAN_by_mean(data,X):
        ''' cette fonction remplace les valeurs manquantes par la moyenne'''
        
        data[X] = data[X].fillna(data[X].mean())
        return data
        
    
    
    def Replacing_NAN_by_medianne(data,X):
        ''' cette fonction remplace les valeurs manquantes par la medianne '''
       
        data[X] = data[X].fillna(data[X].median())
         
        return data 
    
    def Replacing_NAN_by_mode(data,X):
        ''' cette fonction remplace les valeurs manquantes par la medianne '''
       
        data[X] = data[X].fillna(data[X].mode())
         
        return data 
    
    def Dropping_NAN_Values(data):
        ''' this fonction dropp all NAN values in dataframe'''
        data_Clean=data.dropna(how='any', axis=1)
        return data_Clean
    
    def Replacing_NAN_by_zero(data):
        ''' THis function replace NAN values with zeros'''
        data = data.fillna(0)
        return data
    
    
    def _Drop_outliers(data, col,Q1,Q3,IQR):
        whisker_width = 1.5
        lower_whisker = Q1 -(whisker_width*IQR)
        upper_whisker = Q3 + (whisker_width*IQR)
        data[col]=np.where(data[col]>upper_whisker,
                           upper_whisker,np.where(data[col]<lower_whisker,
                                                  lower_whisker,data[col]))
        return data,lower_whisker,upper_whisker
    
    def _replacing_outliers_with_median(data,median,col,high,low):
        ''' Replacing outliers with median'''
        data[col] = np.where(data[col] >high, median,data[col])
        data[col] = np.where(data[col] <low, median,data[col])
        return data

    
    
 
    
  


# In[4]:


class Data_exploration:
    
    def Scanning_product(data,x):
        
        '''This function allows to scan the product and returns its nutriscore value'''
         
        i = data.index
        index = data["code"] == x
        result = i[index]
        result.tolist()
        
        Nutriscore=data._get_value(result[0], "nutrition-score-fr_100g")
        print("The product has nutriscore value of =", Nutriscore)
        
        return(Nutriscore)
    
    
    def Grouping_data(data):
        '''cette fonction permet de classifier les nourriture based on the score of 
        nutrition-score-uk_100g ==nutrition-score-fr_100g'''
        
        data_grouped = data.groupby('nutrition-score-fr_100g')
        
        return data_grouped
    
    def Get_similar_product(data,nutriscore_value):
        
        data=data[(data["nutrition-score-fr_100g"]==nutriscore_value) & (data["cholesterol_100g"]==0)]
        print("Vous avez choisi un bon produit contre les maladies cardiaques")
        data=data.reset_index()
        nber_Show=int(input("Veuillez entrer le nombre d'articles similaires que vous souhaitez voir:"))
        display(data.head(nber_Show))
        return data
    


# In[5]:


class Linear_Regression_:
    def Selecting_data_to_fit(data):
        data_selected=data[data["fat_100g"] >20]
        ## On reindexe
        
        data_selected = data_selected.reset_index(drop = True)
        return data_selected
    
    def _3D_plot(colx,coly,colz): #
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure().gca(projection='3d')
        datax = colx[:]
        datay=coly
        dataz=colz
        fig.scatter(datax, datay, dataz, c=datax, cmap="viridis")
        plt.show()
    def spliting_data_into_train_test(data, col1, col2,col3):
        xtrain, xtest, ytrain, ytest = train_test_split(data[[col1,col2 ]],data[[col3]], test_size=0.3)
        return xtrain, xtest, ytrain, ytest
    
    def Linear_regression(xtrain,xtest,ytrain,ytest,col1):
        
        lr = LinearRegression()
        lr_baseline = lr.fit(xtrain[[col1]], ytrain)
        baseline_pred = lr_baseline.predict(xtest[[col1]])
        plt.plot(xtest[[col1]], ytest, 'bo', markersize = 5)
        plt.plot(xtest[[col1]], baseline_pred, color="skyblue", linewidth = 2)
        return baseline_pred
        
    def sumsq(x,y):
        return sum((x - y)**2)
    
    def r2score(pred, target):
        return 1 - Linear_Regression_.sumsq(pred, target) / Linear_Regression_.sumsq(target, np.mean(target))
    
    def Quadratic_sum_of_the_residuals(baseline_pred,ytest,col1):
        score_bl = Linear_Regression_.r2score(baseline_pred[:,0], ytest[col1])
        return score_bl 

    


# In[6]:


class _CAH_:
    def _values_of_dataframe(data, colstoDelete):
        X = Dealing_with_defect.Drop_column(data,cols)
        X=X.values
        X[np.isnan(X)] = 0
        print("verfions bien notre array:\n",X[:10])
        return X
    def Name_of_product(data):
        themes =data.product_name
        themes=themes.values
        print("verfions bien que c'est le nom de produit:\n",themes[:10])
        return themes
    def Name_of_country(data):
        names =data.countries_fr
        names=names.values
        print("verfions bien que c'est le nom du pays:\n",names[:10])
        return names
    
    def Data_scaling(X):
        std_scale = preprocessing.StandardScaler()
        std_scale.fit(X)
        X_scaled = std_scale.transform(X)
        print("verfier bien que la moyenne est zero:\n")
        display(pd.DataFrame(X_scaled).describe().round(2).iloc[1:3:, : ])
        print("Reverfier les données  rescaled:\n",X_scaled[:10])
        return X_scaled
    def Linking_node_par_Ward(X_scaled):
        Z = linkage(X_scaled, method="ward")
        print("Z=\n",Z[:10])
        return Z
    def Linking_node_par_Single(X_scaled):
        Z = linkage(X_scaled, method="single")
        print("Z=\n",Z[:10])
        return Z
    def Plotting_nodes(Z):
        # les arguments p=10, truncate_mode="lastp" signifient que l'on ne va afficher que 20 clusters
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        _ = dendrogram(Z, p=20, truncate_mode="lastp", ax=ax)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.ylabel("Distance.")
        plt.show()
    
    def _Heat_map(Z,k):
        clusters = fcluster(Z, k, criterion='maxclust')
        print("Cluster:\n", clusters)
        crosstab = pd.crosstab(themes, clusters, dropna=False)
        crosstab.rename_axis(columns="cluster", index="theme", inplace=True)
        print("table de contingence\n",crosstab)
        return crosstab,clusters
      
    def Plotting_Heat_map(crosstab):
        fig, ax = plt.subplots(1,1, figsize=(12,6))
        ax = sns.heatmap(crosstab, vmin=0.1, vmax=14, annot=True, cmap="Purples")
        plt.show() 
    
    def _printing_class_(clusters,names,themes):
        df = pd.DataFrame({"name" : names, "theme" : themes, "cluster" : clusters})
        print(df.head())
        
        for i in range(1, 13) : 
            # on fait une selection
            sub_df = df.loc[df.cluster == i]
            # le cluster en question
            print(f"cluster : {i}")
            # on extrait les noms et les themes de chaque ligne
            names_list = sub_df.name.values
            themes_list = sub_df.theme.values
            # on créé une liste de couple nom/theme
            ziped = zip(names_list, themes_list) 
            txt = [f"{n} ({t})" for n, t in ziped]
            # on transforme en str
            txt = " / ".join(txt)
            # on print
            print(txt)
            print("\n\n")

        


# In[7]:


class _ACP_:
    def _data_to_use_(data,ColtoDelete):
        data=Dealing_with_defect.Drop_column(data,ColtoDelete)
        X = data.values
        #X[np.isnan(X)] = 0
        print('c est bien an array:\n',type(X))
        print(X[:10])
        print('Les dimension de array:\n',X.shape)
        names =data.index
        return data,X,names
    
    def _PCA_(n_components,X_scaled):
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        pca.explained_variance_ratio_
        pca.explained_variance_ratio_
        
        scree = (pca.explained_variance_ratio_*100).round(2)
        scree
        scree_cum = scree.cumsum().round()
        scree_cum
        x_list = range(1, n_components+1)
        list(x_list)
        return x_list,scree,scree_cum,pca
    def Plotting_Inertia_percentage(x_list,scree,scree_cum):
        plt.bar(x_list, scree)
        plt.plot(x_list, scree_cum,c="red",marker='o')
        plt.xlabel("rang de l'axe d'inertie")
        plt.ylabel("pourcentage d'inertie")
        plt.title("Eboulis des valeurs propres")
        plt.show(block=False)
        
    def PCA_components(data,pca):
        pcs = pca.components_
        pcs
        pcs = pd.DataFrame(pcs)
        pcs
        features = data.columns
        features
        pcs.columns = features
        pcs.index = [f"F{i}" for i in x_list]
        pcs.round(2)
        pcs=pcs.T
        return pcs,features
    def _plot_Heat_map(pcs):
        fig, ax = plt.subplots(figsize=(20, 6))
        sns.heatmap(pcs, vmin=-1, vmax=1, annot=True, cmap="coolwarm", fmt="0.2f")
        
    def _Plotting_F2_function_F1(x,y):
        fig, ax = plt.subplots(figsize=(10, 9))
        for i in range(0, pca.components_.shape[1]):
            ax.arrow(0,
                     0,  # Start the arrow at the origin
                     pca.components_[0, i],  #0 for PC1
                     pca.components_[1, i],  #1 for PC2
                     head_width=0.07,
                     head_length=0.07, 
                     width=0.02,   )
        plt.text(pca.components_[0, i] + 0.05,pca.components_[1, i] + 0.05,features[i])
        
        # affichage des lignes horizontales et verticales
        plt.plot([-1, 1], [0, 0], color='grey', ls='--')
        plt.plot([0, 0], [-1, 1], color='grey', ls='--')
        
        # nom des axes, avec le pourcentage d'inertie expliqué
        plt.xlabel('F{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
        plt.ylabel('F{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))
        plt.title("Cercle des corrélations (F{} et F{})".format(x+1, y+1))
        
        an = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
        plt.axis('equal')
        plt.show(block=False)
        
    def correlation_graph(pca, x_y, features) :
        
        """Affiche le graphe des correlations
        
        Positional arguments : 
        -----------------------------------
        pca : sklearn.decomposition.PCA : notre objet PCA qui a été fit
        x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2
        features : list ou tuple : la liste des features (ie des dimensions) à représenter"""
        
        # Extrait x et y 
        x,y=x_y
        # Taille de l'image (en inches)
        fig, ax = plt.subplots(figsize=(10, 9))
        
        # Pour chaque composante : 
        
        for i in range(0, pca.components_.shape[1]):
            
            # Les flèches
            ax.arrow(0,0, pca.components_[x, i],pca.components_[y, i],
                     head_width=0.07,head_length=0.07, width=0.02, )
            # Les labels
            plt.text(pca.components_[x, i] + 0.05, pca.components_[y, i] + 0.05,features[i])
        
        # Affichage des lignes horizontales et verticales
        plt.plot([-1, 1], [0, 0], color='grey', ls='--')
        plt.plot([0, 0], [-1, 1], color='grey', ls='--')
        # Nom des axes, avec le pourcentage d'inertie expliqué
        plt.xlabel('F{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
        plt.ylabel('F{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))
        
        # J'ai copié collé le code sans le lire
        plt.title("Cercle des corrélations (F{} et F{})".format(x+1, y+1))
        # Le cercle 
        an = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
        # Axes et display
        plt.axis('equal')
        plt.show(block=False)
        
    def display_factorial_planes( X_projected, 
                                x_y, 
                                pca=None, 
                                labels = None,
                                clusters=None, 
                                alpha=1,
                                figsize=[10,8], 
                                marker="." ):
        """Affiche la projection des individus
        
        Positional arguments :
        -------------------------------------
        
        X_projected : np.array, pd.DataFrame, list of list : la matrice des points projetés
        x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2
        
        Optional arguments : 
        -------------------------------------
        pca : sklearn.decomposition.PCA : un objet PCA qui a été fit, cela nous permettra d'afficher la variance de chaque composante, default = None
        labels : list ou tuple : les labels des individus à projeter, default = None
        clusters : list ou tuple : la liste des clusters auquel appartient chaque individu, default = None
        alpha : float in [0,1] : paramètre de transparence, 0=100% transparent, 1=0% transparent, default = 1
        figsize : list ou tuple : couple width, height qui définit la taille de la figure en inches, default = [10,8] 
        marker : str : le type de marker utilisé pour représenter les individus, points croix etc etc, default = "."
        """

        # Transforme X_projected en np.array
        X_ = np.array(X_projected)
        # On définit la forme de la figure si elle n'a pas été donnée
        if not figsize: 
            figsize = (7,6)
        # On gère les labels
        
        if  labels is None : 
            labels = []
        try : 
            len(labels)
        except Exception as e : 
            raise e
        
        # On vérifie la variable axis 
        if not len(x_y) ==2 : 
            raise AttributeError("2 axes sont demandées")   
        if max(x_y )>= X_.shape[1] : 
            raise AttributeError("la variable axis n'est pas bonne")
        # on définit x et y
        
        x, y = x_y
        
        # Initialisation de la figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # On vérifie s'il y a des clusters ou non
        c = None if clusters is None else clusters
        
        # Les points
        
        # plt.scatter(   X_[:, x], X_[:, y], alpha=alpha,
        
        #                     c=c, cmap="Set1", marker=marker
        sns.scatterplot(data=None, x=X_[:, x], y=X_[:, y], hue=c)
        
        # Si la variable pca a été fournie, on peut calculer le % de variance de chaque axe
        if pca :
            v1 = str(round(100*pca.explained_variance_ratio_[x]))  + " %"
            v2 = str(round(100*pca.explained_variance_ratio_[y]))  + " %"
        else :
            v1=v2= ''
        # Nom des axes, avec le pourcentage d'inertie expliqué
        ax.set_xlabel(f'F{x+1} {v1}')
        ax.set_ylabel(f'F{y+1} {v2}')
        
        # Valeur x max et y max
        x_max = np.abs(X_[:, x]).max() *1.1
        y_max = np.abs(X_[:, y]).max() *1.1
        
        # On borne x et y 
        ax.set_xlim(left=-x_max, right=x_max)
        ax.set_ylim(bottom= -y_max, top=y_max)
        
        # Affichage des lignes horizontales et verticales
        plt.plot([-x_max, x_max], [0, 0], color='grey', alpha=0.8)
        plt.plot([0,0], [-y_max, y_max], color='grey', alpha=0.8)
        
        # Affichage des labels des points
        if len(labels) :
            # j'ai copié collé la fonction sans la lire
            for i,(_x,_y) in enumerate(X_[:,[x,y]]):
                plt.text(_x, _y+0.05, labels[i], fontsize='14', ha='center',va='center') 
            # Titre et display
        plt.title(f"Projection des individus (sur F{x+1} et F{y+1})")
        plt.show()
    


# In[8]:


class _plot_:
    
    def figure_custom():
        """figure customisation"""
        fig = plt.figure(figsize =(12, 7))
        ax = fig.add_subplot(111)
        return fig,ax
    
    def Box_plot(data,fig,ax):
        
        """Cette fonction permet de tracer les diagrame de moustaches"""
        # Creating axes instance
        bp = ax.boxplot(data, patch_artist = True,notch ='True', vert = 0)
        colors = ['#0000FF', '#00FF00','#FFFF00', '#FF00FF']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            # changing color and linewidth of
            # whiskers
        for whisker in bp['whiskers']:
            whisker.set(color ='#8B008B',linewidth = 1.5,linestyle =":")
        # changing color and linewidth of
        # caps
        for cap in bp['caps']:
            cap.set(color ='#8B008B',linewidth = 2)
        # changing color and linewidth of
        # medians
        
        for median in bp['medians']:
            median.set(color ='red',linewidth = 3)
        # changing style of fliers
        
        for flier in bp['fliers']:
            flier.set(marker ='D',color ='#e7298a',alpha = 0.5)
        # Removing top axes and right axes
        # ticks
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

