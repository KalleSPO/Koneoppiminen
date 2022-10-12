from array import array
from re import X
from typing_extensions import assert_type
from unicodedata import category
import sklearn # This is anyway how package is imported
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn import metrics


'''
Install scikit-learn module with pip install scikit-learn command.

Tehtävät:
1. Luetaan dataout.csv tiedosto pandas data frameksi siten, että tiedostosta luetaan vai
   sarakkeet xyz ja labels. Eli jätetään se indeksi sarake, joka koostuu 0,1,2 jonosta pois. 
   Käytä dataframen read_csv funktiota ja sieltä parametreja delimiter=, header=, usecols=
'''
data = pd.read_csv('./dataout.csv',sep='\t', header=0, usecols=['x','y','z','labels'])
#data = pd.read_csv('./dataout.csv',sep='\t', header=None, usecols=[1,2,3,4], names=['x','y','z','labels'])



'''
2. Poistetaan edellä luetusta dataframesta sen ensimmäinen rivi, jossa siis xyz ja labels tieto.
   Tämä siksi, että jäljelle jäänyttä 60,3 matriisia ja string saraketta käytetään eri algoritmien
   opettamiseen. Käytä dataframe iloc metodia
'''
#dataPP = data.iloc[1:]
#print(dataPP)
dataPP = data


'''
3. Seuraavaksi suodatetaan dataframesta pois sellaiset rivit, joissa x,y tai z arvo on suurempi
   kuin 1023, mikä on Arduinon analogia muuntimen maksimi lukema. Eli poistetaan virheelliset 
   mittaustulokset. Tulosta dataframe rivistä 40 eteenpäin (iloc käsky) ennen suodatusta ja 
   suodatuksen jälkeen, jotta varmistut siitä, että osa riveistä poistuu suodatuksen avulla
   Selvitä internetin avulla kuinka pandas dataframen sarakkeen arvoja voi suodattaa.
'''


#print(dataPP.iloc[39:])

#dataPP = dataPP.drop(dataPP.index[[dataPP['x'] >1023] or [dataPP['y'] >1023] or [dataPP['z'] >1023]])   #ei onnistu

dataPP = dataPP.drop(dataPP.index[dataPP['x'] >1023])
dataPP = dataPP.drop(dataPP.index[dataPP['y'] >1023])
dataPP = dataPP.drop(dataPP.index[dataPP['z'] >1023])
print(dataPP)
# print("---------")
# print(dataPP.iloc[39:])

'''

4. Seuraavaksi irroitetaan dataframesta labels tiedot left, right, up ja down tietoja
   kertova sarake (sen pitäisi olla neljäs sarake. Voit kokeilla esim print(df[4]) komennolla)
   Muutetaan sarakkeen tyyppi as_type komennolla 'category' tyypiksi ja luodaan dataframeen
   vielä viides sarake ja alustetaan sinne df[4].cat.codes funktion avulla numeeriset arvot
   left, rigth, up ja down arvoja vastaamaan.
'''
#print(dataPP['labels'])

#print(dataPP['labels'].dtypes)

dataPP['labels']=dataPP['labels'].astype('category')
#print('----------')
#print(dataPP['labels'].dtypes)

dataPP['codes']=dataPP['labels'].cat.codes
# print(dataPP)
# down   =  0
# left   =  1
# right  =  2
# up     =  3
'''


5. Seuraavaksi "irroitetaan" dataframesta x,y,z sarakkeet ja muodostetaan niistä yksi 
   NumPy array, jossa on kolme saraketta ja N kpl rivejä. Tämä array = matriisi = data on sitten
   se, mitä käytetään eri mallien datana opettamiseen. Irroitetaan myös numpy arrayksi
   se viides sarake joka edellisessä vaiheessa saatiin tehtyä. Ja tätä käytetään opetuksessa
   kertomaan, mitä kukin data matriisin rivi edustaa = labels. Ja muutetaan molemmat irroitetut
   data ja labels int tyyppisiksi.
'''

#xyzArray = dataPP(['x','y','z']).to_numpy

xyz = dataPP.iloc[:40,0:3].values   # jotku random arvot sekotti tuota ite oppimista
code = dataPP.iloc[:40,-1].values

#print(code.dtype)
xyz = xyz.astype('int')
code = code.astype('int')     #int32
#print(code.dtype)
# print(xyz)
# print("-------")
# print(code)
# print((xyz.size/3))
# print(code.size)


'''
6. Ja nyt vihdoin data on käsitelty algoritmin opettamiseen sopivaksi. Jaetaan data vielä
   training ja test datasetteihin ja käytetään siihen sklearn kirjaston train_test_split luokkaa
   jonka voi importata komennolla from sklearn.model_selection import train_test_split. Tee
   sellainen jako, että datasta 20% jätetään testaukseen ja 80% datasta käytetään opetukseen.
   Netistä löytyy taas hyviä esimerkkejä, miten tämä tehtään: https://realpython.com/train-test-split-python-data/
'''


xyz_train, xyz_test, code_train, code_test = train_test_split(xyz, code, test_size=0.2, random_state=4)

#print(xyz_test," ",code_test)
#print(xyz_train.shape)
print(type(xyz_test))
'''

7. Ja lopuksi testataan random forest ja K-means algoritmien toimivuutta. Eli opetetaan opetusdatalla
   x_train,y_train sekä random forest että K-means malli. Ja sen jälkeen testataan mallin tarkkuus
   x_test,y_test datalla. Ja ylimääräisenä tehtävänä voi vielä mitata kummastakin algoritmista kuinka
   kauaan mallin opettaminen kestää ja kuinka kauan yhden ennustuksen tekeminen mallilla kestää. Ja
   apuja löytyy taas netistä seuraavasti:
   K-means: https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75
   Random Forests:https://www.datacamp.com/tutorial/random-forests-classifier-python

       
'''

# k_range = range(1,38)
# scores = {}
# scores_list = []

# for k in k_range:
#        knn = KNeighborsClassifier(n_neighbors=4)
#        knn.fit(xyz_train, code_train)
#        code_pred=knn.predict(xyz_test)
#        scores[k] = metrics.accuracy_score(code_test,code_pred)
#        scores_list.append(metrics.accuracy_score(code_test,code_pred))


# plt.plot(k_range,scores_list)
# plt.show()

# print(scores)
# print(scores_list)

# KMeans

# knn = KNeighborsClassifier(n_neighbors=4)
# start = time.time()

# knn.fit(xyz_train,code_train)

# print("Training kmeans model took =", (time.time()-start)*1000,"ms")

# code_pred = knn.predict(xyz_test)
# code_predP = knn.predict_proba(xyz_test)
# code_predL=[]
# code_predPL=[]
# print("Testi data: ",code_test)

# for x in range(code_test.size):
#    code_predL.append(code_pred[x])
#    code_predPL.append(code_predP[x])

# print("Mallin output: ", code_predL)
# print("Mallin todennäköisyys: ", code_predPL)


# Random Forest

from sklearn.ensemble import RandomForestClassifier
rForest = RandomForestClassifier(n_estimators=2)
start = time.time()
rForest.fit(xyz_train, code_train)
print("Training random forest model took =", (time.time()-start)*1000,"ms")
print("Random Forest malling tarkkus = ", rForest.score(xyz_test,code_test))

# start=time.time()
print("Testikoodi: ",code_test)

codePredict = rForest.predict(xyz_test)
print("Random forest output: ",codePredict)
