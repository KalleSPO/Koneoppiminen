import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


""" 
Tehtävä 1: 
- lataa Latest https://covidtracking.com/data/download/national-history.csv
  tiedosto pandas kirjaston avulla Pandas dataframeksi. 
- "Irroita" siitä ladattaessa'date','deaths','hospitalInc','hospitalNow' sarakkeet
- Piirrä matplotlib.pyplot kirjaston ja plt.subplot, plt.plot, plt.title, plt.show 
  komentojen avulla kuva, josta nähdään kuolleiden lukumäärät, sairaalapotilaiden
  inkrementaalinen kasvu ja kuinka paljon sairaalassa on potilaita eri päivinä.
- Selvitä minä päivänä potilaiden kasvu on ollut suurinta ja mikä on tuon päivän potilasmäärä
"""
nh = pd.read_csv('C:/OAMK/Koneoppimisen Perusteet/Koneoppiminen/koodit/national-history.csv', usecols=['date','death','hospitalizedIncrease','hospitalizedCurrently'])
nh.fillna(0, inplace=True)
print(nh)
#T2
date = pd.to_datetime(nh.date)
death = nh['death'].to_numpy()
inc = nh['hospitalizedIncrease'].to_numpy()
cur = nh['hospitalizedCurrently'].to_numpy()

#print(death)
#print(date)

fig, joo= plt.subplots(3,1)   #rivit, sarakkeet
#fig.suptitle
joo[0].plot(date, death,'')
joo[0].set_ylabel('Kuolleet')
joo[1].plot(date, inc,'' )
joo[1].set_ylabel('Sairaalahoitoon / vrk')
joo[2].plot(date, cur,'')
joo[2].set_ylabel('Sairaalahoidossa')

#plt.xlabel("Päivämäärä")
#plt.ylabel("")
plt.show()




"""
Tehtävä 2:
- Muuta kaikki dataFramen sarakkeet numpy arrayksi to_numpy() funktion avulla
- Tulosta kuolleiden määrä ja sairaalassa olleiden lukumäärät oikeassa järjestyksessä
  (huom päivämäärät ovat tiedostossa viimeisin päivämäärä ensin. Eli käännä tulostusjärjestys
   siten, että kuvaan tulostetaan deaths sarakkeen viimeisin alkio ensin jne.)
""" 

