from turtle import color
import numpy as np
import matplotlib.pyplot as plt
'''
Let's look in to matplotlib library properties and train ourselves
with Numpy array manipulations. Your tasks are the following:
1) Make 256*256 pixel picture (instructions given below). Current
   example plots RGB picture where Red component is all ones and G,B components are zero
   thus picture is all red. Your task is to modify picture in such a way that pixels
   at range 0-127 (rows), 0-127 (columns) show red picture area
   at range 0-127 (rows), 128-256 (columns) show green picture area
   at range 128-256 (row), 0-127 (columns) show blue picture area
   at range 128-256 (row), 128-256 (columns) show gray picture area 

'''
# size = 256

# kuva = np.zeros((size,size,3)) #luodaan 3d array, 3
# # kuva[_alku:_loppu, _alku:_loppu, kerros]
# kuva[0:128, 0:128,0]= np.ones((128,128))  #vasen ylä punanen
# kuva[0:128,128:,1]=np.ones((128,128))  #oikea ylhäältä vihreä
# kuva[128:,0:128,2]=np.ones((128,128)) #vasen ala siniseksi
# kuva[128:,128:,:]=np.ones((128,128,3))*0.5
# plt.imshow(kuva)
# plt.show()

'''
Let's look at matplotlib library basic usage tutorials. There is an example
how to create 3 figures as shown below

fig1 = plt.figure()  # an empty figure with no Axes
fig2, ax = plt.subplots()  # a figure with a single Axes
fig3, axs = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes

Your tasks are the following:
1) create 4 NumPy arrays containing 1 second sine signals with 1, 2, 3, 4 Hz
    HINT:
    create first time t = np.arange(0,1,0.01), where t starts from 0, ends at 0.99
    and then use np.sin and np.pi functions to genereate certain sine signals with certain
    frequencies. And remember the equation = sin(2*pi*f*t)
2) print 1 Hz sine signal to fig1
3) print 2 Hz sine signal to fig2
4) print 1,2,3,4 Hz sine signals to fig3 subplots. HINT: remember that axs is 2*2 matrix. 
   thus accessing second subplot goes with axs[0,1], where 0 = first row, 1 = second column.
5) print titles to all 3 figures
6) change line type and color of line of all 4 subplots at fig3 

7) And finally as we hopefully saw during the lectures that one can put
   many signals to a matrix and then just print matrix and all columns are
   seen as separate signals. Make signal matrix where you have a 1 second 2 Hz sine
   signal and its second power. And print those to a same figure.

'''
t=np.arange(0,1,0.01)
sin1= np.sin(2*np.pi*1*t)
sin2= np.sin(2*np.pi*2*t)
sin3= np.sin(2*np.pi*3*t)
sin4= np.sin(2*np.pi*4*t)

#1
fig1 = plt.figure()
fig1.supylabel("1hz")
plt.plot(t,sin1,'')   #rivit, sarakkeet


#2
fig2, ax = plt.subplots()
ax.set_ylabel('2hz')
plt.plot(t,sin2,'')

#3,4,5,6
fig3, sinplot1234= plt.subplots(2,2, layout='constrained')   #rivit, sarakkeet

sinplot1234[0,0].plot(t,sin1,'*',color="green")
sinplot1234[0,0].set_title('1hz')

sinplot1234[0,1].plot(t,sin2,'.',color="red")
sinplot1234[0,1].set_xlabel('2hz')

sinplot1234[1,0].plot(t,sin3,'^',color="black")
sinplot1234[1,0].set_ylabel('3hz')

sinplot1234[1,1].plot(t,sin4,'-',color="magenta")
sinplot1234[1,1].set_xlabel('4hz')

#7
sin2p2 = np.power(sin2,2)     #sin2 potenssiin 2

matrix = np.zeros((t.size,2)) #luodaan matriisi
matrix[:,0]=sin2              #asetetaan matriisiin sini aallot
matrix[:,1]=sin2p2

fig4, ax2 = plt.subplots(layout='constrained')
plt.plot(matrix)
ax2.set_xlabel("mS")
ax2.set_ylabel("Voltage")
plt.show()

kurvijuttu= np.exp(2*np.pi*1*t)
fig5, kj=plt.plot(t,kurvijuttu)
plt.show()