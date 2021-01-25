import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from pmdarima.utils import diff_inv
import time
import warnings
fmpi = 1

warnings.simplefilter('ignore')

def NN_Skit(x_train, y_train, x_test): 
	from sklearn.neural_network import MLPRegressor
	m = MLPRegressor(hidden_layer_sizes=(100,100,100),alpha=0.001,solver='adam',activation='relu',max_iter=1,random_state=1,verbose=True)
	#m = MLPRegressor(hidden_layer_sizes=(100,100,100,100,),random_state=1)
	#m = MLPRegressor(random_state=1)
	m.fit(x_train,y_train)
	print("Neural Network Score: {}".format(m.score(x_train,y_train)))
	print("Hyperparameters: {}".format(m.get_params()))
	mean_train = m.predict(x_train)
	mean_test = m.predict(x_test)
	return mean_train, mean_test, m

start = time.time()

dfo=pd.read_csv('train.csv')
dft=pd.read_csv('test.csv')
#dfs=pd.read_csv('train_targets_scored.csv')
#dfnet=pd.read_csv('network.csv')

dfot=pd.concat([dfo,dft])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for encl in ['lineName','trainNo','stopStation']:
    le.fit(dfot[encl].unique())
    dfo[encl]=le.transform(dfo[encl])
    dft[encl]=le.transform(dft[encl])

import datetime
base_time = pd.to_datetime('00:00', format='%H:%M')
dfo['planArrival']=pd.to_datetime(dfo['planArrival'], format='%H:%M') - base_time
dfo['planArrival']=dfo['planArrival'].dt.total_seconds()
dft['planArrival']=pd.to_datetime(dft['planArrival'], format='%H:%M') - base_time
dft['planArrival']=dft['planArrival'].dt.total_seconds()

dfo1 = dfo.drop('id', axis=1)
dft1 = dft[dft['target'] == 0].drop('target', axis=1).drop('id', axis=1).dropna(how='any')
dft2 = dft[dft['target'] == 1].drop('target', axis=1).drop('id', axis=1)
dfoID = dfo['id']
dftID = dft['id']
dft1ID = dft[dft['target'] == 0]['id']
dft2ID = dft[dft['target'] == 1]['id']

#----------date bound----------------#
#dfo1 = dfo1[dfo1['date'] <= 20200430]
#dft1 = dft1[dft1['date'] <= 20200430]

#----------sample diff.--------------#
dfo1['dd']=dfo1['delayTime'].diff().fillna(0)
dfo1['td']=dfo1['trainNo'].diff().fillna(0)
ddTemp=dfo1.loc[~(dfo1['td'] == 0),'dd']
dfo1T=dfo1.copy()
dfo1.loc[~(dfo1['td'] == 0),'dd']=0
dfo1=dfo1.drop('td', axis=1)#.drop('delayTime', axis=1)

dft1['dd']=dft1['delayTime'].diff().fillna(0)
dft1['td']=dft1['trainNo'].diff().fillna(0)
ddTemp=dft1.loc[~(dft1['td'] == 0),'dd']
dft1T=dft1.copy()
dft1.loc[~(dft1['td'] == 0),'dd']=0
dft1=dft1.drop('td', axis=1)#.drop('delayTime', axis=1)

dft2['dd']=dft2['delayTime']

print(dfo1.describe())
print(dft1.describe())
print(dft2.describe())

mdis=2
ntar=mdis
ttar=mdis

df=pd.concat([dfo1,dft1])
h=df.columns.values
#--Train--#
nd=df.loc[:,h].values
dat=nd[:,:]

k=dat.shape
ndis=k[1]-ntar
nsam=k[0]
tdis=ndis
  
#--Test--#
ndt=dft2.loc[:,h].values
datt=ndt[:,:]

tk=datt.shape
tdis=tk[1]-ttar
tsam=tk[0]

#--PrePro--#
#sc=MinMaxScaler()
sc=StandardScaler()
#sc=QuantileTransformer()
#--Train--# 
std=sc.fit_transform(dat)
std0=std.T[0:ndis]
std1=std.T[ndis:ndis+ntar]

#--Test--# 
stdt=sc.transform(datt)
stdt0=stdt.T[0:tdis]
stdt1=stdt.T[tdis:tdis+ttar]

print("|-----------------------|")
print("|--Parameter Dimension--|")
print("|-----------------------|")
print("Train Sample: {}".format(nsam))
print("Train Discripter: {}".format(ndis))
print("Train Target: {}".format(ntar))
print("Test Sample: {}".format(tsam))
print("Test Discripter: {}".format(tdis))
print("Test Target: {}".format(ttar))

#def dPCA(XPCA, YPCA, XTPCA, NPC): 
#        from sklearn.decomposition import PCA 
#        from sklearn.decomposition import KernelPCA
#        from sklearn.decomposition import FastICA
#        from sklearn.metrics import explained_variance_score
#        h=[]
#        for i in range(1,NPC,1):
#            h.append("PC" + str(i))
#        h.append("TG")
#        XXPCA=np.vstack([XPCA,XTPCA])
#        #decomp = PCA(n_components=NPC, random_state=0).fit(XXPCA)
#        decomp = FastICA(n_components=NPC, random_state=0).fit(XXPCA)
#        #decomp = LDA(n_components=NPC).fit(XXPCA, YPCA)
#        #decomp = KernelPCA(n_components=NPC, kernel="rbf", fit_inverse_transform=True, alpha=0.01, gamma=1).fit(XXPCA)
#        #decomp = KernelPCA(n_components=NPC, kernel="linear", fit_inverse_transform=True, alpha=0.001).fit(XXPCA)
#        dd_data = decomp.transform(XXPCA)
#        d_data = dd_data[:nsam,:]
#        dt_data = dd_data[nsam:,:]
#        evs = explained_variance_score(XXPCA, decomp.inverse_transform(dd_data))
#        #evs_ratio = decomp.explained_variance_ratio_/evs
#        evs_ratio = np.var(dd_data, axis=0) / np.sum(np.var(dd_data, axis=0))
#        print("|-----------------------|")
#        print("|--Decomposition Param--|")
#        print("|-----------------------|")
#        print("Score: {}".format(evs))
#        print("Score Ratio: {}".format(evs_ratio))
#        print("Shape: {}".format(dd_data.shape))
#        #scc=MinMaxScaler()
#        #scc.fit(dd_data)
#        #d_data=scc.transform(dd_data)
#        return d_data, dt_data, h, decomp

#spl0=std0.T[:,:3]
#spl1=std0.T[:,3:3+772]
#spl2=std0.T[:,3+772:]
#tspl0=stdt0.T[:,:3]
#tspl1=stdt0.T[:,3:3+772]
#tspl2=stdt0.T[:,3+772:]
#d_data, dt_data, h, decomp = dPCA(XPCA=spl1, YPCA=std1.T, XTPCA=tspl1, NPC=150)
#d_data2, dt_data2, h, decomp = dPCA(XPCA=spl2, YPCA=std1.T, XTPCA=tspl2, NPC=20)
#tmp1=np.hstack([spl0,d_data])
#tmp2=np.hstack([tmp1,d_data2])
#ttmp1=np.hstack([tspl0,dt_data])
#ttmp2=np.hstack([ttmp1,dt_data2])
#np.savetxt("Train_ICA150-20.dat",tmp2)
#np.savetxt("Test_ICA150-20.dat",ttmp2)
#X_train=tmp2
#Y_train=std1.T
#X_test=ttmp2

x_train=std0.T
y_train=std1.T
x_test=stdt0.T

pndis=ndis
k=X_train.shape
Nsam=k[0]
Ndis=k[1]

from sklearn.model_selection import train_test_split
LEND = 1
for Repeat in range(0, LEND, 1):
  rank = 0
  #FLAGMPILrank=rank
  LRepeat = Repeat + 10 ** rank
  x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=LRepeat)
  id_train, id_test, nodat1, nodat2 = train_test_split(dfoID.values, Y_train, test_size=0.2, random_state=LRepeat)
  print(x_train.shape)
  print(y_train.shape)
  print(x_test.shape)
  print(y_test.shape)
  print(id_train.shape)
  print(id_test.shape)

  k=x_train.shape
  nsam=k[0]
  ndis=k[1]

  mean_train, mean_test, m = NN_Skit(x_train, y_train, x_test)
  nmean_train = np.array(mean_train)
  nmean_test = np.array(mean_test)
  mean_train = nmean_train.reshape(-1,ntar)
  mean_test = nmean_test.reshape(-1,ntar)
 
  ##Prediction Score
  from sklearn.metrics import mean_absolute_error
  from sklearn.metrics import r2_score
  print("|-----------------------|")
  print("|------delta delay------|")
  print("|-----------------------|")
  print("|------Train Score------|")
  print("|-----------------------|")
  R2=np.hstack([[[1]*pndis]*x_train.shape[0],y_train])
  p1=sc.inverse_transform(R2)
  Ry_train=np.round(p1)
  #Ry_train=np.where(p1 < 0,0,p1)
  R2=np.hstack([[[1]*pndis]*x_train.shape[0],mean_train])
  p1=sc.inverse_transform(R2)
  Rmean_train=np.round(p1)
  #Rmean_train=np.where(p1 < 0,0,p1)
  mse_train = mean_absolute_error(Ry_train[:,pndis],Rmean_train[:,pndis])
  print("Mean Squared Error: {}".format(mse_train))
  mse_train = mean_absolute_error(Ry_train[:,pndis+1],Rmean_train[:,pndis+1])
  print("Mean Squared Error 2: {}".format(mse_train))
  
  print("|-----------------------|")
  print("|------Test Score-------|")
  print("|-----------------------|")
  R2=np.hstack([[[1]*pndis]*x_test.shape[0],y_test])
  p1=sc.inverse_transform(R2)
  Ry_test=np.round(p1)
  #Ry_test=np.where(p1 < 0,0,p1)
  R2=np.hstack([[[1]*pndis]*x_test.shape[0],mean_test])
  p1=sc.inverse_transform(R2)
  Rmean_test=np.round(p1)
  #Rmean_test=np.where(p1 < 0,0,p1)
  mse_test = mean_absolute_error(Ry_test[:,pndis],Rmean_test[:,pndis])
  print("Mean Squared Error: {}".format(mse_test))
  mse_test = mean_absolute_error(Ry_test[:,pndis+1],Rmean_test[:,pndis+1])
  print("Mean Squared Error 2: {}".format(mse_test))

  print("|-----------------------|")
  print("|------delay Time-------|")
  print("|-----------------------|")
  train=np.hstack([x_train,mean_train])
  test=np.hstack([x_test,mean_test])
  inv1=np.vstack([train,test])
  inv2=sc.inverse_transform(inv1)
  inv3=np.round(inv2)
  inv4=np.concatenate([id_train,id_test])
  inv=np.hstack([inv4.reshape(-1,1),inv3])
  dfx=dfo.copy().assign(dd=0)
  dfi = pd.DataFrame(data=inv, index=None, columns=dfx.columns) 
  dfi=dfi.sort_values('id').reset_index()
  dfi['td']=dfo1T['td']
  #dfi=dfi.rename(columns={'delayTime': 'dd'})
  #dfi['delayTime']=dfo1T['delayTime']
  des=0
  tdll=dfi['td'].values
  dell=dfi['dd'].values
  dTll=dfi['delayTime'].values
  bl=[]
  for i in range(len(dfi)):
      td=tdll[i]
      de=dell[i]
      dT=dTll[i]
      if td != 0: 
          des=dT
      else:
          des=des+de
      bl.append(des)
  dfi['ddlay']=bl
  dfi.to_csv("dfi2.csv")
  diffD=pd.DataFrame(index=None,columns=(['id','error']))
  diffD['error']=dfi['delayTime']-dfi['ddlay']
  diffD['id']=dfi['id']
  diffD_train=diffD[diffD['id'].isin(id_train)]
  diffD_test=diffD[diffD['id'].isin(id_test)]
  print(diffD_train)
  print(diffD_test)
  mse_all=diffD['error'].abs().mean()
  print("Mean Squared Error: {}".format(mse_all))

  print("|-----------------------|")
  print("|------Train Score------|")
  print("|-----------------------|")
  mse_train=diffD_train['error'].abs().mean()
  print("Mean Squared Error: {}".format(mse_train))
  
  print("|-----------------------|")
  print("|------Test Score-------|")
  print("|-----------------------|")
  mse_test=diffD_test['error'].abs().mean()
  print("Mean Squared Error: {}".format(mse_test))



elapsed_time = time.time() - start
print ("elapsed_time : {0:.4f}".format(elapsed_time) + "[sec]")
