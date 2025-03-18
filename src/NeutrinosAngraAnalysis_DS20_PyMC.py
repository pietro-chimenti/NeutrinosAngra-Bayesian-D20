# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 06:05:08 2025

@author: Usuario
"""
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns

import pymc as pm

from pymc import HalfCauchy, Model, Normal, sample, Uniform, Gamma

###
### Parametros da simulação
###

chains  = 8
samples = 10000
tune    = 2000
run = False
file_basis = 'NA_DS20_N12_20250214_15_good'

# _6: 1000 samples, 1 chain
# _7: 1000 samples, 8 chains
#_8 _9 : testing
# _10: 1000 samples, 1 chains
# _11: 2000 samples, 8 chains
# _12: 1 chains, 1000 samples, 1000 tunes, larger priors
# _13: 1 chain, 2000 tune, 10000 sample, optimized priors
# _14: 1 chain, 2000 tune, 10000 sample, improving prior on Noise_a
# _15: 8 chain, 2000 tune, 10000 sample, final production?


file_name = file_basis+'.nc'
file_text = file_basis+'.txt'

'''Aqui definimos as listas com os trÊs tipos de dados:'''

eventos = np.array([290083, 300194, 300615, 304790, 289274, 293423, 282796,
           283150, 300899, 291846, 298127, 303175, 257279, 303144,
           306551, 303524, 297078, 308897, 300085, 296226, 307124, 
           297733, 303829, 292895, 293881, 295790, 295201 ])

cut = np.array([ 7490227, 7665584, 7711305, 7733262, 7438325, 7534427,
        7376137, 7416514, 7758711, 7457013, 7444900, 6987591,
        5868817, 7096703, 7244451, 7036721, 6812952, 7222445,
        6902124, 7121617, 7230945, 6957214, 7152199, 7110956,
        7551162, 7178880, 6649510 ])

indices_off = np.array([1,2,3,4,5,6,7])
indices_on = np.array([10,11,13,14,15,16,17,18,19,20,21,22,23,24,25])

dados_ibd_off = np.zeros(indices_off.size)
dados_ibd_on = np.zeros(indices_on.size)
dados_cut_off = np.zeros(indices_off.size)
dados_cut_on = np.zeros(indices_on.size)

for i in range(indices_off.size):
    dados_cut_off[i] = cut[indices_off[i]] 
    dados_ibd_off[i] = eventos[indices_off[i]] 

for i in range(indices_on.size):
    dados_cut_on[i] = cut[indices_on[i]] 
    dados_ibd_on[i] = eventos[indices_on[i]] 

print(dados_ibd_off)
print(dados_ibd_on)
print(dados_cut_off)
print(dados_cut_on)


#dados_ibd_off = np.array([300194, 300615, 304790, 289274, 293423, 282796, 283150])
#dados_ibd_on = np.array( [ 298127,303175,303144,306551,303524,297078,308897,300085,296226,307124,297733,303829,292895,293881,295790 ] )

#dados_cut_off = np.array( [7665584,7711305,7733262,7438325,7534427,7376137,7416514] )
#dados_cut_on  = np.array( [7444900,6987591,7096703,7244451, 7036721,6812952,7222445,6902124,7121617,7230945,6957214,7152199,7110956,7551162,7178880] )

datas_intervalos = [
    ('15/06/2020', '22/06/2020'),
    ('22/06/2020', '29/06/2020'),
    ('29/06/2020', '06/07/2020'),
    ('06/07/2020', '13/07/2020'),
    ('13/07/2020', '20/07/2020'),
    ('20/07/2020', '27/07/2020'),
    ('27/07/2020', '03/08/2020'),
    ('03/08/2020', '10/08/2020'),
    ('10/08/2020', '17/08/2020'),
    ('17/08/2020', '25/08/2020'),
    ('25/08/2020', '01/09/2020'),
    ('01/09/2020', '08/09/2020'),
    ('08/09/2020', '13/09/2020'),
    ('24/09/2020', '01/10/2020'),
    ('01/10/2020', '08/10/2020'),
    ('08/10/2020', '15/10/2020'),
    ('15/10/2020', '22/10/2020'),
    ('22/10/2020', '29/10/2020'),
    ('29/10/2020', '05/11/2020'),
    ('05/11/2020', '12/11/2020'),
    ('12/11/2020', '19/11/2020'),
    ('19/11/2020', '27/11/2020'),
    ('27/11/2020', '04/12/2020'),
    ('04/12/2020', '11/12/2020'),
    ('11/12/2020', '18/12/2020'),
    ('18/12/2020', '25/12/2020'),
    ('25/12/2020', '31/12/2020')
]

seleção = ['15/06/2020', '22/06/2020', '29/06/2020', '06/07/2020', '13/07/2020', '20/07/2020', 
           '27/07/2020', '03/08/2020', '10/08/2020', '17/08/2020', '25/08/2020', '01/09/2020',
           '08/09/2020', '13/09/2020', '24/09/2020', '01/10/2020', '08/10/2020', '15/10/2020',
           '22/10/2020', '29/10/2020', '05/11/2020', '12/11/2020', '19/11/2020',
           '27/11/2020', '04/12/2020', '11/12/2020', '18/12/2020', '25/12/2020', '31/12/2020']


print(dados_ibd_off.size)
print(dados_ibd_on.size)
print(dados_cut_off.size)
print(dados_cut_on.size)

'''Defimos os parametros de simulação (número de amostras e walkers)'''

with Model() as model:  # model specifications in PyMC are wrapped in a with-statement

    #priors
    #signal
    signal =  Gamma("Signal", mu=11000, sigma = 5000 )
    #background in not-ibd time window
    mean_cut_noise = Gamma("Mean_cut",  mu = 5e6, sigma = 5e6 )
    sigma_cut_noise = Gamma("Sigma_cut",  mu = 1e6, sigma = 5e6 )

    # linear relation with ibd windows
    noise_a = Gamma("Noise_a", mu=0.03, sigma=0.002)
    noise_b = Normal("Noise_b", mu=1e5, sigma=1e5)
    noise_c = Gamma("Noise_c", mu = 10000, sigma = 10000)

    
    # hierarchycal variables
    taxa_cut_on  = Normal("taxas_cut_on", mu=mean_cut_noise, sigma=sigma_cut_noise, shape = dados_cut_on.size)
    taxa_cut_off = Normal("taxas_cut_off", mu=mean_cut_noise, sigma=sigma_cut_noise, shape = dados_cut_off.size )


    residuals_noise_on = Normal("residuals_noise_on", mu = 0, sigma = noise_c, shape=dados_cut_on.size)
    residuals_noise_off = Normal("residuals_noise_off", mu = 0, sigma = noise_c, shape=dados_cut_off.size)

    taxa_ibd_on = noise_a * taxa_cut_on + noise_b + residuals_noise_on
    taxa_ibd_off = noise_a * taxa_cut_off + noise_b + residuals_noise_off


    likelihood_cut_on  = Normal("Dados_cut_on", mu=taxa_cut_on, sigma=np.sqrt(np.abs(taxa_cut_on)), observed=dados_cut_on)
    likelihood_cut_off = Normal("Dados_cut_off", mu=taxa_cut_off, sigma=np.sqrt(np.abs(taxa_cut_off)), observed=dados_cut_off)


    likelihood_ibd_on  = Normal("Dados_ibd_on", mu=taxa_ibd_on+signal, sigma=np.sqrt(np.abs(taxa_ibd_on)), observed=dados_ibd_on)
    likelihood_ibd_off = Normal("Dados_ibd_off", mu=taxa_ibd_off, sigma=np.sqrt(np.abs(taxa_ibd_off)), observed=dados_ibd_off)

    if run : 
        idata = sample(samples, tune = tune, chains = chains, progressbar="combined+stats", cores = 1)
        pm.compute_log_likelihood(idata)
        pm.sample_posterior_predictive(idata, model=model, extend_inferencedata=True)
        idata.to_netcdf(file_name)
        #idata = az.from_netcdf("NA_DS20_N12_20250214_15.nc")
        #chains_to_keep = [0,1,2,4,6,7]
        #idata_subset = idata.isel(chain=chains_to_keep)
        #idata_subset.to_netcdf("NA_DS20_N12_20250214_15_good.nc")
    else:  
        idata =  az.from_netcdf(file_name)
        #idata = temp_idata.sel(chain=[0, 1, 5, 6]) 
# Sampling posterior predictive

#graph = pm.model_to_graphviz(model)
#graph.render("graphname", format="png")

### First Summaries
with open(file_text, 'w') as f:
    print('NeutrinosAngra Analysis.', file=f)
    print(az.summary(idata,round_to = None, fmt='wide').to_string(), file=f)
    print("rhat values:", file=f)
    print(az.rhat(idata), file=f)
    print(az.rhat(idata).to_dataframe().to_string(), file=f)
    #print(az.rhat(idata)['taxas_cut_off'].to_numpy(), file=f)

### Now trace plots

fig, axes = plt.subplots(10,2, figsize=(6,15))
az.plot_trace(idata,var_names=None, filter_vars="like", axes=axes)
fig.tight_layout() 
fig.canvas.draw() 
az.plot_trace(idata, var_names=["Signal"],filter_vars="like")
az.plot_trace(idata, var_names=["Mean","Sigma"],filter_vars="like")
az.plot_trace(idata, var_names=["taxas"],filter_vars="like")
az.plot_trace(idata, var_names=["residuals"],filter_vars="like")

fig, axes = plt.subplots(3,2, figsize=(12,6))
az.plot_trace(idata, var_names=["Noise"],filter_vars="like",axes=axes)
axes[0,0].set_xlabel('test')
axes[1,0].set_xlabel('test')
axes[2,0].set_xlabel('test')
fig.tight_layout() 



estimates_ibd_off = np.zeros(dados_ibd_off.size)
estimates_ibd_on  = np.zeros(dados_ibd_on.size)
estimates_ibd_on_bck  = np.zeros(dados_ibd_on.size)
estimates_cut_off = np.zeros(dados_cut_off.size)
estimates_cut_on  = np.zeros(dados_cut_on.size)

errors_ibd_off = np.zeros(dados_ibd_off.size)
errors_ibd_on  = np.zeros(dados_ibd_on.size)
errors_ibd_on_bck  = np.zeros(dados_ibd_on.size)
errors_cut_off = np.zeros(dados_cut_off.size)
errors_cut_on  = np.zeros(dados_cut_on.size)

"""
for i in range(dados_cut_off.size):
    ob_val = idata.observed_data['Dados_cut_off'].sel(Dados_cut_off_dim_0=i).values
    ppc_values = idata.posterior['taxas_cut_off'].sel(taxas_cut_off_dim_0=i).values
    estimates_cut_off[i]=np.mean(ppc_values)
    errors_cut_off[i]=np.std(ppc_values)
    fig, ax = plt.subplots(tight_layout=True)
    #sns.displot(np.reshape(ppc_values , (ppc_values.size)), x="ppc", kind="kde",ax=ax)
    sns.kdeplot(np.reshape(ppc_values , (ppc_values.size)), ax=ax)
    #hist = ax.hist( np.reshape(ppc_values , (ppc_values.size)) , bins = 1000)
    plt.axvline(x=ob_val, color='red', linestyle='--')
    plt.show()

for i in range(dados_cut_on.size):
    ob_val = idata.observed_data['Dados_cut_on'].sel(Dados_cut_on_dim_0=i).values
    ppc_values = idata.posterior['taxas_cut_on'].sel(taxas_cut_on_dim_0=i).values
    estimates_cut_on[i]=np.mean(ppc_values)
    errors_cut_on[i]=np.std(ppc_values)
    fig, ax = plt.subplots(tight_layout=True)
    hist = ax.hist( np.reshape(ppc_values , (ppc_values.size)) , bins = 1000)
    plt.axvline(x=ob_val, color='red', linestyle='--')
    plt.show()

for i in range(dados_ibd_off.size):
    ob_val = idata.observed_data['Dados_ibd_off'].sel(Dados_ibd_off_dim_0=i).values
    ppc_noise_a = idata.posterior['Noise_a'].values
    ppc_noise_b = idata.posterior['Noise_b'].values
    ppc_residuals = idata.posterior['residuals_noise_off'].sel(residuals_noise_off_dim_0=i).values
    ppc_values = np.multiply(ppc_noise_a,(idata.posterior['taxas_cut_off'].sel(taxas_cut_off_dim_0=i).values))+ppc_noise_b+ppc_residuals
    estimates_ibd_off[i]=np.mean(ppc_values)
    errors_ibd_off[i]=np.std(ppc_values)
    fig, ax = plt.subplots(tight_layout=True)
    hist = ax.hist( np.reshape(ppc_values , (ppc_values.size)) , bins = 1000)
    plt.axvline(x=ob_val, color='red', linestyle='--')
    plt.show()
    
for i in range(dados_ibd_on.size):
    ob_val = idata.observed_data['Dados_ibd_on'].sel(Dados_ibd_on_dim_0=i).values
    signal = idata.posterior['Signal'].values
    ppc_noise_a = idata.posterior['Noise_a'].values
    ppc_noise_b = idata.posterior['Noise_b'].values
    ppc_residuals = idata.posterior['residuals_noise_on'].sel(residuals_noise_on_dim_0=i).values
    ppc_values = np.multiply(ppc_noise_a,(idata.posterior['taxas_cut_on'].sel(taxas_cut_on_dim_0=i).values))+ppc_noise_b+ppc_residuals+signal
    ppc_values_bck = np.multiply(ppc_noise_a,(idata.posterior['taxas_cut_on'].sel(taxas_cut_on_dim_0=i).values))+ppc_noise_b+ppc_residuals
    estimates_ibd_on[i]=np.mean(ppc_values)
    errors_ibd_on[i]=np.std(ppc_values)
    estimates_ibd_on_bck[i] = np.mean(ppc_values_bck)
    errors_ibd_on_bck[i] = np.std(ppc_values_bck)
    fig, ax = plt.subplots(tight_layout=True)
    hist = ax.hist( np.reshape(ppc_values , (ppc_values.size)) , bins = 1000)
    plt.axvline(x=ob_val, color='red', linestyle='--')
    plt.show()
"""
        
print("estimates_ibd_off ",estimates_ibd_off )
print("estimates_ibd_on ",estimates_ibd_on  )
print("estimates_cut_off ",estimates_cut_off )
print("estimates_cut_on ",estimates_cut_on  )

print("errors_ibd_off ",errors_ibd_off )
print("errors_ibd_on ",errors_ibd_on  )
print("errors_cut_off ",errors_cut_off )
print("errors_cut_on ",errors_cut_on  )




# Converter datas para objetos datetime
datas_inicio = pd.to_datetime([intervalo[0] for intervalo in datas_intervalos], dayfirst=True)
datas_fim = pd.to_datetime([intervalo[1] for intervalo in datas_intervalos], dayfirst=True)

# Calcular os valores médios dos intervalos de datas para plotar os pontos
datas_media_intervalo = (datas_inicio + (datas_fim - datas_inicio) / 2)
datas_analysis_off = [0] * indices_off.size
datas_analysis_on = [0] * indices_on.size
for i in range(indices_off.size):
    datas_analysis_off[i] = datas_media_intervalo[indices_off[i]]
for i in range(indices_on.size):
    datas_analysis_on[i] = datas_media_intervalo[indices_on[i]]


# Transformar em array de uma dimensão e depois em lista de strings
datas_flat = np.array(datas_intervalos).flatten().tolist()
datas_total = pd.to_datetime(datas_flat, dayfirst=True)

#Dados Selecionados a mão, retirando os vizinhos (para formar o eixo x)
seleção = ['15/06/2020', '22/06/2020', '29/06/2020', '06/07/2020', '13/07/2020', '20/07/2020', 
           '27/07/2020', '03/08/2020', '10/08/2020', '17/08/2020', '25/08/2020', '01/09/2020',
           '08/09/2020', '13/09/2020', '24/09/2020', '01/10/2020', '08/10/2020', '15/10/2020',
           '22/10/2020', '29/10/2020', '05/11/2020', '12/11/2020', '19/11/2020',
           '27/11/2020', '04/12/2020', '11/12/2020', '18/12/2020', '25/12/2020', '31/12/2020']

datas_selecionadas = pd.to_datetime(seleção, dayfirst=True)
 
# Limites do eixo y (regular caso queira gerar um gráfico só com um tipo de dado)
y_max = 500000
y_min = 0

#regular tranparencia da cor de fundo
alpha= 0.4

'''Gerar o gráfico'''
plt.figure(figsize=(12, 6))

#intervalos de faixas pintadas
primeiros_eventos = eventos[:9]  
plt.fill_between(datas_selecionadas[1:9], y_max, color='tomato', alpha=alpha, label = 'reactor off')
segundos_eventos = eventos[7:10]
plt.fill_between(datas_selecionadas[10:13], y_max, color='skyblue', alpha=alpha, label = 'reactor on')
terceiros_eventos = eventos[8:]
plt.fill_between(datas_selecionadas[14:-1], y_max, color='skyblue', alpha=alpha)

#Dados como scatter e linha dot conectando
print("Datas medias intervalo dim ", datas_media_intervalo.size , " valores ", datas_media_intervalo)
print("Eventos dim ", eventos.size , " valores ", eventos)

plt.scatter(datas_media_intervalo, eventos, color='black', label = 'Candidates IBD')
plt.plot(datas_media_intervalo, eventos, linestyle='--', color='black',alpha = 0.8)
plt.errorbar(datas_analysis_off,estimates_ibd_off,yerr=errors_ibd_off, color='b', label = 'Bayes Posterior Predictive Estimate')
plt.errorbar(datas_analysis_on,estimates_ibd_on,yerr=errors_ibd_on, color='b')
plt.errorbar(datas_analysis_on,estimates_ibd_on_bck ,yerr=errors_ibd_on_bck, color='r', label = 'Bayes Posterior Predictive Background')


#Ajustes do gráfico 
plt.xlabel('Dates')
plt.ylabel('Event Number')
plt.title('Neutrinos Angra - Low Multiplicity Analysis (N>=12)')
plt.xticks(ticks=datas_selecionadas, labels=[data.strftime('%d/%m/%Y') for data in datas_selecionadas], rotation=60)
plt.grid(axis='x')
plt.ylim(y_min,y_max)
plt.legend()
plt.show()

noise_a = idata.posterior['Noise_a'].values.flatten()[::10]
noise_b = idata.posterior['Noise_b'].values.flatten()[::10]
noise_c = idata.posterior['Noise_c'].values.flatten()[::10]
signal = idata.posterior['Signal'].values.flatten()[::10]
Mean_cut = idata.posterior['Mean_cut'].values.flatten()[::10]
Sigma_cut = idata.posterior['Sigma_cut'].values.flatten()[::10]

print(noise_a.shape)


posterior_dist = np.stack((signal,Mean_cut,Sigma_cut,noise_a,noise_b,noise_c))
pd_post = pd.DataFrame(posterior_dist.T , columns=['signal','Mean_cut','Sigma_cut','noise_a','noise_b','noise_c'])
sns.pairplot(pd_post, kind = 'kde', corner = True)
plt.show()
"""
with open(file_text, 'w') as f:
    print('Dados_cut_on',az.loo(idata,var_name='Dados_cut_on'),file=f)
    print('Dados_cut_off',az.loo(idata,var_name='Dados_cut_off'),file=f)
    print('Dados_ibd_on',az.loo(idata,var_name='Dados_ibd_on'),file=f)
    print('Dados_ibd_off',az.loo(idata,var_name='Dados_ibd_off'),file=f)
"""