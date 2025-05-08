"""
This script produces the plots of the low multiplicity analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

import NA_DS20_Data

show_ts_ibd = False #time serie of IBD candidates
show_ts_acc = False #time serie of acc candidates
show_scatt_ibd_acc = True # scatter plot between IBD and Acc for reactor off 


ibd_off = np.zeros(NA_DS20_Data.indices_off.size)
ibd_on = np.zeros(NA_DS20_Data.indices_on.size)
acc_off = np.zeros(NA_DS20_Data.indices_off.size)
acc_on = np.zeros(NA_DS20_Data.indices_on.size)

for i in range(NA_DS20_Data.indices_off.size):
    acc_off[i] = NA_DS20_Data.acc_lowMult[NA_DS20_Data.indices_off[i]] 
    ibd_off[i] = NA_DS20_Data.ibd_lowMult[NA_DS20_Data.indices_off[i]] 

for i in range(NA_DS20_Data.indices_on.size):
    acc_on[i] = NA_DS20_Data.acc_lowMult[NA_DS20_Data.indices_on[i]] 
    ibd_on[i] = NA_DS20_Data.ibd_lowMult[NA_DS20_Data.indices_on[i]] 

# Limites do eixo y (regular caso queira gerar um gráfico só com um tipo de dado)
y_max_ibd = 500000
y_min_ibd = 0

y_max_acc = 10000000
y_min_acc = 0

#regular tranparencia da cor de fundo
alpha= 0.4

if show_ts_ibd:
    # IBD canddidates plot
    plt.figure(figsize=(12, 6))

    plt.fill_between(NA_DS20_Data.datas_selecionadas[1:9], y_max_ibd, color='tomato', alpha=alpha, label = 'reactor off')
    plt.fill_between(NA_DS20_Data.datas_selecionadas[0:2], y_max_ibd, color='yellow', alpha=alpha, label = 'reactor ramping')
    plt.fill_between(NA_DS20_Data.datas_selecionadas[8:11], y_max_ibd, color='yellow', alpha=alpha)
    plt.fill_between(NA_DS20_Data.datas_selecionadas[10:13], y_max_ibd, color='skyblue', alpha=alpha, label = 'reactor on')
    plt.fill_between(NA_DS20_Data.datas_selecionadas[12:14], y_max_ibd, color='green', alpha=alpha, label = 'incomplete week')
    plt.fill_between(NA_DS20_Data.datas_selecionadas[14:-1], y_max_ibd, color='skyblue', alpha=alpha)
    plt.fill_between(NA_DS20_Data.datas_selecionadas[-2:], y_max_ibd, color='green', alpha=alpha)

    plt.scatter(NA_DS20_Data.datas_media_intervalo, NA_DS20_Data.ibd_lowMult, color='black', label = 'Candidates IBD')
    plt.plot(NA_DS20_Data.datas_media_intervalo, NA_DS20_Data.ibd_lowMult, linestyle='--', color='black',alpha = 0.8)

    plt.xlabel('Dates')
    plt.ylabel('Event Number')
    plt.title('Neutrinos Angra - 2020 Dataset - IBD Candidates - Low Multiplicity Analysis')
    plt.xticks(ticks=NA_DS20_Data.datas_selecionadas, labels=[data.strftime('%d/%m/%Y') for data in NA_DS20_Data.datas_selecionadas], rotation=60)
    plt.grid(axis='x')
    plt.ylim(y_min_ibd,y_max_ibd)
    plt.legend()
    plt.tight_layout()
    plt.show()


if show_ts_acc:
    # Acc canddidates plot
    plt.figure(figsize=(12, 6))

    plt.fill_between(NA_DS20_Data.datas_selecionadas[1:9], y_max_acc, color='tomato', alpha=alpha, label = 'reactor off')
    plt.fill_between(NA_DS20_Data.datas_selecionadas[0:2], y_max_acc, color='yellow', alpha=alpha, label = 'reactor ramping')
    plt.fill_between(NA_DS20_Data.datas_selecionadas[8:11], y_max_acc, color='yellow', alpha=alpha)
    plt.fill_between(NA_DS20_Data.datas_selecionadas[10:13], y_max_acc, color='skyblue', alpha=alpha, label = 'reactor on')
    plt.fill_between(NA_DS20_Data.datas_selecionadas[12:14], y_max_acc, color='green', alpha=alpha, label = 'incomplete week')
    plt.fill_between(NA_DS20_Data.datas_selecionadas[14:-1], y_max_acc, color='skyblue', alpha=alpha)
    plt.fill_between(NA_DS20_Data.datas_selecionadas[-2:], y_max_acc, color='green', alpha=alpha)

    plt.scatter(NA_DS20_Data.datas_media_intervalo, NA_DS20_Data.acc_lowMult, color='black', label = 'Accidentals')
    plt.plot(NA_DS20_Data.datas_media_intervalo, NA_DS20_Data.acc_lowMult, linestyle='--', color='black',alpha = 0.8)

    plt.xlabel('Dates')
    plt.ylabel('Event Number')
    plt.title('Neutrinos Angra - 2020 Dataset - Accidentals - Low Multiplicity Analysis')
    plt.xticks(ticks=NA_DS20_Data.datas_selecionadas, labels=[data.strftime('%d/%m/%Y') for data in NA_DS20_Data.datas_selecionadas], rotation=60)
    plt.grid(axis='x')
    plt.ylim(y_min_acc,y_max_acc)
    plt.legend()
    plt.tight_layout()
    plt.show()

if show_scatt_ibd_acc:
    x = np.array(np.take(NA_DS20_Data.acc_lowMult,NA_DS20_Data.indices_off))
    y = np.array(np.take(NA_DS20_Data.ibd_lowMult,NA_DS20_Data.indices_off))
    x_err = np.sqrt(x)
    y_err = np.sqrt(y)
    print(x)
    print(y)


    # Define the linear function for curve_fit
    def linear(x, a, b):
        return a * x + b

    # Fit using curve_fit (with errors only in y)
    popt, pcov = op.curve_fit(linear, x, y, sigma=y_err, absolute_sigma=True)

    # Extract fit results and uncertainties
    slope, intercept = popt
    slope_err, intercept_err = np.sqrt(np.diag(pcov))

    print(f"Slope:     {slope:.4f} ± {slope_err:.4f}")
    print(f"Intercept: {intercept:.4f} ± {intercept_err:.4f}")

    # Plotting
    x_fit = np.linspace(min(x) - 1, max(x) + 1, 100)
    y_fit = linear(x_fit,slope,intercept)

    plt.figure(figsize=(6, 6))
    plt.scatter( np.take(NA_DS20_Data.acc_lowMult,NA_DS20_Data.indices_off), np.take(NA_DS20_Data.ibd_lowMult,NA_DS20_Data.indices_off), color='black', label = 'Reactor off')
    plt.plot(x_fit, y_fit, 'r-', label='linear LS fit')
    plt.xlabel('Accidentals')
    plt.ylabel('IBD')
    plt.title('Neutrinos Angra - 2020 Dataset - IBD Vs Accidentals - Low Multiplicity Analysis')
    plt.grid(axis='x')
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()
    plt.show()