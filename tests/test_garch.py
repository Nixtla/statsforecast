import warnings
warnings.simplefilter('ignore')

import matplotlib.pyplot as plt
# GARCH(1,1) 
n = 1000 
w = 0.5 
alpha = np.array([0.3])
beta = np.array([0.4])

y = generate_garch_data(n, w, alpha, beta) 

plt.figure(figsize=(10,4))
plt.plot(y)
# Coefficients from arch library 
np.array([0.4307, 0.3050, 0.4243])
# GARCH(2,2) 
n = 1000 
w = 0.5
alpha = np.array([0.1, 0.2])
beta = np.array([0.3, 0.1])

y = generate_garch_data(n, w, alpha, beta) 

plt.figure(figsize=(10,4))
plt.plot(y)
x0 = np.array([0.5, 0.1, 0.2, 0.3, 0.1])
p = 2
q = 2 

sigma2 = garch_sigma2(x0, y, p, q)
garch_cons(x0)
garch_loglik(x0, y, p, q) 
mod = garch_model(y, p, q)
np.around(mod['coeff'], 5)
# Coefficients from arch library 
np.array([0.5300, 0.0920, 0.3039, 0.2856, 2.7330e-15])
h = 50
fcst = garch_forecast(mod, h)
fig, ax = plt.subplots(1, 1, figsize = (20,7))
plt.plot(np.arange(0, len(y)), y) 
plt.plot(np.arange(len(y), len(y) + h), fcst['mean'], label='point forecast')
plt.legend()
fig, ax = plt.subplots(1, 1, figsize = (20,7))
plt.plot(np.arange(0, len(y)), y) 
plt.plot(np.arange(0, len(y)), fcst['fitted'], label='fitted values') 
plt.legend()
# p = q 
mod = garch_model(y, 1, 1) 
print('StatsForecast\'s coefficients: ') 
print(np.around(mod['coeff'], 5))
print('')
print('arch\'s coefficients: ') 
print(np.array([0.3238, 0.1929, 0.6016]))
# p > q 
mod = garch_model(y, 2, 1) 
print('StatsForecast\'s coefficients: ') 
print(np.around(mod['coeff'], 5))
print('')
print('arch\'s coefficients: ') 
print(np.array([0.5299, 0.0920, 0.3039, 0.2846])) 
# p < q 
mod = garch_model(y, 1, 2) 
print('StatsForecast\'s coefficients: ') 
print(np.around(mod['coeff'], 5))
print('') 
print('arch\'s coefficients: ') 
print(np.array([0.3238, 0.1930, 0.6015, 9.2320e-13]))
# q = 0 
mod = garch_model(y, 1, 0) 
print('StatsForecast\'s coefficients: ') 
print(np.around(mod['coeff'], 5))
print('') 
print('arch\'s coefficients: ') 
print(np.array([1.3503, 0.1227]))
