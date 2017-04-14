"""
Ensemble model
"""
import pyflux as pf
from infodenguepredict.models import GAS, arima, sarimax
from infodenguepredict.data.infodengue import get_alerta_table

data = get_alerta_table(3304557)

model1 = arima.build_model(data, 2, 2, 1, 'casos')
model2 = GAS.build_model(data, ar=2, sc=6, target='casos')
model3 = sarimax.build_model(data, 'casos', [])

mix = pf.Aggregate(learning_rate=1.0, loss_type='squared')
mix.add_model(model1)
mix.add_model(model2)
# mix.add_model(model3)

mix.tune_learning_rate(52)
print(mix.learning_rate)
mix.plot_weights(h=52, figsize=(15, 5))

print(mix.summary(h=52))

# Previsoes

print(mix.predict_is(h=6))

print(mix.predict(h=6))
