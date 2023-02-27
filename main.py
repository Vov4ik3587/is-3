# %% Импортируем библиотеки
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import boston_housing
import matplotlib.pyplot as plt

# %% Загружаем датасет и выполняем его нормализацию
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print(train_data.shape)
print(test_data.shape)
print(test_targets)

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


# %% Определяем функцию построения модели
def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# %% Обучим модель без кросс-валидации. Далее построим графики функции ошибки и метрики
model = build_model()
H = model.fit(train_data, train_targets, epochs=20, batch_size=1, validation_data=(test_data, test_targets))
results = model.evaluate(test_data, test_targets, batch_size=1)
print(f'Without cross-validation MAE = {results[1]}')

loss = H.history['loss']
val_loss = H.history['val_loss']
mae = H.history['mae']
val_mae = H.history['val_mae']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training MSE')
plt.plot(epochs, val_loss, 'b', label='Validation MSE')
plt.title('Training and validation MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()

plt.clf()
plt.plot(epochs, mae, 'bo', label='Training MAE')
plt.plot(epochs, val_mae, 'b', label='Validation MAE')
plt.title('Training and validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

# %% Выполняем кросс-валидацию
k = 4
num_val_samples = len(train_data) // k
num_epochs = 20
all_scores = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
    axis=0)
    partial_train_targets = np.concatenate(
    [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1)
    val_mse, val_mae = model.evaluate(val_data, val_targets)
    all_scores.append(val_mae)
print(f'MAE = {all_scores}')
print(f'Mean MAE = {np.mean(all_scores)}')
