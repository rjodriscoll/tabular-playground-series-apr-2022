import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from utility_funcs import get_train_labels_test, split_train_data, scale_and_as_array

import random 

random.seed(42)


train, labels, test = get_train_labels_test(is_py=True)

features = [f for f in train.columns if "sensor" in f]

groups = train["sequence"]
train = train.drop(["sequence", "subject", "step"], axis=1).values
test = test.drop(["sequence", "subject", "step"], axis=1).values
labels = labels["state"]

scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)


train = train.reshape(-1, 60, 13)
test = test.reshape(-1, 60, 13)

assert train.shape[0] == labels.shape[0]



# helpers 
def train_model(model_in, test_pred_mode = False, n_folds=5):

    gkf = GroupKFold(n_folds)
    store = []

    model_in.summary()

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(train, labels, groups.unique())
    ):
        print(f"Fitting fold {fold} for {model_in.name}...")
        model = keras.models.clone_model(model_in)
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=[keras.metrics.AUC()]
        )

        X_train, X_val = train[train_idx], train[val_idx]
        y_train, y_val = labels.iloc[train_idx], labels.iloc[val_idx]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            verbose=0,
            batch_size=128,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=10, monitor="val_loss", restore_best_weights=True
                )
            ],
        )
        auc = roc_auc_score(y_val, model.predict(X_val).squeeze())
        print(f"The val auc for fold {fold}, {model_in.name} is {auc}")

        if not test_pred_mode:
            plot_model(history, model, fold)

        if test_pred_mode:
            store.append(model.predict(test).squeeze())
        else:
            store.append(auc)
            
    result = sum(store) / n_folds # if test mode we want the prediction
    return result


def plot_model(history, model, fold):

    l_name = list(history.history.keys())[0]
    vl_name = list(history.history.keys())[2]
    a_name = list(history.history.keys())[1]
    al_name = list(history.history.keys())[3]

    loss, val_loss = history.history[l_name], history.history[vl_name]
    auc, val_auc = history.history[a_name], history.history[al_name]
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(15, 8))
    plt.plot(
        epochs,
        loss,
        color="tab:blue",
        linestyle="-",
        linewidth=2,
        marker="*",
        label="Training loss",
    )
    plt.plot(
        epochs,
        val_loss,
        color="tab:orange",
        linestyle="-",
        marker="o",
        label="Validation loss",
    )
    plt.title(f"Training and validation loss, {model.name}, fold {fold}", fontsize=16)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(loc="upper right", fontsize="x-large")
    plt.tick_params(labelsize=14)
    plt.savefig(f'Loss_plots/loss_{model.name}_fold{fold}', bbox_inches='tight')
    #plt.show()
    plt.clf()

    plt.figure(figsize=(15, 8))
    plt.plot(
        epochs,
        auc,
        color="tab:blue",
        linestyle="-",
        linewidth=2,
        marker="*",
        label="Training auc",
    )
    plt.plot(
        epochs,
        val_auc,
        color="tab:orange",
        linestyle="-",
        marker="o",
        label="Validation auc",
    )
    plt.title(f"Training and validation auc, {model.name}, fold {fold}", fontsize=16)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Auc", fontsize=16)
    plt.legend(loc="upper left", fontsize="x-large")
    plt.tick_params(labelsize=14)
    plt.savefig(f'Loss_plots/auc_{model.name}_fold_{fold}', bbox_inches='tight')
    #plt.show()
    plt.clf()

# models 
model_1 = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=(60, train.shape[2])),
        keras.layers.Dense(50, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ],
    name="Dense_model_1",
)


model_2 = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=(60, train.shape[2])),
        keras.layers.Dense(200, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(50, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ],
    name="Dense_model_2",
)

model_3 = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=(60, train.shape[2])),
        keras.layers.Dense(400, activation="swish"),
        keras.layers.Dense(200, activation="swish"),
        keras.layers.Dense(100, activation="swish"),
        keras.layers.Dense(50, activation="swish"),
        keras.layers.Dense(1, activation="sigmoid"),
    ],
    name="Dense_model_3",
)


model_4 = keras.models.Sequential([
    keras.layers.Input(shape=(60, train.shape[2])),
    keras.layers.LSTM(128, return_sequences=True), 
    keras.layers.Flatten(),
    keras.layers.Dense(50, activation="swish"),
    keras.layers.Dense(1, activation="sigmoid")
], name = 'RNN_model_1')



model_5 = keras.models.Sequential([
    keras.layers.Input(shape=(60, train.shape[2])),
    keras.layers.LSTM(256, return_sequences=True), 
    keras.layers.LSTM(128, return_sequences=True), 
    keras.layers.Flatten(),
    keras.layers.Dense(150, activation="swish"),
    keras.layers.Dense(50, activation="swish"),
    keras.layers.Dense(1, activation="sigmoid")
], name = 'RNN_model_2')

model_6 = keras.models.Sequential([
    keras.layers.Input(shape=(60, train.shape[2])),
    keras.layers.LSTM(512, return_sequences=True), 
    keras.layers.LSTM(256, return_sequences=True), 
    keras.layers.LSTM(128, return_sequences=True), 
    keras.layers.Flatten(),
    keras.layers.Dense(150, activation="swish"),
    keras.layers.Dense(50, activation="swish"),
    keras.layers.Dense(1, activation="sigmoid")
], name = 'RNN_model_3')


model_7 = keras.models.Sequential([
    keras.layers.Input(shape=(60, train.shape[2])),
    keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)), 
    keras.layers.Flatten(),
    keras.layers.Dense(50, activation="swish"),
    keras.layers.Dense(1, activation="sigmoid")
], name = 'RNN_model_4')

model_8 = keras.models.Sequential([
    keras.layers.Input(shape=(60, train.shape[2])),
    keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True)), 
    keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)), 
    keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)), 
    keras.layers.Flatten(),
    keras.layers.Dense(150, activation="swish"),
    keras.layers.Dense(50, activation="swish"),
    keras.layers.Dense(1, activation="sigmoid")
], name = 'RNN_model_5')

model_9 = keras.models.Sequential([
    keras.layers.Input(shape=(60, train.shape[2])),
    keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True)), 
    keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)), 
    keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)), 
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.GRU(32, return_sequences=True)),
    keras.layers.Flatten(),
    keras.layers.Dense(200, activation="swish"),
    keras.layers.Dense(150, activation="swish"),
    keras.layers.Dense(50, activation="swish"),
    keras.layers.Dense(1, activation="sigmoid"),
], name = 'RNN_model_6')


models = [model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9] 
model_dict = {}
for i, model_in in enumerate(models):
    val_auc = train_model(model_in)
    model_dict[model_in.name] = val_auc



names, counts = zip(*model_dict.items())
plt.bar(names, counts)
plt.title(f"AUCs for each model", fontsize=16)
plt.xlabel("Model", fontsize=16)
plt.ylabel("AUC", fontsize=16)
plt.tick_params(labelsize=10)
plt.xticks(rotation=30)
plt.savefig('Loss_plots/auc_averages', bbox_inches='tight')