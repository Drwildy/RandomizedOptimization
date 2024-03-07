import optuna
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def objective(trial):
    activation = 'tanh'
    hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [(10,), (50,), (100,), (10,5), (50,25), (100,50)])
    # batch_size = trial.suggest_int('batch_size', 200, 1600)
    learning_rate_init = trial.suggest_categorical('learning_rate_init', [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3])


    model = MLPClassifier(activation=activation, hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate_init, max_iter=5000)

    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    scoreMean = score.mean()

    return scoreMean


dataset = pd.read_csv('insurance.csv')

bins = [0, 7500, 15000, float('inf')]
labels = ['low', 'medium', 'high']

dataset['chargesClass'] = pd.cut(dataset['charges'], bins=bins, labels=labels, right=False)
dataset = dataset.drop(['charges'], axis=1)
# print(dataset)

X = dataset.drop(['chargesClass'], axis=1)
y = dataset['chargesClass']

# print(y.value_counts())

Le = LabelEncoder()
for col in X.columns:
    # Check if the column data type is object (string)
    if X[col].dtype == 'object':
        # Use LabelEncoder to transform the string values to numerical values
        X[col] = Le.fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)

# Normalize feature data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One hot encode target values
one_hot = OneHotEncoder()
y_train_hot = one_hot.fit_transform(y_train.values.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.values.reshape(-1, 1)).todense()

# Create an MLPClassifier
model = MLPClassifier(activation='tanh', hidden_layer_sizes=(10,5), learning_rate_init=0.001, max_iter=5000, random_state=60)



# Train the model
start_time = time.time()
model.fit(X_train_scaled, np.asarray(y_train_hot))
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
# trainaccuracy = accuracy_score(y_train, y_predtrain)
accuracy = accuracy_score(np.asarray(y_test_hot), y_pred)
precision = precision_score(np.asarray(y_test_hot), y_pred, average='weighted')
recall = recall_score(np.asarray(y_test_hot), y_pred, average='weighted')
f1Score = f1_score(np.asarray(y_test_hot), y_pred, average='weighted')

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1Score:.4f}")

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('Learning Curve')
plt.legend()
# plt.show()


# Validation Curve
# param_range = [(100,), (50,50), (100,50,25), (100,75,50,25)]
# train_scores, test_scores = validation_curve(
#     model, X_train, y_train, param_name='hidden_layer_sizes', param_range=param_range, cv=5, scoring='accuracy')

# # Convert tuple values to string for plotting purposes
# param_range_str = [str(x) for x in param_range]

# plt.figure(figsize=(10, 6))
# plt.plot(param_range_str, np.mean(train_scores, axis=1), marker='o', label='Training Score')
# plt.plot(param_range_str, np.mean(test_scores, axis=1), marker='s', label='Cross-Validation Score')
# plt.xticks(param_range_str, param_range_str) # Set x-tick labels to clearly show different configurations
# plt.xlabel('Hidden Layer Sizes')
# plt.ylabel('Accuracy Score')
# plt.title('Validation Curve for MLPClassifier')
# plt.legend()

# param_range = np.arange(100, 2000, 100)
# train_scores, test_scores = validation_curve(
#     model, X_train, y_train, param_name='batch_size', param_range=param_range, cv=5, scoring='accuracy')

# plt.figure(figsize=(10, 6))
# plt.plot(param_range, np.mean(train_scores, axis=1), label='Training Score')
# plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-Validation Score')
# plt.xlabel('Batch Size')
# plt.ylabel('Accuracy Score')
# plt.title('Validation Curve')
# plt.legend()

# param_range = np.array([0.0001, 0.001, 0.01, 0.1])
# train_scores, test_scores = validation_curve(
#     model, X_train, y_train, param_name='learning_rate_init', param_range=param_range, cv=5, scoring='accuracy')

# plt.figure(figsize=(10, 6))
# plt.plot(param_range, np.mean(train_scores, axis=1), label='Training Score')
# plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-Validation Score')
# plt.xlabel('Learning Rate')
# plt.ylabel('Accuracy Score')
# plt.title('Validation Curve')
# plt.legend()

# plt.show()


# ----- Optuna-------------# #

# study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(42))

# study.optimize(objective, n_trials=100)

# print(study.best_params)
# best_params = study.best_params

# fig = optuna.visualization.plot_optimization_history(study)
# fig.show()

# fig2 = optuna.visualization.plot_slice(study, params=['hidden_layer_sizes', 'learning_rate_init'])
# fig2.show()

# fig3 = optuna.visualization.plot_param_importances(study)
# fig3.show()