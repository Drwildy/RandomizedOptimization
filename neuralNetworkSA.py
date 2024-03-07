import mlrose_hiive as mlrose
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
        
class SACustomNeuralNetwork(mlrose.NeuralNetwork):
    def __init__(self, hidden_nodes=None,
                 activation='tanh',
                 algorithm='simulated_annealing',
                 max_iters=10000,
                 bias=True,
                 is_classifier=True,
                 learning_rate=0.1,
                 early_stopping=False,
                 clip_max=1e+10,
                 restarts=0,
                 schedule=mlrose.GeomDecay(),
                 pop_size=200,
                 mutation_prob=0.1,
                 max_attempts=10,
                 random_state=None,
                 curve=False):
        self.classes_ = [0, 1]
        super().__init__(
            hidden_nodes=hidden_nodes,
            activation=activation,
            algorithm=algorithm,
            max_iters=max_iters,
            bias=bias,
            is_classifier=is_classifier,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            clip_max=clip_max,
            restarts=restarts,
            schedule=schedule,
            pop_size=pop_size,
            mutation_prob=mutation_prob,
            max_attempts=max_attempts,
            random_state=random_state,
            curve=curve)
#########################################################
        
def objective(trial):
    activation = 'tanh'
    hidden_nodes = trial.suggest_categorical('hidden_nodes', [(10,), (50,), (100,), (10,5), (50,25), (100,50)])
    learning_rate_init = trial.suggest_categorical('learning_rate_init', [0.5, 0.75, 1, 2, 4, 6, 8, 10])
    temperature = trial.suggest_int('temperature', 100, 1000)
    decay = trial.suggest_categorical('decay', [0.99, 0.9, 0.8, 0.7, 0.6, 0.5])
    max_iters = trial.suggest_int('max_iters', 500, 10000)


    model = SACustomNeuralNetwork(max_iters=max_iters,hidden_nodes=hidden_nodes, activation=activation, learning_rate=learning_rate_init, schedule=mlrose.GeomDecay(init_temp=temperature, decay=decay), random_state=60)

    score = cross_val_score(model, X_train_scaled, np.asarray(y_train_hot), cv=5, scoring='accuracy')
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

model = SACustomNeuralNetwork(hidden_nodes=(10,), learning_rate=0.5, schedule=mlrose.GeomDecay(init_temp=911, decay=.7), max_iters=6500, max_attempts=10, random_state=60)
model.get_params()   

start_time = time.time()
model.fit(X_train_scaled, np.asarray(y_train_hot))
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

# Make predictions on the test set
# y_predtrain = model.predict(X_train)
y_pred = model.predict(X_test_scaled)

# Evaluate the model
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
    model, X_train_scaled, np.asarray(y_train_hot), cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('Learning Curve')
plt.legend()
plt.show()

# Validation Curve
# param_range = [(100,), (50,50), (100,50,25), (100,75,50,25)]
# train_scores, test_scores = validation_curve(
#     model, X_train_scaled, np.asarray(y_train_hot), param_name='hidden_nodes', param_range=param_range, cv=5, scoring='accuracy')

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

# plt.show()

# ----- Optuna-------------# #

# study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(42))

# study.optimize(objective, n_trials=100)

# print(study.best_params)
# best_params = study.best_params

# fig = optuna.visualization.plot_optimization_history(study)
# fig.show()

# fig2 = optuna.visualization.plot_slice(study, params=['hidden_nodes', 'learning_rate_init', 'temperature', 'decay', 'max_iters'])
# fig2.show()

# fig3 = optuna.visualization.plot_param_importances(study)
# fig3.show()