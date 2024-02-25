import csv # Data format
import matplotlib.pyplot as plt # Graph plotting
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
### https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.inspection import permutation_importance # Feature importance

# Hyperparameters
"""
Sampling:
random_state (set randomness)
max_samples 

Feature Selection:
1. Assess individual class separability score of each metric
2. Then, perform greedy Sequential Forward Selection based on these results
3. Compare the performance with the more resource intensive BB algorithm
    - Use the score from (1) to order the nodes in the tree
"""
# -- Meta Knowledge
N_TRAIN_SAMPLES = 5000
N_FEATURES = 108

# -- Tree Model --
N_TREES = 100 # default: 100 (only 10 before Version 0.22)
MIN_SAMPLES_PER_LEAF = 1 # default: 1
MIN_WEIGHT_FRACTION_PER_LEAF = 0.0 # default: 0.0
MAX_DEPTH = None # default: None

# -- Splitting Constraints
MIN_SAMPLES_FOR_SPLIT = 2 # default: 2
MIN_IMPURITY_DECREASE_PER_SPLIT = 0.0 # default: 0.0

# -- Splitting Procedure
CRITERION = 'gini' # ["gini" (def), "entropy", "log_loss"]
MAX_FEATURES_PER_SPLIT = 1.0 # ['sqrt' (def), 'log2', int, float, None]

# -- Sampling
RANDOM_SEED = None # [int, None (def)]
MAX_SAMPLES = None # [int, None (def)]

# -- Training --

# -- Cross Validation --
DO_CROSS_VAL = False
CV_ORDER = 10

# -- Testing --

# -- Logging --
WRITE_RES_TO_FILE = True
PLOT_RES = True
PLOT_SEPARATE = True
SCATTER_POINT_SIZE = 5
LOG_FEATURES = True
LOG_FEATURES_CUTOFF = 10

LOG_HYPERPARAMS = {
    "TREES": N_TREES,
    "CRIT": CRITERION,
    "MAXFT": MAX_FEATURES_PER_SPLIT,
    "MAXSMP": MAX_SAMPLES,
    "DEP": MAX_DEPTH,
    "SMPSPLIT": MIN_SAMPLES_FOR_SPLIT,
    "IMPSPLIT": MIN_IMPURITY_DECREASE_PER_SPLIT,
    "SMPLF": MIN_SAMPLES_PER_LEAF
}

# Train the model with each combination of the experiment values: 

EXPERIMENT_VALUES = {
    # "MAXFT": [0.3],
    # "MAX_SAMPLES":[None, 0.3 * N_TRAIN_SAMPLES],
    # "SMPSPLIT": [None],
    # "IMPSPLIT": [None, 0.2],
    # "DEP": [5, 10, 20],
    # "SMPLF": [20]
}

# feature_importances_ is an array with shape (n_features,) whose values sum to 1.0. 
# The higher the value, the more important is the contribution of the matching feature to the prediction function.

def preprocess(data, labels, desc):
    # labels_out = [0 if row[0] == '0' else 1 for row in labels]
    labels_out = [[1,0] if row[0] == '0' else [0,1] for row in labels]
    labels_out = labels_out[1:]

    desc_out = [row for row in desc]
    data_out = [row for row in data]
    data_out = data_out[1:]

    n_data = len(labels_out)
    n_features = len(desc_out) - 1 

    feature_names = []

    # Feature scaling
    for i in range(n_features):
        feature_desc = desc_out[i + 1]
        feature_type = feature_desc[2]
        feature_names.append(feature_desc[0])
        if feature_type == "binary":
            for j in range(n_data):
                row = data_out[j]
                row[i] = float(row[i])
            continue
        
        # Get max and min feature value
        ymin = ymax = float(data_out[0][i])
        for j in range(n_data):
            row = data_out[j]
            row[i] = float(row[i])
            ymin = min(row[i], ymin)
            ymax = max(row[i], ymax)

        yspan = ymax - ymin
        
        # Rescale features to [0,1]
        for j in range(n_data):
            row = data_out[j]
            row[i] = (row[i] - ymin) / yspan

    return data_out, labels_out, feature_names

def evaluate_pred(predictions, labels):
    score = [[0,0],[0,0]] # Positives and negatives for each class
    for pred, label in zip(predictions, labels):
        class_idx = label[1]
        correct = (class_idx == 1 and pred >= 0.5) or (class_idx==0 and pred<0.5)
        score[class_idx][1 if correct else 0] += 1
    return score

def main():
    if not WRITE_RES_TO_FILE:
        print("No results are written to the textfiles. Set WRITE_RES_TO_FILE to True to enable this function.")

    with open('data1forEx1to4/train1_icu_data.csv') as fdata1, \
        open('data1forEx1to4/train1_icu_label.csv') as flabel1, \
        open('data1forEx1to4/feature_description.csv') as fdescr1, \
        open('data1forEx1to4/test1_icu_data.csv') as tdata1, \
        open('data1forEx1to4/test1_icu_label.csv') as tlabel1, \
        open('data1forEx1to4/feature_description.csv') as fdescr2, \
        open('results_train_test.txt', "a") as resFile, \
        open('results_crossval.txt', "a") as cvResFile, \
        open('results_feature_importance.txt', "a") as ftResFile:
        data1 = csv.reader(fdata1)
        label1 = csv.reader(flabel1)
        desc1 = csv.reader(fdescr1)
        
        tdata1 = csv.reader(tdata1)
        tlabel1 = csv.reader(tlabel1)
        desc2 = csv.reader(fdescr2)

        print("\r\n--- PREPROCESSING PHASE ---")
        data, labels, feature_names = preprocess(data1, label1, desc1)
        tdata, tlabels, _ = preprocess(tdata1, tlabel1, desc2)

        hyperparam_combinations = [{}]
        for param in EXPERIMENT_VALUES:
            temp = []
            for combi in hyperparam_combinations:
                for val in EXPERIMENT_VALUES[param]:
                    ext_combi = combi.copy()
                    ext_combi[param] = val
                    temp.append(ext_combi)
            hyperparam_combinations = temp

        for combi in hyperparam_combinations:
            print("\r\nSelected hyperparameters in this iteration:")
            for param_name in combi:
                LOG_HYPERPARAMS[param_name] = combi[param_name]
                print(f"{param_name}: {combi[param_name]}")
            
            print("\r\n--- TRAINING PHASE ---")
            model = RandomForestClassifier(
                n_estimators=N_TREES,
                criterion=CRITERION,
                max_depth = MAX_DEPTH,
                min_samples_split=MIN_SAMPLES_FOR_SPLIT,
                min_samples_leaf=MIN_SAMPLES_PER_LEAF,
                min_weight_fraction_leaf=MIN_WEIGHT_FRACTION_PER_LEAF,
                max_features=MAX_FEATURES_PER_SPLIT,
                min_impurity_decrease=MIN_IMPURITY_DECREASE_PER_SPLIT,
                random_state=RANDOM_SEED,
                max_samples=MAX_SAMPLES
            )

            model = model.fit(data, labels)

            if PLOT_RES:
                plotParams = ""
                for key in LOG_HYPERPARAMS:
                    plotParams += f"{key}{LOG_HYPERPARAMS[key]}_"
                predictions = model.predict_proba(data)
                predictions_survived = predictions[1]
                predictions_survived_1 = [predictions_survived[i][1] for i in range(predictions_survived.shape[0])]
                color_list = ['red' if label[0]==1 else 'blue' for label in labels]
                plt.scatter(range(len(predictions_survived_1)), predictions_survived_1, s=SCATTER_POINT_SIZE, c=color_list)
                plt.ylabel("Mean Vote Towards Survival")
                plt.xlabel("Training Sample Index")
                plt.title(f"Classification of ICU Patient Training Samples")
                plt.suptitle(f"Red: died, blue: survived")
                plt.draw()
                plt.savefig(fr'Latex Report\Figures\VOTING_TOTAL_{plotParams}.png')
                plt.clf()

            hyperParamHeader = ""
            for key in LOG_HYPERPARAMS:
                hyperParamHeader += f"{key}: {LOG_HYPERPARAMS[key]}, "
            hyperParamHeader += "\r\n"
            predictions = model.predict(data)
            predictions_survived = [predictions[i][1] for i in range(predictions.shape[0])]
            score = evaluate_pred(predictions_survived, labels)

            # Positives are Survivors
            resString = f"Training Results: TP: {score[1][1]}, FP: {score[0][0]}, FN: {score[1][0]}, TN: {score[0][1]}, "\
            f"Accuracy: {(score[1][1] + score[0][1]) / (score[1][0] + score[0][0] + score[1][1] + score[0][1]):.4f}"
            if WRITE_RES_TO_FILE:
                resFile.write(hyperParamHeader)
                resFile.write(resString)
                resFile.write("\r\n")
            print(resString)

            if LOG_FEATURES:
                
                # Permutation Importance
                result = permutation_importance(
                    model, tdata, tlabels, n_repeats=10, n_jobs=2
                )

                sorted_importances_idx = result.importances_mean.argsort()
                importances = pd.DataFrame(
                    result.importances[sorted_importances_idx].T,
                    columns=np.array(feature_names)[sorted_importances_idx],
                )
                ax = importances.plot.box(vert=False, whis=10)
                ax.set_title("Permutation Importances (test set)")
                ax.axvline(x=0, color="k", linestyle="--")
                ax.set_xlabel("Decrease in accuracy score")
                ax.figure.tight_layout()
                

                feature_importances = model.feature_importances_
                ftResFile.write(hyperParamHeader)
                for i in range(LOG_FEATURES_CUTOFF):
                    best_idx = feature_importances.argmax()
                    ftResFile.write(f"{i+1}. {feature_names[best_idx]}, {feature_importances[best_idx]}\r\n")
                    feature_importances[best_idx] = 0
                ftResFile.write("\r\n")
                

            print("\r\n--- CROSS VALIDATION PHASE ---")
            if not DO_CROSS_VAL:
                print(f"Cross Validation skipped. Set DO_CROSS_VAL to True to perform Cross Validation.\r\n")
            else:
                n_data = len(data)
                partition_size = n_data // CV_ORDER
                tr_score = []
                ts_score = []
                for i in range(CV_ORDER):
                    # Divide into training and test data
                    train_data = data[0 : i * partition_size] + data[(i+1) * partition_size : n_data]
                    test_data = data[i * partition_size: (i+1) * partition_size]
                    train_labels = labels[0 : i * partition_size] + labels[(i+1) * partition_size : n_data]
                    test_labels = labels[i * partition_size : (i+1) * partition_size]
                    
                    print(f"Iteration {i+1}...")        
                    model = RandomForestClassifier(
                                n_estimators=N_TREES,
                                criterion=CRITERION,
                                max_depth = MAX_DEPTH,
                                min_samples_split=MIN_SAMPLES_FOR_SPLIT,
                                min_samples_leaf=MIN_SAMPLES_PER_LEAF,
                                min_weight_fraction_leaf=MIN_WEIGHT_FRACTION_PER_LEAF,
                                max_features=MAX_FEATURES_PER_SPLIT,
                                min_impurity_decrease=MIN_IMPURITY_DECREASE_PER_SPLIT,
                                random_state=RANDOM_SEED,
                                max_samples=MAX_SAMPLES
                            )

                    model = model.fit(train_data, train_labels)
                    predictions = model.predict(train_data)
                    predictions_survived = [predictions[i][1] for i in range(predictions.shape[0])]
                    score = evaluate_pred(predictions_survived, train_labels)
                    accuracy = (score[1][1] + score[0][1]) / (score[1][0] + score[0][0] + score[1][1] + score[0][1])
                    tr_score.append(accuracy)

                    predictions = model.predict(test_data)
                    predictions_survived = [predictions[i][1] for i in range(predictions.shape[0])]
                    score = evaluate_pred(predictions_survived, test_labels)
                    accuracy = (score[1][1] + score[0][1]) / (score[1][0] + score[0][0] + score[1][1] + score[0][1])
                    ts_score.append(accuracy)
                
                if WRITE_RES_TO_FILE:
                    cvResFile.write(hyperParamHeader)
                    tr_mean = sum(tr_score)/float(len(tr_score))
                    ts_mean = sum(ts_score)/float(len(ts_score))
                    resString = f"Training Score: {tr_score}, Mean: {tr_mean:.4f}\r\n"\
                                f"Test Score: {ts_score}, Mean: {ts_mean:.4f}\r\n"
                    if WRITE_RES_TO_FILE:
                        cvResFile.write(resString)
                    print(resString)

            # Test
            print("\r\n--- TEST PHASE ---")
            predictions = model.predict(tdata)
            predictions_survived = [predictions[i][1] for i in range(predictions.shape[0])]
            score = evaluate_pred(predictions_survived, tlabels)
            resString = f"Testing Results: TP: {score[1][1]}, FP: {score[0][0]}, FN: {score[1][0]}, TN: {score[0][1]}, "\
            f"Accuracy: {(score[1][1] + score[0][1]) / (score[1][0] + score[0][0] + score[1][1] + score[0][1]):.4f}"
            if WRITE_RES_TO_FILE:
                resFile.write(resString)
                resFile.write("\r\n")
            print(resString)

            if PLOT_RES:
                plotParams = ""
                for key in LOG_HYPERPARAMS:
                    plotParams += f"{key}{LOG_HYPERPARAMS[key]}_"
                predictions = model.predict_proba(tdata)
                predictions_survived = predictions[1]
                predictions_survived_1 = [predictions_survived[i][1] for i in range(predictions_survived.shape[0])]
                color_list = ['red' if label[0]==1 else 'blue' for label in tlabels]
                plt.scatter(range(len(predictions_survived_1)), predictions_survived_1, s=SCATTER_POINT_SIZE, c=color_list)
                plt.ylabel("Mean Vote Towards Survival")
                plt.xlabel("Testing Sample Index")
                plt.title(f"Classification of ICU Patient Test Samples")
                plt.suptitle(f"Red: died, blue: survived")
                plt.draw()
                plt.savefig(fr'Latex Report\Figures\VOTING_TOTAL_TEST_{plotParams}.png')
                plt.clf()

if __name__=="__main__":
    main()