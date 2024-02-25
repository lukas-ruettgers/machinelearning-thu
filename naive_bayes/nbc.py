import csv # Data format
import matplotlib.pyplot as plt # Graph plotting
import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB

# HYPERPARAMETERS

# -- Meta Knowledge
N_TRAIN_SAMPLES = 5000
N_FEATURES = 108

# -- Model --
PRIORS = [0.5,0.5]
FIT_PRIOR = True
NUM_MODEL = "numeric"
BIN_MODEL = "binary"
BIN_MODEL_BERNOULLI = False
COMP_MODEL = "composite"
MODELTYPES_LAYER1 = [NUM_MODEL, BIN_MODEL]
MODELTYPES_ALL = [NUM_MODEL, BIN_MODEL, COMP_MODEL]


# -- Cross Validation --
DO_CROSS_VAL = False
CV_ORDER = 10

# -- Testing --
RISK = [
    # Columns: true class, Lines: decision (0: died, 1: survived)
    [0.05,10],
    [1,0]
]
TEST_RISK_DECISION = True

# -- Logging --
WRITE_RES_TO_FILE = False
PLOT_RES = False
PLOT_SEPARATE = True
SCATTER_POINT_SIZE = 5

# Use these hyperparams to uniquely identify each logged results
LOG_HYPERPARAMS = {
    "UseBernoulli": BIN_MODEL_BERNOULLI
}

# Train the model with each combination of the following experiment values: 
EXPERIMENT_VALUES = {
    "UseBernoulli": [True, False]
}

def preprocess(data, labels, desc):
    labels_out = [0 if row[0] == '0' else 1 for row in labels]
    labels_out = labels_out[1:]

    desc_out = [row for row in desc]
    data_out = [row for row in data]
    data_out = data_out[1:]

    n_data = len(labels_out)
    n_features = len(desc_out) - 1 

    feature_names = []
    
    # 1 = binary, 0 = numeric
    feature_is_binary = [0] * n_features
    
    # Feature scaling
    for i in range(n_features):
        feature_desc = desc_out[i + 1]
        feature_type = feature_desc[2]
        feature_names.append(feature_desc[0])
        if feature_type == BIN_MODEL:
            feature_is_binary[i] = 1
        
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

    # Split features into binary and numeric features
    data_out_numeric = []
    data_out_binary = []
    for i in range(n_data):
        feature_vec_num = []
        feature_vec_bin = []
        row = data_out[i]
        for j in range(n_features):
            if feature_is_binary[j]:
                feature_vec_bin.append(row[j])
            else:
                feature_vec_num.append(row[j])
        data_out_binary.append(feature_vec_bin)
        data_out_numeric.append(feature_vec_num)

    data_out = {
        BIN_MODEL: data_out_binary,
        NUM_MODEL: data_out_numeric
    }

    return data_out, labels_out, feature_names

def evaluate_pred(predictions, labels):
    score = [[0,0],[0,0]] # Positives and negatives for each class
    for pred, label in zip(predictions, labels):
        correct = (label == 1 and pred >= 0.5) or (label==0 and pred<0.5)
        score[label][1 if correct else 0] += 1
    return score

def evaluate_pred_risk(predictions, labels):
    score = [[0,0],[0,0]] # Positives and negatives for each class
    risk_threshold = (RISK[0][1]-RISK[1][1])/(RISK[1][0]-RISK[0][0])
    for pred, label in zip(predictions, labels):
        # Prevent numerical instability (zero division)
        pred[1] = max(pred[1], 1e-8)
        
        # Posterior odds
        odds = pred[0] / pred[1]
        class_idx = 0 if odds > risk_threshold else 1
        score[label][class_idx == label] += 1
    return score

def main():
    if not WRITE_RES_TO_FILE:
        print("No results are written to the textfiles. Set WRITE_RES_TO_FILE to True to enable this function.")

    if not PLOT_RES:
        print("No figures will be plotted. Set PLOT_RES to True to enable this function.")

    with open('data1forEx1to4/train1_icu_data.csv') as fdata1, \
        open('data1forEx1to4/train1_icu_label.csv') as flabel1, \
        open('data1forEx1to4/feature_description.csv') as fdescr1, \
        open('data1forEx1to4/test1_icu_data.csv') as tdata1, \
        open('data1forEx1to4/test1_icu_label.csv') as tlabel1, \
        open('data1forEx1to4/feature_description.csv') as fdescr2, \
        open('results_train_test.txt', "a") as resFile, \
        open('results_crossval.txt', "a") as cvResFile:
        data1 = csv.reader(fdata1)
        label1 = csv.reader(flabel1)
        desc1 = csv.reader(fdescr1)
        
        tdata1 = csv.reader(tdata1)
        tlabel1 = csv.reader(tlabel1)
        desc2 = csv.reader(fdescr2)

        print("\r\n--- PREPROCESSING PHASE ---")
        data, labels, feature_names = preprocess(data1, label1, desc1)

        tdata, tlabels, _ = preprocess(tdata1, tlabel1, desc2)
        
        # Run for each combination of hyperparameters
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
            

            models = {
                NUM_MODEL: GaussianNB(
                    priors=PRIORS
                ),
                BIN_MODEL: BernoulliNB(
                    class_prior=PRIORS,
                    fit_prior=FIT_PRIOR
                ) if LOG_HYPERPARAMS["UseBernoulli"] 
                else MultinomialNB(
                    class_prior=PRIORS,
                    fit_prior=FIT_PRIOR
                ),
                COMP_MODEL: GaussianNB(
                    priors=PRIORS
                )
            }
            
            preds_layer1 = {}
            for model_type in MODELTYPES_LAYER1:
                models[model_type] = models[model_type].fit(data[model_type], labels)
                preds_layer1[model_type] = models[model_type].predict_proba(data[model_type])
            

            data[COMP_MODEL] = np.array(
                [[preds_layer1[NUM_MODEL][i][1],preds_layer1[BIN_MODEL][i][1]] for i in range(preds_layer1[NUM_MODEL].shape[0])]
                )
            models[COMP_MODEL] = models[COMP_MODEL].fit(data[COMP_MODEL], labels)
            preds_layer1[COMP_MODEL] = models[COMP_MODEL].predict_proba(data[COMP_MODEL])

            for model_type in MODELTYPES_ALL:
                if PLOT_RES:
                    plotParams = f"MODEL{model_type}_"
                    for key in LOG_HYPERPARAMS:
                        plotParams += f"{key}{LOG_HYPERPARAMS[key]}_"
                    predictions = preds_layer1[model_type]
                    predictions_survived = [predictions[i][1] for i in range(predictions.shape[0])]
                    color_list = ['red' if label==0 else 'blue' for label in labels]
                    plt.scatter(range(len(predictions_survived)), predictions_survived, s=SCATTER_POINT_SIZE, c=color_list)
                    plt.ylabel("Probability of Survival")
                    plt.xlabel("Training Sample Index")
                    plt.title(f"Survival Prediction of ICU Patient Training Samples")
                    plt.suptitle(f"Red: died, blue: survived")
                    plt.draw()
                    plt.savefig(fr'Latex Report\Figures\TRAIN_PRED_{plotParams}.png')
                    plt.clf()

                hyperParamHeader = f"\r\n{model_type}_"
                for key in LOG_HYPERPARAMS:
                    hyperParamHeader += f"{key}: {LOG_HYPERPARAMS[key]}, "
                hyperParamHeader += "\r\n"
                predictions = models[model_type].predict(data[model_type])
                score = evaluate_pred(predictions, labels)

                # Positives are Survivors
                resString = f"Training Results: TP: {score[1][1]}, FP: {score[0][0]}, FN: {score[1][0]}, TN: {score[0][1]}, "\
                f"Accuracy: {(score[1][1] + score[0][1]) / (score[1][0] + score[0][0] + score[1][1] + score[0][1]):.4f}"
                if WRITE_RES_TO_FILE:
                    resFile.write(hyperParamHeader)
                    resFile.write(resString)
                    resFile.write("\r\n")
                print(resString)

            
            print("\r\n--- CROSS VALIDATION PHASE ---")
            if not DO_CROSS_VAL:
                print(f"Cross Validation skipped. Set DO_CROSS_VAL to True to perform Cross Validation.\r\n")
            else:
                n_data = len(data[COMP_MODEL])
                partition_size = n_data // CV_ORDER
                tr_score = []
                ts_score = []
                for i in range(CV_ORDER):
                    # Divide into training and test data
                    train_data = {}
                    test_data = {}
                    train_labels = labels[0 : i * partition_size] + labels[(i+1) * partition_size : n_data]
                    test_labels = labels[i * partition_size : (i+1) * partition_size]
                    
                    print(f"Iteration {i+1}...")        
                    cv_models = {
                        NUM_MODEL: GaussianNB(
                            priors=PRIORS
                        ),
                        BIN_MODEL: BernoulliNB(
                            class_prior=PRIORS,
                            fit_prior=FIT_PRIOR
                        ) if LOG_HYPERPARAMS["UseBernoulli"]  
                        else MultinomialNB(
                            class_prior=PRIORS,
                            fit_prior=FIT_PRIOR
                        ),
                        COMP_MODEL: GaussianNB(
                            priors=PRIORS
                        )
                    }
                    
                    preds_layer1 = {}
                    for model_type in MODELTYPES_LAYER1:
                        train_data[model_type] = data[model_type][0 : i * partition_size] + data[model_type][(i+1) * partition_size : n_data]
                        test_data[model_type] = data[model_type][i * partition_size: (i+1) * partition_size]
                        cv_models[model_type] = cv_models[model_type].fit(train_data[model_type], train_labels)
                        preds_layer1[model_type] = cv_models[model_type].predict_proba(train_data[model_type])
                    

                    train_data[COMP_MODEL] = [
                        [preds_layer1[NUM_MODEL][i][1], preds_layer1[BIN_MODEL][i][1]] for i in range(preds_layer1[NUM_MODEL].shape[0])
                        ]    
                    cv_models[COMP_MODEL] = cv_models[COMP_MODEL].fit(train_data[COMP_MODEL], train_labels)
                    predictions = cv_models[COMP_MODEL].predict(train_data[COMP_MODEL])
                    score = evaluate_pred(predictions, train_labels)
                    accuracy = (score[1][1] + score[0][1]) / (score[1][0] + score[0][0] + score[1][1] + score[0][1])
                    tr_score.append(accuracy)

                    test_preds_layer1 = {}
                    for model_type in MODELTYPES_LAYER1:
                        test_preds_layer1[model_type] = cv_models[model_type].predict_proba(test_data[model_type])
                    test_data[COMP_MODEL] = [
                        [test_preds_layer1[NUM_MODEL][i][1], test_preds_layer1[BIN_MODEL][i][1]] for i in range(test_preds_layer1[NUM_MODEL].shape[0])
                        ]    
                    
                    predictions = cv_models[COMP_MODEL].predict(test_data[COMP_MODEL])
                    score = evaluate_pred(predictions, test_labels)
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
            test_preds_layer1 = {}
            for model_type in MODELTYPES_LAYER1:
                test_preds_layer1[model_type] = models[model_type].predict_proba(tdata[model_type])
            tdata[COMP_MODEL] = [
                [test_preds_layer1[NUM_MODEL][i][1], test_preds_layer1[BIN_MODEL][i][1]] for i in range(test_preds_layer1[NUM_MODEL].shape[0])
                ] 
            predictions = models[COMP_MODEL].predict(tdata[COMP_MODEL])
            score = evaluate_pred(predictions, tlabels)
            resString = f"Testing Results: TP: {score[1][1]}, FP: {score[0][0]}, FN: {score[1][0]}, TN: {score[0][1]}, "\
            f"Accuracy: {(score[1][1] + score[0][1]) / (score[1][0] + score[0][0] + score[1][1] + score[0][1]):.4f}"
            if WRITE_RES_TO_FILE:
                resFile.write(resString)
                resFile.write("\r\n")
            print(resString)

            if PLOT_RES:
                plotParams = f"{COMP_MODEL}_"
                for key in LOG_HYPERPARAMS:
                    plotParams += f"{key}{LOG_HYPERPARAMS[key]}_"
                predictions = models[COMP_MODEL].predict_proba(tdata[COMP_MODEL])
                predictions_survived = [predictions[i][1] for i in range(predictions.shape[0])]
                color_list = ['red' if label==0 else 'blue' for label in tlabels]
                plt.scatter(range(len(predictions_survived)), predictions_survived, s=SCATTER_POINT_SIZE, c=color_list)
                plt.ylabel("Probability of Survival")
                plt.xlabel("Test Sample Index")
                plt.title(f"Survival Prediction of ICU Patient Test Samples")
                plt.suptitle(f"Red: died, blue: survived")
                plt.draw()
                plt.savefig(fr'Latex Report\Figures\TEST_PRED_{plotParams}.png')
                plt.clf()

            if TEST_RISK_DECISION:
                predictions = models[COMP_MODEL].predict_proba(tdata[COMP_MODEL])
                score = evaluate_pred_risk(predictions, tlabels)
                resString = f"Risk Decision Test Results (lambda00={RISK[0][0]}, lambda01={RISK[0][1]}, lambda10={RISK[1][0]}, lambda11={RISK[1][1]}): TP: {score[1][1]}, FP: {score[0][0]}, FN: {score[1][0]}, TN: {score[0][1]}, "\
                f"Accuracy: {(score[1][1] + score[0][1]) / (score[1][0] + score[0][0] + score[1][1] + score[0][1]):.4f}"
                if WRITE_RES_TO_FILE:
                    resFile.write(resString)
                    resFile.write("\r\n")
                print(resString)

if __name__=="__main__":
    main()