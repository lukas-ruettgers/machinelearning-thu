import csv # Data format
import time
import matplotlib.pyplot as plt # Graph plotting
from sklearn.neural_network import MLPClassifier # Multilayer Perceptron
from sklearn.model_selection import learning_curve


# Hyperparameters
# -- Model --
ACTIVATION = 'relu' # ['relu', 'logistic', 'identity', 'tanh']
N_LAYERS = 1
HIDDEN_LAYER_SIZE = 100

# -- Training --
SOLVER = 'sgd' # ['sgd', 'adam', 'lbfgs']
LEARNING_RATE = 5e-2 # Default: 1e-3
LEARNING_RATE_SCHEDULE = 'constant' # ['constant', 'adaptive',...]
MOMENTUM_RATE = 0.9 # Default: 0.9
MAX_TRAIN_ITERATIONS = 800 # Default: 200
MAX_TRAIN_ITERATIONS_BASE = 200
REGULARIZATION_RATE = 1e-4 # Default: 1e-4
LEARNING_SCORE_METRIC = 'neg_log_loss' # ['accuracy', 'f1', 'neg_log_loss',...]
UNIMPROVED_STEPS_FOR_CONVERGENCE = 100 # Default: 10
UNIMPROVED_LOSS_THRESHOLD_FOR_CONVERGENCE = 1e-4 # Default: 1e-4

# -- Cross Validation --
DO_CROSS_VAL = True
CV_ORDER = 10
CV_TRAIN_PROPORTIONS = [1.0]

# -- Testing --

# -- Logging --
WRITE_RES_TO_FILE = True
PLOT_RES = True

def preprocess(data, labels, desc):
    labels_out = [0 if row[0] == '0' else 1 for row in labels]
    labels_out = labels_out[1:]

    desc_out = [row for row in desc]
    data_out = [row for row in data]
    data_out = data_out[1:]

    n_data = len(labels_out)
    n_features = len(desc_out) - 1 

    # Feature scaling
    for i in range(n_features):
        feature_desc = desc_out[i + 1]
        feature_type = feature_desc[2]
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

    return data_out, labels_out

def evaluate_pred(predictions, labels):
    n_classes = len(predictions[0])
    score = [[0,0] for _ in range(n_classes)] # Positives and negatives for each class
    for pred, label in zip(predictions, labels):
        if pred[label] > 0.5:
            score[label][0] += 1 # 0 = Positive
        else:
            score[label][1] += 1 # 1 = Negative
    return score

def main():
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
        tdata, tlabels = preprocess(tdata1, tlabel1, desc2)

        print("\r\n--- PREPROCESSING PHASE ---")
        data, labels = preprocess(data1, label1, desc1)

        for mr in [0.001]:
            MOMENTUM_RATE = mr
            for maxiter_scale in range(5,6):
                MAX_TRAIN_ITERATIONS = MAX_TRAIN_ITERATIONS_BASE * maxiter_scale
                print("\r\n--- TRAINING PHASE ---")
                model = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER_SIZE,
                                    activation=ACTIVATION,
                                    solver=SOLVER,
                                    alpha=REGULARIZATION_RATE,
                                    learning_rate_init=LEARNING_RATE,
                                    learning_rate=LEARNING_RATE_SCHEDULE,
                                    max_iter=MAX_TRAIN_ITERATIONS,
                                    momentum=MOMENTUM_RATE,
                                    tol=UNIMPROVED_LOSS_THRESHOLD_FOR_CONVERGENCE,
                                    n_iter_no_change=UNIMPROVED_STEPS_FOR_CONVERGENCE)

                model = model.fit(data, labels)
                if PLOT_RES:
                    plt.plot(model.loss_curve_, label=f"LR_{LEARNING_RATE}_TI{MAX_TRAIN_ITERATIONS}_ACT{ACTIVATION}")
                    plt.title("Training Loss Curve")
                    plt.ylabel("Log loss")
                    plt.xlabel("Iterations")
                    plt.legend()
                    plt.draw()
                    plt.savefig(fr'Latex Report\Figures\TRAIN_LOSS_LR{LEARNING_RATE}_'\
                                fr'TI{MAX_TRAIN_ITERATIONS}_ACT{ACTIVATION}_MR{MOMENTUM_RATE}_RR{REGULARIZATION_RATE}.png')
                
                # resFile.write(f"\r\n-- Timestamp: {time.strftime("%Y%m%d%H%M%S")}\r\n")
                hyperParamHeader = f"\r\nLR: {LEARNING_RATE}, TI: {MAX_TRAIN_ITERATIONS}, ACT: {ACTIVATION}, MR: {MOMENTUM_RATE}, RR: {REGULARIZATION_RATE}\r\n"
                predictions = model.predict_proba(data)
                score = evaluate_pred(predictions, labels)
                resString = f"Training Results: TP: {score[1][0]}, FP: {score[1][1]}, FN: {score[0][1]}, TN: {score[0][0]}, "\
                f"Accuracy: {(score[1][0] + score[0][0]) / (score[1][0] + score[0][0] + score[1][1] + score[0][1]):.4f}"
                if WRITE_RES_TO_FILE:
                    resFile.write(hyperParamHeader)
                    resFile.write(resString)
                    resFile.write("\r\n")
                print(resString)

                print("\r\n--- CROSS VALIDATION PHASE ---")
                if not DO_CROSS_VAL:
                    print(f"Cross Validation skipped. Set DO_CROSS_VAL to True to perform Cross Validation.\r\n")
                else:
                    cv_model = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER_SIZE,
                                        activation=ACTIVATION,
                                        solver=SOLVER,
                                        alpha=REGULARIZATION_RATE,
                                        learning_rate_init=LEARNING_RATE,
                                        learning_rate=LEARNING_RATE_SCHEDULE,
                                        max_iter=MAX_TRAIN_ITERATIONS,
                                        momentum=MOMENTUM_RATE,
                                        tol=UNIMPROVED_LOSS_THRESHOLD_FOR_CONVERGENCE,
                                        n_iter_no_change=UNIMPROVED_STEPS_FOR_CONVERGENCE) 
                    train_size, train_scores, test_scores = learning_curve(cv_model, data, labels,
                                                                        train_sizes=CV_TRAIN_PROPORTIONS,
                                                                        cv=CV_ORDER,
                                                                        scoring=LEARNING_SCORE_METRIC)
                    
                    if WRITE_RES_TO_FILE:
                        cvResFile.write(hyperParamHeader)
                    for t_size, tr_score, ts_score in zip(train_size, train_scores, test_scores):
                        resString = f"Training Size: {t_size},\r\n" \
                            f"Training Score: {tr_score}, Mean: {tr_score.mean():.4f}\r\n"\
                            f"Test Score: {ts_score}, Mean: {ts_score.mean():.4f}\r\n"
                        if WRITE_RES_TO_FILE:
                            cvResFile.write(resString)
                        print(resString)

                # Test
                print("\r\n--- TEST PHASE ---")
                predictions = model.predict_proba(tdata)
                score = evaluate_pred(predictions, tlabels)
                resString = f"Testing Results: TP: {score[1][0]}, FP: {score[1][1]}, FN: {score[0][1]}, TN: {score[0][0]}, "\
                f"Accuracy: {(score[1][0] + score[0][0]) / (score[1][0] + score[0][0] + score[1][1] + score[0][1]):.4f}"
                if WRITE_RES_TO_FILE:
                    resFile.write(resString)
                    resFile.write("\r\n")
                print(resString)

if __name__=="__main__":
    main()