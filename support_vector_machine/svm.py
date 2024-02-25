import csv # Data format
import matplotlib.pyplot as plt # Graph plotting
from sklearn.svm import SVC

# Hyperparameters
# -- Model --
REGULARIZATION_RATE = 1.0 # default: 1.0
KERNEL = 'poly' # ['rbf', 'linear', 'poly', 'sigmoid']
POLY_KERNEL_DEG = 3 # default: 3
KERNEL_COEF = 'scale' # ['scale', 'auto', float]

# -- Training --
SHRINKING = True # default: True
MAX_TRAIN_ITERATIONS = -1 # Default: -1 (no limit)
MAX_TRAIN_ITERATIONS_BASE = 200
UNIMPROVED_LOSS_THRESHOLD_FOR_CONVERGENCE = 1e-3 # Default: 1e-3

# -- Cross Validation --
DO_CROSS_VAL = False
CV_ORDER = 10
CV_TRAIN_PROPORTIONS = [1.0]
CV_SCORE_METRIC = 'neg_log_loss' # ['accuracy', 'f1', 'neg_log_loss',...]

# -- Testing --

# -- Logging --
WRITE_RES_TO_FILE = True
PLOT_RES = True
SCATTER_POINT_SIZE = 5

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
    score = [[0,0],[0,0]] # Positives and negatives for each class
    for pred, label in zip(predictions, labels):
        if pred > 0:
            if label == 1:
                score[label][0] += 1 # TP
            else:
                score[label][1] += 1 # FP
        else:
            if label == 0:
                score[label][0] += 1 # TN
            else:
                score[label][1] += 1 # FN
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

        for _ in ['rbf']:
            for maxiter_scale in range(5,6):
                # MAX_TRAIN_ITERATIONS = MAX_TRAIN_ITERATIONS_BASE * maxiter_scale
                print("\r\n--- TRAINING PHASE ---")
                model = SVC(
                    C = REGULARIZATION_RATE,
                    kernel = KERNEL,
                    degree = POLY_KERNEL_DEG,
                    gamma = KERNEL_COEF,
                    shrinking = SHRINKING,
                    tol = UNIMPROVED_LOSS_THRESHOLD_FOR_CONVERGENCE,
                    max_iter = MAX_TRAIN_ITERATIONS
                )

                model = model.fit(data, labels)
                if PLOT_RES:
                    plotParams = f"KN{KERNEL}_KNC{KERNEL_COEF}_REG{REGULARIZATION_RATE}_TI{MAX_TRAIN_ITERATIONS}_PKD{POLY_KERNEL_DEG}"
                    predictions = model.decision_function(data)
                    predictions = list(predictions)
                    predictions0 = []
                    predictions1 = []
                    for i, label in enumerate(labels):
                        if label == 1:
                            predictions1.append(predictions[i])
                        else:
                            predictions0.append(predictions[i])

                    plt.scatter(range(len(predictions0)), predictions0, s=SCATTER_POINT_SIZE, c='red')
                    plt.title("Classification of ICU Patient Training Samples that Died")
                    plt.ylabel("Decision function f(x)")
                    plt.xlabel("Training Sample Index")
                    plt.draw()
                    plt.savefig(fr'Latex Report\Figures\DEC_FUN_0_{plotParams}.png')
                    plt.clf()

                    plt.scatter(range(len(predictions1)), predictions1, s=SCATTER_POINT_SIZE, c='blue')
                    plt.title("Classification of ICU Patient Training Samples that Survived")
                    # plt.title("Classification of ICU Patient Training Samples")
                    plt.ylabel("Decision function f(x)")
                    plt.xlabel("Training Sample Index")
                    plt.draw()
                    plt.savefig(fr'Latex Report\Figures\DEC_FUN_1_{plotParams}.png')
                    plt.clf()
                
                hyperParamHeader = f"\r\nKN: {KERNEL}, KNC: {KERNEL_COEF}, REG: {REGULARIZATION_RATE}, TI: {MAX_TRAIN_ITERATIONS}, SHR: {SHRINKING}, PKD: {POLY_KERNEL_DEG}\r\n"
                predictions = model.predict(data)
                score = evaluate_pred(predictions, labels)
                resString = f"Training Results: TP: {score[1][0]}, FN: {score[1][1]}, FP: {score[0][1]}, TN: {score[0][0]}, "\
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
                        model = SVC(
                                    C = REGULARIZATION_RATE,
                                    kernel = KERNEL,
                                    degree = POLY_KERNEL_DEG,
                                    gamma = KERNEL_COEF,
                                    shrinking = SHRINKING,
                                    tol = UNIMPROVED_LOSS_THRESHOLD_FOR_CONVERGENCE,
                                    max_iter = MAX_TRAIN_ITERATIONS
                                )

                        model = model.fit(train_data, train_labels)
                        predictions = model.predict(train_data)
                        score = evaluate_pred(predictions, train_labels)
                        accuracy = (score[1][0] + score[0][0]) / (score[1][0] + score[0][0] + score[1][1] + score[0][1])
                        tr_score.append(accuracy)

                        predictions = model.predict(test_data)
                        score = evaluate_pred(predictions, test_labels)
                        accuracy = (score[1][0] + score[0][0]) / (score[1][0] + score[0][0] + score[1][1] + score[0][1])
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
                score = evaluate_pred(predictions, tlabels)
                resString = f"Testing Results: TP: {score[1][0]}, FN: {score[1][1]}, FP: {score[0][1]}, TN: {score[0][0]}, "\
                f"Accuracy: {(score[1][0] + score[0][0]) / (score[1][0] + score[0][0] + score[1][1] + score[0][1]):.4f}"
                if WRITE_RES_TO_FILE:
                    resFile.write(resString)
                    resFile.write("\r\n")
                print(resString)

if __name__=="__main__":
    main()