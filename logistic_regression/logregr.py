import csv # Data format
import math # Log and exp function
import random # Random samples
import matplotlib.pyplot as plt # Graph plotting

# Hyperparameters
ERROR = 1e-1
LEARNING_RATE = 1e-1
CROSS_VAL_ORDER = 10
CROSS_VAL_TRAIN_ITERATIONS = 300
TRAIN_ITERATIONS = 3000

def preprocess(data, labels, desc):
    labels_out = [-1 if row[0] == '0' else 1 for row in labels]
    labels_out = labels_out[1:]

    desc_out = [row for row in desc]
    data_out = []
    for row in data:
        row.append(1.0) 
        # Bias weight w0 and constant 1 in each feature vector are put at the end(!) for convenience
        data_out.append(row)
    data_out = data_out[1:]

    n_data = len(labels_out)
    n_features = len(desc_out) - 1
    
    # Include bias weight w_0 in weight vector
    weights = [0.0] * (n_features + 1) 

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

    return data_out, labels_out, weights

def get_scalar_prod(feature_vec, label, weights):
    n_weights = len(weights)
    sc_prod = 0.0
    for i in range(n_weights):
        sc_prod += feature_vec[i] * weights[i]
    return label * sc_prod

def sigmoid(x):
    # Numerically stable computation that avoids overflows
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    return math.exp(x) / (1 + math.exp(x))

def predict(data, labels, weights):
    n_data = len(data)
    tp = fp = tn = fn = 0
    for i in range(n_data):
        feature_vector = data[i]
        label = labels[i]
        scalar_prod = get_scalar_prod(feature_vector, label, weights)
        pred = sigmoid(scalar_prod)
        if pred <= 0.5:
            if label < 0:
                fp += 1
            else:
                fn += 1
        else:
            if label < 0:
                tn += 1
            else:
                tp += 1
    # print(f"tp: {tp}, fp: {fp}, fn: {fn}, tn: {tn}")
    return tp, fp, tn, fn

def update_loss(data, labels, weights):
    n_data = len(data)
    n_weights = len(weights)
    loss = 0.0
    inv = 1.0 / n_data
    indices = [i for i in range(n_data)]
    random.shuffle(indices)
    
    # For deterministic sampling, iterate directly over range(n_data)
    # for i in range(n_data):
    for i in indices:
        # i = random.randint(0, n_data - 1) 
        feature_vector = data[i]
        label = labels[i]
        scalar_prod = get_scalar_prod(feature_vector, label, weights)
        
        # Numerically stable computation of the gradient
        if scalar_prod >= 0:
            expterm = math.exp((-1) * scalar_prod)
            loss += inv * math.log(1 + expterm)
            gradLossFactor = (-1) * inv * (expterm / (1 + expterm))
        else:
            expterm = math.exp(scalar_prod)
            loss += (-1) * inv * math.log(expterm / (1 + expterm))
            gradLossFactor = (-1) * inv / (1 + expterm)
        
        # Update weights
        for j in range(n_weights):
            weights[j] -= LEARNING_RATE * label * feature_vector[j] * gradLossFactor

    return loss

def train(data, labels, weights, epochs = TRAIN_ITERATIONS):
    err = ERROR
    log_err = []
    t = 0
    step_limit = epochs
    while err >= ERROR and t <= step_limit:
        t += 1
        err = update_loss(data, labels, weights)
        log_err.append(err)
    
    return weights, log_err

def cross_val(data, labels, weights):
    n_data = len(data)
    n_weights = len(weights)
    loss_log = []
    partition_size = n_data // CROSS_VAL_ORDER
    test_results = []
    for i in range(CROSS_VAL_ORDER):
        # Divide into training and test data
        train_data = data[0 : i * partition_size] + data[(i+1) * partition_size : n_data]
        test_data = data[i * partition_size: (i+1) * partition_size]
        train_labels = labels[0 : i * partition_size] + labels[(i+1) * partition_size : n_data]
        test_labels = labels[i * partition_size : (i+1) * partition_size]
        
        print(f"Iteration {i+1}...")        
        weights, loss_data = train(train_data, train_labels, [0.0] * n_weights, epochs=CROSS_VAL_TRAIN_ITERATIONS)
        loss_log += loss_data # Append loss values

        tp, fp, tn, fn = predict(test_data, test_labels, weights)
        test_results.append({"tp":tp, "fp":fp, "tn":tn, "fn":fn})

    return loss_log, test_results


def main():
    with open('data1forEx1to4/train1_icu_data.csv') as fdata1, \
        open('data1forEx1to4/train1_icu_label.csv') as flabel1, \
        open('data1forEx1to4/feature_description.csv') as fdescr1, \
        open('data1forEx1to4/test1_icu_data.csv') as tdata1, \
        open('data1forEx1to4/test1_icu_label.csv') as tlabel1, \
        open('data1forEx1to4/feature_description.csv') as fdescr2:
        data1 = csv.reader(fdata1)
        label1 = csv.reader(flabel1)
        desc1 = csv.reader(fdescr1)

        print("\r\n--- PREPROCESSING PHASE ---")
        data, labels, weights = preprocess(data1, label1, desc1)

        print("\r\n--- TRAINING PHASE ---")
        weights, loss_log = train(data, labels, weights)
        
        plt.plot(loss_log)
        plt.ylabel("E(w)")
        plt.title("Training Error")
        plt.show()

        
        print("\r\n--- CROSS VALIDATION PHASE ---")
        loss_log, test_results = cross_val(data, labels, weights)
        for i, result in enumerate(test_results):
            print(f"CV Iteration {i}")
            print(f"tp: {result["tp"]}, fp: {result["fp"]}, fn: {result["fn"]}, tn: {result["tn"]}")
            false_count = result["fp"] + result["fn"]
            correct_count = result["tp"] + result["tn"]
            print(f"Accuracy: {correct_count / (correct_count + false_count)}")
            print("")

        # Test
        print("\r\n--- TEST PHASE ---")
        tdata1 = csv.reader(tdata1)
        tlabel1 = csv.reader(tlabel1)
        desc2 = csv.reader(fdescr2)
        tdata, tlabels, _ = preprocess(tdata1, tlabel1, desc2)
        tp, fp, tn, fn = predict(tdata, tlabels, weights)
        print(f"Test Results: TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

        # ROC curve
        w0 = weights[-1]
        tpr_data = [float(tp) / float(tp + fn)]
        fpr_data = [float(fp) / float(tn + fp)]
        while float(tn) / float(tn + fp) >= 0.05:
            weights[-1] = weights[-1] + 0.1
            # print(f"w0: {weights[-1]}")
            tp, fp, tn, fn = predict(tdata, tlabels, weights)
            tpr_data.append(float(tp) / float(tp + fn))
            fpr_data.append(float(fp) / float(tn + fp))

        weights[-1] = w0
        tpr_data.reverse()
        fpr_data.reverse()
        while float(tp) / float(tp + fn) >= 0.05:
            weights[-1] = weights[-1] - 0.1
            # print(f"w0: {weights[-1]}")
            tp, fp, tn, fn = predict(tdata, tlabels, weights)
            tpr_data.append(float(tp) / float(tp + fn))
            fpr_data.append(float(fp) / float(tn + fp))
        
        plt.plot(fpr_data, tpr_data)
        plt.xlabel("False positive rate (FPR)")
        plt.ylabel("True positive rate (TPR)")
        plt.title("ROC curve")
        plt.show()

if __name__=="__main__":
    main()