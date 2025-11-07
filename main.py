import argparse
import pickle

nbc_counts = {}
nbc_probabilites = {}
attributes = []

def train_and_save_model(data_file, meta_file, model_output=None):
    """
    Train a Naive Bayes model and save it to a file
    
    Args:
        data_file (str): Path to training data file
        meta_file (str): Path to meta file
        model_output (str): Path to save the trained model
    """
    reset_globals()

    data = []

    if model_output is None:
        model_output = 'nbc_model.pkl'
    
    # Load training data
    with open(data_file) as file:
        lines = file.readlines()
        data = [line.strip().split(",") for line in lines]
    
    print(f"Loading training data from {data_file}...")
    print(f"Training on {len(data)} instances")
    
    # Train the model
    build_nbc(data, meta_file)
    
    print("Training completed!")
    
    # Save the trained model
    save_model(model_output)
    
    # statistics
    print(f"\nModel Statistics:")
    print(f"Classes: {list(nbc_counts.keys())}")
    print(f"Attributes: {attributes}")
    print(f"Total training instances: {get_total_instances(nbc_counts)}")
    
    return model_output

def load_and_predict(input_file, output_file, model_file="nbc_model.pkl"):
    """
    Load a trained model and make predictions on new data
    
    Args:
        input_file (str): Path to file with data to classify
        output_file (str): Path to save predictions
        model_file (str): Path to saved model
    """

    # Load the trained model
    load_model(model_file)

     # Load data to classify
    data = []
    with open(input_file) as file:
        lines = file.readlines()
        data = [line.strip().split(",") for line in lines]

    print(f"Classifying {len(data)} instances...")
    
    # Check if data has labels or not
    has_labels = len(data[0]) == len(attributes) + 1
    
    # Remove labels if present (for prediction)
    if has_labels:
        data_unlabeled = [instance[:-1] for instance in data]
    else:
        data_unlabeled = data
    
    # Make predictions
    predictions = nbc_predict(data_unlabeled)
    
    # Save predictions to output file
    with open(output_file, "w") as file:
        for instance, pred_label in zip(data, predictions):
            if has_labels:
                # Replace existing label with prediction
                instance[-1] = pred_label
            else:
                # Append prediction to unlabeled data
                instance.append(pred_label)
            
            file.write(",".join(instance))
            file.write("\n")
    
    print(f"Predictions saved to {output_file}")
    print(f"Classified {len(predictions)} instances")
    
    return predictions

def train_and_evaluate(train_file, test_file, meta_file):
    """
    Train a model on training data and evaluate on test data with confusion matrix
    
    Args:
        train_file (str): Path to training data file
        test_file (str): Path to test data file
        meta_file (str): Path to meta file
    """
    reset_globals()

    train_data = []
    test_data = []

    # Load training data
    print(f"Loading training data from {train_file}...")
    with open(train_file) as file:
        lines = file.readlines()
        train_data = [line.strip().split(",") for line in lines]
    
    print(f"Training on {len(train_data)} instances...")
    
    # Train the model
    build_nbc(train_data, meta_file)
    
    print("Training completed!")
    
    # Load test data
    print(f"\nLoading test data from {test_file}...")
    with open(test_file) as file:
        lines = file.readlines()
        test_data = [line.strip().split(",") for line in lines]
    
    print(f"Evaluating on {len(test_data)} instances...")
    
    # Remove class labels from test data for prediction
    test_data_unlabeled = [item[:-1] for item in test_data]
    predictions = nbc_predict(test_data_unlabeled)
    
    # Calculate accuracy
    accuracy_score = get_accuracy(test_data, predictions)
    error_rate = 100 - accuracy_score
    
    # Print results
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy Score: {accuracy_score}%")
    print(f"Error Rate: {error_rate:.2f}%")
    
    # Generate and display confusion matrix
    confusion_matrix = get_confusion_matrix(predictions, test_data)
    display_confusion_matrix(confusion_matrix)
    
    # Calculate and display metrics
    metrics = get_cm_metrics(confusion_matrix)
    
    print(f"\n{'='*50}")
    print("DETAILED METRICS")
    print(f"{'='*50}")
    for class_label in metrics:
        print(f"{class_label}:")
        print(f"  Precision:   {metrics[class_label]['precision']:.4f}")
        print(f"  Recall:      {metrics[class_label]['recall']:.4f}")
        print(f"  Specificity: {metrics[class_label]['specificity']:.4f}")
        print(f"  F1-Score:    {metrics[class_label]['f1']:.4f}")
        print()
    
    return accuracy_score, metrics


def load_meta_data(meta_data):
    """Helper Function to load meta data into nbc_counts
    
    Args:
        meta_data (str): a *.meta file containing the meta data with a list
            of attributes ands possible values
    """
    

    with open(meta_data, "r") as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines] # Strip new lines

        # Get class_labels
        _, classes = lines[-1].split(":")
        classes = classes.split(",")

        for class_label in classes:
            nbc_counts[class_label] = {}
            nbc_probabilites[class_label] = {}

        lines.pop(-1)

        # Add each attributes to each class and intialize values to zero
        for line in lines:
            attribute, values = line.split(":")
            
            values = values.split(",")
            attributes.append(attribute)

            # print(attribute, values)
            for class_label in classes:
                nbc_counts[class_label][attribute] = {value: 0 for value in values}
                nbc_probabilites[class_label][attribute] = {value: 0 for value in values}

def learn_data(data):
    """Function to learn data
    
    Args:
        data (list): a list of instances containing the data to be learn
    """
    for instance in data:
        classification = instance[-1]
        for attribute, value in zip(attributes, instance[:-1]):
            nbc_counts[classification][attribute][value] += 1

    for class_label in nbc_counts:
        for attribute in nbc_counts[class_label]:
            class_label_total_instances = sum(nbc_counts[class_label][attribute].values())
            num_of_possible_values = len(nbc_counts[class_label][attribute])

            for value in nbc_counts[class_label][attribute]:
                nbc_probabilites[class_label][attribute][value] =  (nbc_counts[class_label][attribute][value] + 1) / (class_label_total_instances + num_of_possible_values)
            
def build_nbc(data, meta_data):
    """Function to build the Naive Bayes Classification System
    
    Args:
        data (list): a list of instances containing the data to be trained on
        meta_data (str): a *.meta file containing the meta data with a list
            of attributes ands possible values
    """

    load_meta_data(meta_data)
    learn_data(data)
            
def nbc_predict(data):
    """Function to make prediction
    
    Args:
        training_data (list) a list of testing instances containing the data to be tested on

    Returns
        list: a list of predicted class_labels
    """
    results = []
    
    # classes_prior_probs = {class_label: (get_class_instances(nbc_counts, class_label) / get_total_instances(nbc_counts)) for class_label in nbc_counts}
    # print(classes_prior_probs)


    for instance in data:
        class_probs = {}

        # iterate thru every single class to calcualate probability for each class label
        for class_label in list(nbc_probabilites.keys()):
            class_label_prob = get_class_instances(nbc_counts, class_label) / get_total_instances(nbc_counts) # get class label prior probability

            for attribute_type, attribute in zip(attributes, instance):
                #calculate the probability by using the look up table iterate thru all attribute get their conditional probability with respect to the current class
                class_label_prob *= nbc_probabilites[class_label][attribute_type][attribute]
            
            class_probs[class_label] = class_label_prob

        # get class with biggest probability and add it to results
        results.append(max(class_probs, key=class_probs.get))

    return results

def run_kfold_cross_validation(data_file, meta_file, k_folds):
    data = []
    scores = []

    # Get data from data_file
    with open(data_file) as file:
        lines = file.readlines()
        data = [line.strip().split(",") for line in lines]

    fold_size = len(data)  // k_folds

    folds = [data[i * fold_size: (i + 1) * fold_size] for i in range(0, k_folds)]
    if(len(data) % k_folds != 0):
        # Put remaining data into the last fold
        folds[-1].extend(data[k_folds * fold_size:])

    for i in range(k_folds):
        reset_globals()

        testing_data = folds[i]
        training_data = []
        for k in range(k_folds):
            if k != i:
                training_data.extend(folds[k])
        
        build_nbc(training_data, meta_file)

        # Remove class labels from training data
        testing_data_unlabel = [item[:-1] for item in testing_data]
        predictions = nbc_predict(testing_data_unlabel)

        # evaluate
        accuracy_score = get_accuracy(testing_data,predictions)
        scores.append(accuracy_score)

        # print accuracy
        print(f"\nAccuracy score for fold {i + 1}: {accuracy_score}")

        confusion_matrix = get_confusion_matrix(predictions, testing_data)
        metrics = get_cm_metrics(confusion_matrix)

        print(f"Metrics for fold {i + 1}: ")
        for class_label in metrics:
            print(f'{class_label} --- Precision: {metrics[class_label]["precision"]} Recall: {metrics[class_label]["recall"]} Specificity: {metrics[class_label]["specificity"]} f1: {metrics[class_label]["f1"]}')

    # print avg score
    avg_accuracy = round(sum(scores) / len(scores), 2)
    print(f"\nAverage accuracy: {avg_accuracy}")

def confusion_matrix():
    data = []
    test_data = []

    # Ask User for the names of the fles with the training data
    data_file, meta_file = input("\nEnter a training file and meta file (*.train *.meta): ").split()
    # Get data from data_file
    with open(data_file) as file:
        lines = file.readlines()
        data = [line.strip().split(",") for line in lines]
    # train model
    build_nbc(data, meta_file)


    # Read set of data to provide classification -- ask user for input and output files
    input_files = input("\nEnter data files to be classify (file1 ...): ").split()
    output_files = input("Enter their respective output files (file1 ...): ").split()

    classify_files(input_files, output_files)


    # Ask Users for .test file, 
    test_file = input("\nEnter a .test file (*.test): ")
    # Get testing data from test_file
    with open(test_file) as file:
        lines = file.readlines()
        test_data = [line.strip().split(",") for line in lines]
    # Remove class labels from training data
    testing_data_unlabel = [item[:-1] for item in test_data]
    predictions = nbc_predict(testing_data_unlabel)
    # Evaluate 
    accuracy_score = get_accuracy(test_data, predictions)

    # print a report of accuracy of the testing data
    print(f"\nAccuracy Score: {accuracy_score}")
    print(f"\nError Rate: {100 - accuracy_score}")

    # generate and print confusion matrix for test data
    confusion_matrix = get_confusion_matrix(predictions, test_data)
    display_confusion_matrix(confusion_matrix)
    metrics = get_cm_metrics(confusion_matrix)

    print("\nMetrics: ")
    for class_label in metrics:
        print(f'{class_label} --- Precision: {metrics[class_label]["precision"]} Recall: {metrics[class_label]["recall"]} Specificity: {metrics[class_label]["specificity"]}')

def classify_files(input_files, output_files):
    """Helper function"""
    for input_file, output_file in zip(input_files, output_files):
        data = []
        with open(input_file) as file:
            lines = file.readlines()
            data = [line.strip().split(",") for line in lines]

        predictions = nbc_predict(data)

        #output result to file
        with open(output_file, "w") as file:
            if len(attributes) != len(data[0]):
                # label is ignored2
                for instance, pred_label in zip(data, predictions):
                    instance[-1] = pred_label
                    file.write(",".join(instance))
                    file.write("\n") # new line
            else:
                # there is no label
                for instance, pred_label in zip(data, predictions):
                    instance.append(pred_label)
                    file.write(",".join(instance))
                    file.write("\n") # new line

def get_confusion_matrix(predictions, test_data):
    """Helper function"""
    # initialize confusion matrix
    confusion_matrix = {actu_label : {pred_label: 0 for pred_label in nbc_counts.keys()} for actu_label in nbc_counts.keys()}

    actual_labels = [instance[-1] for instance in test_data]
    for predicted_label, actual_label in zip(predictions, actual_labels):
        if predicted_label and actual_label:
            confusion_matrix[actual_label][predicted_label] += 1

    return confusion_matrix
    
def display_confusion_matrix(confusion_matrix):
    """Helper function"""

    print("\nConfusion Matrx")
    # Header
    header = "Actual/Predicted"
    for label in confusion_matrix.keys():
        header += f"{label:^10}"
    print(header)
    # Rows
    for actual_label in confusion_matrix.keys():
        row = f"{actual_label:16}"
        for pred_label in confusion_matrix.keys():
            row += f"{confusion_matrix[actual_label][pred_label]:^10}"
        print(row)

def get_accuracy(data, predictions):
    """Helper function"""
    correct_count = 0

    for instance, predicted_label in zip(data, predictions):
        if(instance[-1] == predicted_label):
            correct_count += 1

    return round((correct_count / len(data)) * 100, 2)

def get_total_instances(data):
    """Helper function"""
    total = 0
    for class_label in data:
        total += get_class_instances(data, class_label)

    return total

def get_class_instances(data, class_label):
    """Helper function"""
    return sum(list(data[class_label].values())[0].values())

def get_cm_metrics(confusion_matrix):
    """helper function"""
    
    metrics = {class_label: {"precision": 0, "recall" : 0, "specificity" : 0, "f1" : 0} for class_label in nbc_counts}

    for class_label in metrics:
        # True Positive
        tp = confusion_matrix[class_label][class_label]

        # False Positive
        fp = 0
        for actual_class_label in confusion_matrix:
            if actual_class_label != class_label:
                fp += confusion_matrix[actual_class_label][class_label]

        # False Negative
        fn = 0
        for predicted_class_label in confusion_matrix[class_label]:
            if predicted_class_label != class_label:
                fn += confusion_matrix[class_label][predicted_class_label]

        # True Negative
        tn = 0
        for actual_class_label in confusion_matrix:
            for predicted_class_label in confusion_matrix[actual_class_label]:
                if predicted_class_label != class_label and actual_class_label != class_label:
                    tn += confusion_matrix[actual_class_label][predicted_class_label]

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

        metrics[class_label]["precision"] = precision
        metrics[class_label]["recall"] = recall
        metrics[class_label]["specificity"] = specificity
        metrics[class_label]["f1"] = 2 * (precision * recall) / (precision + recall) if precision > 0 and recall > 0 else 0

    return metrics

def save_model(filename='nbc_model.pkl'):
    """
    Save trained model to file
    
    Args:
        filename (str): Path to save the model
    """
    model_data = {
        'counts': nbc_counts,
        'probabilities': nbc_probabilites,
        'attributes': attributes
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {filename}")

def load_model(filename='nbc_model.pkl'):
    """
    Load pre-trained model from file
    
    Args:
        filename (str): Path to the saved model
    """
    global nbc_counts, nbc_probabilites, attributes
    
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    
    nbc_counts = model_data['counts']
    nbc_probabilites = model_data['probabilities']
    attributes = model_data['attributes']
    
    print(f"Model loaded from {filename}")

def main():
    parser = argparse.ArgumentParser(description='Naive Bayes Classifier')
    parser.add_argument('--mode', choices=['train', 'predict', 'kfold', 'evaluate'], 
                       required=True, help='Operation mode')
    parser.add_argument('--data', help='Data file path')
    parser.add_argument('--meta', help='Meta file path')

    parser.add_argument('--train', help='Training file path (for evaluate mode)')
    parser.add_argument('--test', help='Test file path (for evaluate mode)')

    parser.add_argument('--input', help='Input file for prediction')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--model', default='nbc_model.pkl', help='Model file path')
    parser.add_argument('--k', type=int, default=5, help='Number of folds')
    
    args = parser.parse_args()
    
    if args.mode == 'kfold':
        if not args.data or not args.meta:
            print("Error: --data and --meta required for kfold mode")
            return
        run_kfold_cross_validation(args.data, args.meta, args.k)
    elif args.mode == 'train':
        if not args.data or not args.meta:
            print("Error: --data and --meta required for train mode")
            return
        train_and_save_model(args.data, args.meta, args.model)
    elif args.mode == 'predict':
        if not args.input or not args.output:
            print("Error: --input and --output required for predict mode")
            return
        load_and_predict(args.input, args.output, args.model)
    elif args.mode == 'evaluate':
        if not args.train or not args.test or not args.meta:
            print("Error: --train, --test, and --meta required for evaluate mode")
            return
        train_and_evaluate(args.train, args.test, args.meta)

def reset_globals():
    """Reset global variables for fresh training"""
    global nbc_counts, nbc_probabilites, attributes
    nbc_counts = {}
    nbc_probabilites = {}
    attributes = []

if __name__ == "__main__":
    main()