import numpy as np
from numpy.typing import NDArray
from typing import Any
from sklearn.metrics import top_k_accuracy_score, make_scorer, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import utils as u
import new_utils as nu
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

"""
   In the first two set of tasks, we will narrowly focus on accuracy - 
   what fraction of our predictions were correct. However, there are several 
   popular evaluation metrics. You will learn how (and when) to use these evaluation metrics.
"""


# ======================================================================
class Section3:
    def __init__(
        self,
        normalize: bool = True,
        frac_train=0.2,
        seed=42,
    ):
        self.seed = seed
        self.normalize = normalize

    def analyze_class_distribution(self, y: NDArray[np.int32]) -> dict[str, Any]:
        """
        Analyzes and prints the class distribution in the dataset.

        Parameters:
        - y (array-like): Labels dataset.

        Returns:
        - dict: A dictionary containing the count of elements in each class and the total number of classes.
        """
        # Your code here to analyze class distribution
        # Hint: Consider using collections.Counter or numpy.unique for counting
        # Hint: Consider using collections.Counter or numpy.unique for counting
        class_counts = {}
        uniq, counts = np.unique(y, return_counts=True)
        for label, count in zip(uniq, counts):
            class_counts[label] = count
        
        num_classes = len(class_counts)
        print(f"{uniq=}", uniq)
        print(f"{counts=}", counts)
        print(f"{np.sum(counts)=}", np.sum(counts))

        return {
            "class_counts": class_counts,  # Replace with actual class counts
            "num_classes": num_classes,  # Replace with the actual number of classes
        }



    # --------------------------------------------------------------------------
    """
    A. Using the same classifier and hyperparameters as the one used at the end of part 2.B. 
       Get the accuracies of the training/test set scores using the top_k_accuracy score for k=1,2,3,4,5. 
       Make a plot of k vs. score for both training and testing data and comment on the rate of accuracy change. 
       Do you think this metric is useful for this dataset?
    """

    def partA(
        self,
        Xtrain: NDArray[np.floating],
        ytrain: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """ """
        # Enter code and return the `answer`` dictionary

        answer = {}
        
        k_values = [1, 2, 3, 4, 5]
        plot_k_vs_score_train = []
        plot_k_vs_score_test = []
        
        # Train the classifier
        clf = DecisionTreeClassifier(random_state=self.seed)
        clf.fit(Xtrain, ytrain)
        
        for k in k_values:
            # Calculate top-k accuracy scores for both training and testing data
            score_train = top_k_accuracy_score(ytrain, clf.predict_proba(Xtrain), k=k)
            score_test = top_k_accuracy_score(ytest, clf.predict_proba(Xtest), k=k)
            plot_k_vs_score_train.append((k, score_train))
            plot_k_vs_score_test.append((k, score_test))
            
            # Save scores in the answer dictionary
            answer[k] = {
                "score_train": score_train,
                "score_test": score_test
                }
        
        # plot k vs. score for both training and testing data
        k_train, score_train = zip(*plot_k_vs_score_train)
        k_test, score_test = zip(*plot_k_vs_score_test)

        plt.plot(k_train, score_train, label='Training Data')
        plt.plot(k_test, score_test, label='Testing Data')
        plt.xlabel('k')
        plt.ylabel('Top-k Accuracy Score')
        plt.title('Top-k Accuracy vs. k')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Comment on the rate of accuracy change and the usefulness of this metric
        text_rate_accuracy_change = "The rate of accuracy change for the testing data increases as k values increase."
        text_is_topk_useful_and_why = "This metric is useful because it allows us to evaulate a model's performance across different hyperparameters and identity the optimal parameter values that result in the ighest k accuracy."

        # Update answer dictionary with comments
        answer["clf"] = clf
        answer["plot_k_vs_score_train"] = plot_k_vs_score_train
        answer["plot_k_vs_score_test"] = plot_k_vs_score_test
        answer["text_rate_accuracy_change"] = text_rate_accuracy_change
        answer["text_is_topk_useful_and_why"] = text_is_topk_useful_and_why

        """
        # `answer` is a dictionary with the following keys:
        - integers for each topk (1,2,3,4,5)
        - "clf" : the classifier
        - "plot_k_vs_score_train" : the plot of k vs. score for the training data, 
                                    a list of tuples (k, score) for k=1,2,3,4,5
        - "plot_k_vs_score_test" : the plot of k vs. score for the testing data
                                    a list of tuples (k, score) for k=1,2,3,4,5

        # Comment on the rate of accuracy change for testing data
        - "text_rate_accuracy_change" : the rate of accuracy change for the testing data

        # Comment on the rate of accuracy change
        - "text_is_topk_useful_and_why" : provide a description as a string

        answer[k] (k=1,2,3,4,5) is a dictionary with the following keys: 
        - "score_train" : the topk accuracy score for the training set
        - "score_test" : the topk accuracy score for the testing set
        """

        return answer, Xtrain, ytrain, Xtest, ytest
        # Enter code and return the `answer`` dictionary


    # --------------------------------------------------------------------------
    """
    B. Repeat part 1.B but return an imbalanced dataset consisting of 90% of all 9s removed.  Also convert the 7s to 0s and 9s to 1s.
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}

        # Answer is a dictionary with the same keys as part 1.B
        # Load and prepare the MNIST dataset, filtering out digits 7 and 9
        X, y, Xtest, ytest = u.prepare_data()
        X, y = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        
        # Scale the data matrix between 0 and 1
        success_train, X = nu.scale_data(X)
        success_test, Xtest = nu.scale_data(Xtest)
        
        # Convert 7s to 0s and 9s to 1s in the labels
        y[y == 7] = 0
        y[y == 9] = 1
        ytest[ytest == 7] = 0
        ytest[ytest == 9] = 1
        
        # Print the length of the filtered X and y, and the maximum value of X for both sets
        print("Length of filtered X:", len(X))
        print("Length of filtered y:", len(y))
        print("Maximum value of filtered X:", np.max(X))
        print("Length of filtered Xtest:", len(Xtest))
        print("Length of filtered ytest:", len(ytest))
        print("Maximum value of filtered Xtest:", np.max(Xtest))
        
        # Update the answer dictionary
        answer["length_Xtrain"] = len(X)
        answer["length_Xtest"] = len(Xtest)
        answer["length_ytrain"] = len(y)
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = np.max(X)
        answer["max_Xtest"] = np.max(Xtest)
        
        return answer, X, y, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    C. Repeat part 1.C for this dataset but use a support vector machine (SVC in sklearn). 
        Make sure to use a stratified cross-validation strategy. In addition to regular accuracy 
        also print out the mean/std of the F1 score, precision, and recall. As usual, use 5 splits. 
        Is precision or recall higher? Explain. Finally, train the classifier on all the training data 
        and plot the confusion matrix.
        Hint: use the make_scorer function with the average='macro' argument for a multiclass dataset. 
    """

    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""

        # Enter your code and fill the `answer` dictionary
        answer = {}

        # Intitialize the classifier
        clf = SVC(random_state=self.seed)
        
        # define evaluation metrics
        scoring = {
            'accuracy': 'accuracy',
            'f1': 'f1_macro',
            'precision': make_scorer(precision_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro')
            }
        
        # Perform stratified k-fold cross-validation
        cv = StratifiedKFold(n_splits=5, random_state=self.seed, shuffle=True)
        cv_results = cross_validate(clf, X, y, cv=cv, scoring=scoring, return_train_score=False)
        
        # Extract mean and standard deviation of evaluation metrics
        mean_scores = {metric: np.mean(cv_results[f'test_{metric}']) for metric in scoring.keys()}
        std_scores = {metric: np.std(cv_results[f'test_{metric}']) for metric in scoring.keys()}

        # Determine if precision is higher than recall
        is_precision_higher = mean_scores['precision'] > mean_scores['recall']
        explain_precision_recall_relationship = "Precision is higher than Recall" if is_precision_higher else "Recall is higher than Precision"

        # Train the classifier on all training data
        clf.fit(X, y)
        
        # Compute confusion matrices
        confusion_matrix_train = confusion_matrix(y, clf.predict(X))
        confusion_matrix_test = confusion_matrix(ytest, clf.predict(Xtest))

        # Prepare answer dictionary
        answer['scores'] = {
            'mean_accuracy': mean_scores['accuracy'],
            'mean_recall': mean_scores['recall'],
            'mean_precision': mean_scores['precision'],
            'mean_f1': mean_scores['f1'],
            'std_accuracy': std_scores['accuracy'],
            'std_recall': std_scores['recall'],
            'std_precision': std_scores['precision'],
            'std_f1': std_scores['f1'],
        }

        answer['cv'] = cv
        answer['clf'] = clf
        answer['is_precision_higher_than_recall'] = is_precision_higher
        answer['explain_is_precision_higher_than_recall'] = "Precision is higher than recall when the classifier prioritizes minimizing false positives over false negatives."
        answer['confusion_matrix_train'] = confusion_matrix_train
        answer['confusion_matrix_test'] = confusion_matrix_test


        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "is_precision_higher_than_recall" : a boolean
        - "explain_is_precision_higher_than_recall" : a string
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        
        answer["scores"] is dictionary with the following keys, generated from the cross-validator:
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1
        """

        return answer
        

    # --------------------------------------------------------------------------
    """
    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the class_weights parameter).  Print out the class weights, and comment on the performance difference. Use the `compute_class_weight` argument of the estimator to compute the class weights. 
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}
        
        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        print("Class weights:", class_weights)

        # Initialize classifier
        clf = SVC(random_state=self.seed)

        # Define stratified k-fold cross-validator
        skf = StratifiedKFold(n_splits=5, random_state=self.seed, shuffle=True)

        # Initialize lists to store evaluation metrics
        accuracy_scores = []
        recall_scores = []
        precision_scores = []
        f1_scores = []

        # Initialize confusion matrix
        confusion_matrix_train = np.zeros((2, 2))
        confusion_matrix_test = np.zeros((2, 2))
        
        # Iterate through cross-validation splits
        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Train classifier
            clf.fit(X_train, y_train)

            # Predict on validation set
            y_pred_val = clf.predict(X_val)

            # Update confusion matrix
            confusion_matrix_val = confusion_matrix(y_val, y_pred_val)
            confusion_matrix_train += confusion_matrix_val

            # Calculate evaluation metrics
            accuracy_scores.append(clf.score(X_val, y_val))
            recall_scores.append(recall_score(y_val, y_pred_val))
            precision_scores.append(precision_score(y_val, y_pred_val))
            f1_scores.append(f1_score(y_val, y_pred_val))

        # Compute mean and standard deviation of evaluation metrics
        mean_scores = {
            'accuracy': np.mean(accuracy_scores),
            'recall': np.mean(recall_scores),
            'precision': np.mean(precision_scores),
            'f1': np.mean(f1_scores),
        }
        
        std_scores = {
            'accuracy': np.std(accuracy_scores),
            'recall': np.std(recall_scores),
            'precision': np.std(precision_scores),
            'f1': np.std(f1_scores),
        }

        # Compute confusion matrix for testing set
        y_pred_test = clf.predict(Xtest)
        confusion_matrix_test = confusion_matrix(ytest, y_pred_test)

        # Check if precision is higher than recall
        is_precision_higher = mean_scores['precision'] > mean_scores['recall']

        # Prepare answer dictionary
        answer['scores'] = {
            'mean_accuracy': mean_scores['accuracy'],
            'mean_recall': mean_scores['recall'],
            'mean_precision': mean_scores['precision'],
            'mean_f1': mean_scores['f1'],
            'std_accuracy': std_scores['accuracy'],
            'std_recall': std_scores['recall'],
            'std_precision': std_scores['precision'],
            'std_f1': std_scores['f1'],
        }

        answer['cv'] = skf
        answer['clf'] = clf
        answer['is_precision_higher_than_recall'] = is_precision_higher
        answer['confusion_matrix_train'] = confusion_matrix_train
        answer['confusion_matrix_test'] = confusion_matrix_test


        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "class_weights" : the class weights
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        - "explain_purpose_of_class_weights" : explanatory string
        - "explain_performance_difference" : explanatory string

        answer["scores"] has the following keys: 
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1

        Recall: The scores are based on the results of the cross-validation step
        """

        return answer
        