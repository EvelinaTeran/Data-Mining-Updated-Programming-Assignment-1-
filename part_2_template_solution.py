# Add your imports here.
# Note: only sklearn, numpy, utils and new_utils are allowed.

import numpy as np
from numpy.typing import NDArray
from typing import Any
import new_utils as nu
import utils as u

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, KFold
# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.


class Section2:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None
        """
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train

    # ---------------------------------------------------------

    """
    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
        all 10 classes by also printing out the number of elements in each class y and 
        print out the number of classes for both training and testing datasets. 
    """

    def partA(
        self,
    ) -> tuple[
        dict[str, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        answer = {}
        # Enter your code and fill the `answer`` dictionary
        # Implementing part 1.B
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        success_train, Xtrain = nu.scale_data(Xtrain)
        success_train, Xtest = nu.scale_data(Xtest)
        
        # Calculate the number of elements in each class for both training and testing datasets
        class_count_train = nu._count_elements(ytrain)
        class_count_test = nu._count_elements(ytest)
        
        # Convert dictionaries to lists
        class_count_train_list = list(class_count_train.values())
        class_count_test_list = list(class_count_test.values())

        
        # Set number of classes in the training and testing sets
        nb_classes_train = len(class_count_train)
        nb_classes_test = len(class_count_test)
        
        # Print the number of elements in each class y and the number of classes
        print("Number of elements in each class (training set):", class_count_train)
        print("Number of classes (training set):", nb_classes_train)
        print("Number of elements in each class (testing set):", class_count_test)
        print("Number of classes (testing set):", nb_classes_test)

        # Set number of elements in the training and testing sets
        # length_Xtrain = Xtrain.shape[0]
        # length_Xtest = Xtest.shape[0]

        length_Xtrain = len(Xtrain)
        length_Xtest = len(Xtest)

        # Set number of labels in the training and testing sets
        length_ytrain = len(ytrain)
        length_ytest = len(ytest)

        # Set maximum value in the training and testing sets
        max_Xtrain = np.max(Xtrain)
        max_Xtest = np.max(Xtest)
        
        print("Length of Xtrain:", length_Xtrain)
        print("Length of Xtest:", length_Xtest)
        print("Length of ytrain:", length_ytrain)
        print("Length of ytest:", length_ytest)
        print("Maximum value of Xtrain:", max_Xtrain)
        print("Maximum value of Xtest:", max_Xtest)

        # `answer` is a dictionary with the following keys:
        # - nb_classes_train: number of classes in the training set
        # - nb_classes_test: number of classes in the testing set
        # - class_count_train: number of elements in each class in the training set
        # - class_count_test: number of elements in each class in the testing set
        # - length_Xtrain: number of elements in the training set
        # - length_Xtest: number of elements in the testing set
        # - length_ytrain: number of labels in the training set
        # - length_ytest: number of labels in the testing set
        # - max_Xtrain: maximum value in the training set
        # - max_Xtest: maximum value in the testing set

        answer["nb_classes_train"] = nb_classes_train
        answer["nb_classes_test"] = nb_classes_test
        answer["class_count_train"] = class_count_train_list
        answer["class_count_test"] = class_count_test_list
        answer["length_Xtrain"] = length_Xtrain
        answer["length_Xtest"] = length_Xtest
        answer["length_ytrain"] = length_ytrain
        answer["length_ytest"] = length_ytest
        answer["max_Xtrain"] = max_Xtrain
        answer["max_Xtest"] = max_Xtest

        # return values:
        # Xtrain, ytrain, Xtest, ytest: the data used to fill the `answer`` dictionary

        Xtrain = Xtest = np.zeros([1, 1], dtype="float")
        ytrain = ytest = np.zeros([1], dtype="int")

        return answer, Xtrain, ytrain, Xtest, ytest

    """
    B.  Repeat part 1.C, 1.D, and 1.F, for the multiclass problem. 
        Use the Logistic Regression for part F with 300 iterations. 
        Explain how multi-class logistic regression works (inherent, 
        one-vs-one, one-vs-the-rest, etc.).
        Repeat the experiment for ntrain=1000, 5000, 10000, ntest = 200, 1000, 2000.
        Comment on the results. Is the accuracy higher for the training or testing set?
        What is the scores as a function of ntrain.

        Given X, y from mnist, use:
        Xtrain = X[0:ntrain, :]
        ytrain = y[0:ntrain]
        Xtest = X[ntrain:ntrain+test]
        ytest = y[ntrain:ntrain+test]
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
        ntrain_list: list[int] = [],
        ntest_list: list[int] = [],
    ) -> dict[int, dict[str, Any]]:
        """ """
        
        # Enter your code and fill the `answer`` dictionary
        answer = {}

        for ntrain in ntrain_list:
            if ntrain not in answer:
                answer[ntrain] = {}
            # selecting training samples
            Xtrain = X[:ntrain]
            ytrain = y[:ntrain]
            #class_count_train = np.array([np.sum(ytrain == c) for c in np.unique(y)])
            # class_count_train = list(nu._count_elements(ytrain).values())
            class_count_train = nu._count_elements(ytrain)
            
            for ntest in ntest_list:
                # selecting testing samples 
                Xtest_subset = Xtest[ntrain:ntrain+ntest]
                ytest_subset = ytest[ntrain:ntrain+ntest]
                #class_count_test = np.array([np.sum(ytest_subset == c) for c in np.unique(y)])
                #class_count_test = list(nu._count_elements(ytest_subset).values())
                class_count_test = nu._count_elements(ytest_subset)
                
                
                # Part C: Decision Tree Classifier with k-fold cross validation
                clf_c = DecisionTreeClassifier(random_state=self.seed)
                kf_c = KFold(n_splits=5, random_state=self.seed, shuffle=True)
                scores_c = u.train_simple_classifier_with_cv(Xtrain=Xtrain, ytrain=ytrain, clf=clf_c, cv=kf_c)

                print("Mean accuracy scores:", scores_c['test_score'].mean())
                print("Standard deviation of accuracy scores:", scores_c['test_score'].std())

                # Part D: Decision Tree classifier with shuffle-split cross validation
                clf_d = DecisionTreeClassifier(random_state=self.seed)
                ss_d = ShuffleSplit(n_splits=5, random_state=self.seed, test_size=self.frac_train)
                scores_d = u.train_simple_classifier_with_cv(Xtrain=Xtrain, ytrain=ytrain, clf=clf_d, cv=ss_d)

                print("Mean accuracy scores:", scores_d['test_score'].mean())
                print("Standard deviation of accuracy scores:", scores_d['test_score'].std())
                
                # Part F: Logistic Regression with 300 iterations
                clf_f = LogisticRegression(max_iter=300, random_state=self.seed)
                ss_f = ShuffleSplit(n_splits=5, random_state=self.seed, test_size=self.frac_train)
                scores_f = u.train_simple_classifier_with_cv(Xtrain=Xtrain, ytrain=ytrain, clf=clf_f, cv=ss_f)

                # Comparing results from part D and part F
                # Determine the model with the highest accuracy on average
                model_highest_accuracy = "Decision Tree" if scores_d['test_score'].mean() > scores_f['test_score'].mean() else "Logistic Regression"
                print(model_highest_accuracy)
                
                # Determine which model has the lowest variance on average
                # model_lowest_variance = "Random Forest" if scores_RF['test_score'].std() < scores_DT['test_score'].std() else "Decision Tree"
                model_lowest_variance = min(scores_d['test_score'].std(), scores_f['test_score'].std())
                print(model_lowest_variance)
                
                # Determine which mdel is faster to train
                # model_fastest = "Random Forest" if scores_RF['fit_time'].mean() < scores_DT['fit_time'].mean() else "Decision Tree"
                model_fastest = min(scores_d['fit_time'].mean(), scores_f['fit_time'].mean())
                print(model_fastest)

                answer[ntrain][ntest] = {"partC": {"clf": clf_c, "cv": kf_c, "scores": scores_c},
                                  "partD": {"clf": clf_d, "cv": ss_d, "scores": scores_d},
                                  "partF": {"clf": clf_f, "cv": ss_f, "scores": scores_f},
                                  "ntrain": ntrain, 
                                  "ntest": ntest, 
                                  "class_count_train": list(class_count_train.values()), 
                                  "class_count_test": list(class_count_test.values())
                                  }

        """
        `answer` is a dictionary with the following keys:
           - 1000, 5000, 10000: each key is the number of training samples

           answer[k] is itself a dictionary with the following keys
            - "partC": dictionary returned by partC section 1
            - "partD": dictionary returned by partD section 1
            - "partF": dictionary returned by partF section 1
            - "ntrain": number of training samples
            - "ntest": number of test samples
            - "class_count_train": number of elements in each class in
                               the training set (a list, not a numpy array)
            - "class_count_test": number of elements in each class in
                               the training set (a list, not a numpy array)
        """

        return answer

