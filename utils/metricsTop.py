import numpy as np
from sklearn.metrics import accuracy_score, f1_score

__all__ = ['MetricsTop']


class MetricsTop():
    def __init__(self, train_mode):
        if train_mode == "regression":
            self.metrics_dict = {
                'MOSI': self.__eval_mosi_regression,
                'MOSEI': self.__eval_mosei_regression,
                'SIMS': self.__eval_sims_regression,
                'SIMSV2': self.__eval_sims_regression,
            }
        else:
            self.metrics_dict = {
                'MOSI': self.__eval_mosi_classification,
                'MOSEI': self.__eval_mosei_classification,
                'SIMS': self.__eval_sims_classification,
                'SIMSV2': self.__eval_sims_classification
            }

    def __eval_mosi_classification(self, y_pred, y_true):
        """
        {
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2
        }
        """
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()

        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
        test_preds_a3 = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth_a3 = np.clip(test_truth, a_min=-1., a_max=1.)
        test_truth_a2 = np.clip(test_truth, a_min=0., a_max=1.)
        test_preds_a2 = np.clip(test_preds, a_min=0., a_max=1.)

        # 使用clip到[-3, 3]后的数据计算MAE
        mae = np.mean(np.absolute(test_preds_a7 - test_truth_a7)).astype(np.float64)
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a7 = self.__multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        mult_a2 = self.__multiclass_acc(test_preds_a2, test_truth_a2)

        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        non_zeros_binary_truth = (test_truth[non_zeros] > 0)
        non_zeros_binary_preds = (test_preds[non_zeros] > 0)

        non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
        non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average='weighted')

        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)
        acc2 = accuracy_score(binary_preds, binary_truth)
        f_score = f1_score(binary_truth, binary_preds, average='weighted')

        # eval_results = {
        #     "Has0_acc_2":  round(acc2, 8),
        #     "Has0_F1_score": round(f_score, 8),
        #     "Non0_acc_2":  round(non_zeros_acc2, 8),
        #     "Non0_F1_score": round(non_zeros_f1_score, 8),
        #     "Mult_acc_5": round(mult_a5, 8),
        #     "Mult_acc_7": round(mult_a7, 8),
        #     "MAE": round(mae, 8),
        #     "Corr": round(corr, 8)
        # }

        eval_results = {
            "Acc_7": round(mult_a7, 8),
            "Acc_5": round(mult_a5, 8),
            "Acc_3": round(mult_a3, 8),
            # "Acc_2_Has0":  round(acc2, 8),
            "Acc_2": round(non_zeros_acc2, 8),  # Acc_2_Non0
            # "Mult_acc_2": round(mult_a2, 8),
            "F1_score": round(non_zeros_f1_score, 8),  # Non0_F1_score
            # "F1_score": round(f_score, 8),
            "Corr": round(corr, 8),
            "MAE": round(mae, 8),
        }

        return eval_results

    def __eval_mosei_classification(self, y_pred, y_true):
        return self.__eval_mosi_classification(y_pred, y_true)

    def __eval_sims_classification(self, y_pred, y_true):
        return self.__eval_mosi_classification(y_pred, y_true)

    def __multiclass_acc(self, y_pred, y_true):
        y_pred = np.array(y_pred).astype(int)
        y_true = np.array(y_true).astype(int)
        return np.mean(y_pred == y_true)

    def __eval_mosei_regression(self, y_pred, y_true, exclude_zero=False):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()

        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
        test_preds_a3 = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth_a3 = np.clip(test_truth, a_min=-1., a_max=1.)
        test_truth_a2 = np.clip(test_truth, a_min=0., a_max=1.)
        test_preds_a2 = np.clip(test_preds, a_min=0., a_max=1.)

        mae = np.mean(np.absolute(test_preds - test_truth)).astype(
            np.float64)  # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a7 = self.__multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        mult_a2 = self.__multiclass_acc(test_preds_a2, test_truth_a2)

        # non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        # non_zeros_binary_truth = (test_truth[non_zeros] > 0)
        # non_zeros_binary_preds = (test_preds[non_zeros] > 0)

        # non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
        # non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average='weighted')

        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)
        f_score = f1_score(binary_truth, binary_preds, average='weighted')

        eval_results = {
            "acc_7": round(mult_a7, 8),
            "acc_5": round(mult_a5, 8),
            "acc_3": round(mult_a3, 8),
            "acc_2": round(mult_a2, 8),
            # "Non0_acc_2":  round(non_zeros_acc2, 8),
            # "Non0_F1_score": round(non_zeros_f1_score, 8),
            "F1_score": round(f_score, 8),
            "Corr": round(corr, 8),
            "MAE": round(mae, 8),
        }

        return eval_results

    def __eval_mosi_regression(self, y_pred, y_true):
        return self.__eval_mosei_regression(y_pred, y_true)

    def __eval_sims_regression(self, y_pred, y_true):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()
        test_preds = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth = np.clip(test_truth, a_min=-1., a_max=1.)

        mae = np.mean(np.absolute(test_preds - test_truth)).astype(np.float64)
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        # two classes{[-1.0, 0.0], (0.0, 1.0]}
        ms_2 = [-1.01, 0.0, 1.01]
        test_preds_a2 = test_preds.copy()
        test_truth_a2 = test_truth.copy()
        for i in range(2):
            test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i + 1])] = i
        for i in range(2):
            test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i + 1])] = i

        # three classes{[-1.0, -0.1], (-0.1, 0.1], (0.1, 1.0]}
        ms_3 = [-1.01, -0.1, 0.1, 1.01]
        test_preds_a3 = test_preds.copy()
        test_truth_a3 = test_truth.copy()
        for i in range(3):
            test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i + 1])] = i
        for i in range(3):
            test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i + 1])] = i

        # five classes{[-1.0, -0.7], (-0.7, -0.1], (-0.1, 0.1], (0.1, 0.7], (0.7, 1.0]}
        ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
        test_preds_a5 = test_preds.copy()
        test_truth_a5 = test_truth.copy()
        for i in range(5):
            test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i + 1])] = i
        for i in range(5):
            test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i + 1])] = i

        # seven classes{[-1.0, -0.7], (-0.7, -0.3], (-0.3, -0.1], (-0.1, 0.1], (0.1, 0.3], (0.3, 0.7], (0.7, 1.0]}
        ms_7 = [-1.01, -0.7, -0.3, -0.1, 0.1, 0.3, 0.7, 1.01]
        test_preds_a7 = test_preds.copy()
        test_truth_a7 = test_truth.copy()
        for i in range(7):
            test_preds_a7[np.logical_and(test_preds > ms_7[i], test_preds <= ms_7[i + 1])] = i
        for i in range(7):
            test_truth_a7[np.logical_and(test_truth > ms_7[i], test_truth <= ms_7[i + 1])] = i

        mult_a2 = self.__multiclass_acc(test_preds_a2, test_truth_a2)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        mult_a7 = self.__multiclass_acc(test_preds_a7, test_truth_a7)
        f_score = f1_score(test_truth_a2, test_preds_a2, average='weighted')

        eval_results = {
            "Acc_7": round(mult_a7, 8),
            "Acc_5": round(mult_a5, 8),
            "Acc_3": round(mult_a3, 8),
            "Acc_2": round(mult_a2, 8),
            "F1_score": round(f_score, 8),
            "Corr": round(corr, 8),
            "MAE": round(mae, 8),
        }
        return eval_results

    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]