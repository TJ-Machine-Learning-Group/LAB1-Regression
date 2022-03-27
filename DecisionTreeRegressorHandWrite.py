from copy import copy
import numpy as np
from numpy import ndarray
class Node:
    attr_names = ("avg", "left", "right", "feature", "split", "mse")
    def __init__(self, avg=None, left=None, right=None, feature=None, split=None, mse=None):
        self.avg = avg
        self.left = left
        self.right = right
        self.feature = feature
        self.split = split
        self.mse = mse

    def __str__(self):
        ret = []
        for attr_name in self.attr_names:
            attr = getattr(self, attr_name)
            if attr is None:
                continue
            if isinstance(attr, Node):
                des = "%s: Node object." % attr_name
            else:
                des = "%s: %s" % (attr_name, attr)
            ret.append(des)

        return "\n".join(ret) + "\n"

    def copy(self, node):
        for attr_name in self.attr_names:
            attr = getattr(node, attr_name)
            setattr(self, attr_name, attr)


class DecisionTreeRegressorHandWrite:
    def __init__(self):
        self.root = Node()
        self.depth = 1
        self._rules = None

    def __str__(self):
        ret = []
        for i, rule in enumerate(self._rules):
            literals, avg = rule

            ret.append("Rule %d: " % i + ' | '.join(
                literals) + ' => y_hat %.4f' % avg)
        return "\n".join(ret)

    @staticmethod
    def _expr2literal(expr: list) -> str:
        feature, operation, split = expr
        operation = ">=" if operation == 1 else "<"
        return "Feature%d %s %.4f" % (feature, operation, split)

    def get_rules(self):
        que = [[self.root, []]]
        self._rules = []
        while que:
            node, exprs = que.pop(0)
            if not(node.left or node.right):
                
                literals = list(map(self._expr2literal, exprs))
                self._rules.append([literals, node.avg])
            if node.left:
                rule_left = copy(exprs)
                rule_left.append([node.feature, -1, node.split])
                que.append([node.left, rule_left])
            if node.right:
                rule_right = copy(exprs)
                rule_right.append([node.feature, 1, node.split])
                que.append([node.right, rule_right])

    @staticmethod
    def _get_split_mse(col: ndarray, label: ndarray, split: float) -> Node:
        label_left = label[col < split]
        label_right = label[col >= split]
        avg_left = label_left.mean()
        avg_right = label_right.mean()
        mse = (((label_left - avg_left) ** 2).sum() +
               ((label_right - avg_right) ** 2).sum()) / len(label)
        node = Node(split=split, mse=mse)
        node.left = Node(avg_left)
        node.right = Node(avg_right)
        return node

    def _choose_split(self, col: ndarray, label: ndarray) -> Node:
        node = Node()
        unique = set(col)
        if len(unique) == 1:
            return node
        unique.remove(min(unique))
        ite = map(lambda x: self._get_split_mse(col, label, x), unique)
        node = min(ite, key=lambda x: x.mse)
        return node

    def _choose_feature(self, data: ndarray, label: ndarray) -> Node:
        _ite = map(lambda x: (self._choose_split(data[:, x], label), x),
                   range(data.shape[1]))
        ite = filter(lambda x: x[0].split is not None, _ite)
        node, feature = min(
            ite, key=lambda x: x[0].mse, default=(Node(), None))
        node.feature = feature

        return node

    def fit(self, data: ndarray, label: ndarray, max_depth=12, min_samples_split=2):
        self.__init__()
        self.root.avg = label.mean()
        que = [(self.depth + 1, self.root, data, label)]
        while que:
            depth, node, _data, _label = que.pop(0)
            if depth > max_depth:
                depth -= 1
                break
            if len(_label) < min_samples_split or all(_label == label[0]):
                continue
            _node = self._choose_feature(_data, _label)
            if _node.split is None:
                continue
            node.copy(_node)
            idx_left = (_data[:, node.feature] < node.split)
            idx_right = (_data[:, node.feature] >= node.split)
            que.append(
                (depth + 1, node.left, _data[idx_left], _label[idx_left]))
            que.append(
                (depth + 1, node.right, _data[idx_right], _label[idx_right]))

        self.depth = depth#更新树深度
        self.get_rules()#更新树规则

    def predict_one(self, row: ndarray) -> float:
        node = self.root
        while node.left and node.right:
            if row[node.feature] < node.split:
                node = node.left
            else:
                node = node.right

        return node.avg

    def predict(self, data: ndarray) -> ndarray:
        return np.apply_along_axis(self.predict_one, 1, data)
    
    def score(reg, X, y):
        if isinstance(y, list):
            y = np.array(y)
        y_hat = reg.predict(X)#预测值
        r2= 1 - ((y - y_hat)**2).sum() / ((y - y.mean())**2).sum()#R2
        return r2

from Data_preprocessing import Data_preprocessing
from Regression import Regression

if __name__ == "__main__":
    data,target=Data_preprocessing("./Concrete_Data.xls")
    model = DecisionTreeRegressorHandWrite()
    Regression(model, data, target, splits=1, size=0.2)

