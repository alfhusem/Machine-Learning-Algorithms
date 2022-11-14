import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class DecisionTree:
    
    def __init__(self):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        
        self.tree = []
        self.branch = []
        self.root_node = None
        
    
    def fit(self, X, y):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        
        node = self.find_node(X, y)
        
        if self.root_node == None:
            self.root_node = node

        for value in X[node].unique(): #eg Sunny, overcast, rain
            Xy = X.copy()
            Xy['Values'] = y
            Xy_values = Xy[Xy[node] == value]
            y_values = Xy_values['Values'].reset_index(drop=True)
            X_values = Xy_values.drop(columns=[node, 'Values'])
            
            if node == self.root_node:
                self.branch.append( (node, value) )
            else:
                if len(self.tree) > 0:
                    for prev in self.tree[-1][0]:
                        if len(list(filter( 
                            lambda p: (p[0] == node), 
                            self.tree[-1][0] 
                        ))) > 0 and prev[0] != node:               

                            self.branch.append( (prev[0], prev[1]) )
                            
                        else:
                            self.branch.append( (node, value) )

                else:

                    self.branch.append( (node, value) )
                    
            if entropy(y_values.value_counts()) == 0:

                self.tree.append( (self.branch, y_values[0]) )

                self.branch = []

            elif entropy(y_values.value_counts()) > 0:

                self.fit(X_values, y)
        
        return
        
        
       
            
    def entropy_value(self, X, y, attr, value):
        X1 = X.copy()
        X1['Values'] = y
        X_attr = X1[X1[attr] == value]
        y_attr = X_attr['Values']
        return entropy(y_attr.value_counts())

    def probability(self, X, attr, value):

        numerator = 0
        denominator = 0
        for i in X.index:
            if X[attr][i] == value: #ex: if Outlook==Sunny
                numerator +=1
            denominator += 1

        if denominator != 0:
            return numerator/denominator
        return 0

    def get_gain(self, X, y, attr):
        gain = 0
        for value in X[attr].unique():
            gain += self.probability(X, attr, value)*self.entropy_value(X, y, attr, value)
        return entropy(y.value_counts()) - gain

    def find_node(self, X, y):
        highest_gain = 0
        root_node = ''
        for attr in X:
            gain = self.get_gain(X, y, attr)
            if gain > highest_gain:
                highest_gain = gain
                root_node = attr
        return root_node
            
            
        
    
    def pred(self, X, tree, attb_list):
        longest_rule = [[],[]]
        for rule in tree:
            rule_len = len(rule[0])
            match_count = 0
            for node in rule[0]:
                if node[0] in attb_list:
                    feature = X[node[0]]
                    if node[1] == feature:
                        match_count += 1
            if (match_count == rule_len):
                return rule[1]
            elif 1/(rule_len - match_count) + rule_len > len(longest_rule[0]):
                longest_rule = rule
        
        return longest_rule[1]


    def predict(self, X):
        """       
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        pred_list=[]
        attr_list=[i for i in X]
        for index, row in X.iterrows():
            pred_list.append(self.pred(row, self.tree, attr_list))
        return np.asarray(pred_list)

        
        
    
    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        return self.tree


# --- Some utility functions 
    
def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))


