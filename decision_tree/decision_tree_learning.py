import numpy as np 
import pandas as pd
from random import randrange

class Node:
    
    def __init__(self, attribute):
        self.attribute = attribute
        self.subtree = {}

    def __str__(self):
        return "Node(" + str(self.attribute) + ")"
        

def decision_tree(examples, attributes, parent_examples):
    """
    X: examples, y: classes
    each line is an example
    each number in example is the value of an attribute
    the last number is the class value
    """
    if len(examples) == 0:
        return plurality_value(parent_examples)
    # if all classifiactions are the same
    elif len(examples.iloc[: , -1].unique()) == 1:
        return examples.iloc[: , -1].unique()[0]

    elif len(attributes) == 0:
        return plurality_value(examples)

    else:
        max_imp = 0
        A = -1
        #print(attributes)
        for a in attributes:

           imp = importance(a, examples)
           if imp > max_imp:
               max_imp = imp
               A = a

        node = Node(A)
        tree = node

        print(A)
        aindex = attributes.index(A)
        attr = attributes[:aindex] + attributes[aindex+1:]
        print(attr)
        print(aindex)
        print()
        # Iterate through possible values of the attribute A
        for v in examples.iloc[:, aindex].unique():
            exs = examples.drop(examples[examples.iloc[:,aindex] != v].index)
            subtree = decision_tree(exs, attr, examples) #attributes[:A] + attributes[A+1:]
            
            # add subtree as branch to node
            node.subtree[v] = subtree
            #print(str(v) + " : " + str(subtree))
            
    # returns node
    return tree
# returns the importance of the attribute
def importance(a, examples):
    return randrange(100)
# returns the most common output value among examples
def plurality_value(examples):
    #get most common value in last collumn
    mode = examples.iloc[:,-1:].mode()
    if len(mode.index) == 1:
        return mode.iloc[0][0]
    #print(len(mode.index))
    return mode.iloc[randrange(len(mode.index))][0]

def entropy(counts):
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))

def print_tree(tree):
    print(tree)
    
    for key in tree.subtree:
        print(" ", key, ":" ,tree.subtree[key])
    print("----")

    for key in tree.subtree:
        if(not isinstance(tree.subtree[key],(int, np.integer))):
            print_tree(tree.subtree[key])

def main():
    data_1 = pd.read_csv('train.csv')
    examples0 = data_1#.iloc[:, :-1]
    #classes = data_1.iloc[:,-1:]
    attributes0 = []
    for i in range(len(examples0.columns) - 1): #-1
        attributes0.append(i)
    tree = decision_tree(examples0, attributes0, examples0)
    print("-----------------")
    print_tree(tree)

main()

