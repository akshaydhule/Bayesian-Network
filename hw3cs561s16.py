from __future__ import print_function
from copy import deepcopy
from timeit import itertools
import sys, re

######################################################################
def event_values(evidence, parents):
    if isinstance(evidence, dict) and len(evidence) == len(parents):
        return evidence    
    else:
        ev_tuple = tuple(evidence[var] for var in parents)
        return ev_tuple

######################################################################
class BayesNode:
#A conditional probability distribution P(X | parents). Part of a BayesNet
    def __init__(self, X, parents, cpt, decision = False):
        self.variable = None
        self.parents = None
        self.cpt = None
        self.decision = False
        if isinstance(parents, str): parents = parents.split()
    
        if isinstance(cpt, (float, int)): # no parents, 0-tuple
            cpt = {(): cpt}
            
        elif isinstance(cpt, dict):
            if cpt and isinstance(cpt.keys()[0], bool): # one parent, 1-tuple
                cpt = dict(((v,), p) for v, p in cpt.items())

        assert isinstance(cpt, dict)
#updates the node
        self.cpt = cpt
        self.parents = parents
        self.variable = X
        self.decision = decision
    #----------------------------------------------------------------
#returns the probability
    def p(self, sign, evidence):
        if self.decision == True:
            return 1.0
        temp = 0.0
        if isinstance(sign, bool):
            ptrue = self.cpt[event_values(evidence, self.parents)]
            if(sign == True):
                temp = float(ptrue)
                
            else:
                temp= 1.0-float(ptrue)
                
        return temp

######################################################################            
class BayesNet:
    def __init__(self, f):
        self.bn = []
        self.vars = []
        self.un = []
        self.buildnw(f)

    #----------------------------------------------------------------
#Add node to file 
    def add(self, node):
        if node.variable not in self.vars:
            self.bn.append(node)                                  
            self.vars.append(node.variable)
    #----------------------------------------------------------------
#returns the node
    def variable_node(self, var):
        for n in self.bn:
            if n.variable == var:
                return n
        raise Exception("No such variable: %s" % var)
    #----------------------------------------------------------------
#enumeration calculation
    def enumeration_ask(self, X, e):   
        x, var = X.split(' = ')
        Q = {}
        var = True if var == '+' else False
        
        for xi in [False, True]:
            
            e = deepcopy(e)
            varset = deepcopy(self.vars)
            e[x] = xi
            Q[xi] = self.enumerate_all(varset, e)
        
        return (Q[var]/(Q[var]+Q[not var]))
         
    #----------------------------------------------------------------
#enumeration calculation
    def enumerate_all(self, variables, e):
        
        if not variables:
            return 1.0
        
        Y, rest = variables[0], variables[1:]
        Ynode = self.variable_node(Y)
        
        if Y in e.keys():
            return Ynode.p(e[Y], e) * self.enumerate_all(rest, e)
        
        else:
            total = 0.0
            for y in [False, True]:
                e = deepcopy(e)
                e[Y] = y
                total += (Ynode.p(y, e) * self.enumerate_all(rest, e))
            return total;

    #----------------------------------------------------------------
#joint probability ask function
    def joint_ask(self, q ):
        varlist = deepcopy(self.vars)
        return self.enumerate_all(varlist, q)
    
    #----------------------------------------------------------------
# computation for single parent
    def one_generate(self, e, utility_node):
        varset = deepcopy(self.vars)
        Q = {}
        result = {}
        e[utility_node.parents[0]] = True
        Q[0] = self.enumerate_all(varset, e)
        e[utility_node.parents[0]] = False
        Q[1] = self.enumerate_all(varset, e)
        Q[0] = Q[0]/(Q[0]+Q[1])
        Q[1] = 1 - Q[0]
        result[(True,)] = Q[0] * float(utility_node.cpt[(True,)])
        result[(False,)] = Q[1] * float(utility_node.cpt[(False,)])
        return result
    #----------------------------------------------------------------
# single parent
    def single_parent(self, utility_node, e, known):
        sign = False
        if len(known) == 1:
            sign = True
        result = self.one_generate(e, utility_node)
        if sign == False:
            return float(result[(False,)]) + float(result[(True, )])
        else:
            return float(result[(known[0],)])
    #----------------------------------------------------------------
# computation for two parents    
    def two_generate(self, e, utility_node):
        varset = deepcopy(self.vars)
        Q = {}
        result = {}
        table =list(itertools.product([True,False],repeat = 2))
        for tuple in range(len(table)):
            (i,j) = table[tuple]
            e[utility_node.parents[0]] =  i
            e[utility_node.parents[1]] =  j 
            Q[tuple] = self.enumerate_all(varset, e)
            result[table[tuple]] = Q[tuple] * float(utility_node.cpt[table[tuple]])
        return result
    #----------------------------------------------------------------
#two parents
    def two_parents(self, utility_node, e, known, known_set):
        result = self.two_generate(e, utility_node)
        table =list(itertools.product([True,False],repeat = 2 - len(known)))
        total = 0.0
        if len(known) == 0:
            for tuple in range(len(table)):
                tuple_set = table[tuple]
                total += result [tuple_set]   
            return total
        if len(known) == 1:
            for tuple in range(len(table)):
                (i,) = table[tuple]
                if known_set[0] == True:
                        total += result [(known[0], i)]
                else:
                        total += result [(i, known[0])]
        else:
            total+= result [(known[0], known[1])]
            
        return total
    
    #----------------------------------------------------------------
#computation for three parents
    def three_generate(self, e, utility_node):
        varset = deepcopy(self.vars)
        Q = {}
        result = {}
        table =list(itertools.product([True,False],repeat = 3))
        for tuple in range(len(table)):
            (i,j, k) = table[tuple]
            e[utility_node.parents[0]] =  i
            e[utility_node.parents[1]] =  j 
            e[utility_node.parents[2]] =  k
            Q[tuple] = self.enumerate_all(varset, e)
            result[table[tuple]] = Q[tuple] * float(utility_node.cpt[table[tuple]])
        return result
    #----------------------------------------------------------------
#three parents
    def three_parents(self, utility_node, e, known, known_set):
        result = self.two_generate(e, utility_node)
        table =list(itertools.product([True,False],repeat = 2 - len(known)))
        total = 0.0
        if len(known) == 0:
            for tuple in range(len(table)):
                tuple_set = table[tuple]
                total += result [tuple_set]   
            return total
        if len(known) == 1:
            for tuple in range(len(table)):
                (i,j) = table[tuple]
                if known_set[0] == True:
                    total += result [(known[0], i, j)]
                if known_set[1] == True:
                    total += result [(i, known[0], j)]
                else:
                    total += result [(i, j, known[0])]
                    
        if len(known) == 2:
            for tuple in range(len(table)):
                (i,) = table[tuple]
                if known_set[0] == True and known_set[1] == True:
                    total+= result [(known[0], known[1], i)]
                if known_set[0] == True and known_set[2] == True:
                    total+= result [(known[0], i, known[1])]
                if known_set[1] == True and known_set[2] == True:
                    total+= result [( i,known[0], known[1])]
            
        else :
            total+= result [(known[0], known[1], known[2])]
            
        return total
    #----------------------------------------------------------------
#expected utility agent
    
    def utilityhandler(self, utility_node, e):
        known  = []
        utility_parents = utility_node.parents
        known_set = []
        for utility_parent in utility_parents:
            if utility_parent in  e.keys():
                known.append(e[utility_parent])
                known_set.append(True)
            else:
                known_set.append(False)
                
        if len(utility_parents)==1:
            return self.single_parent(utility_node, e, known)
        
        if len(utility_parents)==2:
            return self.two_parents(utility_node, e, known, known_set)
        
        else:
            return self.three_parents(utility_node, e, known, known_set)
    
    def utilityagent(self, find_value, known_dict, utility_node):
        e = deepcopy(known_dict)
        for find_val in find_value:
            find_val = find_val.split(' = ')
            e[find_val[0]] = True if find_val[1] == '+' else False
             
        return self.utilityhandler(utility_node, e)
        
    #----------------------------------------------------------------
    def one_decision(self, utility_node, e, decision_nodes):
        result = {}
        max_result = 0.0
        sign = None
        for y in [False, True]:
            e1 = deepcopy(e)
            e1[decision_nodes[0]] = y 
            result[(y,)] = self.utilityhandler(utility_node, e1)
            if(result[(y,)] > max_result):
                max_result = result[(y,)]
                sign = '+ ' if y == True else '- ' 
        return sign + str(int(round(max_result)))
    
    def two_decision(self, utility_node, e, decision_nodes):
        result = {}
        max_result = 0.0
        sign1 = None
        sign2 = None
        table =list(itertools.product([True,False],repeat = 2))
        for tuple in range(len(table)):
            (i,j) = table[tuple]
            e1 = deepcopy(e)
            e1[decision_nodes[0]] =  i
            e1[decision_nodes[1]] =  j
            result[tuple] = self.utilityhandler(utility_node, e1)
            #print result[tuple]
            if(result[tuple] > max_result):
                max_result = result[tuple]
                sign1 = '+ ' if i == True else '- '
                sign2 = '+ ' if j == True else '- '
        
        return sign1 + sign2 + str(int(round(max_result)))
    
    def three_decision(self, utility_node, e, decision_nodes):
        result = {}
        max_result = 0.0
        sign1 = None
        sign2 = None
        sign3 = None
        table =list(itertools.product([True,False],repeat = 3))
        for tuple in range(len(table)):
            (i,j,k) = table[tuple]
            e1 = deepcopy(e)
            e1[decision_nodes[0]] =  i
            e1[decision_nodes[1]] =  j
            e1[decision_nodes[2]] =  k
            result[tuple] = self.utilityhandler(utility_node, e1)
            if(result[tuple] > max_result):
                max_result = result[tuple]
                sign1 = '+ ' if i == True else '- '
                sign2 = '+ ' if j == True else '- '
                sign3 = '+ ' if k == True else '- '
                
        return sign1 + sign2 + sign3 + str(int(round(max_result)))
        
    def maxutilityagent(self, find_value, known_dict, utility_node):
        e = deepcopy(known_dict)
        decision_nodes =[]
        for find_val in find_value:
            find_val = find_val.split(' = ')
            #e = {find_val[0] : ()}
            for i in range(len(self.bn)):
                if self.bn[i].decision == True and self.bn[i].variable == find_val[0]:
                    decision_nodes.append(self.bn[i].variable)
            
        
        for i in range(len(self.bn)):
            if(self.bn[i].decision == True and self.bn[i].variable not in decision_nodes):
                decision_nodes.append(self.bn[i].variable)
                 
        if(len(decision_nodes)== 1):
            return self.one_decision(utility_node, e, decision_nodes)
        
        if(len(decision_nodes)== 2):
            return self.two_decision(utility_node, e, decision_nodes)
        
        if(len(decision_nodes)== 3):
            return self.two_decision(utility_node, e, decision_nodes)
        
#read file and build network
    def buildnw(self,f):  
        while True:
            decision = False
            sentence = f.readline()            #reading line
            if sentence == '' or sentence == "******":
                break
            else:
                sentence = sentence.strip('\n')
                if(sentence != "***"):
                    prob_set = {}
                    variable = sentence.strip()
                    variables = variable.split(' | ')
                    #if known conditional probability
                    if len(variables)>1:                              
                        node = str(variables[0]).strip("['").strip("']")
                        parents = str(variables[1:]).strip("['").strip("']").split(' ')
                        
                        while True:
                            truth_statement = f.readline().strip('\n') #number of combinations
                            sentence = truth_statement
                            if truth_statement == '******' or truth_statement == '':
                                break
                            #if there is utility info in BN or End of BN
                            else:
                                if  (truth_statement!= "***"):
                                    prob_val = truth_statement.split(' ')
                                    prob_set[tuple((True if x == '+' else False) for x in prob_val[1:])] = prob_val[0]
                                    
                                else:
                                    break
                                 
                        self.add(BayesNode(node, parents, prob_set, decision))
                
                    else: 
                        #unconditional prob / No parent  
                        node = str(variables).strip("['").strip("']")
                        prob_val = f.readline().strip()
                        if prob_val == 'decision' :    # for decision node in expected utility
                            prob_val = 1.0
                            decision = True
                        prob_set = {():prob_val}
                        self.add(BayesNode(node, [], prob_set, decision))
                        
            if sentence == '' or sentence == "******":
                break            
                            
            
        return self.bn        
            
        
# class network ends here


def query(query_set, q, query_type):
    if(q != "******"):  
        #q = q.strip('P(').strip(')')
        known_val = []
        known_dict = {}
        setval = q.split(' | ')
        
        if len(setval)>1:                                 #if conditional query
            find_val = str(setval[0]).strip("['").strip("']")
            if (query_type== 'MEU' or query_type== 'EU'):
                find_val = find_val.split(', ')
            known_val = str(setval[1:]).strip("['").strip("']").split(', ')
            known_dict = dict((x.split(' = ')[0], True if x.split(' = ')[1] == '+' else False) for x in known_val) #True/false values
            #append in the dict of query set
            tup = (query_type, find_val, known_dict)
            query_set.append(tuple(tup))
        
        else:                                   #if single query or product query
            find_val = q.split(', ')
            if (query_type== 'MEU' or query_type== 'EU'):
                tup = (query_type, find_val , {})
            else:
                find_dict = dict((x.split(' = ')[0], True if x.split(' = ')[1] == '+' else False) for x in find_val) #True/false values
                #append in the dict of query set
                tup = (query_type, find_dict , {})
            query_set.append(tuple(tup))


def Build_utility(f):
    utility_set = {}
    utility_node = None
    while True:
        utility = f.readline().strip('\n')
        if (utility!= ''):
            utility = utility.split(' | ')
            node =  str(utility[0]).strip("['").strip("']")
            parents = str(utility[1:]).strip("['").strip("']")
            
            while True:
                utility_val = f.readline().strip()
                if (utility_val == ''):
                    break
                
                else:
                    utility_val = utility_val.split(' ')
                    utility_set[tuple((True if x == '+' else False) for x in utility_val[1:])] = utility_val[0]
                    
            utility_node =(BayesNode(node, parents, utility_set))
        else:
            break
    return utility_node

######################################################################
def readfile(f):
    query_set = []
    
    while True:
        q = f.readline().strip('\n') #query
        if(q != "******"):
            if q.startswith('P'):
                q = q.strip('P(').strip(')')
                query(query_set, q, 'P')
                
            if q.startswith('EU'):
                q = q.strip('EU(').strip(')')
                query(query_set, q, 'EU')
                    
            if q.startswith('MEU'):
                q = q.strip('MEU(').strip(')')
                query(query_set, q, 'MEU')
        
        else:
            break
    #for building a network
    bn = BayesNet(f)
    #building utility dictionary
    utility_node = Build_utility(f)
    #for every query in n/w
    for query_type, find_value, known_dict in query_set:
        if query_type == 'P':
            if (known_dict == {}):
                value = bn.joint_ask(find_value)
                
            else:
                value = bn.enumeration_ask(find_value, known_dict)
            
            value = format(value, '0.2f')
            print (value)
            
        if query_type == 'EU':
            value = bn.utilityagent(find_value, known_dict, utility_node)
            print (int(round(value)))
            
        if query_type == 'MEU':
            value = bn.maxutilityagent(find_value, known_dict, utility_node)
            print (value)
	

######################################################################
def main():
	orig_stdout = sys.stdout
	fout = file('output.txt', 'w')
	sys.stdout = fout
	fp = open (sys.argv[2],'r')
	#fp = open ("C:\Users\e\Desktop\AI\HW3\HW3_samples\sample05.txt",'r')
	readfile(fp)
	fp.close()
	sys.stdout = orig_stdout
	fout.close()
main()