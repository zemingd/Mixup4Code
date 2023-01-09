import javalang
import secrets
import random
import json

from os import listdir
from os.path import isfile, join


def get_radom_var_name():
    res_string = ''
    for x in range(8):
        res_string += random.choice('abcdefghijklmnopqrstuvwxyz')
    return res_string


def get_dead_for_condition():
    var = get_radom_var_name()
    return "int "+var+" = 0; "+var+" < 0; "+var+"++"


def get_random_false_stmt():
    res = [random.choice(["True", "False"]) for x in range(10)]
    res.append("False")
    res_str = " and ".join(res)
    return res_str


def get_tree(data):
    tokens = javalang.tokenizer.tokenize(data)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    return tree


def verify_method_syntax(data):
    try:
        tokens = javalang.tokenizer.tokenize(data)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
        print("syantax check passed")
    except:
        print("syantax check failed")


def get_random_type_name_and_value_statment():
    datatype = random.choice(
        'byte,short,int,long,float,double,boolean,char,String'.split(','))
    var_name = get_radom_var_name()

    if datatype == "byte":
        var_value = get_random_int(-128, 127)
    elif datatype == "short":
        var_value = get_random_int(-10000, 10000)
    elif datatype == "boolean":
        var_value = random.choice(["True", "False"])
    elif datatype == "char":
        var_value = str(random.choice(
            'a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z'.split(',')))
        var_value = '"'+var_value+'"'
    elif datatype == "String":
        var_value = str(get_radom_var_name())
        var_value = '"'+var_value+'"'
    else:
        var_value = get_random_int(-1000000000, 1000000000)

    mutant = str(var_name) + ' = ' + str(var_value)
    return mutant


def generate_file_name_list_file_from_dir(method_path):
    filenames = [f for f in listdir(
        method_path) if isfile(join(method_path, f))]
    with open(method_path+'\\'+'all_file_names.txt', 'w') as f:
        f.write(json.dumps(filenames))
    print("done")


def get_file_name_list(method_path):
    with open(method_path+'\\'+'all_file_names.txt') as f:
        data = json.load(f)
    return data


def get_random_int(min, max):
    return random.randint(min, max)


def format_code_chuncks(code_chuncks):
    for idx, c in enumerate(code_chuncks):
        c = c.replace(' . ', '.')
        c = c.replace(' ( ', '(')
        c = c.replace(' ) ', ')')
        c = c.replace(' ;', ';')
        c = c.replace('[ ]', '[]')
        code_chuncks[idx] = c
    return code_chuncks


def format_code(c):
    c = c.replace(' . ', '.')
    c = c.replace(' ( ', '(')
    c = c.replace(' ) ', ')')
    c = c.replace(' ;', ';')
    c = c.replace('[ ]', '[]')
    return c


def get_method_header(string):
    method_header = ''
    tree = get_tree(string)

    tokens = list(javalang.tokenizer.tokenize(string))

    chunck_start_poss = [s.position.column for s in tree.body]
    if len(chunck_start_poss) > 0:
        method_header = ' '.join([t.value for t in tokens
                                  if t.position.column < chunck_start_poss[0]])

    method_header = format_code_chuncks([method_header])[0]
    return method_header


def get_method_statement(string):
    code_chuncks = []
    tree = get_tree(string)
    tokens = list(javalang.tokenizer.tokenize(string))
    chunck_start_poss = [s.position.column for s in tree.body]

    if len(chunck_start_poss) > 1:
        for idx, statement in enumerate(chunck_start_poss[:-1]):
            statment = ' '.join([t.value for t in tokens
                                 if t.position.column >= chunck_start_poss[idx]
                                 and t.position.column < chunck_start_poss[idx+1]])
            code_chuncks.append(statment)

        last_statment = ' '.join([t.value for t in tokens
                                  if t.position.column >= chunck_start_poss[-1]][:-1])
        code_chuncks.append(last_statment)

    if len(chunck_start_poss) == 1:
        last_statment = ' '.join([t.value for t in tokens
                                  if t.position.column >= chunck_start_poss[0]][:-1])
        code_chuncks.append(last_statment)
    code_chuncks = format_code_chuncks(code_chuncks)
    return code_chuncks


def scan_tree(tree):
    for path, node in tree:
        print("=======================")
        print(node)


def get_all_type(tree):
    res_list=[]
    for path, node in tree.filter(javalang.tree.ReferenceType):
        if node.name != None:
            res_list.append(node.name)
    return list(set(res_list))


def scan_local_vars(tree):
    for path, node in tree.filter(javalang.tree.LocalVariableDeclaration):
        print("name=========type=============")
        print(node.declarators[0].name, "\t", node.type.name)


def get_local_vars(tree):
    var_list = []
    for path, node in tree.filter(javalang.tree.LocalVariableDeclaration):
        var_list.append([node.declarators[0].name, node.type.name])
    return var_list


def get_local_assignments(tree):
    var_list = []
    for path, node in tree.filter(javalang.tree.Assignment):
        var_list.append([node.declarators[0].name, node.type.name])
    return var_list


def get_branch_if_else_mutant():
    mutant = get_random_type_name_and_value_statment() + ' if '+get_random_false_stmt() + ' else ' + str(get_random_int(-1000000000, 1000000000))
    return mutant


def get_branch_if_mutant():
    mutant = 'if '+get_random_false_stmt()+': ' + \
        get_random_type_name_and_value_statment()
    return mutant


def get_branch_while_mutant():
    mutant = 'while '+get_random_false_stmt()+': ' + \
        get_random_type_name_and_value_statment()
    return mutant


def get_branch_for_mutant():
    var = get_radom_var_name()
    mutant = 'for ' + var + ' in range(0): ' + get_random_type_name_and_value_statment()
    return mutant


