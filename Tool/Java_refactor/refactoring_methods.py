import os, random, re

from processing_source_code import *


def rename_local_variable(method_string):
    local_var_list = extract_local_variable(method_string)
    if len(local_var_list) == 0:
        return method_string

    mutation_index = random.randint(0, len(local_var_list) - 1)
    return method_string.replace(local_var_list[mutation_index], word_synonym_replacement(local_var_list[mutation_index])[0])


def add_local_variable(method_string):
    local_var_list = extract_local_variable(method_string)
    if len(local_var_list) == 0:
        return method_string

    mutation_index = random.randint(0, len(local_var_list) - 1)
    match_ret      = re.search('.+' + local_var_list[mutation_index] + '.+;', method_string)
    if match_ret:
        var_definition      = match_ret.group()
        new_var_definition  = var_definition.replace(local_var_list[mutation_index], word_synonym_replacement(local_var_list[mutation_index])[0])
        method_string       = method_string.replace(var_definition, var_definition + '\n' + new_var_definition)
        return method_string
    else:
        return method_string


def duplication(method_string):
    local_var_list = extract_local_variable(method_string)
    if len(local_var_list) == 0:
        return method_string
    mutation_index = random.randint(0, len(local_var_list) - 1)
    match_ret = re.search('.+' + local_var_list[mutation_index] + '.+;', method_string)
    if match_ret:
        var_definition = match_ret.group()
        new_var_definition = var_definition
        method_string = method_string.replace(var_definition, var_definition + '\n' + new_var_definition)
        # print(method_string)
        return method_string
    else:
        # print(method_string)
        return method_string


def rename_api(method_string):
    match_ret      = re.findall('\.\s*\w+\s*\(', method_string)
    if match_ret != []:
        api_name = random.choice(match_ret)[1:-1]
        return method_string.replace(api_name,word_synonym_replacement(api_name)[0])
    else:
        return method_string


def rename_method_name(method_string):
    method_name = extract_method_name(method_string)
    if method_name:
        return method_string.replace(method_name, word_synonym_replacement(method_name)[0])
    else:
        return method_string


def rename_argument(method_string):
    arguments_list = extract_argument(method_string)
    if len(arguments_list) == 0:
        return method_string

    mutation_index = random.randint(0, len(arguments_list) - 1)
    # print(method_string.replace(arguments_list[mutation_index],word_synonym_replacement(arguments_list[mutation_index])[0]))
    return method_string.replace(arguments_list[mutation_index],word_synonym_replacement(arguments_list[mutation_index])[0])


def return_optimal(method_string):
    if 'return ' in method_string:
        return_statement  = method_string[method_string.find('return ') : method_string.find(';', method_string.find('return ') + 1)]
        return_object     = return_statement.replace('return ','')
        if return_object == 'null':
            return method_string
        optimal_statement = 'if (' + return_object + ' == null){\n\t\t\treturn 0;\n\t\t}\n' + return_statement
        method_string = method_string.replace(return_statement, optimal_statement)
    return method_string


def enhance_for_loop(method_string):
    for_loop_list = extract_for_loop(method_string)
    if for_loop_list == []:
        return method_string
    mutation_index = random.randint(0, len(for_loop_list) - 1)
    for_text = for_loop_list[mutation_index]
    for_info = for_text[for_text.find('(') + 1 : for_text.find(')')]
    for_body = for_text[for_text.find('{') + 1 : for_text.rfind('}',-1,10)]
    if ':' in for_info:
        loop_bar = for_info.split(':')[-1].strip()
        loop_var = for_info.split(':')[0].strip().split(' ')[-1].strip()
        if loop_bar == None or loop_bar == '' or loop_var == None or loop_var == '':
            return method_string
        new_for_info = 'int i = 0; i < ' + loop_bar + '.size(); i ++'
        method_string = method_string.replace(for_info, new_for_info)
        method_string = method_string.replace(for_body,for_body.replace(loop_var, loop_bar + '.get(i)'))

        return method_string

    else:
        return method_string


def add_print(method_string):
    statement_list = method_string.split(';')
    mutation_index = random.randint(1, len(statement_list) - 1)
    statement      = statement_list[mutation_index]
    new_statement  = '\t' + 'System.out.println("' + str(random.choice(word_synonym_replacement(statement)[1])) + '");'
    method_string = method_string.replace(statement, '\n' + new_statement + '\n' + statement)
    return method_string


def enhance_if(method_string):
    if_list = extract_if(method_string)
    mutation_index = random.randint(0, len(if_list) - 1)
    if_text = if_list[mutation_index]
    if_info = if_text[if_text.find('(') + 1 :if_text.find('{')][:if_text.rfind(')',-1,5) -1]
    new_if_info = if_info
    if 'true' in if_info:
        new_if_info = if_info.replace('true','(0==0)')
    if 'flase' in if_info:
        new_if_info = if_info.replace('flase','(1==0)')
    if '!' in if_info and '!=' not in if_info and '(' not in if_info and '&&' not in if_info and '||' not in if_info:
        new_if_info = if_info.replace('!', 'flase == ')
    if '<' in if_info and '<=' not in if_info and '(' not in if_info and '&&' not in if_info and '||' not in if_info:
        new_if_info = if_info.split('<')[1] + ' > ' + if_info.split('<')[0]
    if '>' in if_info and '>=' not in if_info and '(' not in if_info and '&&' not in if_info and '||' not in if_info:
        new_if_info = if_info.split('>')[1] + ' < ' + if_info.split('>')[0]
    if '<=' in if_info and '(' not in if_info and '&&' not in if_info and '||' not in if_info:
        new_if_info = if_info.split('<=')[1] + ' >= ' + if_info.split('<=')[0]
    if '>=' in if_info and '(' not in if_info and '&&' not in if_info and '||' not in if_info:
        new_if_info = if_info.split('>=')[1] + ' <= ' + if_info.split('>=')[0]
    if '.equals(' in if_info:
        new_if_info = if_info.replace('.equals', '==')

    return method_string.replace(if_info,new_if_info)


def add_argumemts(method_string):
    arguments_list = extract_argument(method_string)
    arguments_info = method_string[method_string.find('(') : method_string.find('{')]
    if len(arguments_list) == 0:
        arguments_info = 'String ' + word_synonym_replacement(extract_method_name(method_string))[0]
        return method_string[0 : method_string.find('()') + 1] + arguments_info + method_string[method_string.find('()') + 1 :]
    mutation_index = random.randint(0, len(arguments_list) - 1)
    org_argument = arguments_list[mutation_index]
    new_argument = word_synonym_replacement(arguments_list[mutation_index])[0]
    new_arguments_info = arguments_info.replace(org_argument,org_argument + ', ' + new_argument)
    method_string = method_string.replace(arguments_info,new_arguments_info)
    return method_string


def enhance_filed(method_string):
    arguments_list = extract_argument(method_string)
    if len(arguments_list) == 0:
        return method_string
    mutation_index = random.randint(0, len(arguments_list) - 1)
    extra_info = "\n\tif (" + arguments_list[mutation_index].strip().split(' ')[-1] + " == null){\n\t\tSystem.out.println('please check your input');\n\t}"
    method_string = method_string[0 : method_string.find(';') + 1] + extra_info + method_string[method_string.find(';') + 1 : ]
    return method_string


def apply_plus_zero_math(data):

    statement_list = data.split(';')
    mutation_index = random.randint(1, len(statement_list) - 1)
    statement = statement_list[mutation_index]

    tree = get_tree(data)
    var_list = get_local_vars(tree)
    var_list = [var for var, var_type in var_list if var_type in (
        "int", "float", "double", "long")]
    if var_list==[]:
        return ""
    for var in var_list:
        mutant = ' ' + str(var) + ' = ' + str(var) + ' + ' + str(0) + ";"

        for idx, _ in enumerate(statement_list):
            if var in _ and idx < len(statement_list) - 1:
                insertion_index = idx + 1
        method_string = data.replace(statement, '\n' + mutant + '\n' + statement)
    return method_string




def dead_branch_if_else(data):
    statement_list = data.split(';')
    mutation_index = random.randint(1, len(statement_list) - 1)
    statement = statement_list[mutation_index]
    new_statement = get_branch_if_else_mutant()
    method_string = data.replace(statement, '\n' + new_statement + '\n' + statement)
    # print(method_string)
    return method_string
    # return data


def dead_branch_if(data):

    statement_list = data.split(';')
    mutation_index = random.randint(1, len(statement_list) - 1)
    statement = statement_list[mutation_index]
    new_statement = get_branch_if_mutant()
    method_string = data.replace(statement, '\n' + new_statement + '\n' + statement)
    return method_string


def dead_branch_while(data):

    statement_list = data.split(';')
    mutation_index = random.randint(1, len(statement_list) - 1)
    statement = statement_list[mutation_index]
    new_statement = get_branch_while_mutant()
    method_string = data.replace(statement, '\n' + new_statement + '\n' + statement)
    return method_string


def dead_branch_for(data):

    statement_list = data.split(';')
    mutation_index = random.randint(1, len(statement_list) - 1)
    statement = statement_list[mutation_index]
    new_statement = get_branch_for_mutant()
    method_string = data.replace(statement, '\n' + new_statement + '\n' + statement)
    return method_string


def dead_branch_switch(data):

    statement_list = data.split(';')
    mutation_index = random.randint(1, len(statement_list) - 1)
    statement = statement_list[mutation_index]
    new_statement = get_branch_switch_mutant()
    method_string = data.replace(statement, '\n' + new_statement + '\n' + statement)
    return method_string



if __name__ == '__main__':
    filename = '**.java'
    open_file = open(filename, 'r', encoding='ISO-8859-1')
    code = open_file.read()
    Class_list, raw_code = extract_class(code)
    for class_name in Class_list:
        function_list, class_name = extract_function(class_name)
    candidate_code = function_list[0]
    print(candidate_code)
    new_code = enhance_for_loop(candidate_code)
    print(new_code)

