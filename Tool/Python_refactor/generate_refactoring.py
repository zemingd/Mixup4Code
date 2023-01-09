import os, random
from shutil import copyfile
from refactoring_methods import *


def return_function_code(code, method_names):
    final_codes = []
    final_names = []
    Class_list, raw_code = extract_class(code)
    for class_name in Class_list:
        function_list, class_name = extract_function(class_name)
    for fun_code in function_list:
        for method_name in method_names:
            method_name_tem = method_name.replace('|', '')
            if method_name_tem.upper() in fun_code.split('\n')[0].upper():

                final_codes.append(fun_code)
                final_names.append(method_name)
    return final_codes, final_names


def generate_adversarial(k, code, method_names):
        method_name = method_names[0]
        function_list = []
        class_name = ''
        Class_list, raw_code = extract_class(code)
        for class_name in Class_list:
            function_list, class_name = extract_function(class_name)

        refac = []
        new_refactored_code = ''
        for code in function_list:
            if method_name not in code.split('\n')[0]:
                continue
            new_rf = code
            new_refactored_code = code
            for t in range(k):
                refactors_list = [rename_argument,
                                  return_optimal,
                                  add_argumemts,
                                  rename_api,
                                  rename_local_variable,
                                  add_local_variable,
                                  rename_method_name,
                                  enhance_if,
                                  add_print,
                                  duplication,
                                  apply_plus_zero_math,
                                  dead_branch_if_else,
                                  dead_branch_if,
                                  dead_branch_while,
                                  dead_branch_for,
                                  # dead_branch_switch
                                  ]#
                vv = 0
                while new_rf == new_refactored_code and vv <= 20:
                    try:
                        vv += 1
                        refactor       = random.choice(refactors_list)
                        print('*'*50 , refactor , '*'*50)
                        new_refactored_code = refactor(new_refactored_code)

                    except Exception as error:
                        print('error:\t', error)

                new_rf = new_refactored_code
                print('----------------------------OUT of WHILE----------------------------------', vv)
                print('----------------------------CHANGED THJIS TIME:----------------------------------', vv)
            refac.append(new_refactored_code)
        code_body = raw_code.strip() + ' ' + class_name.strip()
        for i in range(len(refac)):
            final_refactor = code_body.replace('vesal' + str(i), str(refac[i]))
            code_body = final_refactor
        return new_refactored_code


def generate_adversarial_json(k, code):
    final_refactor = ''
    function_list = []
    class_name = ''
    vv = 0
    if len(function_list) == 0:
        function_list.append(code)
    refac = []
    for code in function_list:
        new_rf = code
        new_refactored_code = code
        for t in range(k):
            refactors_list = [rename_argument,
                              return_optimal,
                              add_argumemts,
                              rename_api,
                              rename_local_variable,
                              add_local_variable,
                              rename_method_name,
                              enhance_if,
                              add_print,
                              duplication,
                              apply_plus_zero_math,
                              dead_branch_if_else,
                              dead_branch_if,
                              dead_branch_while,
                              dead_branch_for,
                              # dead_branch_switch
                              ]  
            vv = 0
            while new_rf == new_refactored_code and vv <= 20:
                try:
                    vv += 1
                    refactor = random.choice(refactors_list)
                    print('*' * 50, refactor, '*' * 50)
                    new_refactored_code = refactor(new_refactored_code)

                except Exception as error:
                    print('error:\t', error)

            new_rf = new_refactored_code
        refac.append(new_refactored_code)

    print("refactoring finished")
    return refac


def generate_adversarial_file_level(k, code):
        new_refactored_code = ''
        new_rf = code
        new_refactored_code = code
        for t in range(k):
            refactors_list = [
                              rename_argument, 
                              return_optimal, 
                              add_argumemts,
                              rename_api, 
                              rename_local_variable,
                              add_local_variable,
                              rename_method_name,
                              enhance_if,
                              add_print,
                              duplication,
                              apply_plus_zero_math,
                              dead_branch_if_else,
                              dead_branch_if,
                              dead_branch_while,
                              dead_branch_for
                              ]  
            vv = 0
            while new_rf == new_refactored_code and vv <= 20:
                try:
                    vv += 1
                    refactor = random.choice(refactors_list)
                    print('*' * 50, refactor, '*' * 50)
                    new_refactored_code = refactor(new_refactored_code)
                except Exception as error:
                    print('error:\t', error)
            new_rf = new_refactored_code
        return new_refactored_code


if __name__ == '__main__':
    K = 1
    filename = '**.py'
    open_file = open(filename, 'r', encoding='ISO-8859-1')
    code = open_file.read()
    new_code = generate_adversarial_file_level(K, code)
    print(new_code)
