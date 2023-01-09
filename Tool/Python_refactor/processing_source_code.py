import re, random
from nltk.corpus import wordnet
import wordninja
from util import *
import ast


def word_synonym_replacement(word):
    if len(word) <= 3:
        return word + '_new'
    word_set = wordninja.split(word)
    while True:
        if word_set == []:
            return word + '_new'
        word_tar = random.choice(word_set)
        word_syn = wordnet.synsets(word_tar)
        if word_syn == []:
            word_set.remove(word_tar)
        else:
            break
    word_ret = []
    for syn in word_syn:
        word_ret = word_ret + syn.lemma_names()
        if word_tar in word_ret:
            word_ret.remove(word_tar)
    try:
        word_new = random.choice(word_ret)
    except:
        word_new = word

    return word.replace(word_tar, word_new), word_ret


def extract_method_name(string):
    match_ret = re.search('\w+\s*\(',string)
    if match_ret:
        method_name = match_ret.group()[:-1].strip()
        return method_name
    else:
        return None


def extract_argument(string):
    end_pos    = string.find(')')
    sta_pas    = string.find('(')
    arguments  = string[sta_pas + 1: end_pos + 1].strip()[:-1]
    arguments_list = arguments.split(',')
    if ' ' in arguments_list:
        arguments_list.remove(' ')
    if '' in arguments_list:
        arguments_list.remove('')
    return arguments_list


def extract_brace_python(string, start_pos):
    fragment = string[start_pos:]
    line_list = fragment.split('\n')
    return_string = ''
    return_string += line_list[0] + '\n'
    space_min = 0
    for _ in range(1, len(line_list)):
        space_count = 0
        for char in line_list[_]:
            if char == ' ':
                space_count += 1
            else:
                break
        if _ == 1:
            space_min = space_count
            return_string += line_list[_] + '\n'
        elif space_count < space_min and space_count != len(line_list[_]):
            break
        else:
            return_string += line_list[_] + '\n'
    return_string = return_string[:-1]
    return return_string


def extract_class(string):

    class_list = []
    while ' class ' in string:
        start_pos  = string.find(' class ')
        class_text = extract_brace_python(string, start_pos)
        class_list.append(class_text)
        string = string.replace(class_text, '')

    while 'class ' in string:
        start_pos  = string.find('class ')
        class_text = extract_brace_python(string, start_pos)
        class_list.append(class_text)
        string = string.replace(class_text, '')

    return class_list, string


def extract_function_python(string):
    i = 0
    function_list = []
    # print(string)
    while True:
        match_ret = re.search('(def).+\s*\(', string)
        # print(match_ret)
        if match_ret:
            function_head = match_ret.group()
            start_pos = string.find(function_head)
            function_text = extract_brace_python(string, start_pos)
            function_list.append(function_text)
            string = string.replace(function_text, 'vesal'+ str(i))
            i+=1
        else:
            break
    return function_list, string


def extract_for_loop(string):

    for_list = []
    while True:
        # match_ret = re.search('for\s+\(', string)
        match_ret = re.search(' for ', string)
        if match_ret:
            for_head = match_ret.group()
            start_pos = string.find(for_head)
            for_text = extract_brace_python(string, start_pos)
            for_list.append(for_text)
            string = string.replace(for_text, '')
        else:
            break
    return for_list


def extract_if(string):

    if_list = []
    while True:
        match_ret = re.search(' if ', string)
        if match_ret:
            if_head = match_ret.group()
            start_pos = string.find(if_head)
            if_text = extract_brace_python(string, start_pos)
            if_list.append(if_text)
            string = string.replace(if_text, '')
        else:
            break
    return if_list


def extract_while_loop(string):

    while_list = []
    while True:
        match_ret = re.search(' while ', string)
        if match_ret:
            while_head = match_ret.group()
            start_pos = string.find(while_head)
            while_text = extract_brace_python(string, start_pos)
            while_list.append(while_text)
            string = string.replace(while_text, '')
        else:
            break
    return while_list, string


def hack(source):
    root = ast.parse(source)

    for node in ast.walk(root):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, ast.FunctionDef):
            yield node.name


def extract_local_variable(string):
    return list(hack(string))


if __name__ == '__main__':
    filename = 'test.py'
    open_file = open(filename, 'r', encoding='ISO-8859-1')
    code = open_file.read()
    Class_list, raw_code = extract_class(code)
    print(Class_list)
    for class_name in Class_list:
        function_list, class_name = extract_function_python(class_name)
    print(function_list)
    print(extract_local_variable(function_list[0]))
