import re, random
from nltk.corpus import wordnet
import wordninja

from util import *

reserved_kws = ["abstract", "assert", "boolean",
                "break", "byte", "case", "catch", "char", "class", "const",
                "continue", "default", "do", "double", "else", "extends", "false",
                "final", "finally", "float", "for", "goto", "if", "implements",
                "import", "instanceof", "int", "interface", "long", "native",
                "new", "null", "package", "private", "protected", "public",
                "return", "short", "static", "strictfp", "super", "switch",
                "synchronized", "this", "throw", "throws", "transient", "true",
                "try", "void", "volatile", "while"]

reserved_cls = ["ArrayDeque", "ArrayList", "Arrays", "BitSet", "Calendar", "Collections", "Currency",
                "Date", "Dictionary", "EnumMap", "EnumSet", "Formatter", "GregorianCalendar", "HashMap",
                "HashSet", "Hashtable", "IdentityHashMap", "LinkedHashMap", "LinkedHashSet",
                "LinkedList", "ListResourceBundle", "Locale", "Observable",
                "PriorityQueue", "Properties", "PropertyPermission",
                "PropertyResourceBundle", "Random", "ResourceBundle", "ResourceBundle.Control",
                "Scanner", "ServiceLoader", "SimpleTimeZone", "Stack",
                "StringTokenizer", "Timer", "TimerTask", "TimeZone",
                "TreeMap", "TreeSet", "UUID", "Vector", "WeakHashMap"
                ]

reserved_kws = reserved_kws + reserved_cls


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


    return word.replace(word_tar,word_new),word_ret


def extract_method_name(string):
    match_ret = re.search('\w+\s*\(',string)
    if match_ret:
        method_name = match_ret.group()[:-1].strip()
        return method_name
    else:
        return None


def extract_argument(string):
    end_pos    = string.find('{')
    sta_pas    = string.find('(')
    arguments  = string[sta_pas + 1 :end_pos].strip()[:-1]
    arguments_list = arguments.split(',')
    if ' ' in arguments_list:
        arguments_list.remove(' ')
    if '' in arguments_list:
        arguments_list.remove('')
    return arguments_list


def extract_brace(string,start_pos):
    length = 0
    brace_l_num = 0
    brace_r_num = 0
    for char in string[start_pos:]:
        if char == '{':
            brace_l_num += 1
        if char == '}':
            brace_r_num += 1
        if brace_l_num == brace_r_num and brace_l_num > 0:
            break;
        length += 1
    return string[start_pos: start_pos + length + 1]

'''
def extract_import(string):
    import_list = re.findall('import .+;',string)
    return import_list,string
'''


def extract_class(string):

    class_list = []
    while ' class ' in string:
        start_pos  = string.find(' class ')
        class_text = extract_brace(string, start_pos)
        class_list.append(class_text)
        string = string.replace(class_text,'')

    return class_list,string


def extract_member_variable(string):

    variable_list = []
    while True:
        match_ret = re.search('(private|public).+;', string)
        if match_ret:
            variable_text = match_ret.group()
            variable_list.append(variable_text)
            string = string.replace(variable_text,'')
        else:
            break
    return variable_list,string


def extract_function(string):
    i = 0
    function_list = []
    while True:
        match_ret = re.search('(protected|private|public).+\s*{', string)
        if match_ret:
            function_head = match_ret.group()
            start_pos = string.find(function_head)
            function_text = extract_brace(string, start_pos)
            function_list.append(function_text)
            string = string.replace(function_text, 'vesal'+ str(i))
            i+=1
        else:
            break
    return function_list, string


def extract_for_loop(string):

    for_list = []
    while True:
        match_ret = re.search('for\s+\(', string)
        print(match_ret)
        if match_ret:
            for_head = match_ret.group()
            start_pos = string.find(for_head)
            for_text = extract_brace(string, start_pos)
            for_list.append(for_text)
            string = string.replace(for_text, '')
        else:
            break
    return for_list


def extract_if(string):

    if_list = []
    while True:
        match_ret = re.search('if\s+\(', string)
        if match_ret:
            if_head = match_ret.group()
            start_pos = string.find(if_head)
            if_text = extract_brace(string, start_pos)
            if_list.append(if_text)
            string = string.replace(if_text, '')
        else:
            break
    return if_list


def extract_while_loop(string):

    while_list = []
    while True:
        match_ret = re.search('while\s+\(', string)
        if match_ret:
            while_head = match_ret.group()
            start_pos = string.find(while_head)
            while_text = extract_brace(string, start_pos)
            while_list.append(while_text)
            string = string.replace(while_text, '')
        else:
            break
    return while_list, string


def extract_local_variable(string):

    local_var_list = []
    statement_list = string.split('\n')
    for line in statement_list:
        match_ret = re.search('[^\s]+\s+\w+\s+=', line)
        if match_ret:
            var_definition = match_ret.group()
            local_var_list.append(var_definition.split(' ')[1])

    return local_var_list



