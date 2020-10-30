import re
import random
from data import *
from samples import *

def valid_eq(eq):
    Z = random.randint(1, 9)
    C = random.randint(1, 9)
    a = random.randint(1, 9)
    try:
        exec(eq)
        return True
    except ZeroDivisionError:
        return False
    except SyntaxError:
        return False
    except TypeError:
        return False
    except NameError:
        return False
    except MemoryError:
        return False
    
def valid_expression(eq):
    eq = eq.replace("o", "+")
    matched = re.match(
        '^(\()(\()?(.+)(\))?(\))(.*)|^(\()(\()(.+)(\))(\))(.*)$', eq)
    if (bool(matched)):
        return True
    else:
        return False
    
def equation_validity(equation):
    print("----- BEGIN equation_validity -----")
    valid = []
    not_valid = []
    
    for i, eq2 in enumerate(equation['EQUATIONS']):  
        eq = eq2.replace("o", "+")
        val_exp = valid_expression(eq)
        val_exec = valid_eq(eq)
        if (val_exp and val_exec):
            print('Valid: ', eq2)
            valid.append(eq2)
        else:
            val_exec = valid_eq(eq)
            if (val_exec):
                print('Valid: ', eq2)
                valid.append(eq2)
            else:
               print('Not valid: ', eq2)
               not_valid.append(eq2)
                
    max_equations = len(valid) + len(not_valid)
    # unique valid equations
    unique_equations = list(set(valid))

    raw_split = re.split("#", raw_text)
    check_end = [i.replace("$", "") for i in raw_split]
    find_equation =  [item for item in check_end if item != '']

    # unique unseen valid equations
    N = [i for i in range(len(unique_equations)) if not unique_equations[i] in find_equation]
    unseenval = []
    for i in N:
        unseenval.append(unique_equations[i])

    print('Genrated equations', max_equations)
    print('Valid equations', len(valid))
    print('Unique valid equations', len(unique_equations))
    print('Unseen valid equations', len(N))
                
    return valid, not_valid, max_equations, unseenval


def take_results(valid, not_valid):
    max_equations = len(valid) + len(not_valid)
    unique_equations = list(set(valid))

    raw_split = re.split("#", raw_text)
    check_end = [i.replace("$", "") for i in raw_split]
    find_equation =  [item for item in check_end if item != '']

    N = [i for i in range(len(unique_equations)) if not unique_equations[i] in find_equation]

    print('Genrated equations', max_equations)
    print('Valid equations', len(valid))
    print('Unique valid equations', len(set(valid)))
    print('Unseen valid equations', len(N))
    

    
    
