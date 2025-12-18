import nltk
from nltk import Tree
import re
from Levenshtein import distance as edit_dist
from typing import List
from copy import deepcopy
import numpy as np
from itertools import product, permutations
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math
import itertools
import time
import os
from collections import defaultdict

MODEL_NAME = 'all-MiniLM-L6-v2'
MODEL_SAVE_PATH = '../assistance/all-MiniLM-L6-v2'

def load_model():
    """
    Load SentenceTransformer model:
    -If the model already exists locally, load it directly
    -If it does not exist, download and save to the specified path
    """
    # 检查本地模型是否存在
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"load from local: {MODEL_SAVE_PATH}")
        model = SentenceTransformer(MODEL_SAVE_PATH)
    else:
        print(f"Model not found locally, downloading and saving to: {MODEL_SAVE_PATH}")
        model = SentenceTransformer(MODEL_NAME, cache_folder=os.path.dirname(MODEL_SAVE_PATH))
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        model.save(MODEL_SAVE_PATH)
        print("Model download and save completed!")
    return model

Unit_Match_Model = load_model()

sym_reg = re.compile(r'[^⊕∨∧→↔∀∃¬(),]+')

cfg_template = """
S -> F | Q F
Q -> QUANT VAR | QUANT VAR Q
F -> '¬' '(' F ')' | '(' F ')' | F OP F | L
OP -> '⊕' | '∨' | '∧' | '→' | '↔'
L -> '¬' PRED '(' TERMS ')' | PRED '(' TERMS ')'
TERMS -> TERM | TERM ',' TERMS
TERM -> CONST | VAR
QUANT -> '∀' | '∃'
"""

op_ls = ['⊕', '∨', '∧', '→', '↔', '∀', '∃', '¬', '(', ')', ',']
# used in perturbation
last_nt_nodes = set(['PRED', 'OP', 'CONST', 'VAR', 'QUANT'])
# used in node insertion
insertable_nt_nodes = set(['Q', 'S', 'TERMS', 'F'])
# used in node deletion
deleteable_nt_nodes = set(['Q', 'TERMS', 'F', 'L'])


def make_cfg_str(token_ls):
    """
    NOTE: since nltk does not support reg strs like \w+, we cannot separately recognize VAR, PRED, and CONST.
    Instead, we first allow VAR, PRED, and CONST to be matched with all symbols found in the FOL; once the tree is
    parsered, we then go back and figure out the exact type of each symbols
    """
    sym_ls = list(set([e for e in token_ls if sym_reg.match(e)]))
    sym_str = ' | '.join(["'%s'" % s for s in sym_ls])
    cfg_str = cfg_template + 'VAR -> %s\nPRED -> %s\nCONST -> %s' % (sym_str, sym_str, sym_str)
    return cfg_str


def msplit(s):
    for op in op_ls:
        s = s.replace(op, ' %s ' % op)
    r = [e.strip() for e in s.split()]
    # remove ' from the string if it contains any: this causes error in nltk cfg parsing
    r = [e.replace('\'', '') for e in r]
    r = [e for e in r if e != '']

    # deal with symbols with spaces like "dc universe" and turn it to "DcUniverse"
    res = []
    cur_str_ls = []
    for e in r:
        if (len(e) > 1) and sym_reg.match(e):
            cur_str_ls.append(e[0].upper() + e[1:])
        else:
            if len(cur_str_ls) > 0:
                res.extend([''.join(cur_str_ls), e])
            else:
                res.extend([e])
            cur_str_ls = []
    if len(cur_str_ls) > 0:
        res.append(''.join(cur_str_ls))

    # re-generate the FOL string
    make_str_ls = []
    for ind, e in enumerate(r):
        if re.match(r'[⊕∨∧→↔]', e):
            make_str_ls.append(' %s ' % e)
        elif re.match(r',', e):
            make_str_ls.append('%s ' % e)
        # a logical variable
        elif (len(e) == 1) and re.match(r'\w', e):
            if ((ind - 1) >= 0) and ((r[ind - 1] == '∃') or (r[ind - 1] == '∀')):
                make_str_ls.append('%s ' % e)
            else:
                make_str_ls.append(e)
        else:
            make_str_ls.append(e)

    return res, ''.join(make_str_ls)


def reorder_quantifiers(rule_str):
    matches = re.findall(r'[∃∀]\w', rule_str)
    for match in matches[::-1]:
        rule_str = '%s ' % match + rule_str.replace(match, '', 1)
    return rule_str


def build_parent_map(tree):

    parent_map = {}

    def traverse(subtree, parent=None):
        if isinstance(subtree, Tree):
            parent_map[id(subtree)] = parent

            for child in subtree:
                traverse(child, subtree)

    traverse(tree)
    return parent_map


def replace_node(main_tree, target, replacement_tree, parent_dict):

    replace_res = False
    for i, main_node in enumerate(main_tree):

        if not isinstance(main_node, Tree):
            continue

        if isinstance(main_node[0], str) and target[:-3] == main_node[0]:
            current_node = main_node

            while current_node is not None:

                replacement_node = replacement_tree
                exit_mark = False
                while isinstance(replacement_node, Tree) and replacement_node.label() != current_node.label():

                    replacement_node = replacement_node[0]
                    if exit_mark:
                        break

                    if len(replacement_node) > 1:
                        exit_mark = True

                if isinstance(replacement_node, Tree) and current_node.label() == replacement_node.label():
                    parent_node = parent_dict[id(current_node)]
                    if parent_node is None:
                        return False
                    index = parent_node.index(current_node)
                    parent_node[index] = replacement_node
                    return True
                else:
                    current_node = parent_dict[id(current_node)]

        replace_res = replace_node(main_node, target, replacement_tree, parent_dict)

        # Early Termination Iteration
        if replace_res:
            return replace_res
    return replace_res


def extract_parentheses(text):

    stack = []
    result_dict = {}
    placeholder_count = 1
    i = 0

    while i < len(text):
        if text[i] == '(':
            stack.append(i)
        elif text[i] == ')':
            if stack:
                start = stack.pop()
                matched_text = text[start:i+1]
                if matched_text.count('(') < 2:
                    i += 1
                    continue
                placeholder = f"PLACEHOLDER#{placeholder_count}(#)"
                result_dict[placeholder] = matched_text
                text = text[:start] + placeholder + text[i+1:]
                i = start + len(placeholder) - 1
                placeholder_count += 1
        i += 1

    result_dict["ROOT"] = text
    return result_dict


initial_tree = None
def parse_text_FOL_to_tree(input_FOL:str=None, root:str='ROOT', chunk=None):
    """
        Parse a text FOL rule into nltk.tree
        Returns: nltk.tree, or None if the parse fails

        1. Find the root of the tree and build the tree
        2. After building the tree, find the leaf PLACEHOLDER
        3. Retrieve the parent node of the tree from the dictionary,
        4. Recursively traverse and replace the node with a new tree
    """

    if chunk is None:
        chunk = extract_parentheses(input_FOL)

    rule_str = chunk[root]
    rule_str = reorder_quantifiers(rule_str)

    r, parsed_fol_str = msplit(rule_str)
    cfg_str = make_cfg_str(r)

    grammar = nltk.CFG.fromstring(cfg_str)
    parser = nltk.EarleyChartParser(grammar)
    tree = parser.parse_one(r)

    if tree is None:
        return None

    global initial_tree
    if initial_tree is None:
        initial_tree = tree

    pattern = r'PLACEHOLDER#\d+\(#\)'
    matches = re.findall(pattern, chunk[root])
    if len(matches) > 0 and tree != initial_tree and tree[0].label()=='Q' and tree[1].label()=='F':
        temp_node = initial_tree[0]
        while temp_node[-1].label() == 'Q':
            temp_node = temp_node[-1]
        temp_node.append(tree[0])
        tree = tree[1]

    for sub in matches:
        sub_tree = parse_text_FOL_to_tree(root=sub, chunk=chunk)
        parent_dict = build_parent_map(tree)
        replace_success = replace_node(tree, sub, sub_tree, parent_dict)
        if not replace_success:
            return None
        parent_dict = None

    return tree


def FOL_to_tree(FOL_text):
    if not isinstance(FOL_text, str):
        return None

    global initial_tree
    initial_tree = None
    return parse_text_FOL_to_tree(FOL_text)


def Compute_LE(pred_text_FOL: str, true_text_FOL: str, match_with_threshold = False, return_Yang: bool = False):
    global initial_tree

    if len(pred_text_FOL) > 180 or len(true_text_FOL) > 180 or true_text_FOL.count('∧') > 5 or pred_text_FOL.count('∧') > 5 \
            or true_text_FOL.count('∨') > 5 or pred_text_FOL.count('∨') > 5:
        return (0., None, None), (0., None, None)
    true_root = FOL_to_tree(true_text_FOL)
    # initial_tree = None

    pred_root = FOL_to_tree(pred_text_FOL)
    # initial_tree = None

    # parsing pred FOL can fail if model produces invalid rule, in which case, LE score is 0
    if pred_root is None or true_root is None :
        return (0., None, None), (0., None, None)

    # if both parsed successfully, then compute LE score
    res_Yang = 0.0
    if return_Yang:
        res_Yang = VecRuleEvaluator.find_best_LE_score_YuanYang(
                true_root,
                pred_root,
                soft_binding=True,
                greedy_match=True,
                top_n=1000
            )
    res_Jiang = VecRuleEvaluator.find_best_LE_score_Jiang(true_root, pred_root, match_with_threshold=match_with_threshold)

    return res_Yang, res_Jiang


def get_unique_permutations(arr):
    # Identify duplicate elements
    from collections import defaultdict
    count = defaultdict(int)
    for num in arr:
        count[num] += 1

    duplicates = [num for num in count if count[num] > 1]

    # Generate possible positional combinations for each repeating element
    position_options = {}
    for num in duplicates:
        positions = [i for i, x in enumerate(arr) if x == num]
        # We need to choose one of the positions to keep num, and set the other positions to ∞
        # Generate an option for each location: (reserved location, other locations)
        options = []
        for keep in positions:
            option = {pos: (num if pos == keep else math.inf) for pos in positions}
            options.append(option)
        position_options[num] = options

    # Generate all possible combinations
    combinations = itertools.product(*position_options.values())

    # Create a result array for each combination
    results = []
    for combo in combinations:
        # Create a copy of the initial array
        result = arr.copy()
        # Apply all replacement rules
        for num_option in combo:
            for pos, value in num_option.items():
                result[pos] = value
        results.append(result)

    return results


class VecRuleEvaluator:
    dummy_input_str: str = '#DUMMY'
    dummy_distance: int = 10000

    @classmethod
    def default_input_similarity(cls, e1: str, e2: str):
        if e1.startswith(cls.dummy_input_str) or e2.startswith(cls.dummy_input_str):
            return cls.dummy_distance
        return edit_dist(e1, e2)

    @classmethod
    def enumerate_bindings_with_greedy_match(cls, ls1: List[str], ls2: List[str], top_n: int):
        """
            Given two lists of strings ls1 and ls2, yields the ind bindings of ls2 strings that matches the strings in
            ls1. I use greedy match and yields starts from the best to the worst binding until full enumeration or hit
            the top_n bound
        """

        used_inds = []

        def _enum_bindings(ind1: int):
            if ind1 == len(ls1):
                yield deepcopy(used_inds)
                return
            e1 = ls1[ind1]
            match_ls = [
                (ind, cls.default_input_similarity(e1, e2))
                for ind, e2 in enumerate(ls2) if ind not in used_inds
            ]
            match_ls.sort(key=lambda x: x[1])
            for ind, dist in match_ls:
                used_inds.append(ind)
                for inds in _enum_bindings(ind1 + 1):
                    yield inds
                used_inds.pop()

        for cnt, ind_ls in enumerate(_enum_bindings(0)):
            yield ind_ls
            if cnt + 1 == top_n:
                break

    @classmethod
    def find_inputs(cls, root, input_set=None):
        if isinstance(root, str):
            return

        label = root.label()

        if label == 'L':
            literal_str = ''.join(root.leaves())
            literal_str = literal_str[1:] if literal_str[0] == '¬' else literal_str
            if input_set is None:
                input_set = set()
            input_set.add(literal_str)
        else:
            for child in root:
                cls.find_inputs(child, input_set)

    @classmethod
    def gen_input_vecs(cls, num_inputs):
        return np.array(list(product([False, True], repeat=num_inputs)))

    @classmethod
    def from_nltk_tree(cls, root, name2ind_dict, input_vecs):
        assert not isinstance(root, str), 'something wrong with the rule or the algo; you should not parse a leave'

        label = root.label()

        if label == 'S':

            return cls.from_nltk_tree(root[-1], name2ind_dict, input_vecs)

        elif label == 'F':

            # the case F -> L
            if (len(root) == 1) and (root[0].label() == 'L'):
                return cls.from_nltk_tree(root[0], name2ind_dict, input_vecs)

            # the case F -> '¬' '(' F ')' | (' F ')'
            elif root[-2].label() == 'F':

                isnegated_rule = isinstance(root[0], str) and (root[0] == '¬')
                res = cls.from_nltk_tree(root[-2], name2ind_dict, input_vecs)

                if isnegated_rule:
                    res = ~res

                return res

            # the case F -> F OP F
            elif root[-2].label() == 'OP':

                p, q = cls.from_nltk_tree(root[0], name2ind_dict, input_vecs), \
                    cls.from_nltk_tree(root[-1], name2ind_dict, input_vecs)

                op = root[1][0]
                if op == '⊕':
                    return np.logical_xor(p, q)
                elif op == '∨':
                    return np.logical_or(p, q)
                elif op == '∧':
                    return np.logical_and(p, q)
                elif op == '→':
                    return np.logical_or(~p, q)
                elif op == '↔':
                    return np.logical_or(np.logical_and(p, q), np.logical_and(~p, ~q))
                else:
                    raise ValueError

        elif label == 'L':

            isnegated_literal = isinstance(root[0], str) and (root[0] == '¬')

            literal_str = ''.join(root.leaves())
            # remove the possible negation at the beginning
            literal_str = literal_str[1:] if isnegated_literal else literal_str

            vec = input_vecs[:, name2ind_dict[literal_str]]

            if isnegated_literal:
                vec = ~vec

            return vec

        else:
            raise ValueError

    @classmethod
    def find_best_LE_score_YuanYang(
            cls,
            true_root,
            pred_root,
            soft_binding: bool,
            greedy_match: bool,
            top_n: int,
            verbose: bool = False
    ):
        """
            Given the groundtruth and the predicted nltk FOL trees, compute the truth tables over all
            literal bindings and returns the best one
        """

        # first we find "inputs" in each tree, i.e. the set of unique literals in a FOL
        true_inputs, pred_inputs = set(), set()
        VecRuleEvaluator.find_inputs(true_root, true_inputs), VecRuleEvaluator.find_inputs(pred_root, pred_inputs)
        true_inputs, pred_inputs = list(true_inputs), list(pred_inputs)
        n_true_inputs, n_pred_inputs = len(true_inputs), len(pred_inputs)
        min_n, max_n = sorted([n_true_inputs, n_pred_inputs])

        best_score, best_binded_pred_inputs = 0., None

        # once we found the inputs, then we deal with the case where # inputs in two trees are different
        # either we do soft binding by adding dummy inputs to the shorter ones, or we simply return 0
        if n_true_inputs != n_pred_inputs:
            if soft_binding:
                # extend the shorter inputs to the max number of inputs by adding dummy input names
                ls_to_extend = true_inputs if n_true_inputs < max_n else pred_inputs
                ls_to_extend.extend([f'{cls.dummy_input_str}_{ind}' for ind in range(max_n - min_n)])
            else:
                return best_score, true_inputs, best_binded_pred_inputs

        # at this point, we have two list ofs inputs of the same length and we will find the input binding that yields
        # the best score
        input_vecs = VecRuleEvaluator.gen_input_vecs(len(true_inputs))
        true_name2ind_dict = dict((e, ind) for ind, e in enumerate(true_inputs))
        true_res_vec = VecRuleEvaluator.from_nltk_tree(true_root, true_name2ind_dict, input_vecs)

        ind_binding_enumerator = \
            cls.enumerate_bindings_with_greedy_match(true_inputs, pred_inputs, top_n) if greedy_match \
                else permutations(list(range(max_n)))

        for cnt_ind, binded_pred_inputs_inds in enumerate(ind_binding_enumerator):
            binded_pred_inputs = [pred_inputs[ind] for ind in binded_pred_inputs_inds]
            pred_name2ind_dict = dict((e, ind) for ind, e in enumerate(binded_pred_inputs))
            pred_res_vec = VecRuleEvaluator.from_nltk_tree(pred_root, pred_name2ind_dict, input_vecs)

            score = (pred_res_vec == true_res_vec).mean(dtype=np.float32).item()

            if verbose:
                print('{0}\n{1}\n{2}\n---\n'.format(
                    score, true_inputs, binded_pred_inputs)
                )

            if score > best_score:
                best_score = score
                best_binded_pred_inputs = binded_pred_inputs

            if cnt_ind + 1 >= top_n:
                break

        return best_score, true_inputs, best_binded_pred_inputs


    @classmethod
    def verify_match(cls, bound_pred_inputs: list, bound_true_inputs:list, match_table_score:defaultdict) -> bool:
        for pred_item, true_item in zip(bound_pred_inputs, bound_true_inputs):
            if true_item.startswith(cls.dummy_input_str) or pred_item.startswith(cls.dummy_input_str):
                continue
            elif match_table_score[true_item][pred_item] < 0.6:
                return False

        return True


    @classmethod
    def find_best_LE_score_Jiang(
            cls,
            true_root,
            pred_root,
            match_with_threshold=False
    ):
        """
            Given the groundtruth and the predicted nltk FOL trees, compute the truth tables over all
            literal bindings and returns the best one
        """

        # first we find "inputs" in each tree, i.e. the set of unique literals in a FOL
        true_inputs, pred_inputs = set(), set()
        VecRuleEvaluator.find_inputs(true_root, true_inputs), VecRuleEvaluator.find_inputs(pred_root, pred_inputs)
        true_inputs, pred_inputs = list(true_inputs), list(pred_inputs)
        n_true_inputs, n_pred_inputs = len(true_inputs), len(pred_inputs)
        min_n, max_n = sorted([n_true_inputs, n_pred_inputs])

        best_score, best_binded_pred_inputs, best_binded_true_inputs, bound_scores= 0., None, None, []

        match_table = defaultdict(dict)
        match_table_score = defaultdict(dict)
        pred_list = []
        for true_item in true_inputs:

            best_match = 0.0
            best_idx = 0
            for idx, pred_item in enumerate(pred_inputs):
                embeddings = Unit_Match_Model.encode([true_item, pred_item])
                similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

                match_table_score[true_item][pred_item] = similarity_score

                if similarity_score > best_match:
                    match_table[true_item]['binded_item'] = pred_item
                    match_table[true_item]['best_match'] = similarity_score
                    best_idx = idx

                    best_match = similarity_score
            pred_list.append(best_idx)

        list_permutations = get_unique_permutations(pred_list)

        # tmp_pred_inputs = pred_inputs, temp_input_vecs = true_inputs
        for i, binded_pred_inputs_inds in enumerate(list_permutations):

            tmp_pred_inputs = deepcopy(pred_inputs)
            temp_true_inputs = deepcopy(true_inputs)

            for j, num in enumerate(binded_pred_inputs_inds):
                if num == math.inf:
                    tmp_pred_inputs.append(f'{cls.dummy_input_str}_{temp_true_inputs[j]}')
                    list_permutations[i][j] = len(tmp_pred_inputs) - 1

            full_set = set(range(max_n))
            given_set = set(binded_pred_inputs_inds)
            missing_numbers = sorted(full_set - given_set)

            if len(missing_numbers) != 0:
                temp_true_inputs.extend([f'{cls.dummy_input_str}_{tmp_pred_inputs[ind]}' for ind in missing_numbers])
                list_permutations[i].extend(missing_numbers)

            input_vecs = VecRuleEvaluator.gen_input_vecs(len(temp_true_inputs))
            true_name2ind_dict = dict((e, ind) for ind, e in enumerate(temp_true_inputs))
            true_res_vec = VecRuleEvaluator.from_nltk_tree(true_root, true_name2ind_dict, input_vecs)

            binded_pred_inputs = [tmp_pred_inputs[ind] for ind in binded_pred_inputs_inds]
            pred_name2ind_dict = dict((e, ind) for ind, e in enumerate(binded_pred_inputs))
            pred_res_vec = VecRuleEvaluator.from_nltk_tree(pred_root, pred_name2ind_dict, input_vecs)

            score = (pred_res_vec == true_res_vec).mean(dtype=np.float32).item()

            if match_with_threshold and not VecRuleEvaluator.verify_match(bound_pred_inputs=binded_pred_inputs,
                                                                      bound_true_inputs=temp_true_inputs,
                                                                      match_table_score=match_table_score):
                continue

            if score > best_score:
                best_score = score
                best_binded_pred_inputs = binded_pred_inputs
                best_binded_true_inputs = temp_true_inputs
                bound_scores = [ match_table_score[true][pred] for pred, true in zip(binded_pred_inputs, temp_true_inputs)
                                 if not (true.startswith(cls.dummy_input_str) or pred.startswith(cls.dummy_input_str))]

        return best_score, best_binded_true_inputs, best_binded_pred_inputs, bound_scores
