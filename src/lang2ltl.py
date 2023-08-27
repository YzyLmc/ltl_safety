# migrated lang2ltl
import logging
from collections import defaultdict
import itertools
import nltk

from utils import load_from_file, save_to_file, prefix_to_infix

PROPS_OBJ = ['A', 'B', 'C', 'D', 'H', 'J', 'K', 'L', 'N', 'O', 'P', 'Q', 'R', 'S', 'Y', 'Z']  # 16
PROPS_PRED = ["a", "b", "c", "d", "h", "j", "k", "l", "n", "o", "p", "q", "r", "s", "y", "z"]
OPERATORS = ["G", "!", "X", "W", "U", "i", "&", "F"]

def rer(rer_module, rer_prompt, input_utts, utt2lmk=None):
    """
    Referring Expression Recognition: extract name entities from input utterances.
    """
    # if rer_model == "gpt3":
    #     rer_module = GPT3(rer_engine)
    # elif rer_model == "gpt4":
    #     rer_module = GPT4(rer_engine)
    # else:
    #     raise ValueError(f"ERROR: RER module not recognized: {rer_model}")

    names, utt2names = set(), []  # name entity list names should not have duplicates
    for idx_utt, utt in enumerate(input_utts):
        logging.info(f"Extracting referring expressions from utterance: {idx_utt}/{len(input_utts)}")
        try:
            names_per_utt = list(utt2lmk[utt])
        except:
            # breakpoint()
            names_per_utt = [name.strip() for name in rer_module.extract_re(f"{rer_prompt.strip()} {utt}\nPropositions:")]
            names_per_utt = list(set(names_per_utt))  # remove duplicated RE

        names.update(names_per_utt)
        utt2names.append((utt, names_per_utt))

    return names, utt2names

def ground_res(res, re2embed_fpath, obj_embed, ground_model, embed_engine, topk=1, re2lmk=None):
    """
    Find groundings (objects in given environment) of referring expressions (REs) extracted from input utterances.
    """
    obj2embed = load_from_file(obj_embed)  # load embeddings of known objects in given environment
    if os.path.exists(re2embed_fpath):  # load cached embeddings of referring expressions
        re2embed = load_from_file(re2embed_fpath)
    else:
        re2embed = {}

    if ground_model == "gpt3":
        ground_module = GPT3(embed_engine)
    else:
        raise ValueError(f"ERROR: grounding module not recognized: {ground_model}")

    re2grounds = {}
    is_new_embed = False
    for re in res:
        logging.info(f"grounding referring expression: {re}")
        try:
            re2grounds[re] = [re2lmk[re]]
        except:
            if re in re2embed:  # use cached RE embedding if exists
                logging.info(f"use cached RE embedding: {re}")
                re_embed = re2embed[re]
            else:
                re_embed = ground_module.get_embedding(re)
                re2embed[re] = re_embed
                is_new_embed = True

            sims = {o: cosine_similarity(e, re_embed) for o, e in obj2embed.items()}
            sims_sorted = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
            re2grounds[re] = list(dict(sims_sorted[:topk]).keys())

            if is_new_embed:
                save_to_file(re2embed, re2embed_fpath)

    return re2grounds

def ground_utterances(input_strs, utt2res, re2grounds):
    """
    Replace referring expressions in input utterances with best matching objects in given env.
    """
    grounding_maps = []  # name to grounding map per utterance
    for _, res in utt2res:
        grounding_maps.append({re: re2grounds[re][0] for re in res})
    # breakpoint()
    output_strs, subs_per_str = substitute(input_strs, grounding_maps, is_utt=True)

    return output_strs, subs_per_str

def translate_modular(ground_utts, trans_module, objs_per_utt, trans_modular_prompt_fpath="prompts/translation/symbolic_trans_navigation.txt", utt2ltl=None):
    """
    Translation language to LTL modular approach.
    :param ground_utts: Input utterances with name entities grounded to objects in given environment.
    :param objs_per_utt: grounding objects for each input utterance.
    :param sym_trans_model: symbolic translation model, gpt3_finetuned, gpt3_pretrained, t5-base.
    :param translation_engine: pretrained T5 model weights, finetuned or pretrained GPT-3 engine to use for translation.
    :param convert_rule: referring expression to proposition conversion rule.
    :param props: all possible propositions.
    :param trans_modular_prompt: prompt for pretrained GPT-3.
    :return: output grounded LTL formulas, corresponding intermediate symbolic LTL formulas, placeholder maps
    """
    trans_modular_prompt = load_from_file(trans_modular_prompt_fpath)
    placeholder_maps, placeholder_maps_inv = [], []
    for objs in objs_per_utt:
        placeholder_map, placeholder_map_inv = build_placeholder_map(objs, props=["A", "B", "C"])
        placeholder_maps.append(placeholder_map)
        placeholder_maps_inv.append(placeholder_map_inv)
    symbolic_utts, _ = substitute(ground_utts, placeholder_maps, is_utt=True)  # replace names by symbols

    symbolic_ltls = []
    for idx, sym_utt in enumerate(symbolic_utts):
        logging.info(f"Symbolic Translation: {idx}/{len(symbolic_utts)}")
        query = sym_utt.translate(str.maketrans('', '', ',.'))
        try:
            ltl = utt2ltl[query]
            # logging.info(f"found utt in previous result: {query}")
        except:
            # logging.info(f"utt not found: {query}")
            # query = f"Utterance: {query}\nLTL:"  # query format for finetuned GPT-3
            ltl = trans_module.generate(f"{trans_modular_prompt} {query}\nLTL:")[0]
            # try:
            #     spot.formula(ltl)
            # except SyntaxError:
            #     ltl = feedback_module(trans_module, query, trans_modular_prompt, ltl)
        # breakpoint()
        # symbolic_ltls.append(prefix_to_infix(ltl))
        symbolic_ltls.append(ltl)
    # breakpoint()
    # output_ltls, _ = substitute(symbolic_ltls, placeholder_maps_inv, is_utt=False)  # replace symbols by props
    output_ltls = simple_sub(symbolic_ltls, placeholder_maps_inv)  # replace symbols by props
    # breakpoint()
    return symbolic_utts, symbolic_ltls, output_ltls, placeholder_maps

def simple_sub(input_strs, substitute_maps):
    """
    simpler substitute function, only check capital characters
    """
    output_strs = []
    for input_str, sub_map in zip(input_strs, substitute_maps):
        for prop, obj in sub_map.items():
            while prop in input_str:
                input_str = input_str.replace(prop, obj)
        output_strs.append(input_str)
    return output_strs

def substitute(input_strs, substitute_maps, is_utt):
    """
    Substitute every occurrence of key in the input string by its corresponding value in substitute_maps.
    :param input_strs: input strings
    :param substitute_maps: map substring to substitutions
    :param is_utt: True if input_strs are utts; False if input_strs are LTL formulas
    :return: substituted strings and their corresponding substitutions
    """
    # breakpoint()

    output_strs, subs_per_str = [], []
    for input_str, sub_map in zip(input_strs, substitute_maps):
        if is_utt:
            out_str, subs_done = substitute_single_word(input_str, sub_map)
        else:
            out_str = substitute_single_letter(input_str, sub_map)
            subs_done = set()
        # out_str = out_str.translate(str.maketrans('', '', ',.'))  # remove comma, period since sym translation module finetuned on utts w/o puns
        output_strs.append(out_str)
        subs_per_str.append(subs_done)
    # breakpoint()

    return output_strs, subs_per_str


def substitute_single_word(in_str, sub_map):
    """
    Substitute words and phrases to words or phrases in a single utterance.
    Assume numbers are not keys of sub_map.
    """
    # breakpoint()

    sub_map = sorted(sub_map.items(), key=lambda kv: len(kv[0]), reverse=True)  # start substitution with long strings
    subs_done = set()

    # swap every k with a unique number
    for n, (k, v) in enumerate(sub_map):
        in_str = in_str.replace(k, f"[{n}]")  # escape number

    # swap every number with corresponding v
    for n, (k, v) in enumerate(sub_map):
        in_str = in_str.replace(f"[{n}]", v)  # escape number
        subs_done.add(v)

    # breakpoint()

    return in_str.strip(), subs_done

def substitute_single_letter(in_str, sub_map):
    """
    :param in_str: input string can utterance or LTL formula.
    :param sub_map: dict maps letters to noun phrases for lmk names (for utterance) or letters (for LTL formula).

    Substitute English letters to letters, words or phrases in a single utterance.
    e.g. input_str="go to a then go to b", sub_map={'a': 'b', 'b': 'a'} -> return="go to a then go to a"

    Require `input_str` to be normalized, i.e. all punctuations removed. If not, "go to a. after that go to b." not tokenized correctly
    Only work with letters, e.g. a, b, c, etc, not phrases, e.g. green one -> green room.
    """
    in_str_list = nltk.word_tokenize(in_str)
    # in_str_list = in_str.split(" ")
    sub_map = sorted(sub_map.items(), key=lambda kv: len(kv[0]), reverse=True)  # start substitution with long strings

    # Record indices of all keys in sub_map in *original* input_str.
    key2indices = defaultdict(list)
    for k, _ in sub_map:
        key2indices[k] = [idx for idx, word in enumerate(in_str_list) if word == k]

    # Loop through indices and keys to replace each key with value.
    for k, v in sub_map:
        indices = key2indices[k]
        for idx in indices:
            in_str_list[idx] = v

    return ' '.join(in_str_list).strip()

def build_placeholder_map(name_entities, props):
    # breakpoint()

    placeholder_map, placeholder_map_inv = {}, {}
    for name, letter in zip(name_entities, props[:len(name_entities)]):
        placeholder_map[name] = letter
        placeholder_map_inv[letter] = name_to_prop(name)
    return placeholder_map, placeholder_map_inv

def name_to_prop(name, convert_rule="lang2ltl"):
    """
    :param name: name, e.g. Canal Street, TD Bank.
    :param convert_rule: identifier for conversion rule.
    :return: proposition that corresponds to input landmark name and is compatible with Spot.
    """
    # return name
    if convert_rule == "lang2ltl":
        # return "_".join(name.translate(str.maketrans('/()-–', '     ', "'’,.!?")).lower().split())
        return "".join(name.translate(str.maketrans('/()-–', '     ', "'’,.!?")).lower().split())
    elif convert_rule == "copynet":
        return f"lm( {name} )lm"
    elif convert_rule == "cleanup":
        return "_".join(name.split()).strip()
    else:
        raise ValueError(f"ERROR: unrecognized conversion rule: {convert_rule}")

## concat formulas and unified placeholder map
def concat_formula(formula_list, format="prefix"):
    if format == "infix":
        formula_list = [f"({formula})" for formula in formula_list]
        return " & ".join(formula_list)
    elif format == "prefix":
        return (len(formula_list)-1)*"& " + " ".join(formula_list)
    else:
        raise Exception("format not defined")

def unify_formula(out_ltls, placeholder_maps, props=PROPS_OBJ):
    map_key_list = [list(map.keys()) for map in placeholder_maps]
    objs = set(list(itertools.chain.from_iterable(map_key_list)))
    unified_mapping = {obj: props[i] for i, obj in enumerate(objs)}
    unified_ltl = concat_formula(out_ltls)
    for obj in unified_mapping.keys():
        while obj in unified_ltl:
            # breakpoint()
            unified_ltl = unified_ltl.replace(obj, unified_mapping[obj])
    return unified_ltl, {v:k for k,v in unified_mapping.items()}

def sub_predicate(unified_ltl, operators=OPERATORS, props=PROPS_PRED):
    ltl_split = unified_ltl.split()
    predicates = set([char for char in ltl_split if char not in operators])
    pred_map = {}
    for i, p in enumerate(predicates):
        pred_map[p] = props[i]
    sym_unified_ltl = simple_sub([unified_ltl], [pred_map])[0]
    inv_pred_map = {v:k for k,v in pred_map.items()} 
    return sym_unified_ltl, inv_pred_map  