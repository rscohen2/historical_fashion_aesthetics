from collections import namedtuple
import stanza
from stanza.server.semgrex import Semgrex
from google.protobuf import json_format


Word = namedtuple("Word", ["start_idx", "end_idx", "text"])
Pair = namedtuple("Pair", ["adjective", "noun", "negated"])


def process_batch(nlp, lines):
    """
    @input:
    - a list of strings
    @outputs:

    """

    def get_word(doc, word) -> Word:
        i, j = word
        word = doc.sentences[i].words[j]
        return Word(word.start_char, word.end_char, word.text)
        # return doc.sentences[i].words[j].text

    def get_compound(doc, compounds, word):
        # build compound string
        output = [word]
        while word in compounds:
            word = compounds[word]
            output.append(word)
        start_idx = output[0][0]
        end_idx = output[-1][1]
        return Word(
            start_idx,
            end_idx,
            " ".join([get_word(doc, word).text for word in output[::-1]]),
        )

    in_docs = [stanza.Document([], text=d.strip()) for d in lines]
    docs = nlp(in_docs)

    def process_doc(doc):
        pairs = set()
        description = dict()
        compound = dict()
        conj = dict()
        matches = sem.process(
            doc,
            "{cpos:ADJ}=adjective <amod=adj {pos:/NN.*|PRP/}=noun",
            "{cpos:ADJ}=adjective >nsubj=adj {pos:/NN.*|PRP/}=noun",
            "{cpos:ADJ}=adjective </acl.*/=adj {pos:/NN.*|PRP/}=noun",
            "{cpos:VERB}=link >xcomp=xcomp {cpos:ADJ}=adjective >obj {pos:/NN.*|PRP/}=noun",
            "{pos:/NN.*|PRP/}=first_noun <compound=compound {pos:/NN.*|PRP/}=second_noun",
            "{cpos:ADJ}=adjective <conj=conj {cpos:ADJ}=second_adjective",
            "{lemma:not}=not <advmod=not {cpos:ADJ}=adjective",
        )
        matches = json_format.MessageToDict(matches)
        # print(matches)
        negations = set()
        for i, result in enumerate(matches["result"]):
            for match in result["result"]:
                if "match" not in match:
                    continue
                for m in match["match"]:
                    match_dict = {}
                    relname = m["reln"][0]["name"]
                    for node in m["node"]:
                        match_dict[node["name"]] = (
                            i,
                            node["matchIndex"] - 1,
                        )  # (sentence_idx, word_idx)
                    if relname == "adj":
                        description[match_dict["adjective"]] = match_dict["noun"]
                    elif relname == "compound":
                        compound[match_dict["second_noun"]] = match_dict["first_noun"]
                    elif relname == "conj":
                        conj[match_dict["adjective"]] = match_dict["second_adjective"]
                    elif relname == "xcomp":
                        description[match_dict["adjective"]] = match_dict["noun"]
                    elif relname == "not":
                        negations.add(match_dict["adjective"])

        for adj in description:
            negation = adj in negations
            pairs.add(
                Pair(
                    get_word(doc, adj),
                    get_compound(doc, compound, description[adj]),
                    negation,
                )
            )
        for adj in conj:
            if conj[adj] not in description:
                continue
            negation = adj in negations or conj[adj] in negations
            pairs.add(
                Pair(
                    get_word(doc, adj),
                    get_compound(doc, compound, description[conj[adj]]),
                    negation,
                )
            )
        return pairs

    with Semgrex() as sem:
        outputs = list(map(process_doc, docs))

    return outputs


if __name__ == "__main__":
    nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse")

    test_batch = [
        "The red dress is beautiful. The blue shoes are ugly.",
        "The red dress is beautiful.",
        "I love the blue shoes and the green hat.",
        "She wore a stylish black jacket with a white shirt.",
        "Lily found herself confused and sad.",
        "This is the wonderful, talented Michelle.",
        "Sam, my sad brother, arrived.",
        "Sam, who was sad, arrived.",
        "Lauren was wonderful.",
        "Lauren was a wonderful student.",
        "The horse rider was lovely.",
        "Lauren and Bob are happy.",
        "I feel happy.",
        "I'm so pretty and witty and bright.",
        "I am not happy.",
    ]

    print(process_batch(nlp, test_batch))
