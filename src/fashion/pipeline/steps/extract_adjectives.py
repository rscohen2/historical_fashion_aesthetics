import argparse
from collections import defaultdict, namedtuple
import itertools
from pathlib import Path

import pandas as pd
import stanza
from google.protobuf import json_format
from stanza.server.semgrex import Semgrex
from tqdm import tqdm

from fashion.distributed import add_distributed_args, run_distributed
from fashion.span_utils import get_overlap_rows_single

Word = namedtuple("Word", ["start_idx", "end_idx", "text"])
Pair = namedtuple("Pair", ["adjective", "noun", "corefs", "negated"])


def process_batch(
    nlp: stanza.Pipeline, lines: list[str], do_coref: bool
) -> list[set[Pair]]:
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
        start_tok = output[-1]
        end_tok = output[0]
        start_idx = get_word(doc, start_tok).start_idx
        end_idx = get_word(doc, end_tok).end_idx
        return Word(
            start_idx,
            end_idx,
            " ".join([get_word(doc, word).text for word in output[::-1]]),
        )

    def get_corefs(doc, compounds, word, coref_dict, coref_spans):
        corefs = set()
        for chain_id in coref_dict[word]:
            corefs.add(chain_id)
        while word in compounds:
            word = compounds[word]
            for chain_id in coref_dict[word]:
                corefs.add(chain_id)
        corefs = sorted(corefs)

        return tuple(
            set(
                [
                    get_word(doc, coref_span)
                    for coref in corefs
                    for coref_span in coref_spans[coref]
                ]
            )
        )

    in_docs = [stanza.Document([], text=d.strip()) for d in lines]
    docs = nlp(in_docs)

    def get_coref_spans(corefs):
        coref_spans = defaultdict(list)
        for (sent_id, word_id), chain_ids in corefs.items():
            for chain_id in chain_ids:
                coref_spans[chain_id].append((sent_id, word_id))
        return coref_spans

    def get_coref_from_chains(doc, do_coref):
        corefs = defaultdict(list)
        index = 0
        for sent_id, sentence in enumerate(doc.sentences):
            for word_id, word in enumerate(sentence.words):
                # only take the minimal coref.
                if not do_coref:
                    corefs[(sent_id, word_id)].append(index)
                    index += 1
                    continue
                min_chain = min(
                    word.coref_chains,
                    key=lambda c: len(c.chain.representative_text),
                    default=None,
                )
                if min_chain is None:
                    continue
                corefs[(sent_id, word_id)].append(min_chain.chain.index)
        return corefs

    def process_doc(doc):
        pairs = set()
        description = dict()
        compound = dict()
        conj = dict()

        corefs = get_coref_from_chains(doc, do_coref)
        coref_spans = get_coref_spans(corefs)

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
                    get_corefs(doc, compound, description[adj], corefs, coref_spans),
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
                    get_corefs(
                        doc, compound, description[conj[adj]], corefs, coref_spans
                    ),
                    negation,
                )
            )
        return pairs

    with Semgrex() as sem:
        outputs = list(map(process_doc, docs))

    return outputs


def test():
    nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse,coref")
    test_batch = [
        # "The red dress is beautiful. The blue shoes are ugly.",
        # "The red dress is beautiful.",
        # "I love the blue shoes and the green hat.",
        # "She wore a stylish black jacket with a white shirt.",
        # "Lily found herself confused and sad.",
        # "This is the wonderful, talented Michelle.",
        # "Sam, my sad brother, arrived.",
        # "Sam, who was sad, arrived.",
        # "Lauren was wonderful.",
        # "Lauren was a wonderful student.",
        # "The horse rider was lovely.",
        # "Lauren and Bob are happy.",
        # "I feel happy.",
        # "I'm so pretty and witty and bright.",
        # "I am not happy.",
        # "Charles' red dress is beautiful, but it is a little dirty. He is sad.",
        #         "The dress being worn by the woman is beautiful, but it is a little dirty. She is sad.",
        #         """He
        # was smoking a cheap cigarette and wore the same soft felt hat he had
        # worn all last winter.""",
        #         """Their clothes and their manners were so fine,
        # and Mrs. Priest IS handsome.""",
        #         """The
        # first (and the worst!) thing that confronted Thea was a suit of clean,
        # prickly red flannel, fresh from the wash."""
        # """Then he
        # walked down Broadway with his hands in his overcoat pockets, wearing a
        # smile which embraced all the stream of life that passed him and the
        # lighted towers that rose into the limpid blue of the evening sky.""",
        #         """The presence of guttural sounds, diacritic aspirations, epenthetic and
        # servile letters in both languages: their antiquity, both having been
        # taught on the plain of Shinar 242 years after the deluge in the seminary
        # instituted by Fenius Farsaigh, descendant of Noah, progenitor of Israel,
        # and ascendant of Heber and Heremon, progenitors of Ireland: their
        # archaeological, genealogical, hagiographical, exegetical, homiletic,
        # toponomastic, historical and religious literatures comprising the works
        # of rabbis and culdees, Torah, Talmud (Mischna and Ghemara), Massor,
        # Pentateuch, Book of the Dun Cow, Book of Ballymote, Garland of Howth,
        # Book of Kells: their dispersal, persecution, survival and revival: the
        # isolation of their synagogical and ecclesiastical rites in ghetto
        # (S. Mary’s Abbey) and masshouse (Adam and Eve’s tavern): the
        # proscription of their national costumes in penal laws and jewish dress
        # acts: the restoration in Chanah David of Zion and the possibility of
        # Irish political autonomy or devolution.""",
        # "In what ultimate ambition had all concurrent and consecutive ambitions\nnow coalesced?\n\nNot to inherit by right of primogeniture, gavelkind or borough English,\nor possess in perpetuity an extensive demesne of a sufficient number\nof acres, roods and perches, statute land measure (valuation £ 42), of\ngrazing turbary surrounding a baronial hall with gatelodge and carriage\ndrive nor, on the other hand, a terracehouse or semidetached villa,\ndescribed as Rus in Urbe or Qui si sana, but to purchase by private\ntreaty in fee simple a thatched bungalowshaped 2 storey dwellinghouse of\nsoutherly aspect, surmounted by vane and lightning conductor, connected\nwith the earth, with porch covered by parasitic plants (ivy or Virginia\ncreeper), halldoor, olive green, with smart carriage finish and neat\ndoorbrasses, stucco front with gilt tracery at eaves and gable, rising,\nif possible, upon a gentle eminence with agreeable prospect from balcony\nwith stone pillar parapet over unoccupied and unoccupyable interjacent\npastures and standing in 5 or 6 acres of its own ground, at such\na distance from the nearest public thoroughfare as to render its\nhouselights visible at night above and through a quickset hornbeam hedge\nof topiary cutting, situate at a given point not less than 1 statute\nmile from the periphery of the metropolis, within a time limit of not\nmore than 15 minutes from tram or train line (e.g., Dundrum, south, or\nSutton, north, both localities equally reported by trial to resemble the\nterrestrial poles in being favourable climates for phthisical subjects),\nthe premises to be held under feefarm grant, lease 999 years, the\nmessuage to consist of 1 drawingroom with baywindow (2 lancets),\nthermometer affixed, 1 sittingroom, 4 bedrooms, 2 servants’ rooms,\ntiled kitchen with close range and scullery, lounge hall fitted\nwith linen wallpresses, fumed oak sectional bookcase containing the\nEncyclopaedia Britannica and New Century Dictionary, transverse obsolete\nmedieval and oriental weapons, dinner gong, alabaster lamp, bowl\npendant, vulcanite automatic telephone receiver with adjacent directory,\nhandtufted Axminster carpet with cream ground and trellis border, loo\ntable with pillar and claw legs, hearth with massive firebrasses and\normolu mantel chronometer clock, guaranteed timekeeper with cathedral\nchime, barometer with hygrographic chart, comfortable lounge settees and\ncorner fitments, upholstered in ruby plush with good springing and sunk\ncentre, three banner Japanese screen and cuspidors (club style, rich\nwinecoloured leather, gloss renewable with a minimum of labour by use of\nlinseed oil and vinegar) and pyramidically prismatic central chandelier\nlustre, bentwood perch with fingertame parrot (expurgated language),\nembossed mural paper at 10/- per dozen with transverse swags of carmine\nfloral design and top crown frieze, staircase, three continuous flights\nat successive right angles, of varnished cleargrained oak, treads\nand risers, newel, balusters and handrail, with steppedup panel dado,\ndressed with camphorated wax: bathroom, hot and cold supply, reclining\nand shower: water closet on mezzanine provided with opaque singlepane\noblong window, tipup seat, bracket lamp, brass tierod and brace,\narmrests, footstool and artistic oleograph on inner face of door:\nditto, plain: servants’ apartments with separate sanitary and hygienic\nnecessaries for cook, general and betweenmaid (salary, rising by\nbiennial unearned increments of £ 2, with comprehensive fidelity\ninsurance, annual bonus (£ 1) and retiring allowance (based on the\n65 system) after 30 years’ service), pantry, buttery, larder,\nrefrigerator, outoffices, coal and wood cellarage with winebin (still\nand sparkling vintages) for distinguished guests, if entertained to\ndinner (evening dress), carbon monoxide gas supply throughout.\n\n",
    ]

    outputs = process_batch(nlp, test_batch, do_coref=True)
    for i, output in enumerate(outputs):
        print(test_batch[i])
        for pair in output:
            negation = " (negated)" if pair.negated else ""
            print(
                f"  Adjective: {pair.adjective.text} ({pair.adjective.start_idx}-{pair.adjective.end_idx}), "
                f"Noun: {pair.noun.text} ({pair.noun.start_idx}-{pair.noun.end_idx}){negation}"
            )
            print("Corefs:", [coref.text for coref in pair.corefs])
        print()


def process_file(noun_mention_file: Path, output_file: Path, do_coref: bool):
    mentions = pd.read_csv(noun_mention_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        print(f"Output file {output_file} already exists. Skipping.")
        return
    rows = list(mentions.itertuples())

    results = []
    nlp = stanza.Pipeline(
        "en", processors="tokenize,pos,lemma,depparse" + (",coref" if do_coref else "")
    )
    batch_size = 8
    for batch in tqdm(
        itertools.batched(rows, batch_size),
        total=(len(rows) // batch_size) + 1,
    ):
        # we do this length check because of James Joyce's Ulysses.
        batch_texts = [
            str(mention.sentence)
            for mention in batch
            if len(str(mention.sentence)) < 1200
        ]
        outputs = process_batch(nlp, batch_texts, do_coref)

        for mention, pairs in zip(batch, outputs):
            if not pairs:
                continue
            pairs = list(pairs)
            for pair in pairs:
                if pair.corefs:
                    span_starts, span_ends = zip(
                        *[(coref.start_idx, coref.end_idx) for coref in pair.corefs]
                    )
                else:
                    span_starts, span_ends = [pair.noun.start_idx], [pair.noun.end_idx]
                overlap = get_overlap_rows_single(
                    (mention.start_idx, mention.end_idx),
                    (span_starts, span_ends),
                    partial=True,
                )
                if not overlap:
                    continue
                results.append(
                    {
                        "mention_id": mention.mention_id,
                        "adjective": pair.adjective.text,
                        "adjective_start_idx": pair.adjective.start_idx,
                        "adjective_end_idx": pair.adjective.end_idx,
                        "noun": pair.noun.text,
                        "noun_start_idx": pair.noun.start_idx,
                        "noun_end_idx": pair.noun.end_idx,
                        "negated": pair.negated,
                        "sentence": mention.sentence,
                        "sentence_start_idx": mention.sentence_start_idx,
                        "sentence_end_idx": mention.sentence_end_idx,
                        "mention_start_idx": mention.start_idx,
                        "mention_end_idx": mention.end_idx,
                        "filename": mention.filename,
                    }
                )

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)


def main(
    noun_mention_dir: Path,
    output_dir: Path,
    num_processes: int,
    concurrent_processes: int,
    do_coref: bool,
):
    def process(subset: list[Path]):
        for noun_mention_file in subset:
            process_file(
                noun_mention_file,
                output_dir / f"{noun_mention_file.stem}.csv",
                do_coref,
            )

    run_distributed(
        process,
        sorted(list(noun_mention_dir.glob("*.csv"))),
        script_path=__file__,
        total_processes=num_processes,
        concurrent_processes=concurrent_processes,
        extra_args=[
            "--noun_mention_dir",
            str(noun_mention_dir),
            "--output_dir",
            str(output_dir),
            "--do_coref" if do_coref else "",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--do_coref",
        "-c",
        action="store_true",
        help="Whether to do coreference resolution.",
    )
    parser.add_argument(
        "--noun_mention_dir",
        type=Path,
        help="Path to the file containing noun mentions.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to the output file where results will be saved.",
    )
    add_distributed_args(parser)
    args = parser.parse_args()
    main(
        args.noun_mention_dir,
        args.output_dir,
        args.num_processes,
        args.concurrent_processes,
        args.do_coref,
    )
