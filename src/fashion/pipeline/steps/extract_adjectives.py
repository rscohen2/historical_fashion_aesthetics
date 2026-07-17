import argparse
import itertools
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import stanza
from google.protobuf import json_format
from stanza.server.semgrex import Semgrex
from tqdm import tqdm

from fashion.distributed import add_distributed_args, run_distributed
from fashion.span_utils import get_overlap_rows_single


@dataclass
class Span:
    start_idx: int
    end_idx: int
    text: str
    corefs: set["Span"] = field(default_factory=set)

    def __hash__(self):
        return hash((self.start_idx, self.end_idx, self.text))

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text


@dataclass
class AdjectivePair:
    adjective: Span
    noun: Span
    negated: bool

    @property
    def corefs(self):
        return self.noun.corefs

    def __hash__(self):
        return hash((self.adjective, self.noun))


@dataclass
class Node:
    name: str
    span: Span

    def __hash__(self):
        return hash(self.span)


@dataclass
class Match:
    nodes: list[Node]
    reln: list[str]


def get_compound(compounds: dict[Span, Span], word: Span) -> Span:
    # build compound string
    output = [word]
    while word in compounds:
        word = compounds[word]
        output.append(word)
    start_tok = output[-1]
    end_tok = output[0]
    start_idx = start_tok.start_idx
    end_idx = end_tok.end_idx
    return Span(
        start_idx,
        end_idx,
        " ".join([word.text for word in output[::-1]]),
        start_tok.corefs | end_tok.corefs,
    )


def create_pairs(
    matches: list[Match], corefs: dict | None = None
) -> set[AdjectivePair]:
    pairs = set()
    description: dict[Span, Span] = {}
    compound: dict[Span, Span] = {}
    conj: dict[Span, Span] = {}

    negations = set()
    for match in matches:
        match_dict: dict[str, Span] = {}
        relname = match.reln[0]
        for node in match.nodes:
            match_dict[node.name] = node.span
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
        noun = get_compound(compound, description[adj])
        pairs.add(
            AdjectivePair(
                adjective=adj,
                noun=noun,
                negated=negation,
            )
        )

    for adj in conj:
        if conj[adj] not in description:
            continue
        negation = adj in negations or conj[adj] in negations
        noun = get_compound(compound, description[conj[adj]])
        pairs.add(
            AdjectivePair(
                adjective=adj,
                noun=noun,
                negated=negation,
            )
        )
    return pairs


class Processor:
    """What to use for parsing the dependency tree."""

    do_coref: bool = False

    def __init__(self, do_coref: bool = False):
        self.do_coref = do_coref

    def match_batch(self, lines: list[str]) -> list[list[Match]]:
        """Process a batch of documents."""
        raise NotImplementedError


class StanzaProcessor(Processor):
    """Use Stanza for parsing the dependency tree."""

    def __init__(self, do_coref: bool = False):
        super().__init__(do_coref)
        self.nlp = stanza.Pipeline(
            "en",
            processors="tokenize,pos,lemma,depparse"
            + (",coref" if self.do_coref else ""),
        )
        self.patterns = [
            # The red dress
            "{cpos:ADJ}=adjective <amod=adj {pos:/NN.*|PRP/}=noun",
            # The dress is beautiful.
            # Lauren was wonderful.
            "{cpos:ADJ}=adjective >nsubj=adj {pos:/NN.*|PRP/}=noun",
            # Sam, who was sad, arrived.
            # "Sam" -> "sad"
            "{cpos:ADJ}=adjective </acl.*/=adj {pos:/NN.*|PRP/}=noun",
            # Lily found herself confused.
            # "Lily" -> "confused"
            "{cpos:VERB}=link >xcomp=xcomp {cpos:ADJ}=adjective >obj {pos:/NN.*|PRP/}=noun",
            # Handle compound nouns
            # The horse rider was lovely
            # "horse rider" -> "lovely"
            "{pos:/NN.*|PRP/}=first_noun <compound=compound {pos:/NN.*|PRP/}=second_noun",
            # Handle conjunctions in adjectives
            # Lily found herself confused and sad.
            # "herself" -> "confused"
            # "herself" -> "sad"
            "{cpos:ADJ}=adjective <conj=conj {cpos:ADJ}=second_adjective",
            # Handle negations
            # I am not happy.
            # "I" -> "happy", negated=True
            "{lemma:not}=not <advmod=not {cpos:ADJ}=adjective",
        ]
        self.semgrex = Semgrex()
        self.semgrex.__enter__()

    def __del__(self):
        if hasattr(self, "semgrex") and self.semgrex is not None:
            self.semgrex.__exit__(None, None, None)
        del self.nlp

    def get_word(self, doc, i, j):
        word = doc.sentences[i].words[j]
        return word

    def get_coref_from_chains(self, doc) -> dict[tuple[int, int], int]:
        corefs = {}
        index = 0
        for sent_id, sentence in enumerate(doc.sentences):
            for word_id, word in enumerate(sentence.words):
                # only take the minimal coref.
                if not self.do_coref:
                    corefs[(sent_id, word_id)] = index
                    index += 1
                    continue
                min_chain = min(
                    word.coref_chains,
                    key=lambda c: len(c.chain.representative_text),
                    default=None,
                )
                if min_chain is None:
                    continue
                corefs[(sent_id, word_id)] = min_chain.chain.index
        return corefs

    def get_coref_spans(
        self, doc, corefs: dict[tuple[int, int], int]
    ) -> dict[int, set[Span]]:
        coref_spans = defaultdict(set)
        for (sent_id, word_id), chain_id in corefs.items():
            word = self.get_word(doc, sent_id, word_id)
            span = Span(
                start_idx=word.start_char,
                end_idx=word.end_char,
                text=word.text,
            )
            coref_spans[chain_id].add(span)
        return coref_spans

    def match_batch(self, lines: list[str]) -> list[list[Match]]:
        in_docs = [stanza.Document([], text=d.strip()) for d in lines]
        docs = self.nlp(in_docs)

        matches: list[list[Match]] = []
        for doc in docs:
            stanza_matches = self.semgrex.process(
                doc,
                *self.patterns,
            )
            stanza_matches = json_format.MessageToDict(stanza_matches)
            doc_matches: list[Match] = []

            for sentence_ind, result in enumerate(stanza_matches["result"]):
                for match in result["result"]:
                    if "match" not in match:
                        continue
                    for m in match["match"]:
                        relname = m["reln"][0]["name"]

                        corefs = self.get_coref_from_chains(doc)
                        coref_spans = self.get_coref_spans(doc, corefs)

                        nodes = []
                        for stanza_node in m["node"]:
                            stanza_word = self.get_word(
                                doc, sentence_ind, stanza_node["matchIndex"] - 1
                            )
                            if (sentence_ind, stanza_node["matchIndex"] - 1) in corefs:
                                node_corefs = coref_spans[
                                    corefs[
                                        (sentence_ind, stanza_node["matchIndex"] - 1)
                                    ]
                                ]
                            else:
                                node_corefs = set()
                            node = Node(
                                name=stanza_node["name"],
                                span=Span(
                                    start_idx=stanza_word.start_char,
                                    end_idx=stanza_word.end_char,
                                    text=stanza_word.text,
                                    corefs=node_corefs,
                                ),
                            )
                            nodes.append(node)

                        doc_matches.append(Match(nodes, [relname]))
            matches.append(doc_matches)
        return matches


def test():
    processor = StanzaProcessor(do_coref=True)
    test_batch = [
        "Lily found herself confused and sad.",
        "The red dress is beautiful. The blue shoes are ugly.",
        "The red dress is beautiful.",
        "I love the blue shoes and the green hat.",
        "She wore a stylish black jacket with a white shirt.",
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
        "Charles' red dress is beautiful, but it is a little dirty. He is sad.",
        # "The dress being worn by the woman is beautiful, but it is a little dirty. She is sad.",
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

    matches = processor.match_batch(test_batch)
    for i, doc_matches in enumerate(matches):
        pairs = create_pairs(doc_matches)
        print(test_batch[i])
        for pair in pairs:
            print(pair)
        print()


def process_file(mentions: pd.DataFrame, output_file: Path, do_coref: bool):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rows = list(mentions.itertuples())

    results = []
    processor = StanzaProcessor(do_coref)
    if output_file.exists():
        print(f"Output file {output_file} already exists. Skipping.")
        return
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
        matches = processor.match_batch(batch_texts)
        pairs = [create_pairs(doc_matches) for doc_matches in matches]

        for mention, pairs in zip(batch, pairs):
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

    # free up memory
    del processor


def main(
    noun_mention_dir: Path,
    output_dir: Path,
    num_processes: int,
    concurrent_processes: int,
    do_coref: bool,
):

    def process(subset: list[dict]):
        print(f"Processing {len(subset)} files.")
        mentions = pd.DataFrame(subset)
        rank = int(os.environ.get("DISTRIBUTED_RANK", "0"))
        process_file(
            mentions,
            output_dir / f"adjectives.{rank}.csv",
            do_coref,
        )

    mentions = pd.concat(
        [
            pd.read_csv(noun_mention_file)
            for noun_mention_file in sorted(list(noun_mention_dir.glob("*.csv")))
        ]
    ).to_dict(orient="records")

    run_distributed(
        process,
        mentions,
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
