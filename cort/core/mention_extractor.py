""" Functions for extracting and filtering mentions in documents. """

from collections import defaultdict
import re

from cort.core import mentions
from cort.core import spans


__author__ = 'smartschat'


def extract_system_mentions(document, filter_mentions=True):
    """ 
    Returns:
        list(Mention): the sorted list of extracted system mentions. Includes a
        "dummy mention".
    """
    system_mentions = [mentions.Mention.from_document(m.span, document)
                       for m in document.annotated_mentions]
    system_mentions = [mentions.Mention.dummy_from_document(document)] \
        + system_mentions
    return system_mentions

def post_process_by_head_pos(system_mentions):
    """ Removes mentions whose head has the part-of-speech tag JJ.

    Args:
        system_mentions (list(Mention): A list of system mentions.

    Returns:
        list(Mention): the filtered list of mentions.
    """
    return sorted(
        [mention for mention
            in system_mentions
            if not re.match("^(JJ)",
                            mention.attributes["pos"][
                                mention.attributes["head_index"]])]
    )


def post_process_by_nam_type(system_mentions):
    """ Removes proper name mentions of types QUANTITY, CARDINAL, ORDINAL,
    MONEY and PERCENT.

    Args:
        system_mentions (list(Mention): A list of system mentions.

    Returns:
        list(Mention): the filtered list of mentions.
    """
    return sorted(
        [mention for mention
            in system_mentions
            if mention.attributes["type"] != "NAM" or
            mention.attributes["ner"][mention.attributes["head_index"]] not in
            ["QUANTITY", "CARDINAL", "ORDINAL", "MONEY", "PERCENT"]]
    )


def post_process_weird(system_mentions):
    """ Removes all mentions which are "mm", "hmm", "ahem", "um", "US" or
    "U.S.".

    Args:
        system_mentions (list(Mention): A list of system mentions.

    Returns:
        list(Mention): the filtered list of mentions.
    """
    return sorted(
        [mention for mention
         in system_mentions
         if " ".join(mention.attributes["tokens"]).lower() not in
         ["mm", "hmm", "ahem", "um"]
         and " ".join(mention.attributes["tokens"]) != "US"
         and " ".join(mention.attributes["tokens"]) != "U.S."]
    )


def post_process_pleonastic_pronoun(system_mentions):
    """ Removes pleonastic it and you.

    These are detected via the following heuristics:
        - it: appears in 'it _ _ that' or 'it _ _ _ that'
        - you: appears in 'you know'

    Args:
        system_mentions (list(Mention): A list of system mentions.

    Returns:
        list(Mention): the filtered list of mentions.
    """
    filtered = []

    for mention in system_mentions:
        if " ".join(mention.attributes["tokens"]).lower() == "it":
            context_two = mention.get_context(2)
            context_three = mention.get_context(3)

            if context_two is not None:
                if context_two[-1] == "that":
                    continue

            if context_three is not None:
                if context_three[-1] == "that":
                    continue

        if " ".join(mention.attributes["tokens"]).lower() == "you":
            if mention.get_context(1) == ["know"]:
                continue

        filtered.append(mention)

    return sorted(filtered)


def post_process_same_head_largest_span(system_mentions):
    """ Removes a mention if there exists a larger mention with the same head.

    Args:
        system_mentions (list(Mention): A list of system mentions.

    Returns:
        list(Mention): the filtered list of mentions.
    """
    head_span_to_mention = defaultdict(list)

    for mention in system_mentions:
        head_span_to_mention[mention.attributes["head_span"]].append(
            (mention.span.end - mention.span.begin, mention))

    return sorted([sorted(head_span_to_mention[head_span])[-1][1]
                   for head_span in head_span_to_mention])


def post_process_embedded_head_largest_span(system_mentions):
    """ Removes a mention its head is embedded in another head.

    Args:
        system_mentions (list(Mention): A list of system mentions.

    Returns:
        list(Mention): the filtered list of mentions.
    """
    map_for_heads = {}

    for mention in system_mentions:
        head_span = mention.attributes["head_span"]
        if head_span.end not in map_for_heads:
            map_for_heads[head_span.end] = []

        map_for_heads[head_span.end].append(head_span.begin)

    post_processed_mentions = []

    for mention in system_mentions:
        head_span = mention.attributes["head_span"]
        head_begins = sorted(map_for_heads[head_span.end])
        if head_begins[0] < head_span.begin:
            continue
        else:
            post_processed_mentions.append(mention)

    return sorted(post_processed_mentions)


def post_process_appositions(system_mentions):
    """ Removes a mention its embedded in an apposition.

    Args:
        system_mentions (list(Mention): A list of system mentions.

    Returns:
        list(Mention): the filtered list of mentions.
    """
    appos = [mention for mention
             in system_mentions if mention.attributes["is_apposition"]]

    post_processed_mentions = []

    for mention in system_mentions:
        span = mention.span
        embedded_in_appo = False
        for appo in appos:
            appo_span = appo.span
            if appo_span.embeds(span) and appo_span != span:
                if len(appo.attributes["parse_tree"]) == 2:
                    embedded_in_appo = True
                elif (mention.attributes["parse_tree"] in
                        appo.attributes["parse_tree"]):
                    embedded_in_appo = True

        if mention.attributes["type"] == "PRO" or not embedded_in_appo:
            post_processed_mentions.append(mention)

    return sorted(post_processed_mentions)
