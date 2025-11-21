"""NOTE: all functions assume correct range and don't test for validity."""
import re
import torch


def unit_to_token_speechgpt(unit_str):
    return 32000 + torch.tensor([
        int(idx.replace("<", "").replace(">", ""))
        for idx in re.findall('<\d+>', unit_str)
    ])


def token_to_unit_speechgpt(token_sequence):
    return "".join([f"<{idx - 32000}>" for idx in token_sequence])


def unit_to_token_spiritlm(unit_str):
    return 32002 + torch.tensor([
        int(idx.replace("[Hu", "").replace("]", ""))
        for idx in re.findall('\[Hu\d+\]', unit_str)
    ])


def token_to_unit_spiritlm(token_sequence):
    return "".join([f"[Hu{idx - 32002}]" for idx in token_sequence])


if __name__ == "__main__":
    unit_str = "<0><1>dfg<><2><3>fwert,<<<4>"
    tokens = unit_to_token_speechgpt(unit_str)
    print(tokens)
    units_again = token_to_unit_speechgpt(tokens)
    print(units_again)

    unit_str = "Specieses have a cellulose wall and other polysaccharides.' true or false?[Speech][Hu409][Hu143][Hu91][Hu490][Hu7"
    tokens = unit_to_token_spiritlm(unit_str)
    print(tokens)
    units_again = token_to_unit_spiritlm(tokens)
    print(units_again)
