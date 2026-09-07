import re
from typing import Sequence


def prepare_response_for_loss_mask(
    llm_raw_response: str,
    special_tokens: Sequence[str],
    mode: str = "default",
) -> str:
    """Insert loss-mask marker tokens into a generated assistant response."""
    if len(special_tokens) != 2:
        raise ValueError("special_tokens must contain begin and end markers")

    sptk_b, sptk_e = special_tokens
    response = llm_raw_response.replace("<image>", "")
    response = response.replace(sptk_b, "").replace(sptk_e, "")

    if mode == "answer_only":
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match is not None:
            start, end = match.span(1)
            return response[:start] + sptk_b + response[start:end] + sptk_e + response[end:]
    elif mode != "default":
        raise ValueError(f"Unknown loss_mask_mode: {mode}")

    return sptk_b + response + sptk_e
