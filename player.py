<<<<<<< HEAD
"""
TransformerPlayer — Magnus Carlsen-style chess player.
Fine-tuned Qwen2.5-0.5B on Magnus Carlsen's Lichess games.

Scores all legal moves by log-probability under the model.
Key optimisation: all legal moves are scored in a single batched
forward pass, so inference is O(1) in the number of legal moves
instead of O(N).  This keeps the tournament clock happy.
"""

from __future__ import annotations
import math
import chess
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

=======
import chess
import random
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

>>>>>>> parent of 49cac54 (transformer model)
from chess_tournament.players import Player


class TransformerPlayer(Player):

    HF_MODEL_ID: str = "Jochemvkem/magnusbot-qwen"

    def __init__(self, name: str = "MagnusBot"):
        super().__init__(name)
        self._model     = None
        self._tokenizer = None
        self._device    = None

    # ------------------------------------------------------------------
    # Lazy model loading (called once on first get_move invocation)
    # ------------------------------------------------------------------
    def _load(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[{self.name}] Loading model on {self._device} ...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.HF_MODEL_ID, trust_remote_code=True
        )
        # Qwen2 models need a pad token
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Use float16 on CUDA, float32 on CPU (avoids NaN on some CPU builds)
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.HF_MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()
        print(f"[{self.name}] Ready.")

    # ------------------------------------------------------------------
    # Batched log-prob scoring
    # ------------------------------------------------------------------
    def _score_moves_batched(self, prompt: str, uci_moves: list) -> list:
        """
        Return a log-probability score for every move in uci_moves with a
        single batched forward pass through the model.

        Strategy
        --------
        For each move we build  full_text = prompt + " " + uci,  pad all
        sequences to the same length (left-padding so that the last token
        positions align), run one forward pass, then read off the
        per-token log-probs that belong to the *move* tokens only.
        """
        import torch
        import torch.nn.functional as F

        tok = self._tokenizer

        # Tokenise prompt once (no BOS to avoid double-counting)
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        n_prompt   = len(prompt_ids)

        # Tokenise each "prompt + move" sequence
        full_ids_list = []
        move_lengths  = []
        for uci in uci_moves:
            full_text = prompt + " " + uci
            ids = tok.encode(full_text, add_special_tokens=False)
            full_ids_list.append(ids)
            move_lengths.append(len(ids) - n_prompt)   # tokens belonging to the move

        # Pad to uniform length (left-pad with pad_token_id)
        max_len   = max(len(ids) for ids in full_ids_list)
        pad_id    = tok.pad_token_id
        input_ids = []
        attn_masks = []
        for ids in full_ids_list:
            pad_len = max_len - len(ids)
            input_ids.append([pad_id] * pad_len + ids)
            attn_masks.append([0] * pad_len + [1] * len(ids))

        input_tensor = torch.tensor(input_ids,  dtype=torch.long).to(self._device)
        attn_tensor  = torch.tensor(attn_masks, dtype=torch.long).to(self._device)

        with torch.no_grad():
            logits    = self._model(input_tensor, attention_mask=attn_tensor).logits
            log_probs = F.log_softmax(logits, dim=-1)

        scores = []
        for b, (ids, mv_len) in enumerate(zip(full_ids_list, move_lengths)):
            score      = 0.0
            move_start = max_len - mv_len  # index of first move token in padded seq
            for t in range(move_start, max_len):
                target_token = input_tensor[b, t].item()
                # The log-prob for token at position t is predicted from position t-1
                score += log_probs[b, t - 1, target_token].item()
            scores.append(score)

        return scores

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def get_move(self, fen: str) -> Optional[str]:
        self._load()

        board       = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        prompt    = f"FEN: {fen}\nMove:"
        uci_moves = [m.uci() for m in legal_moves]

        try:
            scores = self._score_moves_batched(prompt, uci_moves)
        except Exception as e:
            # Graceful fallback: score one by one (slower but safe)
            print(f"[{self.name}] Batched scoring failed ({e}), falling back to sequential.")
            scores = [self._score_move_single(prompt, uci) for uci in uci_moves]

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return uci_moves[best_idx]

    # ------------------------------------------------------------------
    # Sequential fallback (kept for robustness)
    # ------------------------------------------------------------------
    def _score_move_single(self, prompt: str, move_uci: str) -> float:
        """Sum of log-probs of the move tokens conditioned on the prompt."""
        import torch

        full_text  = prompt + " " + move_uci
        prompt_ids = self._tokenizer.encode(prompt + " ", return_tensors="pt").to(self._device)
        full_ids   = self._tokenizer.encode(full_text,    return_tensors="pt").to(self._device)
        n_prompt   = prompt_ids.shape[1]

        with torch.no_grad():
            logits    = self._model(full_ids).logits[0]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        score = 0.0
        for i in range(n_prompt - 1, full_ids.shape[1] - 1):
            score += log_probs[i, full_ids[0, i + 1].item()].item()
        return score
