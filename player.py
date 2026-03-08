import os
import sys
import importlib.util
import logging
import math
import yaml
import chess
import random
import torch
import torch.nn.functional as F

from huggingface_hub import hf_hub_download
from chess_tournament.players import Player


REPO_ID = "Jochemvkem/magnusbot"

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Heuristic bonus weights (in log-prob units; tune between 0–3 to taste)
# ─────────────────────────────────────────────────────────────────────────────
# These are *added* to the model's log-prob score before the argmax, so a
# bonus of +1.0 has roughly the same influence as the model being ~2.7× more
# confident about that move.  Negatives penalise moves.
W_CAPTURE        =  1.2   # capturing any piece
W_CAPTURE_QUEEN  =  1.8   # capturing a queen specifically
W_PROMO          =  2.0   # pawn promotion (any piece)
W_PROMO_QUEEN    =  0.8   # extra bonus on top of W_PROMO for queen promotion
W_CHECK          =  0.8   # move gives check
W_CENTER         =  0.4   # move lands on or attacks a central square (d4/d5/e4/e5)
W_DEVELOP        =  0.5   # first move of a minor piece from back rank (opening)
W_CASTLE         =  1.0   # castling move
W_HANGING        = -1.5   # moving TO a square defended by the opponent
W_BLUNDER_KING   = -2.0   # moving the king to an attacked square (non-castling)
W_REPEAT         = -0.6   # move leads to a position seen before (crude repetition)

# Piece values used for MVV-LVA (most-valuable-victim / least-valuable-attacker)
PIECE_VALUE = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK:   5,
    chess.QUEEN:  9,
    chess.KING:   0,   # kings can't really be captured; included for completeness
}

CENTER_SQUARES = {chess.D4, chess.D5, chess.E4, chess.E5}
NEAR_CENTER    = {chess.C3, chess.C4, chess.C5, chess.C6,
                  chess.D3,                     chess.D6,
                  chess.E3,                     chess.E6,
                  chess.F3, chess.F4, chess.F5, chess.F6}


def _heuristic_bonus(board: chess.Board, move: chess.Move) -> float:
    """
    Return a scalar bonus (possibly negative) for *move* on *board*.

    The bonus is expressed in natural-log units so it lives in the same space
    as the model's log-prob scores.  It is deliberately small relative to a
    typical log-prob sum so the model retains final say on move ordering.
    """
    bonus = 0.0
    to_sq   = move.to_square
    from_sq = move.from_square
    piece   = board.piece_at(from_sq)
    if piece is None:
        return 0.0

    # ── Captures ──────────────────────────────────────────────────────────────
    victim = board.piece_at(to_sq)
    if board.is_en_passant(move):
        bonus += W_CAPTURE          # en-passant is a pawn capture
    elif victim is not None:
        bonus += W_CAPTURE
        # MVV-LVA: scale bonus by (victim value − attacker value / 10)
        # so winning a queen with a pawn scores higher than winning it with a queen
        mvv_lva = PIECE_VALUE[victim.piece_type] - PIECE_VALUE[piece.piece_type] / 10.0
        bonus  += mvv_lva * 0.15
        if victim.piece_type == chess.QUEEN:
            bonus += W_CAPTURE_QUEEN

    # ── Promotion ─────────────────────────────────────────────────────────────
    if move.promotion is not None:
        bonus += W_PROMO
        if move.promotion == chess.QUEEN:
            bonus += W_PROMO_QUEEN

    # ── Castling ──────────────────────────────────────────────────────────────
    if board.is_castling(move):
        bonus += W_CASTLE

    # ── Check ─────────────────────────────────────────────────────────────────
    board.push(move)
    gives_check = board.is_check()
    seen_before = board.is_repetition(2)   # has this position occurred ≥2 times?
    board.pop()

    if gives_check:
        bonus += W_CHECK
    if seen_before:
        bonus += W_REPEAT

    # ── Central control ───────────────────────────────────────────────────────
    if to_sq in CENTER_SQUARES:
        bonus += W_CENTER
    elif to_sq in NEAR_CENTER:
        bonus += W_CENTER * 0.4

    # ── Development (minor pieces off back rank in the first 15 moves) ────────
    if board.fullmove_number <= 15:
        if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            back_rank = chess.BB_RANK_1 if piece.color == chess.WHITE else chess.BB_RANK_8
            if chess.BB_SQUARES[from_sq] & back_rank:
                bonus += W_DEVELOP

    # ── Safety: penalise moving into defended squares ─────────────────────────
    opponent = not piece.color
    if board.is_attacked_by(opponent, to_sq):
        # Moving a high-value piece to an attacked square is especially bad
        bonus += W_HANGING * (PIECE_VALUE[piece.piece_type] / 9.0)

    # ── King safety: penalise walking the king into check (non-castling) ──────
    if piece.piece_type == chess.KING and not board.is_castling(move):
        if board.is_attacked_by(opponent, to_sq):
            bonus += W_BLUNDER_KING

    return bonus


def _load_module_from_path(module_name: str, file_path: str):
    """
    Load a Python module directly from a file path without mutating sys.path.
    This keeps the import fully scoped and avoids global side effects.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TransformerPlayer(Player):
    """
    A chess player that uses a Transformer model downloaded from HuggingFace
    to score and select legal moves.

    The final move score is:

        score(move) = log_prob(move | position)  +  λ · heuristic_bonus(move)

    where λ (heuristic_lambda) controls how much the hand-crafted bonuses
    influence the decision relative to the model.  λ=0 reverts to pure
    model scoring; λ=1 (default) blends both equally in log-prob space.

    Use the factory method `TransformerPlayer.from_hub()` to construct an
    instance.  Direct instantiation via `__init__` expects pre-loaded
    components and is intended for testing or dependency injection.
    """

    def __init__(
        self,
        name: str,
        model: torch.nn.Module,
        tokenizer,
        device: torch.device,
        temperature: float = 1.0,
        heuristic_lambda: float = 1.0,
    ):
        super().__init__(name)
        self.model            = model
        self.tokenizer        = tokenizer
        self.device           = device
        self.temperature      = temperature
        self.heuristic_lambda = heuristic_lambda

    # ─────────────────────────────────────────────────────────────────────────
    # Factory
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_hub(
        cls,
        name: str = "MagnusBot",
        temperature: float = 1.0,
        heuristic_lambda: float = 1.0,
        repo_id: str = REPO_ID,
    ) -> "TransformerPlayer":
        """
        Download scripts, config, and weights from HuggingFace and return a
        fully initialised TransformerPlayer.

        Artifacts are cached locally by `hf_hub_download`; subsequent calls
        are fast when the cache is warm.  Pass `local_files_only=True` (via
        environment variable HF_HUB_OFFLINE=1) to enforce offline use.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Step 1: download scripts from HF ──────────────────────────────────
        tok_path  = hf_hub_download(repo_id=repo_id, filename="scripts/tokenizer.py")
        arch_path = hf_hub_download(repo_id=repo_id, filename="scripts/architecture.py")

        # ── Step 2: import without mutating sys.path ───────────────────────────
        tokenizer_mod = _load_module_from_path("magnusbot.tokenizer", tok_path)

        _TOKENIZER_ALIAS = "tokenizer"
        sys.modules[_TOKENIZER_ALIAS] = tokenizer_mod
        try:
            architecture_mod = _load_module_from_path("magnusbot.architecture", arch_path)
        finally:
            sys.modules.pop(_TOKENIZER_ALIAS, None)

        ChessTokenizer = tokenizer_mod.ChessTokenizer
        Transformer    = architecture_mod.Transformer

        # ── Step 3: download config and weights ───────────────────────────────
        config_path  = hf_hub_download(repo_id=repo_id, filename="opt-configs.yml")
        weights_path = hf_hub_download(repo_id=repo_id, filename="magnusbot.pth")

        with open(config_path) as f:
            settings = yaml.safe_load(f)

        # ── Step 4: build tokenizer ────────────────────────────────────────────
        tokenizer = ChessTokenizer()

        # ── Step 5: build model ────────────────────────────────────────────────
        model = Transformer(
            src_vocab_size = tokenizer.vocab_size,
            tgt_vocab_size = tokenizer.vocab_size,
            d_model        = settings["d_model"],
            num_heads      = settings["num_heads"],
            num_layers     = settings["num_layers"],
            d_ff           = settings["d_ff"],
            max_seq_length = 100,
            dropout        = settings["dropout"],
        ).to(device)

        # ── Step 6: load weights ───────────────────────────────────────────────
        state = torch.load(weights_path, map_location=device, weights_only=True)

        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}

        model.load_state_dict(state)
        model.eval()

        logger.info("[%s] Ready on %s.", name, device)
        print(f"[{name}] Ready on {device}.")

        return cls(
            name=name,
            model=model,
            tokenizer=tokenizer,
            device=device,
            temperature=temperature,
            heuristic_lambda=heuristic_lambda,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Log-prob scoring
    # ─────────────────────────────────────────────────────────────────────────

    def _encode_moves(self, legal_moves: list[str]) -> dict[str, list[int]]:
        """
        Encode each UCI move string into a list of token ids.

        Raises ValueError if any character in a move is absent from the
        tokenizer vocabulary, which would silently corrupt scoring otherwise.
        """
        encoded = {}
        for move in legal_moves:
            tokens = []
            for c in move:
                if c not in self.tokenizer.char_to_int:
                    raise ValueError(
                        f"Character {c!r} in move {move!r} is not in the tokenizer "
                        f"vocabulary. The model may be incompatible with this position."
                    )
                tokens.append(self.tokenizer.char_to_int[c])
            encoded[move] = tokens
        return encoded

    def score_legal_moves(self, src_tensor: torch.Tensor, legal_moves: list[str]) -> dict[str, float]:
        """
        Score each legal move by summing the log-probabilities of its character
        tokens under the model.

        Moves are grouped by token length so a single batched forward pass is
        used per character position within each length group, reducing the total
        number of forward passes from O(n_moves × move_length) to
        O(n_unique_lengths × max_move_length).
        """
        encoded = self._encode_moves(legal_moves)

        by_length: dict[int, list[str]] = {}
        for move, tokens in encoded.items():
            by_length.setdefault(len(tokens), []).append(move)

        scores: dict[str, float] = {}

        with torch.no_grad():
            for length, group in by_length.items():
                tgt_ids   = torch.tensor(
                    [[1] + encoded[m] for m in group], dtype=torch.long
                ).to(self.device)
                src_batch = src_tensor.expand(len(group), -1)

                log_probs = torch.zeros(len(group), device=self.device)

                for i in range(length):
                    output         = self.model(src_batch, tgt_ids[:, :i + 1])
                    step_log_probs = F.log_softmax(output[:, -1, :] / self.temperature, dim=-1)
                    next_token     = tgt_ids[:, i + 1].unsqueeze(1)
                    log_probs     += step_log_probs.gather(1, next_token).squeeze(1)

                for move, lp in zip(group, log_probs.tolist()):
                    scores[move] = lp

        return scores

    # ─────────────────────────────────────────────────────────────────────────
    # Main API
    # ─────────────────────────────────────────────────────────────────────────

    def get_move(self, fen: str) -> str | None:
        board       = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]

        if not legal_moves:
            return None

        encoded_fen = self.tokenizer.encode(fen, is_target=False)
        src_tensor  = torch.tensor(encoded_fen, dtype=torch.long).unsqueeze(0).to(self.device)

        try:
            scores = self.score_legal_moves(src_tensor, legal_moves)
        except (ValueError, RuntimeError) as exc:
            logger.warning(
                "[%s] Falling back to random move due to scoring error: %s",
                self.name, exc,
            )
            return random.choice(legal_moves)

        # ── Blend model log-probs with heuristic bonuses ──────────────────────
        if self.heuristic_lambda != 0.0:
            move_objects = {m.uci(): m for m in board.legal_moves}
            blended: dict[str, float] = {}
            for uci, lp in scores.items():
                bonus = _heuristic_bonus(board, move_objects[uci])
                blended[uci] = lp + self.heuristic_lambda * bonus
            return max(blended, key=blended.get)

        return max(scores, key=scores.get)
