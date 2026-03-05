"""
TransformerPlayer, a Magnus Carlsen style chess player
Fine-tuned Qwen2.5-0.5B on 400k Magnus Carlsen moves

How it works:
It scores all legal moves by log-probability under the model in a single
batched forward pass, then adjusts each score with a set of heuristic
bonuses and penalties.  The move with the highest adjusted score is played.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every legal move receives a final score:

    score = model_log_prob
          + BONUS_CHECKMATE        if the move delivers checkmate
          + BONUS_QUEEN_PROMOTION  if the move promotes to queen
          + BONUS_FREE_CAPTURE     × captured piece value  (undefended pieces)
          + BONUS_SACRIFICE        if captured piece > moving piece in value
          + ENDGAME_WEIGHT         × endgame_heuristic     if in endgame
          - PENALTY_DRAW           if the move leads to a draw
          - PENALTY_HANG           if the move walks our piece into an attack
          - loop_penalty           based on move history and position counts

The model always runs for every move; heuristics only tilt the odds.

Because grandmaster level chess games rarely end in checkmate, there was a lack
of end-game strategy in the training data. In observing the model play chess
(there's a very cool library for visualising) I noted some weird behaviors that
I tried to remove with these heuristics. In my own testing with the chess colab
it can quite reliably beat stockfish_weak!! That was a pretty cool moment.

I've also organised some small tournaments in a friend group who all take the course,
so I hope I win. Anyways good luck you who read this and good luck little magnus bot!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations
import time
import chess
import torch
import torch.nn.functional as F
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from chess_tournament.players import Player
from collections import defaultdict

# ── Time budget ────────────────────────────────────────────────────────────────
MOVE_TIME_BUDGET = 5.0

# ── Heuristic bonuses (added to model score) ──────────────────────────────────
BONUS_CHECKMATE       = 1000.0  # virtually guarantees checkmate is always chosen
BONUS_QUEEN_PROMOTION =   20.0  # strongly prefer promoting to queen
BONUS_FREE_CAPTURE    =    12.0  # multiplied by piece value for undefended captures
BONUS_SACRIFICE       =    4.0  # capturing a more-valuable piece with a lesser one

# ── Heuristic penalties (subtracted from model score) ─────────────────────────
PENALTY_DRAW          =   10.0  # moves that lead to repetition / 50-move draw
PENALTY_HANG          =    5.0  # moves that walk our piece into an attacked square

# ── Endgame weight ─────────────────────────────────────────────────────────────
# The endgame heuristic score (king proximity + pawn advancement + captures) is
# multiplied by this and added to the model score when in an endgame position.
# It blends endgame-specific knowledge into the model score rather than replacing it.
ENDGAME_WEIGHT        =    4

# ── Move history / loop penalties ─────────────────────────────────────────────
HISTORY_DEPTH  = 6
PENALTY_RECENT = 4.0
PENALTY_OLDER  = 2.0

# ── Material values ────────────────────────────────────────────────────────────
PIECE_VALUES: dict[int, int] = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK:   5,
    chess.QUEEN:  9,
    chess.KING:   0,
}


class TransformerPlayer(Player):

    HF_MODEL_ID: str = "Jochemvkem/magnusbot-qwen"

    def __init__(self, name: str = "MagnusBot"):
        super().__init__(name)
        self._model     = None
        self._tokenizer = None
        self._device    = None

        self._position_counts = defaultdict(int)
        self._move_history    = []

    def reset_game(self):
        self._position_counts.clear()
        self._move_history.clear()

    # ──────────────────────────────────────────────────────────────────────────
    # Lazy model loading
    # ──────────────────────────────────────────────────────────────────────────
    def _load(self):
        if self._model is not None:
            return

        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        print(f"[{self.name}] Loading model on {self._device} ...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.HF_MODEL_ID, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = torch.float32 if self._device == "cpu" else torch.float16
        self._model = AutoModelForCausalLM.from_pretrained(
            self.HF_MODEL_ID,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self._device)
        self._model.eval()
        print(f"[{self.name}] Ready.")

    # ──────────────────────────────────────────────────────────────────────────
    # Core inference — batched log-prob scoring
    # ──────────────────────────────────────────────────────────────────────────
    def _score_moves_batched(self, prompt: str, uci_moves: list) -> list[float]:
        tok = self._tokenizer

        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        n_prompt   = len(prompt_ids)

        full_ids_list = []
        move_lengths  = []
        for uci in uci_moves:
            ids = tok.encode(prompt + " " + uci, add_special_tokens=False)
            full_ids_list.append(ids)
            move_lengths.append(len(ids) - n_prompt)

        max_len    = max(len(ids) for ids in full_ids_list)
        pad_id     = tok.pad_token_id
        input_ids  = []
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

        for uci, mv_len in zip(uci_moves, move_lengths):
            assert (max_len - mv_len) > 0, (
                f"Move '{uci}' tokenises to {mv_len} token(s), "
                f"leaving no room before padding boundary (max_len={max_len}). "
                f"Prompt may be too short or move too long."
            )

        scores = []
        for b, (ids, mv_len) in enumerate(zip(full_ids_list, move_lengths)):
            score      = 0.0
            move_start = max_len - mv_len
            for t in range(move_start, max_len):
                score += log_probs[b, t - 1, input_tensor[b, t].item()].item()
            scores.append(score)

        return scores

    # ──────────────────────────────────────────────────────────────────────────
    # Sequential fallback
    # ──────────────────────────────────────────────────────────────────────────
    def _score_move_single(self, prompt: str, move_uci: str) -> float:
        """Fallback if batched scoring raises an exception (e.g. OOM)."""
        full_text  = prompt + " " + move_uci
        prompt_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
        full_ids   = self._tokenizer.encode(
            full_text, add_special_tokens=False, return_tensors="pt"
        ).to(self._device)
        n_prompt = len(prompt_ids)

        with torch.no_grad():
            logits    = self._model(full_ids).logits[0]
            log_probs = F.log_softmax(logits, dim=-1)

        score = 0.0
        for i in range(n_prompt - 1, full_ids.shape[1] - 1):
            score += log_probs[i, full_ids[0, i + 1].item()].item()
        return score

    # ──────────────────────────────────────────────────────────────────────────
    # Heuristic adjustments
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _piece_value(piece: Optional[chess.Piece]) -> int:
        if piece is None:
            return 0
        return PIECE_VALUES.get(piece.piece_type, 0)

    def _heuristic_adjustment(self, board: chess.Board, move: chess.Move) -> float:
        """
        Compute the total heuristic adjustment for a single move.

        Positive values make the move more attractive; negative values less so.
        All signals are summed and returned as a single float to be added to the
        model log-prob score.

        Signals:
          +BONUS_CHECKMATE        — move delivers immediate checkmate
          +BONUS_QUEEN_PROMOTION  — move promotes a pawn to queen
          +BONUS_FREE_CAPTURE × v — undefended enemy piece of value v captured
          +BONUS_SACRIFICE        — captured piece is more valuable than ours
          +ENDGAME_WEIGHT × h     — endgame positional heuristic h
          -PENALTY_DRAW           — move leads to threefold repetition / 50-move
          -PENALTY_HANG           — move places our piece on an attacked square
          -loop_penalty           — move repeats recent history or revisits position
        """
        adjustment = 0.0
        opponent   = not board.turn

        # ── Checkmate bonus ────────────────────────────────────────────────
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()
        if is_mate:
            return BONUS_CHECKMATE  # dominates everything; return early

        # ── Draw penalty ───────────────────────────────────────────────────
        board.push(move)
        if board.is_repetition(count=3) or board.can_claim_fifty_moves():
            adjustment -= PENALTY_DRAW
        board.pop()

        # ── Queen promotion bonus ──────────────────────────────────────────
        if move.promotion == chess.QUEEN:
            adjustment += BONUS_QUEEN_PROMOTION

        # ── Capture bonuses ────────────────────────────────────────────────
        if board.is_capture(move):
            our_piece = board.piece_at(move.from_square)

            # En-passant: the captured pawn is not on to_square.
            if board.is_en_passant(move):
                captured_value = PIECE_VALUES[chess.PAWN]
            else:
                captured_value = self._piece_value(board.piece_at(move.to_square))

            # Free-capture bonus: scale with the value of the undefended piece.
            board.push(move)
            if not board.is_attacked_by(opponent, move.to_square):
                adjustment += BONUS_FREE_CAPTURE * captured_value
            board.pop()

            # Sacrifice bonus: we give up a lesser piece to take a greater one.
            if captured_value > self._piece_value(our_piece):
                adjustment += BONUS_SACRIFICE

        # ── Hang penalty ───────────────────────────────────────────────────
        board.push(move)
        if board.is_attacked_by(opponent, move.to_square):
            adjustment -= PENALTY_HANG
        board.pop()

        # ── Endgame heuristic bonus ────────────────────────────────────────
        if self._is_endgame(board):
            adjustment += ENDGAME_WEIGHT * self._endgame_heuristic(board, move)

        # ── Loop / repetition penalty ──────────────────────────────────────
        adjustment -= self._loop_penalty(move.uci(), board, move)

        return adjustment

    # ──────────────────────────────────────────────────────────────────────────
    # Endgame helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _is_endgame(self, board: chess.Board) -> bool:
        """Return True when 6 or fewer major/minor pieces remain."""
        major_pieces = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
        count = sum(
            len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))
            for pt in major_pieces
        )
        return count <= 6

    def _endgame_heuristic(self, board: chess.Board, move: chess.Move) -> float:
        """
        Positional score for endgame moves. Three components:
          1. Captures    — 10× piece value
          2. King proximity — reward closing in on the enemy king
          3. Pawn advancement — reward pushing pawns toward promotion

        board.turn flips after push(), so 'our' colour is `not board.turn` inside.
        """
        score = 0.0
        board.push(move)

        if board.is_capture(move):
            score += self._piece_value(board.piece_at(move.to_square)) * 10

        our_king_sq   = board.king(not board.turn)
        enemy_king_sq = board.king(board.turn)
        if our_king_sq and enemy_king_sq:
            score += 14 - chess.square_distance(our_king_sq, enemy_king_sq)

        our_color = not board.turn
        for sq in board.pieces(chess.PAWN, our_color):
            rank = chess.square_rank(sq)
            score += (rank if our_color == chess.WHITE else 7 - rank) * 0.5

        board.pop()
        return score

    # ──────────────────────────────────────────────────────────────────────────
    # Loop prevention
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _position_key(board: chess.Board) -> str:
        return board.board_fen()

    def _loop_penalty(self, uci: str, board: chess.Board, move: chess.Move) -> float:
        """
        Penalty from two independent signals:
          1. Move history  — penalises recently repeated UCI move strings,
                             catching A→B→A oscillations.
          2. Position counts — penalises returning to positions seen this game,
                               catching loops that travel different move paths.
        """
        penalty = 0.0

        recent = self._move_history[-HISTORY_DEPTH:]
        if uci in recent[-2:]:
            penalty += PENALTY_RECENT
        elif uci in recent:
            penalty += PENALTY_OLDER

        board.push(move)
        penalty += self._position_counts[self._position_key(board)] * 2.0
        board.pop()

        return penalty

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────
    def get_move(self, fen: str) -> Optional[str]:
        """Return the best UCI move string for the given FEN."""
        t0 = time.time()
        self._load()

        board       = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        self._position_counts[self._position_key(board)] += 1

        uci_moves = [m.uci() for m in legal_moves]

        # ── Model scoring ──────────────────────────────────────────────────
        try:
            scores = self._score_moves_batched(fen, uci_moves)
        except Exception as e:
            print(f"[{self.name}] Batched scoring failed ({e}), falling back to sequential.")
            scores = [self._score_move_single(fen, uci) for uci in uci_moves]

        # ── Heuristic adjustments ──────────────────────────────────────────
        adjusted = [
            score + self._heuristic_adjustment(board, move)
            for score, move in zip(scores, legal_moves)
        ]

        best_idx = max(range(len(adjusted)), key=lambda i: adjusted[i])
        chosen   = uci_moves[best_idx]

        self._move_history.append(chosen)
        print(f"[{self.name}] Move: {chosen} | Time: {time.time() - t0:.2f}s")
        return chosen
