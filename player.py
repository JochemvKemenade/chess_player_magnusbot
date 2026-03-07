import chess
import random
import re
import sys
import os
import yaml
import torch
from typing import Optional

from huggingface_hub import snapshot_download

try:
    from chess_tournament.players import Player
except ImportError:
    from abc import ABC, abstractmethod
    class Player(ABC):
        def __init__(self, name: str):
            self.name = name
        @abstractmethod
        def get_move(self, fen: str) -> Optional[str]:
            pass


REPO_ID = "Jochemvkem/magnusbot"

# Token IDs as defined in ChessTokenizer
PAD_ID   = 0
START_ID = 1
END_ID   = 2


class TransformerPlayer(Player):
    """
    MagnusBot — custom encoder-decoder Transformer trained on chess positions.

    Downloads weights + scripts from HuggingFace on first use:
        https://huggingface.co/Jochemvkem/magnusbot

    Initializable with only a name, as required by the competition:
        player = TransformerPlayer("MagnusBot")
    """

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    def __init__(
        self,
        name: str = "MagnusBot",
        repo_id: str = REPO_ID,
        max_new_tokens: int = 10,
        retries: int = 5,
    ):
        super().__init__(name)
        self.repo_id        = repo_id
        self.max_new_tokens = max_new_tokens
        self.retries        = retries
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model     = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Lazy loading — downloads the full repo snapshot on first call
    # ------------------------------------------------------------------
    def _load(self):
        if self._model is not None:
            return

        print(f"[{self.name}] Downloading model from '{self.repo_id}'...")
        local_dir = snapshot_download(repo_id=self.repo_id)
        print(f"[{self.name}] Cached at: {local_dir}")

        # Make the repo's scripts/ package importable
        if local_dir not in sys.path:
            sys.path.insert(0, local_dir)

        from scripts.tokenizer    import ChessTokenizer  # noqa: E402
        from scripts.architecture import Transformer     # noqa: E402

        # Read hyper-parameters saved by the trainer
        config_path = os.path.join(local_dir, "opt-configs.yml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Character-level tokenizer — vocab is fixed at construction
        self._tokenizer = ChessTokenizer()

        # Rebuild the exact architecture used during training
        model = Transformer(
            src_vocab_size=self._tokenizer.vocab_size,
            tgt_vocab_size=self._tokenizer.vocab_size,
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            d_ff=config["d_ff"],
            max_seq_length=100,
            dropout=0.0,    # disable dropout at inference time
        )

        weights_path = os.path.join(local_dir, "magnusbot.pth")
        state_dict = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self._model = model
        print(f"[{self.name}] Ready on {self.device}.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _random_legal(self, fen: str) -> Optional[str]:
        try:
            board = chess.Board(fen)
            moves = list(board.legal_moves)
            return random.choice(moves).uci() if moves else None
        except Exception:
            return None

    def _is_legal(self, fen: str, uci: str) -> bool:
        try:
            board = chess.Board(fen)
            return chess.Move.from_uci(uci) in board.legal_moves
        except Exception:
            return False

    def _extract_uci(self, text: str) -> Optional[str]:
        m = self.UCI_REGEX.search(text)
        return m.group(1).lower() if m else None

    # ------------------------------------------------------------------
    # Autoregressive decoding
    # ------------------------------------------------------------------
    def _decode(self, fen: str, sample: bool = False) -> Optional[str]:
        """
        Encode the FEN, then autoregressively decode a move token-by-token.

        Source encoding uses encode(fen, is_target=False) — no START/END wrapping.
        Target decoding starts with <START> (id=1) and stops at <END> (id=2).
        Both behaviours match ChessDataset.__getitem__ exactly.
        """
        tok   = self._tokenizer
        model = self._model

        # Encode FEN as source sequence (no START/END tokens)
        src_ids = tok.encode(fen, is_target=False)
        src = torch.tensor([src_ids], dtype=torch.long, device=self.device)  # (1, S)

        # Target starts with <START>
        tgt_ids = [START_ID]

        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                tgt = torch.tensor([tgt_ids], dtype=torch.long, device=self.device)
                logits  = model(src, tgt)         # (1, T, vocab_size)
                next_lg = logits[0, -1, :]        # (vocab_size,)

                if sample:
                    probs   = torch.softmax(next_lg, dim=-1)
                    top_k   = torch.topk(probs, k=5)
                    choice  = torch.multinomial(top_k.values, num_samples=1)
                    next_id = top_k.indices[choice].item()
                else:
                    next_id = next_lg.argmax().item()

                if next_id == END_ID:
                    break
                tgt_ids.append(next_id)

        # tok.decode skips PAD/START and stops at END automatically
        raw = tok.decode(tgt_ids[1:])
        return self._extract_uci(raw) if raw else None

    # ------------------------------------------------------------------
    # Public API — required by the competition
    # ------------------------------------------------------------------
    def get_move(self, fen: str) -> Optional[str]:
        try:
            self._load()
        except Exception as e:
            print(f"[{self.name}] Model load failed ({e}). Using random fallback.")
            return self._random_legal(fen)

        for attempt in range(1, self.retries + 1):
            try:
                # Attempt 1: greedy (deterministic best guess)
                # Attempts 2+: top-5 sampling (explore alternatives)
                move = self._decode(fen, sample=(attempt > 1))

                if move and self._is_legal(fen, move):
                    return move

            except Exception as e:
                print(f"[{self.name}] Attempt {attempt} error: {e}")

        # Final fallback — random legal move so the player is never disqualified
        return self._random_legal(fen)
