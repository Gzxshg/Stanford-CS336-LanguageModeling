from typing import Iterable, Iterator
import json
import regex as re

class tokenizer:
    def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None=None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.special_tokens_set=set(self.special_tokens)

        if self.special_tokens:
            next_id=max(self.vocab.keys())+1 if self.vocab else 0
            existing_values=list(self.vocab.values())

            for st in self.special_tokens:
                st_bytes=st.encode('utf-8')
                if st_bytes not in existing_values:
                    self.vocab[next_id]=st_bytes
                    existing_values.append(st_bytes)
                    next_id+=1

        self.inv_vocab={v:k for k,v in self.vocab.items()}
        # Create a mapping for merges to their rank (index in the merges list)
        self.merges_rank={pair: i for i, pair in enumerate(self.merges)}

    @classmethod
    def from_files(
            cls,
            vocab_path: str,
            merges_filepath: str,
            special_tokens: list[str] | None=None
    ):
        vocab={}
        with open(vocab_path,'r',encoding='utf-8') as f:
            vocab_raw=json.load(f)
            for k, v in vocab_raw.items():
                vocab[int(k)]=v.encode('utf-8') if isinstance(v, str) else bytes(v)

        merges=[]
        with open(merges_filepath,'r',encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if not line:
                    continue
                parts=line.split(' ')
                if len(parts)>=2:
                    p1=parts[0].encode('utf-8')
                    p2=parts[1].encode('utf-8')
                    merges.append((p1, p2))
        return cls(vocab, merges, special_tokens)

    def _encode_chunk(self,text_bytes: bytes) -> list[int]:
        if not text_bytes:
            return []
        
        tokens=[bytes([b]) for b in text_bytes]

        while len(tokens)>=2:
            min_rank=float('inf')
            best_pair_index=-1

            for i in range(len(tokens)-1):
                pair=(tokens[i], tokens[i+1])
                rank=self.merges_rank.get(pair, float('inf'))
                if rank<min_rank:
                    min_rank=rank
                    best_pair_index=i

            if best_pair_index==-1:
                break

            pair=(tokens[best_pair_index], tokens[best_pair_index+1])
            new_token=pair[0]+pair[1]
            tokens=tokens[:best_pair_index]+[new_token]+tokens[best_pair_index+2:]
        return [self.inv_vocab[token] for token in tokens]
    
    def encode(
            self,
            text: str
    )-> list[int]:
        if not self.special_tokens:
            return self._encode_chunk(text.encode('utf-8'))
        escaped_special=[re.escape(st) for st in self.special_tokens]
        # Implementation for handling special tokens would go here

    def encode_iterable(
            self,
            iterable: Iterable[str]
        ) -> Iterator[list[int]]:
        pass

    def decode(
            self,
            ids: list[int]
    ) -> str:
        pass
    