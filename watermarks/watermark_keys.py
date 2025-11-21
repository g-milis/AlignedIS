from .base import AbstractWatermarkKey,AbstractContextCodeExtractor
import torch


class NGramHashing(AbstractWatermarkKey):
    def __init__(self,context_code_extractor:AbstractContextCodeExtractor,ignore_history:bool) -> None:
        self.context_code_extractor=context_code_extractor
        self.ignore_history=ignore_history
        self.cc_history=[]
        
    def __repr__(self) -> str:
        return f"NGramHashing(context_code_extractor={repr(self.context_code_extractor)},ignore_history={self.ignore_history})"
        
    def reset(self,batch_size):
        self.cc_history = [set() for _ in range(batch_size)]

    def generate_key_and_mask(self,input_id,batch_idx):
        context_code=self.context_code_extractor.extract(input_id)
        
        mask = (context_code in self.cc_history[batch_idx])
        if not self.ignore_history:
            self.cc_history[batch_idx].add(context_code)
        return mask, context_code
    
    
class FixedKeySet(AbstractWatermarkKey):
    def __init__(self,private_key_set) -> None:
        self.private_key_set=private_key_set
        
    def __repr__(self) -> str:
        return f"FixedKeySet(len(private_key_set)={len(self.private_key_set)})"
        
    def reset(self,batch_size):
        pass
    
    def generate_key_and_mask(self,input_id,batch_idx):
        selected_key_idx = torch.randint(
            low=0, high=len(self.private_key_set), size=(1,)
        )
        mask=0
        watermark_key=self.private_key_set[selected_key_idx]
        return mask,watermark_key


class KeySequence(AbstractWatermarkKey):
    def __init__(self,key_sequence_len):
        self.key_sequence=self._generate_key_sequence(key_sequence_len)
        self.idx_cnt=None
    
    
    def _generate_key_sequence(self,key_sequence_len):
        import random
        key_set=[]
        seed_set=list(range(42,42+key_sequence_len))
        
        for seed in seed_set:
            random.seed(seed)
            cur_private_key = random.getrandbits(1024).to_bytes(128, "big")
            key_set.append(cur_private_key)
        return key_set

    
    
    def __repr__(self):
        return f"KeySequence(key_sequence_len={len(self.key_sequence)})"
    
    def reset(self,batch_size):
        self.idx_cnt=[0 for _ in range(batch_size)]
        
    def generate_key_and_mask(self, input_id, batch_idx):
        random_offset=torch.randint(low=0,high=len(self.key_sequence)-1,size=(1,))
        
        key_idx=(random_offset+self.idx_cnt[batch_idx])%(len(self.key_sequence))
        watermark_key=self.key_sequence[key_idx]
        
        self.idx_cnt[batch_idx]+=1
        
        mask=0
        return mask,watermark_key
        
        
class PositionHashing(AbstractWatermarkKey):
    def __init__(self):
        self.position_cnt=None
    
    def __repr__(self) -> str:
        return "PositionHashing()"
    
    def reset(self,batch_size):
        self.position_cnt=[0 for _ in range(batch_size)]
        
    def generate_key_and_mask(self, input_id, batch_idx):
        mask=0
        self.position_cnt[batch_idx]+=1
        return mask,str(self.position_cnt[batch_idx]).encode()
    
    

class NoKey(AbstractWatermarkKey):
    def __init__(self) -> None:
        pass
        
    def __repr__(self) -> str:
        return "NoKey()"
    
    def reset(self,batch_size):
        pass
    
    def generate_key_and_mask(self, input_id, batch_idx):
        mask=0
        watermark_key=None
        return mask,watermark_key