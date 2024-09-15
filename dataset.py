#from typing import Any
import torch
from torch import nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self,  ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # Token -> Number 

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype = torch.int64)
    
    def __len__(self):
        return len(self.ds)
    # Get sentences -> Source and Target -> Tokenize them -> Convert them to ids
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids # Gives Input IDs of each word in the original language sentence --- array
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Need to pad sentence to reach seq_len ---> If not enough words, we will pad the sentence by calculating it ---
        enc_num_padding_token = self.seq_len - len(enc_input_tokens) - 2 # -2 because of SOS and EOS token 
        dec_num_padding_token = self.seq_len - len(dec_input_tokens) -1 # In decoder, we add only the EOS token and not SOS

        # We need to ensure whether the seq_len that we have chosen is enough to represent all the sentences appropriately (they should fit)

        if enc_num_padding_token < 0 or dec_num_padding_token < 0:
            raise ValueError("Sentence is too long. Enter a shorter sentence perhaps to find the result.")
        
        # Building the 3 tensors for the encoder input and decoder input but also for the label. 
        # 1 sentence for encoder, 1 sentence for decoder, 1 sentence as target label -> the thing we want to predict --

        # Input to the encoder found by using torch.cat() to concatenate the tokens --> SOS and EOS also used -->
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_token, dtype = torch.int64)
            ]
        )
        # Notice the difference between decoder and label --->
        # Add only SOS -->
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_token, dtype = torch.int64)
            ]
        )
        # Add only EOS -->
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]* dec_num_padding_token, dtype = torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        # Padding tokens must not participate in self attention mechanism -- we use the mask for it 
        return {
            "encoder_input" : encoder_input, # (seq_len)
            "decoder_input" : decoder_input, # (seq_len)
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # add sequence dim and batch dim later -> (1, 1, seq_len)
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, Seq_len) & (1, seq_len, seq_len)
            "label" : label, # seq_len 
            "src_text" : src_text,
            "tgt_text" : tgt_text

        }

# Did not understand how it is achieving the condition we need in the decoder_mask value -> & ?
def causal_mask(size):
    mask = torch.triu(torch.ones(1,size,size), diagonal = 1).type(torch.int)
    return mask == 0 









