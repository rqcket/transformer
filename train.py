import torch
from torch import nn

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from config import get_weights_file_path, get_config # For Preloader -> to restore status of the model --

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
# For splitting the word according to whitespace ---
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import warnings
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

def greedy_decode(model, source, source_mask, tokeinzer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Pre Compute the encoder output and resuse it every token we get from the decoder --
    encoder_output = model.encode(source, source_mask)
    # HOW DO WE DO INFERENCING --
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for the target -- we don't want it to see the future word
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculating the output of the decoder ---

        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the next token --
        prob = model.project(out[:,-1])
        # Selecting the token with max probability ---
        _, next_word = torch.max(prob, dim = 1)
        decoder_input = torch.concat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim = 1)

        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples = 2):
    model.eval()
    count = 0

    # source_texts = []
    # expected = []
    # predicted = []

    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size == 1, "batch_size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            # Comparing the output with the label ---

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            # Converting the tokens to the output sentence 
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # source_texts.append(source_text)
            # expected.append(target_text)
            # predicted.append(model_out_text)

            # Printing on the console -- not good to use print with tqdm loading bar is running --
            print_msg("-"*console_width)
            print_msg(f"SOURCE: {source_text}")
            print_msg(f"TARGET: {target_text}")
            print_msg(f"SOURCE: {model_out_text}")

            if count == num_examples:
                break


# Methods to create tokenizer -- 
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Maps Unknown Tokenizer to [UNK]
        tokenizer = Tokenizer(WordLevel(unk_token = '[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        # PAD used for training the transformer ---
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)

        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

# To load the dataset 

def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split = "train")

    # Build the tokenizer ---
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keeping 90% for training and the other for validation --- 
    train_ds_size = int(0.9*(len(ds_raw)))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # Now creating a dataset -> billiungual dataset.py 
    # here after creating dataset.py

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max Length for source sentence : {max_len_src}")
    print(f"Max Length for target sentence : {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle = True)
    # 1 batch size because we want to process sentence 1 by 1 
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(f" Using Device : {device}")

    Path(config['model_folder']).mkdir(parents = True, exist_ok = True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # TensorBoard -> helps in visualising the loss and the chrats ---

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(params = model.parameters(), lr = config['lr'], eps = 1e-9)
    # Since we also have config that allows us to resume the training, we will implement it. Helps in restoring state of the model and the optimizer -->

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading the model : {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    # Asking the model to not mind the padding bits --- We will be using label smoothing - to make our model be less confident about its decision - helps reduce overfitting 
    loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer_src.token_to_id('[PAD]'), label_smoothing = 0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            model.train()

            encoder_input = batch['encoder_input'].to(device) # (BATCH, SEQ_LEN)
            decoder_input = batch['decoder_input'].to(device) # (BATCH, SEQ_LEN)
            encoder_mask = batch['encoder_mask'].to(device) # (BATCH, 1,1 ,SEQ_LEN)
            decoder_mask = batch['decoder_mask'].to(device) # (BATCH, 1, seq_len ,SEQ_LEN)

            # Run the tensors through the transformer 
            encoder_output = model.encode(encoder_input, encoder_mask) # (Batch ,seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (Batch ,seq_len, d_model)

            proj_output = model.project(decoder_output) # (Batch , seq_len, tgt_vocab_size)

            # To compare the output with the label --

            label = batch['label'].to(device) # (Batch ,seq_len)
            # To compare -- (Batch ,seq_len) and (Batch , seq_len, tgt_vocab_size)

            # (Batch , seq_len, tgt_vocab_size) converted to (Batch*seq_len , tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix({f"loss":f"{loss.item(): 6.3f}"})

            # Log the loss into tensorboard ---

            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()


            global_step += 1 # tensorboard is using global_step to keep track of the loss 
        # save the model, also to resume the progress of the model training --> this is how you save -> need to save optimizer and model dicts 

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'global_step' : global_step
        }, model_filename)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)



















