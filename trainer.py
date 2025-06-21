import torch 
from torch import nn 
import torch.nn.functional as F
from tqdm import tqdm

    
def generate_padding_mask(tensor:torch.tensor, pad_token:int):
    return (tensor == pad_token).float()


def train_batch(encoder:torch.nn.Module,
                  decoder:torch.nn.Module,
                  batch:tuple,
                  loss_fn:torch.nn.Module,
                  optimizer:torch.optim.Optimizer,
                  scaler:torch.cuda.amp.GradScaler,
                  device:torch.device):
    
    encoder.train()
    decoder.train()
    
    img, input_tokens, output_tokens = batch[0].to(device), batch[1].to(device), batch[2].to(device)
    output_mask = generate_padding_mask(output_tokens, pad_token=2)
    
    with torch.cuda.amp.autocast():
        memory = encoder(img, input_tokens)
        decoder_output = decoder(output_tokens, memory, output_mask)
        output_tokens = output_tokens.view(-1).long()
        decoder_output = decoder_output.view(-1, 50)
        loss = loss_fn(decoder_output, output_tokens)
        
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    del img, input_tokens, output_tokens, output_mask, memory, decoder_output
    return loss.item()



def model_trainer(encoder:torch.nn.Module,
                  decoder:torch.nn.Module,
                  dataloader:torch.utils.data.DataLoader,
                  loss_fn:torch.nn.Module,
                  optimizer:torch.optim.Optimizer,
                  scaler:torch.cuda.amp.GradScaler,
                  device:torch.device,
                  epochs:int,
                  encoder_save_path:str=None,
                  decoder_save_path:str=None):
    
    
    for epoch in range(1, epochs+1):
        loss = 0
        with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
            for i, batch in t:
                batch_loss = train_batch(encoder=encoder,
                                         decoder=decoder,
                                         batch=batch,
                                         loss_fn=loss_fn,
                                         optimizer=optimizer,
                                         scaler=scaler,
                                         device=device)

                loss += batch_loss
            
                t.set_description(f'Epoch [{epoch}/{epochs}]')
                t.set_postfix({
                    'Batch Loss' : batch_loss,
                    'Train Loss' : loss/(i+1)
                })
        
                if i%100 == 0 and encoder_save_path:
                    torch.save(obj=encoder.state_dict(), f=encoder_save_path)
                if i%100 == 0 and decoder_save_path:
                    torch.save(obj=decoder.state_dict(), f=decoder_save_path)
                
                if i%100 == 0:
                    img, input_tokens, output_tokens = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    with torch.inference_mode():
                        memory = encoder(img[0].unsqueeze(dim=0), input_tokens[0].unsqueeze(dim=0))
                        current_tokens = torch.tensor([[0]], device=device)
                        i = 1
                        while current_tokens[0][-1] != 1 and i<=64:
                            decoder_output = decoder(current_tokens, memory)[:, -1, :]
                            decoder_output = decoder_output.unsqueeze(dim=1)
                            new_token = torch.argmax(decoder_output, dim=2)
                            current_tokens = torch.cat([current_tokens, new_token], dim=1)
                            i+=1
                    with open('sample_output.txt', 'w') as f:
                        f.write('Expected tokens : ' + str(output_tokens[0].tolist()))
                        f.write('\nPredicted tokens : ' + str(current_tokens[0].tolist()))