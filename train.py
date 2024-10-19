import statistics

import torch
from torch.optim import AdamW
from accelerate import Accelerator

from art.data import load_data, train_val_split, normalize_data, create_dataloader
from art.models import create_model

# Set these parameters
data_dir = 'data'
batch_size = 10
timesteps = 40  # Set to None to do entire history
epochs = 10
lr = 1e-5
print_interval = 500
clip_gradient = True
model_params = {
    'hidden_size': 384,
    'vocab_size': 1,
    'action_tanh': False,
    'n_positions': 1024,
    'n_layer': 6,
    'n_head': 6,
    'n_inner': None,
    'resid_pdrop': 0.1,
    'embd_pdrop': 0.1,
    'attn_pdrop': 0.1,
}
seed = 42

torch.manual_seed(seed)
raw_data = load_data(data_dir)
train_indices, val_indices = train_val_split(raw_data['states'])
normalized_train_data, norm_stats = normalize_data(raw_data, train_indices, timesteps=timesteps)
normalized_val_data, _ = normalize_data(raw_data, val_indices, timesteps=timesteps, norm_stats=norm_stats)
train_dataloader = create_dataloader(normalized_train_data, batch_size=batch_size, shuffle=True)
val_dataloader = create_dataloader(normalized_val_data, batch_size=batch_size, shuffle=False)

model = create_model(n_state=normalized_train_data['states'].shape[-1],
                     n_action=normalized_train_data['actions'].shape[-1],
                     max_ep_len=raw_data['timesteps'].shape[-1],
                     **model_params,
                     )
optimizer = AdamW(model.parameters(), lr=lr)
accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=8)
model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader)

@torch.no_grad()
def evaluate(val_dataloader):
    model.eval()

    epoch_losses_action1 = []
    epoch_losses_action2 = []
    epoch_losses_state = []
    epoch_losses = []

    for batch in (val_dataloader):
        states, actions, rtgs, ctgs, attention_mask, timesteps, ix = batch
        with torch.no_grad():
            state_preds, action_preds = model(
              states=states,
              actions=actions,
              returns_to_go=rtgs,
              constraints_to_go=ctgs,
              timesteps=timesteps,
              attention_mask=attention_mask,
              return_dict=False,
            )
        # Mask for actions
        loss_action1 = torch.nn.functional.mse_loss(attention_mask * action_preds[:, :, 0], attention_mask * actions[:, :, 0], reduction='mean') / 2
        loss_action2 = torch.nn.functional.mse_loss(attention_mask * action_preds[:, :, 1], attention_mask * actions[:, :, 1], reduction='mean') / 2

        # Mask for states
        expanded_mask = attention_mask.unsqueeze(-1).expand_as(state_preds)
        loss_state = torch.nn.functional.mse_loss(expanded_mask[:,1:,:] * state_preds[:,:-1,:], expanded_mask[:,1:,:] * states[:,1:,:], reduction='mean')

        loss = loss_action1 + loss_action2 + loss_state

        epoch_losses_action1.append(loss_action1.item())
        epoch_losses_action2.append(loss_action2.item())
        epoch_losses_state.append(loss_state.item())
        epoch_losses.append(loss.item())

    model.train()
    return epoch_losses_action1, epoch_losses_action2, epoch_losses_state, epoch_losses


model.train()
losses_action1, losses_action2, losses_state, losses = [], [], [], []
val_losses_action1, val_losses_action2, val_losses_state, val_losses = [], [], [], []
completed_steps = 0

for epoch in range(epochs):
    epoch_losses_action1, epoch_losses_action2, epoch_losses_state, epoch_losses = [], [], [], []

    print(f'==== Epoch: {epoch + 1} ====')
    for step, batch in enumerate(train_dataloader, start=0):
        states, actions, rtgs, ctgs, attention_mask, timesteps, ix = batch
        state_preds, action_preds = model(
            states=states,
            actions=actions,
            returns_to_go=rtgs,
            constraints_to_go=ctgs,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
        )

        # Mask for actions
        loss_action1 = torch.nn.functional.mse_loss(attention_mask * action_preds[:, :, 0],
                                                    attention_mask * actions[:, :, 0], reduction='mean') / 2
        loss_action2 = torch.nn.functional.mse_loss(attention_mask * action_preds[:, :, 1],
                                                    attention_mask * actions[:, :, 1], reduction='mean') / 2

        # Mask for states
        expanded_mask = attention_mask.unsqueeze(-1).expand_as(state_preds)
        loss_state = torch.nn.functional.mse_loss(expanded_mask[:, 1:, :] * state_preds[:, :-1, :],
                                                  expanded_mask[:, 1:, :] * states[:, 1:, :], reduction='mean')

        loss = loss_action1 + loss_action2 + loss_state

        epoch_losses_action1.append(loss_action1.item())
        epoch_losses_action2.append(loss_action2.item())
        epoch_losses_state.append(loss_state.item())
        epoch_losses.append(loss.item())

        if completed_steps % print_interval == 0:
            accelerator.print(
                {
                    "steps": completed_steps,
                    "loss action1": loss_action1.item(),
                    "loss action2": loss_action2.item(),
                    "loss state": loss_state.item(),
                    "loss total": loss.item(),
                })

        accelerator.backward(loss)
        if clip_gradient:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        completed_steps += 1

    # Epoch complete, print train metrics across entire epoch
    losses_action1.append(statistics.mean(epoch_losses_action1))
    losses_action2.append(statistics.mean(epoch_losses_action2))
    losses_state.append(statistics.mean(epoch_losses_state))
    losses.append(statistics.mean(epoch_losses))
    accelerator.print(
        {
            "epoch": epoch + 1,
            "loss action1": losses_action1[-1],
            "loss action2": losses_action2[-1],
            "loss state": losses_state[-1],
            "loss total": losses[-1],
        })

    # Epoch complete, print val metrics
    val_epoch_losses_action1, val_epoch_losses_action2, val_epoch_losses_state, val_epoch_losses = evaluate(val_dataloader)
    val_losses_action1.append(statistics.mean(val_epoch_losses_action1))
    val_losses_action2.append(statistics.mean(val_epoch_losses_action2))
    val_losses_state.append(statistics.mean(val_epoch_losses_state))
    val_losses.append(statistics.mean(val_epoch_losses))

    accelerator.print(
        {
            "epoch": epoch + 1,
            "loss action1": val_losses_action1[-1],
            "loss action2": val_losses_action2[-1],
            "loss state": val_losses_state[-1],
            "loss total": val_losses[-1],
        })

    accelerator.save_state(f'checkpoints/checkpoint_epoch_{epoch + 1}')