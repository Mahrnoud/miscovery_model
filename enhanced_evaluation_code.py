import torch


def generate_text_optimized(model, src, tokenizer, max_length=256, device='cuda',
                            temperature=1.0, top_k=50, top_p=0.95, repetition_penalty=1.05,
                            do_sample=True):
    """
    Memory-optimized text generation function with early stopping
    """
    model.eval()

    # Determine appropriate start token
    if tokenizer.bos_token_id is not None:
        start_token_id = tokenizer.bos_token_id
    elif tokenizer.cls_token_id is not None:
        start_token_id = tokenizer.cls_token_id
    elif tokenizer.pad_token_id is not None:
        start_token_id = tokenizer.pad_token_id
    else:
        start_token_id = 0

    # Early stopping tokens
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)
    if tokenizer.sep_token_id is not None:
        stop_token_ids.append(tokenizer.sep_token_id)

    with torch.no_grad():
        decoder_input = torch.tensor([[start_token_id]], device=device)
        generated_ids = [start_token_id]

        # Use a fixed buffer size for repetition detection
        repetition_window = []
        max_window_size = 8

        # Use autocast for mixed precision
        with torch.amp.autocast('cuda', ):
            for _ in range(max_length):
                try:
                    # Create masks
                    src_mask, tgt_mask = None, None
                    if hasattr(model, 'create_masks'):
                        src_mask, tgt_mask = model.create_masks(src, decoder_input)

                    # Encoder output
                    if hasattr(model, 'encoder'):
                        enc_output = model.encoder(src, src_mask) if src_mask is not None else model.encoder(src)
                    else:
                        enc_output = None

                    # Decoder output
                    if hasattr(model, 'decoder') and enc_output is not None:
                        dec_output = model.decoder(decoder_input, enc_output, src_mask, tgt_mask)
                    else:
                        dec_output = model(src, decoder_input)

                    # Next token logits (only get the last position)
                    next_token_logits = dec_output[:, -1, :]

                    # Apply temperature
                    next_token_logits = next_token_logits / temperature

                    # Apply repetition penalty
                    if repetition_penalty != 1.0:
                        for prev_token in set(generated_ids[-10:]):  # Only check recent tokens
                            next_token_logits[:, prev_token] /= repetition_penalty

                    # Apply top-k filtering
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                        next_token_logits.scatter_(1, top_k_indices, top_k_logits)

                    # Apply nucleus sampling (top-p)
                    if top_p < 1.0 and do_sample:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                        # Remove tokens above threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[:, indices_to_remove] = float('-inf')

                    # Sample or greedy decoding
                    if do_sample:
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)

                    # Add to output
                    decoder_input = torch.cat([decoder_input, next_token], dim=1)
                    next_token_id = next_token.item()
                    generated_ids.append(next_token_id)

                    # Update repetition tracker with fixed size
                    repetition_window.append(next_token_id)
                    if len(repetition_window) > max_window_size:
                        repetition_window.pop(0)

                    # Stop if end token
                    if next_token_id in stop_token_ids:
                        break

                    # Stop for repetition (same token repeated)
                    # if len(repetition_window) >= 4 and len(set(repetition_window[-4:])) == 1:
                    #     break
                    #
                    # Stop for pattern repetition
                    # if len(repetition_window) >= 6:
                    #     half = len(repetition_window) // 2
                    #     if repetition_window[-half:] == repetition_window[-2 * half:-half]:
                    #         break

                except Exception as e:
                    print(f"Error in generation: {e}")
                    break

        # Decode text
        try:
            generated_text = tokenizer.decode(decoder_input[0].tolist(), skip_special_tokens=True)
            return generated_text if generated_text.strip() else "[Empty generation]"
        except Exception as e:
            return f"[Error: {str(e)}]"
