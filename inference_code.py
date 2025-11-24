# 7. 추론 : 텍스트 생성 - 윤희빈

def mask_inputs(input_ids, t, mask_token_id, prompt_length):
    B, L = input_ids.shape
    gen_region = torch.zeros_like(input_ids, dtype=torch.bool)
    gen_region[:, prompt_length:] = True  

#mask_input() 안에서 매 스텝마다 랜덤 마스킹 발생 -> 모델이 다시 채워넣는 단어도 달라짐.
    rand = torch.rand((B, L), device=input_ids.device)
    step_mask = rand < t.view(B, 1)
    mask_pos = gen_region & step_mask

    noised = input_ids.clone()
    noised[mask_pos] = mask_token_id
    return noised, mask_pos


def sample_from_model(model, tokenizer, prompt_ids,
                      response_length=50, steps=10, device='cuda'):
    model.eval()
    B, Lp = prompt_ids.shape
    R = response_length

    response = torch.full((B, R),
                          tokenizer.mask_token_id,
                          dtype=torch.long,
                          device=device)

    combined = torch.cat([prompt_ids.to(device), response], dim=1)

    t_schedule = torch.linspace(1.0, 0.0, steps, device=device)

    for step in range(steps):
        t = t_schedule[step].expand(B)
        noised_inputs, mask_pos = mask_inputs(
            combined, t, tokenizer.mask_token_id, Lp
        )
        logits = model(noised_inputs)
        preds = logits.argmax(-1)
        combined[mask_pos] = preds[mask_pos]

    # 생성된 토큰 부분만 반환
    return combined[:, Lp:]
