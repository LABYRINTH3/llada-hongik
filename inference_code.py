# 7. 추론 : 텍스트 생성 - 윤희빈

def mask_inputs(input_ids, t, mask_token_id, prompt_length):
    B, L = input_ids.shape
    gen_region = torch.zeros_like(input_ids, dtype=torch.bool)
    # 프롬프트 영역(0~Lp)을 제외한 생성 영역(Lp 이후)만 True
    gen_region[:, prompt_length:] = True  

    # t 확률에 따라 마스킹할 위치를 랜덤으로 선택
    rand = torch.rand((B, L), device=input_ids.device)
    step_mask = rand < t.view(B, 1)
    
    # 생성 영역(gen_region)이면서 랜덤으로 선택된(step_mask) 위치만 마스킹 위치로 확정
    mask_pos = gen_region & step_mask

    # 확정된 위치를 [MASK] 토큰 ID로 대체
    noised = input_ids.clone()
    noised[mask_pos] = mask_token_id
    return noised, mask_pos


def sample_from_model_with_log(model, tokenizer, prompt_ids,
                               response_length=20, # 최대 문장 길이 20으로 설정 (요청 반영)
                               steps=40, device='cuda'):
    model.eval()
    B, Lp = prompt_ids.shape
    R = response_length

    # 생성할 토큰(R)만큼 [MASK] 토큰으로 채워진 초기 응답 생성
    response = torch.full((B, R),
                          tokenizer.mask_token_id,
                          dtype=torch.long,
                          device=device)

    # 프롬프트 + [MASK] 응답 결합
    combined = torch.cat([prompt_ids.to(device), response], dim=1)

    # 노이즈 스케줄 (1.0에서 0.0으로 점진적 감소)
    t_schedule = torch.linspace(1.0, 0.0, steps, device=device)

    print(f"\n🚀 텍스트 생성 정제 과정 시작 (Steps: {steps}, Response Length: {R})")
    print("----------------------------------------------------------------------")
    
    # 초기 프롬프트 텍스트 출력 (요청 반영)
    initial_prompt = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
    print(f"Initial Prompt: '{initial_prompt}'")
    print("----------------------------------------------------------------------")


    for step in range(steps):
        t = t_schedule[step].expand(B)
        
        # 1. 현재 상태를 기반으로 t 확률만큼 랜덤 마스킹
        noised_inputs, mask_pos = mask_inputs(
            combined, t, tokenizer.mask_token_id, Lp
        )
        
        # 2. 마스크된 입력에 대한 모델의 예측 로짓 획득
        logits = model(noised_inputs)
        preds = logits.argmax(-1)
        
        # 3. 정제: 마스크된 위치만 모델의 예측값으로 업데이트
        combined[mask_pos] = preds[mask_pos]
        
        # 4. 매 스텝마다 현재 상태 출력 (t 값 제거 요청 반영)
        current_text = tokenizer.decode(combined[0], skip_special_tokens=True)
        print(f"Step {step+1}/{steps}: {current_text}") 
        
    print("----------------------------------------------------------------------")
    
    # 생성된 토큰 부분만 반환
    return combined[:, Lp:]
