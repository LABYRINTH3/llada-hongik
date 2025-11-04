# 전처리 - 토크나이징 (Tokenizing) - 김기현

# 0. 기본 설정


# 1. 토크나이저 (Tokenizer) 설정 - 김기현



# 2.1 모델 정의: Masked Diffusion Transformer - 정연욱

# 2.2손실 함수 (Loss Function) 정의 - 정연욱

# 3. 데이터셋 로딩 함수 - 정연욱



# 4. 학습 함수 (Training) - 이태훈
def train_stage(model, dataloader, optimizer, scheduler):
    # 모델을 학습 모드로 설정 - Dropout 활성화
    model.train()
    total_loss = 0.0

    for batch in dataloader:

        input_ids = batch["input_ids"].to(device) # input_ids - 마스크된 토큰이 포함된 문장
        labels = batch["labels"].to(device) # labels - 마스크 위치의 정답 토큰 (정답을 제외한 나머지는 -100)

        # 마스크 위치 정보
            # 원래 토크나이징 하면서 마스크 위치 정보도 넘겨줄까 고려했지만 데이터셋 크기를 고려해 배치 내에서 생성
        mask_pos = labels.ne(-100)
        
        # Forward Propagation 및 Loss 계산
        logits = model(input_ids) # logits: [배치크기 (문장 개수), 시퀀스길이(한문장이 토큰수 고정), 단어사전크기(bert 사전크기)]의 점수
        loss = diffusion_loss(logits, labels, mask_pos)
        
        # Backpropagation
        optimizer.zero_grad() # 옵티마이저 초기화
        loss.backward() # 역전파
        optimizer.step() # 가중치 업데이트
        scheduler.step() # 학습 진행할 수록 lr 감소
        
        total_loss += loss.item()
    return total_loss / len(dataloader)


# 5. 검증 함수 (Validation) - 이태훈

# 6. 커리큘럼 학습 - 이태훈



# 7. 추론 : 텍스트 생성 - 윤희빈 


# 8. 메인 실행 코드 - 윤희빈

