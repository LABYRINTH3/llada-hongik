from transformers import AutoTokenizer

# 전처리 - 토크나이징 (Tokenizing) - 김기현

# 0. 기본 설정
# 학습 하이퍼파라미터
epochs_per_stage = 2
batch_size = 256
lr = 1e-4

tinystory_probs = [0.2, 0.4, 0.6, 0.8]
# 1. 토크나이저 (Tokenizer) 설정 - 이태훈
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
mask_token_id = tokenizer.mask_token_id


# 2.1 모델 정의: Masked Diffusion Transformer - 정연욱
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
import os

# ----------------------------------------
#  1. Transformer 인코더 블록 직접 구현
# ----------------------------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        # (1) Multi-Head Self-Attention
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        # (2) Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        # attention_mask: 1은 유지, 0은 패딩 (BERT와 동일한 규칙)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()  # (1->False, 0->True)

        # (1) Self-Attention + 잔차 연결 + 정규화
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # (2) FFN + 잔차 연결 + 정규화
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


# ----------------------------------------
#  2. 전체 모델 구조 정의
# ----------------------------------------
class MaskedDiffusionTransformer(nn.Module):
    def __init__(self, vocab_size=30522, hidden_dim=512, num_layers=6, num_heads=8, ffn_dim=2048, max_length=512):
        super().__init__()

        # (1) 임베딩: 단어 + 위치 정보 결합
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_length, hidden_dim)

        # (2) 직접 구성한 Transformer 인코더 레이어 6개
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # (3) 출력층: 각 토큰별로 단어 예측 (vocab 크기만큼 확률)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)

        # (1) 입력 임베딩
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        # (2) 인코더 블록 통과
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)

        x = self.norm(x)

        # (3) 출력층
        logits = self.output_layer(x)  # [batch, seq_len, vocab_size]
        return logits

# 장치 설정 (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MaskedDiffusionTransformer().to(device)
print(" 모델 초기화 완료 (device =", device, ")")

# 2.2손실 함수 (Loss Function) 정의 - 정연욱
import torch.nn.functional as F

def diffusion_loss(logits, labels, mask_pos=None):
    """
    logits: [batch, seq_len, vocab_size]
    labels: [batch, seq_len]
    mask_pos: 
    -100인 토큰은 무시하고 손실 계산
    """
    vocab_size = logits.size(-1)
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100
    )
    return loss

# 3. 데이터셋 로딩 함수 - 정연욱
BASE_DIR = "data/tinystories_export"
device = "cuda" if torch.cuda.is_available() else "cpu"
# 3. 데이터셋 로딩 함수 - 정연욱
from datasets import load_from_disk
import os

def load_dataset_by_mask_prob(mask_prob):
    print(f"===== 마스크 {mask_prob}% 데이터셋 로딩 시작 =====")
    # 1. 인풋(mask_prob)을 사용해 경로 생성
    path = f"{BASE_DIR}/train_tok_{mask_prob}"
    # 2. 해당 데이터셋 로드
    if not os.path.exists(path):
        print(f"경고: {path}를 찾을 수 없습니다. 경로를 확인하세요.")
        raise FileNotFoundError(f"데이터셋 경로를 찾을 수 없음: {path}")
    ds = load_from_disk(path)
    print(f"  -> {path} 로드 완료 (샘플 수: {len(ds)})")
    # 3. PyTorch 형식 설정
    ds.set_format(
        type="torch", 
        columns=["input_ids", "labels", "attention_mask"]
    )
    print("  -> PyTorch 텐서 형식으로 설정 완료")
    return ds

# 4. 학습 함수 (Training) - 이태훈
def train_stage(model, dataloader, optimizer, scheduler):
    # 모델을 학습 모드로 설정 - Dropout 활성화
    model.train()
    total_loss = 0.0

    for batch in dataloader:

        input_ids = batch["input_ids"].to(device) # input_ids - 마스크된 토큰이 포함된 문장
        labels = batch["labels"].to(device) # labels - 마스크 위치의 정답 토큰 (정답을 제외한 나머지 -100)

        # 마스크 위치 정보 만들기
            # 원래 토크나이징 하면서 마스크 위치 정보도 넘겨줄까 고려했지만 데이터셋 크기를 고려해 배치 내에서 생성
        mask_pos = labels.ne(-100)
        
        # Forward Propagation 및 Loss 계산
        
        # logits: [배치크기 (문장 개수), 시퀀스길이(한문장이 토큰수 고정), 단어사전크기(bert 사전크기)]의 점수
        # 즉, logits[i][j][k]는 i번째 문장의 j번째 토큰이 k번째 단어일 점수
        logits = model(input_ids)
        loss = diffusion_loss(logits, labels, mask_pos) # 연욱님이 만든 함수
        
        # Backpropagation
        optimizer.zero_grad() # 옵티마이저 초기화
        loss.backward() # 역전파
        optimizer.step() # 가중치 업데이트

        scheduler.step() # 학습 진행할 수록 lr 감소시킴
        
        total_loss += loss.item()
    return total_loss / len(dataloader)


# 5. 검증 함수 (Validation) - 이태훈
def evaluate(model, dataloader, device='cuda'):
    model.eval()
    # 평가 모드 - Dropout 비활성화
    total_loss = 0.0
    
    # 그래디언트 계산 비활성화
    with torch.no_grad():

        # 위 와 동일
        for batch in dataloader:

            input_ids = batch["input_ids"].to(device) 
            labels = batch["labels"].to(device)
            mask_pos = labels.ne(-100)
            
            logits = model(input_ids)
            loss = diffusion_loss(logits, labels, mask_pos)
            
            total_loss += loss.item()
    return total_loss / len(dataloader)


# 6. 커리큘럼 학습 - 이태훈
def curriculum_train():
    # 모델 생성 후 GPU로 이동
    model = MaskedDiffusionTransformer(tokenizer.vocab_size).to(device)
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    print(f" 모델 생성 파라미터 수: {total_params:,}개")
    
    model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Adam 옵티마이저 사용
    
    # 체크포인트 경로
    prev_ckpt = None
    
    dataset_name = "tinystories_masked_p"
    probs = tinystory_probs # 확률 리스트 [0.2, 0.4, 0.6, 0.8]

    # 각 마스킹 확률에 대해 순차적으로 학습
    for p in probs:
        # 전체 데이터셋 로드
        ds_all = load_dataset(dataset_name, p)
        
        # 데이터셋을 학습/검증 세트로 분할
        split = ds_all.train_test_split(test_size=0.1, seed=77)
        ds_train = split["train"] # 학습 데이터셋 (90%)
        ds_validation = split["test"] # 검증 데이터셋 (10%)

        print(f" 학습 데이터: {len(ds_train)}개")
        print(f" 검증 데이터: {len(ds_validation)}개")

        # 학습용 DataLoader
        train_loader = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4, # 코랩 결제하면 cpu 코어수 4개...
            prefetch_factor=4 # 각 워커가 미리 로드하는 배치 수
        )
        
        # 검증용 DataLoader
        val_loader = DataLoader(
            ds_validation,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            prefetch_factor=4
        )
        
        # 이전 단계 체크포인트 로드
        if prev_ckpt:
            model.load_state_dict(torch.load(prev_ckpt, map_location=device))
            print(f" 이전 단계 가중치 로드: {prev_ckpt}")
        
        # 스케줄러 설정
        total_steps = epochs_per_stage * len(train_loader)

        # 스케줄러 설정
        # 허깅페이스 라이브러리 가져옴 - 모델 학습률 동적으로 조절
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
        
        # 학습 루프
        for epoch in range(1, epochs_per_stage + 1):
            print(f"\n Epoch {epoch}/{epochs_per_stage}")
            train_loss = train_stage(model, train_loader, optimizer, scheduler) # 학습 - 이태훈
            val_loss = evaluate(model, val_loader, device) # 검증 함수 - 이태훈

            print(f" Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 체크포인트 저장
        ckpt_path = os.path.join(checkpoint_dir, f"tinystories_mask{int(p*100)}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"\n 체크포인트 저장: {ckpt_path}")

        # 이전 단계 체크포인트 업데이트
        prev_ckpt = ckpt_path
    print(f" 학습 완료!!!!!!!!!!")


# 7. 추론 : 텍스트 생성 - 윤희빈
from transformers import AutoTokenizer
import torch

def mask_inputs(input_ids, t, mask_token_id, prompt_length):
    """
    입력 시퀀스 중 일부를 확률 t에 따라 [MASK]로 변환하는 함수
    """
    B, L = input_ids.shape  # 배치 크기, 시퀀스 길이
    gen_region = torch.zeros_like(input_ids, dtype=torch.bool)
    gen_region[:, prompt_length:] = True  # 프롬프트 이후 구간만 생성 대상

    rand = torch.rand((B, L), device=input_ids.device)
    step_mask = rand < t.view(B, 1)  # 확률 t보다 작은 위치 마스킹
    mask_pos = gen_region & step_mask

    noised = input_ids.clone()
    noised[mask_pos] = mask_token_id  # [MASK] 토큰으로 덮기
    return noised, mask_pos


def sample_from_model(model, tokenizer, prompt_text="Once upon a time", steps=10, response_length=30):
    """
    Diffusion 방식으로 텍스트를 점진적으로 복원하는 함수
    - 처음엔 [MASK]로 가득 채워진 문장에서 시작
    - step마다 일부를 예측값으로 교체하며 점점 완성
    """
    model.eval()
    with torch.no_grad():
        # 프롬프트 토큰화
        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        B, Lp = prompt_ids.shape  # 배치 크기, 프롬프트 길이
        R = response_length       # 생성할 토큰 수

        # 생성 영역을 전부 [MASK]로 초기화
        response = torch.full((B, R), tokenizer.mask_token_id, dtype=torch.long, device=device)
        combined = torch.cat([prompt_ids, response], dim=1)

        # t 값(마스킹 확률)을 1.0 → 0.0으로 점차 줄임
        t_schedule = torch.linspace(1.0, 0.0, steps, device=device)

        print(f"\n[시작 프롬프트]: {prompt_text}\n" + "-" * 60)
        for step, t in enumerate(t_schedule):
            t = t.expand(B)
            # 1 일부를 [MASK]로 다시 덮음
            noised_inputs, mask_pos = mask_inputs(combined, t, tokenizer.mask_token_id, Lp)

            # 2 모델 예측 수행
            logits = model(noised_inputs)
            preds = logits.argmax(-1)  # 가장 확률 높은 단어 선택

            # 3 마스크된 부분만 예측값으로 교체
            combined[mask_pos] = preds[mask_pos]

            # 중간 결과 출력
            current_text = tokenizer.decode(combined[0, Lp:], skip_special_tokens=True)
            print(f"Step {step+1:02d}/{steps} (t={t[0]:.2f}): {current_text}")

        # 최종 결과 반환
        return tokenizer.decode(combined[0, Lp:], skip_special_tokens=True)



# 8. 메인 실행 코드 - 윤희빈
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Diffusion 기반 텍스트 생성 테스트 시작")
    print("=" * 70)

    # ✅ 모델 초기화 (랜덤 상태)
    model = MaskedDiffusionTransformer().to(device)

    generated_text = sample_from_model(
        model=model,
        tokenizer=tokenizer,
        prompt_text="Once upon a time",  # 시작 문장
        steps=5,                         # Diffusion 스텝 수 (많을수록 점진적 복원)
        response_length=40                    # 출력 토큰 최대 길이
    )

    print("\n[최종 생성 결과]")
    print("=" * 70)
    print(generated_text)
    print("=" * 70)


'''
# 데이터 로드 확인 (테스트용)
from datasets import load_from_disk
import os

base_dir = os.path.join(os.getcwd(), "tinystories_export")
test_path = os.path.join(base_dir, "test_tok_60")

print(f"데이터 경로 확인: {test_path}")
ds = load_from_disk(test_path)
print(f"샘플 개수: {len(ds)}")
print("features:", ds.column_names)
print("첫 샘플 예시:")
print({k: ds[0][k][:10] if isinstance(ds[0][k], list) else ds[0][k] for k in ds.column_names})
'''
