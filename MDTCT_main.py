# 전처리 - 토크나이징 (Tokenizing) - 김기현

# 0. 기본 설정


# 1. 토크나이저 (Tokenizer) 설정 - 김기현



# 2.1 모델 정의: Masked Diffusion Transformer - 정연욱
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

class MaskedDiffusionTransformer(nn.Module):
    def __init__(self, vocab_size=30522, hidden_dim=512, num_layers=6, num_heads=8, ffn_dim=2048):
        super().__init__()
        # BERT 설정값
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=ffn_dim
        )

        #Transformer 인코더 (BERT 기반)
        self.encoder = BertModel(config)

        #출력층: 각 토큰별 단어 예측 (vocab 크기만큼 확률)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  #[batch, seq_len, hidden_dim]
        logits = self.output_layer(hidden_states)  #[batch, seq_len, vocab_size]
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
"""
from datasets import load_from_disk, concatenate_datasets
import os

print("모든 train 데이터셋을 로딩합니다...")

base_dir = "/content/drive/MyDrive/tinystories_export" #경로 상대 경로로 수정 필요?
data_paths = [
    f"{base_dir}/train_tok_20",
    f"{base_dir}/train_tok_40",
    f"{base_dir}/train_tok_60",
    f"{base_dir}/train_tok_80"
]

dataset_list = []
for path in data_paths:
    if os.path.exists(path):
        ds = load_from_disk(path) 
        print(f"  -> {path} 로드 완료 (샘플 수: {len(ds)})")
        dataset_list.append(ds)
    else:
        print(f"경고: {path}를 찾을 수 없습니다.")

# 4개의 데이터셋을 하나로 합치기
all_train_data = concatenate_datasets(dataset_list)

print(f"\n모든 데이터셋 통합 완료 (총 샘플 수: {len(all_train_data)})")

# 통합된 전체 데이터셋에 PyTorch 형식 설정
all_train_data.set_format(
    type="torch", 
    columns=["input_ids", "labels", "attention_mask"]
)
print("PyTorch 텐서 형식으로 설정 완료")
"""


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
        logits = model(input_ids) # logits: [배치크기 (문장 개수), 시퀀스길이(한문장이 토큰수 고정), 단어사전크기(bert 사전크기)]의 점수
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



# 7. 추론 : 텍스트 생성 - 윤희빈 


# 8. 메인 실행 코드 - 윤희빈

