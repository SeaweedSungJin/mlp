#대조학습 으로 MLP 훈련하는 코드
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

# ================= [설정] =================
DATA_DIR = "/data/dataset/mlp"

INPUT_FILE = os.path.join(DATA_DIR, "vectors_student_aligned.pt")
TARGET_FILE = os.path.join(DATA_DIR, "vectors_teacher_aligned.pt")
SAVE_MODEL_PATH = os.path.join(DATA_DIR, "eva_projector_contrastive.pth")

# [핵심] 대조 학습은 배치 사이즈가 클수록 성능이 좋습니다. (Negative가 많아짐)
# V100 8장이면 메모리가 넉넉하니 4096~8192까지 키워도 됩니다.
BATCH_SIZE = 4096  
EPOCHS = 50        
LEARNING_RATE = 1e-3
# =======================================

class EmbeddingProjector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=4096, dropout=0.1):
        super().__init__()
        # 1. MLP 구조 (기존과 동일)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 2. [추가] 학습 가능한 Temperature Parameter (Logit Scale)
        # 초기값 0.07 (ln(1/0.07) = 2.659)
        # CLIP 논문에서 사용된 트릭으로, 모델이 '얼마나 깐깐하게 구분할지'를 스스로 학습합니다.
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x):
        return self.net(x)

def contrastive_loss(student_out, teacher_out, logit_scale):
    """
    양방향(Bi-directional) InfoNCE Loss 계산
    """
    # 1. 정규화 (L2 Normalize) - 코사인 유사도를 위해 필수
    # 방향만 비교하기 위해 길이를 1로 맞춤
    student_out = F.normalize(student_out, dim=-1)
    teacher_out = F.normalize(teacher_out, dim=-1)

    # 2. Temperature 적용 (최대 100으로 클램핑하여 발산 방지)
    logit_scale = logit_scale.clamp(max=4.605) # ln(100) = 4.605
    scale = logit_scale.exp()

    # 3. 유사도 행렬 계산 (Batch Size x Batch Size)
    # logits_per_student: Student가 Teacher들을 보고 맞추기
    # logits_per_teacher: Teacher가 Student들을 보고 맞추기
    logits_per_student = scale * torch.matmul(student_out, teacher_out.t())
    logits_per_teacher = logits_per_student.t()

    # 4. 정답 라벨 (대각선이 정답)
    # 0번 Student의 짝은 0번 Teacher, 1번은 1번...
    batch_size = student_out.shape[0]
    labels = torch.arange(batch_size, device=student_out.device)

    # 5. 양방향 Cross Entropy Loss
    loss_s = F.cross_entropy(logits_per_student, labels)
    loss_t = F.cross_entropy(logits_per_teacher, labels)
    
    # 두 Loss의 평균
    return (loss_s + loss_t) / 2

def main():
    # 1. GPU 설정 (8개 다 쓰기)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"사용 가능한 GPU 개수: {num_gpus}개")

    # 2. 데이터 로드
    print("벡터 파일 로딩 중...")
    try:
        inputs = torch.load(INPUT_FILE)
        targets = torch.load(TARGET_FILE)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {DATA_DIR}")
        return

    print(f"전체 데이터 개수: {inputs.shape[0]}개")
    
    # 데이터셋 생성 (9:1 분할)
    split_idx = int(inputs.shape[0] * 0.9)
    train_ds = TensorDataset(inputs[:split_idx], targets[:split_idx])
    val_ds = TensorDataset(inputs[split_idx:], targets[split_idx:])
    
    # DataLoader
    # num_workers를 늘려 GPU에 데이터를 빠르게 공급
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # 3. 모델 초기화 및 DataParallel 적용
    model = EmbeddingProjector(inputs.shape[1], targets.shape[1])
    
    # [중요] 8개 GPU 병렬 처리
    if num_gpus > 1:
        print(f"DataParallel 활성화: {num_gpus} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # 4. Optimizer & Scheduler
    # DataParallel 사용 시 파라미터 접근 주의 (model.module)
    if isinstance(model, nn.DataParallel):
        params = model.module.parameters()
    else:
        params = model.parameters()
        
    optimizer = optim.AdamW(params, lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 5. 학습 루프
    print(f"\n학습 시작 (Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS})")
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for x, y_teacher in train_loader:
            x, y_teacher = x.to(device, non_blocking=True), y_teacher.to(device, non_blocking=True)
            
            # Forward (Student -> Projected)
            y_pred = model(x)
            
            # Loss 계산 (여기서 Contrastive Loss 호출)
            # DataParallel이 감싸고 있으므로 logit_scale 접근 시 module 필요
            if isinstance(model, nn.DataParallel):
                current_logit_scale = model.module.logit_scale
            else:
                current_logit_scale = model.logit_scale
                
            loss = contrastive_loss(y_pred, y_teacher, current_logit_scale)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        scheduler.step()
            
        # 검증 (Validation)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y_teacher in val_loader:
                x, y_teacher = x.to(device, non_blocking=True), y_teacher.to(device, non_blocking=True)
                
                y_pred = model(x)
                
                if isinstance(model, nn.DataParallel):
                    current_logit_scale = model.module.logit_scale
                else:
                    current_logit_scale = model.logit_scale
                    
                loss = contrastive_loss(y_pred, y_teacher, current_logit_scale)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # 현재 Temperature 값 확인 (학습이 잘 되는지 지표가 됨)
        with torch.no_grad():
            if isinstance(model, nn.DataParallel):
                temp_val = model.module.logit_scale.exp().item()
            else:
                temp_val = model.logit_scale.exp().item()

        # Best Model 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 저장할 때는 DataParallel 껍데기 벗기고 저장
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, SAVE_MODEL_PATH)
            print(f"Epoch [{epoch+1}/{EPOCHS}] Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Temp: {temp_val:.2f} (★ Saved)")
        else:
            if (epoch+1) % 1 == 0:
                 print(f"Epoch [{epoch+1}/{EPOCHS}] Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Temp: {temp_val:.2f}")

    print(f"\n최종 학습 완료! Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()